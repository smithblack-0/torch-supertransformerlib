"""

Torch does not normally like to script text
strings, which is a pity. This file helps to
fix that.

"""
import importlib
import io
import os
import pathlib
import sys
import tempfile
import inspect
from typing import Optional, Dict, Any, Union, List, Type, Callable, Tuple

import torch.jit
import torch

class StringScriptContext():
    """
    Introduction
    ############
    The class is a context manager which may be initialized with
    the source string to execute, along with environmental parameters
    such as local and global details. It can then be used to
    torchscript compile such strings as though they were defined in
    a module.
    It has several modes. One can independently compile
    and retrieve just a single entity using the get method.
    after initialization. Alternatively, one can open a context
    using a "with" statement, which will give the user full
    access to everything declared within the string. The user
    can then decide themselves what to compile and when.
    Sharp bits
    ##########
    Do note that the environmental compiler can cause side
    effects if you are not careful. While execution will not
    alter the locals and globals in your root environment,
    it MAY alter things within them.
    For example, this will produce a side effect:
    ```
    list_for_effect = []
    source = \"\"\"
    def cause_side_effect():
        list_for_side_effect.append(5)
    \"\"\"
    with StringScriptContext(source, globals(), locals()) as module:
        pass
    print(list_for_effect[0]) # Prints 5.
    ```
    Be aware thus that although the environment will not
    be directly modified by code in the string, the objects
    within it CAN be.
    Examples
    ########
    **Example: Scripting a single entity and retrieving the result**
    ```
    .. testcode::
        source = \"\"\"
        import torch
        def add_five(input: torch.Tensor)->torch.Tensor:
            return input + 5
        \"\"\"
        scripted_add_five = StringScriptContext(source).get("add_five")
        test_tensor = torch.tensor(0)
        print(scripted_add_five(test_tensor)) #Returns tensor(5)
    ```
    **Example: Scripting by opening a context manager
    ```
    .. testcode::
        source = \"\"\"
        def add_five(input: torch.Tensor)->torch.Tensor:
            return input + 5
        \"\"\"
        add_five = None
        with StringScriptContext(source) as module:
            add_five = torch.jit.script(module.add_five)
    ```
    **Example: Scripting with environmental dependencies**
    ```
    .. testcode::
        def redirect():
            return 4
        source = \"\"\"\
        def get_4():
            return redirect()
        \"\"\"
        with StringScriptContext(source, globals(), locals()) as module:
            get = torch.jit.script(module.get_4)
            print(get()) #Returns 4
    ```
    """
    #Debug info
    # Important things to note for maintainers and troubleshooters:
    #   This function edits sys.module. It will remove the edit later, but be careful here
    #   This func

    def exec_module(self, module):
        """
        NOT USER SERVICEABLE
        Compiles and executes the
        source code underlying the given module
        Modules, as entering this method, are
        literally a raw defined object. This
        method will compile and execute code
        from source, then transfer the novel
        results onto the new module.
        """

        execution_globals = self.globals.copy()
        execution_locals = self.locals.copy()

        source = open(module.__file__).read()
        code = compile(source, filename=module.__file__, mode="exec")
        exec(code, execution_globals, execution_locals)

        novel_globals = {key : value for key, value in execution_globals.items()
        if value not in self.globals.values()}
        novel_locals = {key : value for key, value in execution_locals.items()
        if value not in self.locals.values()}


        for key, value in novel_locals.items():
            setattr(module, key, value)
        for key, value in novel_globals.items():
            setattr(module, key, value)
        return module

    def get_handle(self)-> io.TextIOWrapper:
        """
        Gets a temporary file handle.
        Tries again if the suggested name has a collision in
        the system module attribute.
        """
        handle = None
        while handle is None:
            #Fetch a collision free module name
            _handle = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False)
            path = _handle.name
            name = pathlib.Path(path).stem
            if name not in sys.modules:
                handle = _handle
            else:
                _handle.close()
        return handle

    def get(self, name: str)->Union[torch.jit.ScriptModule, torch.jit.ScriptFunction]:
        """
        Get a particular compiled feature
        out of the source string. Accepts the name
        it was declared as in the original string.
        :param name: The name of the thing to compile
        :return: A script feature
        :raises: AssertionError:
        """
        output = None
        assert isinstance(name, str), "Name must be string"
        with self as module:
            output = getattr(module, name)
            output.__module__ = self.__module__ #Small hack to ensure qualified names resolves.
            output = torch.jit.script(output)
        return output

    def __init__(self,
                 source: str,
                 globals: Optional[Dict[str, Any]] = None,
                 locals: Optional[Dict[str, Any]] = None,
                 retain_file_on_error: bool = False,
                 ):
        """
        :param source: A string of source code to execute.
        :param globals: The globals in the environment. May be none, yielding a blank environment
        :param locals: The locals in the environment. May be none, yielding a blank environment.
        :param retain_file_on_error: Whether or not to retain a file so that the stack trace is easy
                to follow

        """

        if globals is None:
            globals = {}
        if locals is None:
            locals = {}

        self.globals = globals
        self.locals = locals
        self.source = source
        self.retain_file = retain_file_on_error

        self.path = None
        self.name = None
        self.module = None

    def __enter__(self):
        """
        Dumps the source string into a temporary file,
        creates an associated module in a global safe
        environment, then returns the new module
        :return: A active module with any generated features.
        """
        #This is the key function of the class, and
        #as such deserves some extra commentary
        #
        #The basic process for handling a source string
        #is to create a temporary file, dump everything
        #into it, create a temporary module backed by the file,
        #and execute everything, transferring it onto the module
        #
        #



        #Write to the temporary, then close it. It will not delete.
        handle = self.get_handle()
        path = handle.name
        handle.write(self.source)
        handle.close()

        #Form module
        name = pathlib.Path(path).stem
        spec = importlib.util.spec_from_file_location(name, handle.name)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module.__name__] = module #This line is required for inspect to work

        try:
            self.exec_module(module)
        except Exception as err:
            if self.retain_file is False:
                os.remove(path)
            raise err

        self.path = path
        self.name = name
        self.module = module
        return module
    def __exit__(self, exc_type, exc_val, exc_tb):
        #TODO: Add some decent error handling.
        sys.modules.pop(self.name)
        if exc_val is None or self.retain_file is False:
            os.remove(self.path)
        self.path = None
        self.name = None

def get_context(frames_up: int) -> Tuple[dict, dict]:
    """
    Returns the global and local context at the specified number of frames up the stack.

    :param frames_up: The number of frames to look up the stack.
    :return: A tuple of the global and local context at the specified number of frames up the stack.
    """
    frame = inspect.currentframe()
    for i in range(frames_up + 1):
        frame = frame.f_back
    return frame.f_globals, frame.f_locals

def exec_scriptable_string(source: str,
                           feature: str)->Union[Type, Callable]:
    """
    A scripting agent designed to execute a source string
    in which is present scriptable code, and retrieve a particular
    object out of the source. The returned object is an unscripted instance,
    but will operate properly when torchscript goes to script it in
    the future.

    Note that the targetted feature must be scriptable if typed in
    normally for this function to work.

    :param source: The source code to work with
    :param feature: The name of the class or function to retrieve
    :return: The function or class, in it's unscripted format.
    """
    globals, locals = get_context(2)
    with StringScriptContext(source, globals, locals) as module:
        item = getattr(module, feature)
        torch.jit.script(item)
    return item