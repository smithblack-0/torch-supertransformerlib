from typing import Optional, List


class ValidationError(Exception):
    """
    An error class for validation problems
    """
    def __init__(self,
                 type: str,
                 reason: str,
                 task: Optional[str] = None
                 ):

        msg = ""
        msg += "A %s error occurred \n" % type
        msg += "The error occurred because: \n\n %s\n" % reason
        if task is not None:
            msg += "This happened while doing: \n %s" % task
        super().__init__(msg)


class StandardizationError(ValidationError):
    """
    Called when something cannot be converted to a
    dynamic_shape tensor for whatever reason
    """
    def __init__(self, reason: str, tasks: Optional[List[str]] = None ):
        type = "StandardizationError"
        super().__init__(type, reason, tasks)