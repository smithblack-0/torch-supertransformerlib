# License

Copyright 2022 Christopher M O'Quinn

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, commercially publish,
distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

## False Representation of Authorship:

Credit for the development of this library must not be falsely represented as performed by anyone but
    the author and/or maintainers. Any user of this library may not falsely
    represent themselves as having developed the code or concept of this library
    in a commercial or academic environment solely by themselves. 
     
* **Okay**: Our team put together a machine learning model out of open source code with amazing properties.
* **Okay**: Our company customized ML model which will serve all your needs. We will offer support as well.
* **Forbidden**: Our team developed from scratch a library with the following property

## Academic Exception:

Long story short, do not scoop me. I need this
to break into academics.

*Definitions*

* **Parameter injection** (PI) is henceforth the idea of feeding a self-attending transformer with a block of parameters
    rather than data from a tensor stream.
* **Autocalibration by Parameter Injection** is the concept in which parameter 
injection is performed on the key and value input of an attention layer, 
ensuring the possible outputs have fixed direction and length. it is performed
* **Summary by Parameter Injection** refers to the process of 
    parameter injection by filling in the "query" input of an attention
    layer with parameters. 
* **Global-Local Job Division** (GLJD) is henceforth the process
    of taking a transformer in a banded environment
* **Localization of information** Consists of taking a banded transformer
in it's state as a local kernel and injecting positional information based on the index of the kernel.
* **Transformer Specialization** is the idea of using the auto calibration abilities of parameter injection as a method to 
increase the degrees of freedom without loss of generalization error,
particularly by means of ensembles with minor exchange
* **Global Strategy** is the idea of using a PI based
summary to consider the situation at the global level for
a text, irregardless of implicit order information.

*Condition*

The usage of **PI** within the literature of the machine learning context is likely to come out soon, as it is a fairly simple development of the transformer system.
Other such developments I have been aiming for, such as banded processing, certainly have. Nonetheless,
a combination of these developments does have a reasonable condition.

Except for the application and restrictions of local and federal law, the publication, presentation, or academic discussion of a system implimenting at least **four**
of the seven defined concepts and directly inspired by thing library and associated documentation
may not be performed without the notification, citation, and consent of the author, "Christopher M O'Quinn",
to the usage of the concepts in such a manner. 

*Permission and Mentoring*

Should a researcher wish to publish a paper or analysis on such a topic, feel
free to contact me at chrisoquinn.2@gmail.com. I would love to contribute to a 
paper and get some mentoring of some sort.

This condition expires on 1/1/2025, excepting any legal liabilities or
violations which occurred while this condition is operational.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.