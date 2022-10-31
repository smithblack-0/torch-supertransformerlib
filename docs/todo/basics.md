Basics will contain basic features such as the linear
operation, padding definitions, sampling manipulation, and
other such basic operations.

Basic layers may 

# Basics

* Layers will accept superposition weights when build with them.
* Layers will display wha

## Linear
* Redevelop documentation
  * Expand documentation on linear docstring to possess examples
dealing with superposition
  * Ensure superposition contains example dealing with 1d case
  * Ensure superposition contains example for batch case
  * Ensure superposition contains example dealing with sparse case

# Local 

* Rename documentation to local_kernel not sample
* Rename test features to local_kernel
* Rename test documentation to local_kernel

* Deep documentation
  * Develop documentation involving figures illustrating dilation,
  striding, etc

* Impliment a local kernel layer with clear documentation.
* Hook up the various padding layers/verify they are hooked up
* Ensure error messages are crystal clear.
* 

# Sampler

Sample is designed to draw from some sort of information source info
related to what tensor elements should and should not be included in 
the next processing step. This might occur as in, for example, top-p
or top-k. It then produces a function which will draw from only
these elements so long as the tensor is the same or correct shape.

* Develop closure 
  * Accepts mask
  * Top-p sampling
  * Top-k sampling
  * retention mode
    * mask
    * compress
  * return mode
    * dense
    * sparse

* Develop layer
  * Constructor
    * sample_dims: int
    * top-p: float
    * top-k: int
    * random: float
    * retention_mode: "mask", "compress"
    * return_mode: "dense", "sparse"
    
  * Forward:
    * Generate masks.
    * Merge together logical or.
    * Generate callable closure
    * Return
    
