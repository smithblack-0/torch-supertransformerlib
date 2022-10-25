
# Core

* Extract superposition from linear and turn it into Parameter.
  * Extract into kernel
  * Move validation logic into core
  * Create new error cases
  * Create tests for error cases.

* Refactor core by moving string utilities into their own section
  * Move dedent, etc into strings
  * Move error cases around

* Develop circular padding
  * Create vector representation of circular padding  
  * Create unit tests for circular padding. based on current implimentation
  * Create sane errors for circular padding.
  * Create circular padding
  * Verify unit test work correctly


* Parameter should be callable with or without superposition definition and
return tensor block
* Refactor core into multiple files.
* Refactor errors to lie by themselves. 

# Basics

* Reformat linear to use superposition kernel.
* Finish rollover local function Finish tests
* Verify errors are sane in local fucntion.
* Verify errors are thorough in local function
* Finish local layer.

# Attention

* Split attention into folder and distribute parts to file
* Redevelop feedforward for superposition.
* Redevelop MHA for superposition.
* Redevelop PISU for superposition.
* Redevelop PIMU for superposition.
* Move adaptive into folder. 

# Superposition

* Develop specification.