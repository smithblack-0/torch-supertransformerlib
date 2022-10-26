
# Core

## Kernel

* Extract superposition from linear and turn it into Parameter.
  * Extract into kernel
  * Move validation logic into core
  * Create new error cases
  * Create tests for error cases.

## Functions

* Develop circular padding: DONE
* Develop get stride and point to for when torch updates
* Develop get size

## Errors

* Move primary error into it's own file.
* Make error type change with type call



## Stringwork

* NIT: Rebuild dedent to match textwrap dedent more closely. Particularly, handled \n\n more elegantly


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