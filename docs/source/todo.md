
# Core

Should only contain functions.
No layers



## Functions

* Develop circular padding: DONE
* Develop get stride: DONE
* Develop get size: DONE
* Develop top-k sampling function: DONE
* Develop top-p sampling function: DONE

## Stringwork

* NIT: Rebuild dedent to match textwrap dedent more closely. Particularly, handled \n\n more elegantly


# Basics

* Layers will accept superposition weights when build with them.
* Layers will display what their superposition and parallel specs are.

## Move

Move reshape into basics
Move Kernel into basics.

## Linear

Rebuild linear to just accept sp

## Kernel

Mov

## Local

* Create circular local sample case
  * Develop error cases.
  * Develop test cases.

## Sample

* Topk, topp, native
* Dense, sparse
* Setup, draw.


## Superpositions

* Dropout
* Global Connections
* Interconnections
* Weights

# Attention

* Split attention into folder and distribute parts to file
* Redevelop feedforward for superposition.
* Redevelop MHA for superposition.
* Redevelop PISU for superposition.
* Redevelop PIMU for superposition.
* Move adaptive into folder. 

# Superposition

* Develop specification.