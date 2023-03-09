## Feedforward

* stuff

## Multi Headed Attention

* stuff
* sparse multiply
* shortmask sparse multiply.

## Doubleheaded Attention

## Banded Attention

* Stuff
* O(N)

## Mergesort Attention

* Reorders the incoming, potentially enormous, sequence of embeddings
or layers into a more refined sequence
* Does this by merge sort, and scoring the positions of each element
with respect to each opposing element. Superposition then occors. 
* This repeats until all elements are properly merged.
* This acts in O(N log(N)) rather than O(N^2)
* What is evaluated is basically what to insert something after.



## Planning Generative Model

* Plan construction
  * Incorporates prior outputs 
  * Incorporates inputs
  * Traverse available information and collect outputs
  * Finished when ???


Plan usage

* Conditioning which is seen on decoder attention
* Loops a certain number of times then outputs.



Planning:
* Accepts prior plan
* Builds one of output. 
* If not done, builds plan addition
* If done, returns output and current plan.



* Construct plan
  * Incorporate pr
* Plan can get longer if this is judged useful
* Plan is transformed by query into output
* Plan accepts prior plan, and will attempt to draw from it first
* Contribute to output

Loop A: Build / Update plan
Loop B: Draw from plan for output


## text_canvasing

* Keep embeddings in large file. 
* Can draw entire blocks 
* Can set entire blocks??

# Loop