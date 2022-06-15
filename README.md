# torch-supertransformerlib


## What is this?

This is a collection of layers designed to achieve state-of-the-art results in the field of natural language processing.
It consists of a collection of layers, loss functions, 
and utilities which either implement recent transformer discoveries,
or which have no analogue in the current NLP literature. 
It is also designed to be fast, scalable, and
torchscript plus ensemble compatible.

## How do I use this?

The library is broken up into several components. These are listed
as follows.

### Attention

The attention library contains the majority of the interesting components.
The items here are layers which impliment a variety of attention and feedforward
mechanisms suitable for some kickass transformers. Notably, every layer possesses
three useful properties.

First, they all possess an 'ensembles' option on initialization, which 
sets the number of parallel layers to setup. This is optional, but quite useful. Ensemble layers are processed completely
in parallel, and must be used by having a tensor of the shape (..., ensemble, words, embeddings), 
vs the standard shape of (..., words, embeddings). It should be noted that if ensembles is not defined,
the system expects the latter shape

Second, each and every layer is torchscript compatible. This is 
required for serious work such as saving to ONNX format, or even 
compiling CUDA with custom kernels. It means you can use torch.jit.script
with little worry.

Third, with the exception of Multiheaded Attention,
every layer listed below impliments a variation of 
transformer existing in an O space of less than O(N^2) with
respect to words provided.

As of 6/14/2022, the layers available are:

* FeedForward: 
* MultiHeadedAttention (MHA)
* Parameter Injection Memory Unit (PIMU)
* Parameter Injection Summary Unit (PISU)
* Local Context Self Attention (LCSA) (banded self attention)
* Ensemble Exchange Self Attention (EESA)
* Global-Local Self Attention (GLSA)
* Global Strategic Processing Unit (GSPU)

Their proper utilization is:

* Feedforward: Use this to make decisions.
* PIMU: Use this when dealing with tasks
which you suspect would best be approached by
subcatagorization.
* PISU: Use this if you need to create a fixed shape
tensor output summarizing global trends in an order
independent manner.
* LCSA: Use this to capture order based contextual information
among nearby tensors. This is a banded attention.
* GLSA: Pair this up with LCSA for handling global 
relations between entities.
* GSPU: Use this, with an internal transformer stack, when the model really
needs to be able to reason about the big picture.

## Glimpse