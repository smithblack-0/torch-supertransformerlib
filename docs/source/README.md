

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
respect to words provided. It should be noted that parameter
usage is frequently moderately higher, and signicantly higher
if a naive ensemble is used.

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
* EESA: Use this only when processing an ensemble tensor. It allows
the exchange of data from lower level ensembles to higher level ones, but not vice versa.
* GLSA: Pair this up with LCSA for handling global 
relations between entities.
* GSPU: Use this, with an internal transformer stack, when the model really
needs to be able to reason about the big picture.

### Linear

Linear is a core layer, and a rebuild of torch's
linear layer in an ensemble capable format. See the 
class for details

### Glimpses

The Glimpses package contains a few functions which
are useful for dealing with local operations and reshaping.

In particular, Glimpses contains an operation called "local" 
which is capable of returning a view of a tensor which is 
exactly the same as would be seen from a convolution kernel.

### Loss

Loss contains some experimental loss functions which may
be useful on an ensemble of outputs.