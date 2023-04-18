# What is this

This is the Background file for working with assistants like chatGPT.
It contains a background summary of what is going on, and can be used
to prime the model for useful action when performing development
Introduction

This document describes a small but highly reusable component
within a larger library designed to facilitate the implementation
of advanced indexing mechanisms, such as Neural Turing Machines (NTM).
Advanced indexing mechanisms often involve stateful components that 
can be challenging to manage. The primary objective of this project 
is to develop a system that enables easy management of stateful information
within such mechanisms.

The motivation behind this project stems from the need for a robust 
and efficient method to handle stateful information in advanced indexing 
systems. By providing a well-structured system to manage state, this 
component aims to simplify the implementation of complex indexing mechanisms
and make them more accessible to users.

The scope and applicability of this system are not limited to NTM-like mechanisms.
Any project using PyTorch that requires some form of "setup" for tensors before
primary processing can benefit from this model. As the setup layer only needs 
to be provided with the required tensors, it offers flexibility and adaptability 
for a wide range of use cases.

In the following sections, we will delve into the details of the state
storage mechanism, configuration and setup process, StateActionLayer,
and other key components that make up this system. By the end of this document,
you will have a thorough understanding of how this component can be employed to
manage stateful information in advanced indexing systems effectively.

# Background

The following is a general summary of what is going on and why. It is
designed to prime a human or AI language model and provide it with what the 
general context of what this project is about, so that it/he/she can assist with coding.

# Objective

In general, the purpose of the code under development is to create a mechanism for 
extendable advanced indexing, similar to that seen under an NTM situation. To do this 
requires a lot of background work, such as keeping track of a stateful memory and
prior state weights. The point is to allow, for example, easy implimention of NTM
and the associated concepts.

In particular, I look for a management system which is capable of setting up in the background
a collection of state information, such as seen in NTM, by creating the layers and updating
a configuration then elegantly preparing a model's state information for usage. It should
then be the case that various layers can act on the state and put information into, or
get information out of, this state.


To summarize, I want to see.

* Some sort of easily passed state collection which all layers interact with
* Individual layers which will do their thing against the state collection, loading things into it,
  pulling things out of it, or shuffling state bits around. These layers may in turn depend on
  state themselves. 
* These layers are usable in theory for NTM, but more generally for advanced indexing processes
* There exists a setup mechanism which will easily and elegantly create the initial state needed to 
  kick off the learning process.
* It should be usable by NTM mechanism

# State Storage Mechanism

Keeping track of a lot of different states by tracking individual tensors is
tricky. As a result, the state information is instead stored in a sort of collection.
The collections are known as "BundleTensors". These are a sort of dictionary-like
object which is distinguished by a dictionary by several primary differences: 

* They are immutable, and .set must be used when doing updates to existing contained tensors,
  and will return a new instance. As a corralary, BundleTensor["tensor_name"] = ... will not work.
* They enforce constaints. If a dimension corrolates with "embedding" on multiple tensors,
    all those tensors must have dimensions of the same length. This is true when updating or setting
* They enforce batch corrolation. If the number of batch dimensions is given as 2, 
  the first two dimensions of all tensors in the collection must have the same shape
* They allow arithmetic. Broadcasting, however, works from the first dimension onward 
  but scalar combinations work without fuss and you can, for instance, superimpose two
  batch tensors together using weights.

Other than this, bundle tensors act as dictionaries containing tensors. The 
init parameters are:

* num_batch_dims: int -> The number of batch dimensions to couple
* tensors: Dict[str, torch.Tensor] -> The various tensors to store together
* constraints: Dict[str, List[str]] -> The various constraints to emplace on the provided tensors.

Note that the constraints work from the last dimension forward. 

### How is the state stored

The state for advanced indexing ntm-like processing is contained within the bundleTensor. This state
will consist of both the memory under processing itself alongside any stateful information
such as prior weights which are needed for the next iteration. For the moment, the stored
information may consist of:

* Memory: Always labeled with the "Memory" keyword in the bundle tensor
  * Logit_Weights: Various persistant weights, in a logit format, which can be softmaxed, sigmoid, or
    otherwise processed to get index weights from prior memory access. An example might
    be "Reader_Logits_1" or "Writer_Logits_0"
* Meta_Weights: Various persistent weights that can be stored away based on existing
  logit weight values, and combined back together to form new ways of focusing on things.

And in general, we can access bundle tensors entries like they are a dictionary

# Configuration and Setup

One thing which needs to occur in just about any state based situation which 
may occur is "setting up" whatever blank or partially filled tensors are needed
in order for further layers to function. For instance, the weights in an NTM
process is needed for the initial focusing round.

Setting this up is required, and a procedure to handle this will be important. As
such, it is important to think about what sort of information is needed when setting
up such a case. 

In general, you could require information from one of several places. What I can
see as important, for right now, is:

* Information external to the model's layers. An example would be tensors 
  containing content data. 
* Information internal to other layers. An example might be the number of heads
  needed based on the configuration.

Additionally, as a matter of sanity, I would like it to be the case

* All the setup can be performed by a single layer passed an appropriate bundle
  of information. The setup should require only one thing passed in on init,
  and perhaps a dict of tensors on forward.

To handle this, I perceive a need for dependency injection and careful design of 
the involved architecture. To handle this, I will use a dependency injection
collection object to collect setup layers and an interface. I will define the interface,
the setup_loader data entity, the setup_config data entity, and the setup latyer


## The SetupCase layer

I will first define the interface for the layers responsible for 
actually setting up a usable tensor. It is a class inhereting from
nn.Module called the SetupCase layer.

This layer is an interface without logic but with prebuilt type requirements
and output requirements. The forward parameters are prebuild to be of type
Dict[str, torch.Tensor] while the expected return will be torch.Tensor. A layer
of this type will be expected to be provided with external information matching
the appropriate tensor, and in turn will return the built information

## The SetupFrame Dataclass

The SetupFrame is a dataclass object which is designed to contain
within it the information needed to integrate a particular SetupCase
into a broader setup mechanism. In particular, it is a object containing

* Layer: The SetupCase layer
* Tensors: A List[str] of names of required tensors. These will be fed in as dict
* Constraints: A List[str] of the constraints to attach to the constructed result

This is used for communication with the broader model, and is expected
to be made while making the setup_case layer

## The SetupConfig Dataclass

A SetupConfig object is an object that should be instanced once before
beginning to build any utility layers. It is a mutable object whose
purpose shall be to make and hold SetupFrames inside an internal dictionary. 

The dictionary will consist of a mapping of state_tensor name : SetupFrame, and
the class will have a method called .add which accepts a tensor name, layer,
list of needed tensors, and list of constraints and stores it away as a new data
frame. 

It will contain reasonable error handling, such as throwing if it sees the same
tensor name twice.

## The Setup Layer

The setup layer is responsible for accepting the injection information
on construction and then performing setup of a blank state tensor when
called upon.

The setup layer will be expected to accept a SetupConfig object. It should then
store that information within itself during construction. The forward method of
the layer must be called with a dictionary of tensors, with the names of the tensors
corrosponding to the required tensors indicated in each SetupFrame. For each 
of the stored SetupFrames, the layer will feed in the required entries as a dictionary,
and store the output as part of a broader state_tensor construction along with the 
constraints. 

Once all layers have run, we make and return a state tensor.

# StateActionLayer

In this architecture, we aim to create a mechanism for handling
stateful information within a model and provide a way for users
to easily attach state initialization behavior when creating a layer.
To achieve this, we introduce the StateLayer concept, which ensures 
that the necessary setup details are consistently available for each 
stateful layer.

The primary goal of the StateLayer is to manage state portions 
of a model and enable state initialization behavior to be easily
attached during the layer's creation. To accomplish this, the 
StateLayer interface expects a SetupConfig data class as its 
first parameter, which contains the required setup specifications. 
To allow usage and extendability, all further parameters are free,
and the implimentee is expected to handle them.

The forward interface of the StateLayer accepts a BundleTensor,
which is the generated state, along with arbitrary additional 
parameters. This design ensures that the configured state tensor
with the defined state information is consistently found
within the same place.

Here is the StateLayer interface:

```python
class StateLayer(nn.Module):
    def __init__(self, setup_config: SetupConfig, *args, **kwargs):
        super().__init__()
        # Initialize with the provided setup_config and any additional parameters.

    def forward(self, bundle_tensor: BundleTensor, *args, **kwargs):
        # Perform the necessary operations on the state using the provided BundleTensor.
        # Return the result, which may not necessarily be a BundleTensor.
```

By using the StateLayer interface, you can create a modular and maintainable 
system for managing stateful information in your model. This design allows for 
easy implementation of state initialization behavior and provides a consistent 
API for working with stateful layers.