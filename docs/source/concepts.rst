Concepts
========

Parameter Injection
-------------------

.. _PI:

Parameter injection (PI) is the idea of feeding a self-attending transformer with a block of parameters
rather than data from a tensor stream. That is, you
feed a tensor of parameters of fixed width in for the
queries, keys, and/or values of a MHA layer.

.. _API:

**Autocalibration by Parameter Injection** is the concept in which parameter
injection is performed on the key and value input of an attention layer.
Due to the fact that such a layer may be seen as selecting
between several different "directions" present in it's values,
the layer has a catagorical influence on the model's
though process.

The implimentation is found in the :ref:`PISU <PISU>` layer

.. _SPI:

**Summary by Parameter Injection** refers to the process of parameter injection by filling in the "query" input of an attention
layer with predefined parameters. This has several effects. First of
all, order information not encoded in the tensors is lost in this
transaction. Second, the output shape of the layer is now
fixed and predictable. Third, the layer, by virtue of choosign
it's queries, has a tendency to perform summary actions.

The implementation is found in :ref:`PISU <PISU>`

**Transformer Specialization** is an option which is opened
up by *API*. It becomes the case with *API* that a layer can now conclusively
decide whether a word belongs to some sort of abstract catagory.

However, A problem may occur. With this system, it is almost
assuredly possible to reach a local minimum that is not optimal.
Thus, some way of avoiding this is needed.

By providing a large number of parameters and ensembles,
one can create a transformer with many, many ways of looking at
the same few tensors to increase accuracy.

For exchanging information between the ensembles, the
:ref:`EESA <EESA>` layer exists. Also, certain ensemble
capable loss functions also are present.

Context Splitting
-----------------

Context Splitting, (CS) is the concept of splitting
a transformer into several different components, to take
care of relative relations, global relations, and decisions.

It should be noted that to convey information efficiently, all of these
elements must be used.

.. LI:

The **Localization of Information** occurs when a layer
is in charge of relative relations between word items. My implimentation
uses a convolution-like kernel and projects words based on
their relative position. It is found in :ref:`LCSA <LCSA>`

Meanwhile, **Global Strategization** consists of drawing a summary
out of all existing tensor embeddings, then performing some
sort of processing on the summary and using it to condition
the individual word tensors. It is performed by the :ref:`GSPU <GSPU>`