.. supertransformerlib documentation master file, created by
   sphinx-quickstart on Wed Jun 15 16:28:56 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to supertransformerlib's documentation!
===============================================

.. toctree::
   :maxdepth: 2

   README.md

What is this?
===========

This is a collection of layers designed to achieve state-of-the-art results in the field of natural language processing.
It consists of a collection of layers, loss functions,
and utilities which either implement recent transformer discoveries,
or which have no analogue in the current NLP literature. It is built in torch.

The library primarily forcuses around working
with the word embeddings.

It has the following desirable properties when used correctly

* torchscript compatible
* O(N) with respect to words input (configured correctly)
* Ensembles natively supported.

