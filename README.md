[![Python test then publish](https://github.com/smithblack-0/torch-supertransformerlib/actions/workflows/python-test-publish.yml/badge.svg?branch=master)](https://github.com/smithblack-0/torch-supertransformerlib/actions/workflows/python-test-publish.yml)


# What is this?

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

## Documentation

See https://torch-supertransformerlib.readthedocs.io/en/latest/ for documentation
