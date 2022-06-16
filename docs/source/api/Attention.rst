Attention
=========

.. automodule:: Attention

Standard
--------

The functions below are simply a reimplimentation
of the standard attention mechanisms in torch. The
reason for this reimplimentation is so that they support
native ensembles.

Feedforward
^^^^^^^^^^^

.. autoclass:: FeedForward
    :members:

MultiheadedAttention
^^^^^^^^^^^^^^^^^^^^

.. autoclass:: MultiHeadedAttention
    :members:

Parameter Injection
-------------------

Parameter Injected Memory Unit
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. _PIMU:

.. autoclass:: PIMU
    :members:

Parameter Injected Summary Unit
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. _PISU:

.. autoclass:: PISU
    :members:


Context Splitting
-----------------

Local Context Self Attention
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. _LCSA:

.. autoclass:: LCSA
    :members:

Global Strategic Processing Unit
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. _GSPU:

.. autoclass:: GSPU
    :members:

Ensembles and Specialization
----------------------------

Ensemble Exchange Self Attention
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. _EESA:

.. autoclass:: EESA
    :members:
