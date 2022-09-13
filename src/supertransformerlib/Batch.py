"""

Batching for more advanced text processing engines
becomes a little bit different then blinding feeding
in new tensors one at a time. Rather, there becomes
something of an element of Random Access to the
game as well.

This requires new logic. The logic consists of
a device to put together raw, but unbatched, tensor
entities into special Batch classes which accept
a query tensor and will return the matching information.

This then can be processed more deeply.

"""

