"""

Focus masking is a process in which a sequence of masks may be generated which
span the length of a context tensor or similar entity, and which adjust the
strength of attention when called upon.

terms:

* Focus_Database: A bundle of tensors containing stateful information used for analysis.
    Is a BundleTensor. It contains:

    * keys: A ...
    * masks:
    * activity:
    * hidden_state:
"""
import torch
from torch import nn
from supertransformerlib import Core, Basics
from typing import Optional

class Setup(nn.Module):
    """
    Will make a new focus_state based
    on a batch of information, containing all
    the various required parameters and such
    """

class ExtendMaskDatabase(nn.Module):
    """
    Will make and insert a new mask entry into the
    focus_state when called upon, or otherwise update
    the underlying mask mechanics. Importantly, it will
    perform a summary action across items dimensions then
    produce only one update based on the result.
    """
    def __init__(self,
                 embedding_width: int,
                 key_width: int,
                 ensemble_shape: Optional[Core.StandardShapeType] = None,
                 dtype: Optional[torch.dtype] = None,
                 device: Optional[torch.device] = None
                 ):

    def forward(self,
                state: torch.Tensor,
                context: torch.Tensor,
                database: Core.BundleTensor)->Core.BundleTensor:
        """

        :param query: A query which can be used to construct a mask database. Shape can
                     be ... x (ensemble_shape...)
        :param context: A context tensor, for which to construct another mask database entry from. It possesses
                        shape of ... x (ensemble_shape...) x context_items x embedding_width
        :param database: The stateful database with current entries
        :return: A new stateful database with updated entries
        """

        # To create the new database entry, I need to summarize the query,
        # generate a key, perform scoring and activation against the context without
        # completing attention, then store away the results along with a fresh activity
        # probability and a new hidden state.



class TrimMaskDatabase(nn.Module):
    """
    Will let the model reject mask database
    entries as irrelevant or not useful. This
    can be useful under some circumstances.
    """

    # To trim the mask database I need to go ahead and
    # make a trim key out of the hidden state, then attend
    # with the keys under sigmoid. Selected keys have their
    # activity probability reduced.

class ConstructFocusMask(nn.Module):
    """
    Creates, out of the current mask database, a mask which
    will be useful in response to a query or sequence
    of queries. This can be used to adjust attention.
    """
    # This is performed by performing attention with respect
    # to some sort of keys


