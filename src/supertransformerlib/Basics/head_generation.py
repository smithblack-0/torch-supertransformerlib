from typing import Optional

import torch
from torch import nn
from supertransformerlib import Core
from supertransformerlib.Basics import linear


class MakeHead(nn.Module):
    """
    A helper layer, this will make a head that is
    usable for an attention mechanism or similar
    by reshaping the last dimension to possess
    additional dimensions or by projecting the last
    dimension to possess additional dimensions.
    """
    def __init__(self,
                 d_model: int,
                 heads: int,
                 mode: str = "project",
                 parallel: Optional[Core.StandardShapeType] = None,
                 dtype: Optional[torch.dtype] = None,
                 device: Optional[torch.device] = None,
                 ):
        super().__init__()

        self.d_model = d_model
        self.heads = heads
        self.mode = mode
        self.parallel = parallel
        self.head_width = d_model // heads

        task = "Running head constructor"
        if self.head_width < 1:
            msg = """\
            The given d_model and given heads parameter are not
            compatible. When d_model is divided by heads, the 
            result is less than one.
            """
            msg = Core.dedent(msg)
            raise Core.Errors.ValidationError("Validation", msg, task)

        if mode == "reshape":
            if d_model % heads != 0:
                msg = """\
                The given d_model and given heads are incompatible with 
                mode 'reshape': It is the case that d_model does not 
                divide cleanly by heads, and so a reshape is not possible
                """
                msg = Core.dedent(msg)
                raise Core.Errors.ValidationError("Validation", msg, task)
            self.transform = Core.Reshape(d_model, [heads, self.head_width])
        elif mode == "project":
            self.transform = linear.Linear(d_model, [heads, self.head_width], parallel,
                                           dtype=dtype, device=device)
        else:
            raise ValueError(f"Invalid mode '{mode}', expected 'project' or 'reshape'")

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.mode == "reshape":
            tensor = self.transform(tensor)
        elif self.mode == "project":
            tensor = tensor.movedim(-2, 0)
            tensor = self.transform(tensor)
            tensor = tensor.movedim(0, -3)
        return tensor


class MergeHeads(nn.Module):
    """
    A helper layer. This layer will
    take in a tensor with attention heads
    on it and merge the heads back together
    either by reshaping or using a linear merge.
    """
    def __init__(self,
                 d_model: int,
                 heads: int,
                 mode: str = "linear",
                 parallel: Optional[torch.Tensor] = None,
                 dtype: Optional[torch.dtype] = None,
                 device: Optional[torch.device] = None,
                 ):
        super().__init__()

        self.d_model = d_model
        self.heads = heads
        self.mode = mode
        self.head_width = d_model // heads

        task = "Running head constructor"
        if self.head_width < 1:
            msg = """\
            The given d_model and given heads parameter are not
            compatible. When d_model is divided by heads, the 
            result is less than one.
            """
            msg = Core.dedent(msg)
            raise Core.Errors.ValidationError("Validation", msg, task)

        if mode == "reshape":
            if d_model % heads != 0:
                msg = """\
                The given d_model and given heads are incompatible with 
                mode 'reshape': It is the case that d_model does not 
                divide cleanly by heads, and so a reshape is not possible
                """
                msg = Core.dedent(msg)
                raise Core.Errors.ValidationError("Validation", msg, task)
            self.merge_heads = Core.Reshape([heads, self.head_width], d_model)
        elif mode == "linear":
            self.merge_heads = linear.Linear([heads, self.head_width], d_model,
                                             parallel, dtype, device)
        else:
            raise ValueError(f"Invalid mode '{mode}', expected 'linear' or 'reshape'")

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """Move the head dimension next to the embedding dimension, merge, return"""
        tensor = tensor.movedim(-2, 0)
        tensor = self.merge_heads(tensor)
        tensor = tensor.movedim(0, -2)
        return tensor
