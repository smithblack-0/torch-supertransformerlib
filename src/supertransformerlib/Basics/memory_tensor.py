"""A memory module designed to store and retrieve sequences of embeddings.

This module provides a mechanism for storing and retrieving sequences of embeddings, ignoring padding elements. It is designed to be used in sequence modeling tasks, such as natural language processing, where inputs and outputs are sequences of variable length. The memory module is implemented using PyTorch, and can be used with any PyTorch model.

The memory module provides several methods for manipulating the memory, including concatenating new sequences, resetting batches, setting specific indices, and retrieving embeddings from specific indices. It also includes a method for generating convolutional kernels centered around specific indices.

The memory is stored as a tensor with a batch dimension and an optional embedding dimension. Padding elements are ignored by the memory module, allowing for efficient storage of variable-length sequences. The memory module includes several validation checks to ensure that the input data is of the correct shape and type.

Example usage:

```
    # Create a memory module with batch size 10 and embedding size 32
    memory = Memory(batch_shape=2, embedding_shape=32)

    #Concatenate a new sequence of length 5 into the memory.
    sequence = torch.randn(2, 5, 32)
    memory.concat(sequence)

    # Concotenates 3 elements along batch 0, but 1 element along batch 1, disregarding
    # the rest as padding.

    sequence = torch.randn(2, 5, 32)
    nonpadding_elements = torch.tensor([3, 2])
    memory.concat(sequence, nonpadding_elements)

    #Retrieve the embeddings for the first two indices in batch 0,and masks out the rest as
    # padding
    indices = torch.tensor([0, 1])
    indices = indices.unsqueeze(0).repeat(10, -1)

    unpadded_elements = torch.full(10, 0)
    unpadded_elements[0] = 2
    unpadded_elements[1] = 2

    embeddings = memory.dereference(indices, unpadded_element)
```

"""

from typing import Union, List, Optional
import torch
from torch import nn
from . import local_sample
from .. import Core

class IndexError(Core.ValidationError):
    """
    An error was found with an index when this is raised
    """
    def __init__(self,
                 msg: str,
                 variety: str,
                 violating_tensor: torch.Tensor
                 ):
        self.violator = violating_tensor
        self.variety = variety
        super().__init__("IndexError", msg)


class DataFormatError(Core.ValidationError):
    """
    An error was found with data which we wish to use in some way.
    """
    def __init__(self, msg: str, variety: str, violator: torch.Tensor):
        super().__init__("DataFormatError", msg)
        self.variety = variety
        self.violator = violator

class MemoryTensor:
    """
    This class is designed to manage a
    situation I keep coming up on in which I need
    to keep various state tensors across batches syncronized
    but also need the flexability of being able to reset
    one of the batches and still have an elegant concatenation
    method.

    It operates by ensuring that upon concatenation elemements
    end up being collocated next to the last empty element for
    a particular batch, which can vary per batch, with
    empty elements being padded.

    Given a batch shape of ...batch_shape, an embeddings shape of ...embedding_shape
    and a tensor with sequential cumulative content of shape ..batch_shape x N x ...embedding_shape
    this class will manage the situation such that one can elegantly append along N
    and expect this appending process to load based on the last active element.

    --- fields ----

    tensor: the stored away tensor. This will be a ...batch_shape x L x ...embedding shape tensor, with
            L being determined by the concatenation history

    last_element: The last filled element, whatever it might be. This will be a ...batch_shape int tensor
                  representing the last element along dimension L, and corrospondingly where to start
                  inserting new informaton.

    --- methods ----

    concat: Performs a cumulative concat across the batches, filling in from last element forward
    reset: Reset the batch information for a particular batch entry or sequence of batch entries
    unsqueeze: Inserts a new batch at the given batch along the given dimension.
    release: Eliminates a particular batch plane completely.
    """

    @property
    def _mem_length(self)->int:
        return self.tensor.size(self._batch_rank)

    def _pad_memory_if_required(self, num_elements: torch.Tensor):
        ####################
        # Extends the internal tensor memory by some amount if it is required.
        # Padding will be required if any of the current last elements,
        # plus the additional elements in the update, exceeds the length
        # of the memory

        required_last_element = int(torch.max(self.last_element + num_elements))
        if required_last_element > self._mem_length:
            required_extension_len = required_last_element - self._mem_length
            padding_directive = [0, required_extension_len]
            for _ in range(self._embedding_rank):
                padding_directive = [0, 0] + padding_directive
            self.tensor = nn.functional.pad(self.tensor, padding_directive, value=self._fill)

    def _expand_by_broadcast(self,
                             tensor: torch.Tensor,
                             dim: int,
                             shape: List[int],
                             )->torch.Tensor:
        ## Takes in a tensor formatted in terms of ..whatever and broadcasts it
        # to a tensor of shape ...whatever x ...embedding_shape. No additional\
        # memory is used.

        if dim < 0:
            # This allows proper functioning when provided a negative insertion point
            dim = dim % tensor.dim() + 1

        expansion_shape = [-1]*tensor.dim()
        for i, dim_len in enumerate(shape):
            tensor = tensor.unsqueeze(dim)
            expansion_shape.insert(dim+i, dim_len)
        output = tensor.expand(expansion_shape)
        return output

    def _validate_index(self, index: torch.Tensor):
        # Validates an index is formatted correctly, and thows a descriptive error if
        # not. An index should be of shape ...batch_shape x L, with L varying depending on the
        # problem.

        if index.dtype != torch.int64:
            msg = f"""\
            It was expected that the dtype for parameter index would be of 
            type int64. However, instead found {index.dtype}
            """
            msg = Core.dedent(msg)
            raise IndexError(msg, "Dtype", index)

        if index.dim() != self._batch_rank + 1:
            msg = f"""\
            It is the case that parameter index must have rank equal to batch rank 
            plus 1, which is {int(self._batch_rank + 1)}. However,
            found rank {int(index.dim())}
            """
            msg = Core.dedent(msg)
            raise IndexError(msg, "Rank", index)

        if list(index.shape[:-1]) != self._batch_shape:
            msg = f"""\
            It is the case that parameter index must have initial dimensions of shape
            {self._batch_shape}, with the final dimension being determined by the problem.
            However, it was actually found to have shape {index.shape[:-1]}.
            """
            msg = Core.dedent(msg)
            raise IndexError(msg, "BatchShape", index)
        if torch.any(index < 0):
            msg = f"""\
            It was expected that all index entries provided should be greater than or
            equal to zero. However, some were negative. 
            """
            msg = Core.dedent(msg)
            raise IndexError(msg, "Value", index)


    def _validate_num_elements(self, num_elements: torch.Tensor):
        # validates a num_elements case is formatted correctly with
        # proper shape and so on. If not, gives an informative error why

        if num_elements.dtype != torch.int64:
            msg = f"""
            It was expected that the dtype for parameter num_elements would be of 
            type int64. However, instead found {num_elements.dtype}
            """
            msg = Core.dedent(msg)
            raise IndexError(msg, "Dtype", num_elements)

        if num_elements.dim() != self._batch_rank:
            msg = f"""
            It is the case that parameter num_elements must have rank equal to batch shape,
            which is {int(self._batch_rank)}. However,  
            found rank {int(num_elements.dim())}
            """
            msg = Core.dedent(msg)
            raise IndexError(msg, "Rank", num_elements)

        if list(num_elements.shape) != self._batch_shape:
            msg = f"""
            It is the case that parameter num_elements must have  dimensions of shape
            {self._batch_shape}. However, it was actually found to have shape
            {num_elements.shape[:]}.
            """
            msg = Core.dedent(msg)
            raise IndexError(msg, "BatchShape", num_elements)

    def _validate_data(self, tensor: torch.Tensor, parameter_name: str):
        # Validates that the embedding data is designed correctly
        # Provides a clear error message on what is wrong if there is
        # an issue

        if tensor.dtype != self.tensor.dtype:
            #Dtype assertion
            msg = f"""
            It was expected that the dtype for parameter {parameter_name} would be of 
            type {self.tensor.dtype}. However, instead found {tensor.dtype}
            """
            msg = Core.dedent(msg)
            raise DataFormatError(msg, "Dtype", tensor)

        if tensor.dim() != self.tensor.dim():
            # Right rank assertion
            msg = f"""
            It is the case that parameter {parameter_name} must have rank equal to batch 
            shape plus index dim plus embedding shape, which is {self.tensor.dim()}. 
            However, found rank {int(tensor.dim())}
            """
            msg = Core.dedent(msg)
            raise DataFormatError(msg, "Rank", tensor)

        if list(tensor.shape[:self._batch_rank]) != self._batch_shape:
            # Batch shape matches
            msg = f"""
            It is the case that parameter {parameter_name} did not have batch shape of shape {self._batch_shape}.
            Instead, it was found to have shape {tensor.shape[:self._batch_rank]}. This is not allowed
            """
            msg = Core.dedent(msg)
            raise DataFormatError(msg, "BatchShape", tensor)
        if list(tensor.shape[(self._batch_rank+1):]) != self._embedding_shape:
            # embedding shape matches
            msg = f"""
            It is the case that parameter {parameter_name} did not have embedding shape of shape 
            {self._embedding_shape} as expected. Instead, it was found to have shape 
            {tensor.shape[(self._batch_rank + 1):]}. This is not allowed.
            """
            msg = Core.dedent(msg)
            raise DataFormatError(msg, "EmbeddingShape", tensor)

    def concat(self,
               tensor: torch.Tensor,
               num_elements: Optional[torch.Tensor] = None):
        """
        Concats a particular sequence of information into the memory, ignoring padding

        :param tensor: A ...batch x L x ...embedding tensor to insert which may have padding element
        :param num_elements: A optional B tensor of ints indicating how many elements are not padding.
                            If not included, it is assumed all elements need to be include
        """

        ##################### How to store #################
        # Padding means that tokens do not just need to be concatenated onto the end
        # of the memory. As a result, we need to jump through some hoops to
        # store away new tokens and new memory
        #
        # First, we figure out by how much we need to extend the memory in order
        # to store everything. This is done by comparing the last element to the
        # unpadded length for each batch dimension. After the extension is done,
        # we then insert the tokens and flags at the appropriate positions on
        # each batch. We finish by returning a new memory instance.

        update_len = tensor.size(self._batch_rank)
        if num_elements is None:
            num_elements = torch.full(self._batch_shape, update_len)
        if self.validate:
            self._validate_num_elements(num_elements)
            self._validate_data(tensor, "tensor")
        self._pad_memory_if_required(num_elements)

        # Generate the scattering mask required for the operation. This
        # will involve two masks, one for the destination in the tensor
        # and one for the input data



        source_mask = num_elements.unsqueeze(-1) > torch.arange(0, update_len, device=tensor.device)

        index = torch.arange(0, update_len)
        for _ in range(self._batch_rank):
            index = index.unsqueeze(0)
        index = index + self.last_element.unsqueeze(-1)
        index = torch.where(source_mask, index, self._mem_length + 1)
        destination_mask = index.unsqueeze(-1) == torch.arange(0, self._mem_length, device=self.tensor.device)
        destination_mask = torch.any(destination_mask, dim=-2)

        # Store away the results using vector indexing

        self.tensor[destination_mask] = tensor[source_mask]
        self.last_element += num_elements

    def reset(self, batch_coordinate: Union[List[int], int]):
        """
        A mechanism for resetting a batch. One can reset the
        batch length so that it can be reused easily.

        :param batch_coordinate: A representation of the batch to reset. When working with a
               single batch dim, a simple int suffices, however multidimensional batches must
               have their coordinates specified as a list of ints.
        """
        if isinstance(batch_coordinate, int):
            batch_coordinate = [batch_coordinate]
        batch_coordinate = list(batch_coordinate)
        self.tensor[batch_coordinate] = self._fill
        self.last_element[batch_coordinate] = 0

    def set_embeddings(self,
                       index: torch.Tensor,
                       new_embeddings: torch.Tensor,
                       num_elements: torch.Tensor):
        """
        The set feature. Designed to set the indicated index entries
        :param index: A ... x N collection of integer indices to set to
        :param new_embeddings: A ... x N x ...embedding collection of elements to set
        :param num_elements: A ...   tensor of integers indicating how many elements are present
                            and not just padding.
        """

        if self.validate:
            self._validate_index(index)
            self._validate_num_elements(num_elements)
            self._validate_data(new_embeddings, "new_embeddings")

        update_len = new_embeddings.size(self._batch_rank)
        source_mask =  num_elements.unsqueeze(-1) > torch.arange(0, update_len, device=new_embeddings.device)
        destination_mask = torch.full(list(self._batch_shape) + [self._mem_length], 0, dtype=torch.bool)
        destination_mask = destination_mask.scatter(dim=-1,
                                                    index=index,
                                                    src=source_mask)
        self.tensor[destination_mask] = new_embeddings[source_mask]

    def dereference(self,
                    index: torch.Tensor,
                    num_elements: Optional[torch.Tensor] = None):
        """
        Fetches the particular embeddings associated with
        the particular provided index. Pads away any
        element excluded by num_elements.

        :param index: A ... batch_shape x L int tensor indicating what to draw from.
        :param num_elements: A optional ... batch_shape int tensor indicating how many nonpadding entries
                            were in the index. If not given, it is assumed everything should be dereferenced.
        :return: A ... x batch_shape x L x ...embedding shape tensor.
        """

        access_len = index.size(-1)
        if num_elements is None:
            num_elements = torch.full(self._batch_shape, access_len)
        if self.validate:
            self._validate_index(index)
            self._validate_num_elements(num_elements)

        #Make the vector mask by first figuring out the unpadded elements
        # then scattering that into a source query
        output_mask = num_elements.unsqueeze(-1) <= torch.arange(0, access_len, device=index.device)

        index = self._expand_by_broadcast(index, -1, self._embedding_shape)
        output_mask = self._expand_by_broadcast(output_mask, -1, self._embedding_shape)

        output = self.tensor.gather(dim=self._batch_rank, index=index)
        output = output.masked_fill(output_mask, self._fill)
        return output

    def dereference_convolution(self,
                                index: torch.Tensor,
                                prior_embeddings: int,
                                post_embeddings: int,
                                dilation: int = 1,
                                nonpadding_lengths: Optional[torch.Tensor] = None):
        """
        Generates a convolutional kernel centered around the provided
        embedding indices. Pads away unused elements

        :param index:  a ...batch_shape x L shaped int tensor indicating the
                      memory indices around which to build.
        :param nonpadding_lengths: A ...batch_shape int tensor indicating how many
                            non-padding elements are present in the input
        :param prior_embeddings: A int indicating how many embeddings from before the
                                reference point to include
        :param post_embedding: A int indicating how many embeddings from after the reference point
                              to include
        :param dilation: A int indicating the desired rate of dilation
        :return: A convolutional kernal of shape ...batch_shape x L x kernel x ...embedding_shape
        """

        access_len = index.size(-1)
        if nonpadding_lengths is None:
            nonpadding_lengths = torch.full(self._batch_shape, access_len)

        if self.validate:
            self._validate_index(index)
            self._validate_num_elements(nonpadding_lengths)


        output_mask = nonpadding_lengths.unsqueeze(-1) <= torch.arange(0, access_len, device=index.device)

        # Create the appropriate convolution module to dereference from
        #
        # This consists of performing a strided withdraw using convolution
        # generation tools which place extra dimensions on the end of the tensor,
        # folllowed by rearrangement.
        #
        # the local sample argument is used for this, which is capable of
        # accepting a tensor and creating a strided view with extra dimensions on
        # the end containing a kernel that is a convolutional view of the tensor

        tensor = self.tensor.movedim(self._batch_rank, -1)
        source = local_sample.local(tensor,
                                    start= -prior_embeddings,
                                    end = post_embeddings,
                                    stride = 1,
                                    dilation=dilation,
                                    offset=0)
        source = source.movedim(-1, self._batch_rank)
        source = source.movedim(-1, self._batch_rank) #...batch_shape x L x local x ...

        # Expand the mask and index to match. Then fetch, mask, and return

        expansion_shape = list(source.shape[(self._batch_rank+1):])
        index = self._expand_by_broadcast(index, -1, expansion_shape)
        output_mask = self._expand_by_broadcast(output_mask, -1, expansion_shape)

        output = source.gather(dim=self._batch_rank, index=index)
        output = output.masked_fill(output_mask, self._fill)
        return output


    def __init__(self,
                 batch_shape: Union[List[int], int],
                 embedding_shape: Union[List[int], int, None],
                 empty_fill: float = 0.0,
                 dtype: Optional[torch.dtype] = None,
                 device: Optional[torch.device] = None,
                 validate: bool = True,
                 ):
        """
        The initialization function for the class.

        It sets up the backend storage along with some
        of the defaults

        :param batch_shape: The shape of the batches we will see. Can be as simple as an int, or a
                            more complex multidimensional batch as a list of ints.
        :param embedding_shape: The embedding shape. Can be None for a tensor with no embedding,
                                a int for a fixed witth embedding, or a list of ints for a
                                multidimensional embedding
        :param empty_fill: The value that tensors should be filled with when declared "empty"
        :param dtype: The dtype for the tensor
        :param device: The device for the tensor.
        """

        # Calculate the expected tensor shape

        if isinstance(batch_shape, int):
            batch_shape = [batch_shape]
        if isinstance(embedding_shape, int):
            embedding_shape = [embedding_shape]
        elif embedding_shape is None:
            embedding_shape = []

        storage_shape = batch_shape + [0] + embedding_shape
        self._fill = empty_fill
        self._batch_rank = len(batch_shape)
        self._batch_shape = batch_shape
        self._embedding_rank = len(embedding_shape)
        self._embedding_shape = embedding_shape
        self.tensor = torch.zeros(storage_shape, dtype=dtype, device=device)
        self.last_element = torch.zeros(batch_shape, dtype=torch.int64, device=device)
        self.validate = validate
