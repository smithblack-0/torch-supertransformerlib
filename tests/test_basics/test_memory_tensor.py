import unittest
import torch
from src.supertransformerlib import Basics

class TestMemoryValidation(unittest.TestCase):
    """
    Test for the validation routines in memory tensor.
    """
    def setUp(self):
        batch_shape = 2
        num_entries = 5
        embedding_shape = 4

        data = torch.randn([batch_shape, num_entries, embedding_shape])
        memory = Basics.MemoryTensor(batch_shape=batch_shape, embedding_shape=embedding_shape)
        memory.concat(data)
        self.memory = memory

    def test_validate_index(self):
        """ Test if the validate index method is operating sanely."""

        index = torch.tensor([[1, 2, 3], [0, 1, 2]])
        memory = self.memory
        self.assertIsNone(memory._validate_index(index))

        with self.assertRaises(Basics.IndexError) as cm:
            # Check for wrong rank
            index = torch.tensor([1])
            memory._validate_index(index)
        self.assertTrue(cm.exception.variety == "Rank")

        with self.assertRaises(Basics.IndexError) as cm:
            # Check for wrong dtype rank
            index = torch.tensor([1.0])
            memory._validate_index(index)
        self.assertTrue(cm.exception.variety == "Dtype")

        with self.assertRaises(Basics.IndexError) as cm:
            # Check for wrong num of dimensions
            index = torch.tensor([[1, 2, 3], [0, 1, 7], [0, 2, 2]])
            memory._validate_index(index)
        self.assertTrue(cm.exception.variety == "BatchShape")

        with self.assertRaises(Basics.IndexError) as cm:
            # Check for out of range low
            index = torch.tensor([[1, 2, 3], [0, 1, -2]])
            memory._validate_index(index)
        self.assertTrue(cm.exception.variety == "Value")

    def test_validate_num_elements(self):

        memory = self.memory
        num_elements = torch.tensor([3, 2])
        self.assertIsNone(memory._validate_num_elements(num_elements))

        with self.assertRaises(Basics.IndexError) as cm:
            # Check for wrong rank
            num_elements = torch.tensor([[[1]]])
            memory._validate_num_elements(num_elements)
        self.assertTrue(cm.exception.variety == "Rank")

        with self.assertRaises(Basics.IndexError) as cm:
            # Check for wrong dtype rank
            num_elements = torch.tensor([1.0, 2.0])
            memory._validate_num_elements(num_elements)
        self.assertTrue(cm.exception.variety == "Dtype")

        with self.assertRaises(Basics.IndexError) as cm:
            # Check for wrong num of dimensions
            num_elements = torch.tensor([1, 2, 3])
            memory._validate_num_elements(num_elements)
        self.assertTrue(cm.exception.variety == "BatchShape")

    def test_validate_data(self):

        batch_shape = [2, 3]
        embedding_shape = [4, 5]
        memory = Basics.MemoryTensor(batch_shape=batch_shape, embedding_shape=embedding_shape)

        data = torch.randn(batch_shape + [6] + embedding_shape)
        self.assertIsNone(memory._validate_data(data, "test"))

        with self.assertRaises(Basics.DataFormatError) as cm:
            # Test bad dtype
            data = torch.randn(batch_shape + [4] + embedding_shape, dtype=torch.complex64)
            memory._validate_data(data, "test")
        self.assertTrue(cm.exception.variety == "Dtype")

        with self.assertRaises(Basics.DataFormatError) as cm:
            data = torch.randn([2, 4 ,5])
            memory._validate_data(data, "test")
        self.assertTrue(cm.exception.variety == "Rank")

        with self.assertRaises(Basics.DataFormatError) as cm:
            data = torch.randn([2, 20] + [7] + embedding_shape)
            memory._validate_data(data, "test")
        self.assertTrue(cm.exception.variety == "BatchShape")

        with self.assertRaises(Basics.DataFormatError) as cm:
            data = torch.randn(batch_shape + [8] + [4, 7])
            memory._validate_data(data, "test")
        self.assertTrue(cm.exception.variety == "EmbeddingShape")


class TestMemoryTensor(unittest.TestCase):
    """
    This is an inline test class for the cumulative
    batch tensor case.
    """
    def test_simple_concat(self):
        """ Test concatenation into an empty memory with no restrictions"""

        batch_dim = 2
        num_items = 2
        fill = 1.0
        embedding_dim = None

        tensor = torch.tensor([[1, 2, 3],[4, 5,6.0]])
        num_elements = torch.tensor([3, 3])

        cumulative_batch = Basics.MemoryTensor(batch_dim, embedding_dim, fill)
        cumulative_batch.concat(tensor, num_elements)

        self.assertTrue(torch.allclose(tensor, cumulative_batch.tensor))

    def test_padded_concat(self):
        """ Test concatenation where not every element is expected to be inserted"""
        batch_dim = 2
        num_items = 2
        fill = 1.0
        embedding_dim = None

        tensor = torch.tensor([[1, 2, 3], [4, 5, 6.0]])
        expected = torch.tensor([[1, 2, 3], [4, fill, fill]])
        num_elements = torch.tensor([3, 1])

        cumulative_batch = Basics.MemoryTensor(batch_dim, embedding_dim, fill)
        cumulative_batch.concat(tensor, num_elements)

        self.assertTrue(torch.allclose(expected, cumulative_batch.tensor))

    def test_double_concat(self):
        """ Test concat works properly when used in succession."""
        batch_dim = 2
        fill = 0.0
        embedding_dim = None

        tensor1 = torch.tensor([[1, 2.0], [4, 5]])
        num_elem1 = torch.tensor([2, 1])

        tensor2 = torch.tensor([[4, 3],[8, 9.0]])
        num_elem2 = torch.tensor([1, 2])

        expected = torch.tensor([[1, 2.0, 4],[4, 8, 9]])

        cumulative_batch = Basics.MemoryTensor(batch_dim, embedding_dim, fill)
        cumulative_batch.concat(tensor1, num_elem1)
        cumulative_batch.concat(tensor2, num_elem2)
        self.assertTrue(torch.allclose(expected, cumulative_batch.tensor))
        self.assertTrue(torch.allclose(torch.tensor([3,3]), cumulative_batch.last_element))

    def test_reset(self):
        """ Test concatenation and reset"""

        batch_dim = 2
        num_items = 2
        fill = 1.0
        embedding_dim = None

        tensor1 = torch.tensor([[1, 2], [4, 5.0]])
        num_elem1 = torch.tensor([2, 2])

        tensor2 = torch.tensor([[3, 0],[0, 1.0]])
        num_elem2 = torch.tensor([0, 2])

        expected = torch.Tensor([[1, 2],[0, 1.0]])

        cumulative_batch = Basics.MemoryTensor(batch_dim, embedding_dim, fill)
        cumulative_batch.concat(tensor1, num_elem1)
        cumulative_batch.reset(1)
        cumulative_batch.concat(tensor2, num_elem2)

        self.assertTrue(torch.allclose(expected, cumulative_batch.tensor))

    def test_multidimensional_reset(self):
        """ Test reset when dealing with multidimensional batches"""

        cumulative_batch = Basics.MemoryTensor([2, 3], None)
        data = torch.randn([2, 3, 5])
        cumulative_batch.concat(data)
        cumulative_batch.reset([0, 0])
    def test_single_embedding_dims(self):
        """ run tests with embeddings active, using random noise"""

        batch_dim = 10
        num_elements = 15
        embedding_dim = 12
        fill = 0.0

        tensor1 = torch.rand([batch_dim, num_elements, embedding_dim])
        num_elements1 = torch.randint(0, num_elements, [batch_dim])

        tensor2 = torch.rand([batch_dim, num_elements, embedding_dim])
        num_elements2 = torch.full([batch_dim], num_elements)

        cumulative_batch = Basics.MemoryTensor(batch_dim,
                                               embedding_dim,
                                               fill)
        cumulative_batch.concat(tensor1, num_elements1)
        cumulative_batch.concat(tensor2)
        self.assertTrue(torch.all(num_elements1 + num_elements2 == cumulative_batch.last_element))
    def test_complex_dims(self):
        """ Run tests with multiple dimensions along the embedding and batch dimensions"""

        batch_dims = [3, 3]
        embedding_dim = [2, 6]
        fill = 0.0

        tensor1 = torch.rand(batch_dims + [4] + embedding_dim)
        num_elements1 = 4* torch.ones(batch_dims, dtype=torch.int64)

        cumulative_batch =  Basics.MemoryTensor(batch_dims,
                                                embedding_dim,
                                                fill)
        cumulative_batch.concat(tensor1, num_elements1)
        cumulative_batch.reset([2, 2])
        cumulative_batch.concat(tensor1, num_elements1)

        expected_last_elem = torch.tensor([[8, 8, 8],[8, 8, 8],[8, 8, 4]])

        self.assertTrue(torch.all(expected_last_elem == cumulative_batch.last_element))

    def test_set(self):
        """Run test using the set feature"""

        batch_dim = 2
        num_items = 2
        fill = 1.0
        embedding_dim = None

        tensor = torch.tensor([[1, 2, 3], [4, 5, 6.0]])
        num_elements = torch.tensor([3, 3])
        cumulative_batch = Basics.MemoryTensor(batch_dim,
                                               embedding_dim,
                                               fill)
        cumulative_batch.concat(tensor, num_elements)

        index = torch.tensor([[0, 1],[1, 2]])
        set = torch.tensor([[0.0, 0.0],[0.0, 0.0]])
        num_elements = torch.tensor([2, 1])
        cumulative_batch.set_embeddings(index, set, num_elements)

        expected = torch.tensor([[0, 0, 3], [4, 0, 6.0]])
        self.assertTrue(torch.allclose(expected, cumulative_batch.tensor))

    def test_dereference(self):
        """ Test that dereference can successfully fetch embeddings by index"""

        # Set up the test feature
        data = torch.tensor([[[1,2,3],[4,5,6],[7,8,9.0]],
                            [[10, 11, 12],[13,14,15],[16, 17, 18]]])
        cumulative_batch = Basics.MemoryTensor(batch_shape=2,
                                               embedding_shape=3)
        cumulative_batch.concat(data)

        # Test dereferencing all

        index = torch.tensor([[0, 2],[1, 1]])
        expected = torch.tensor([[[1,2,3],[7,8,9.0]],
                                [[13,14,15],[13, 14, 15]]])
        output = cumulative_batch.dereference(index)
        self.assertTrue(torch.all(expected == output))

        # Test partial dereferencing

        num_elements = torch.tensor([2, 1])
        expected = torch.tensor([[[1,2,3],[7,8,9.0]],
                                [[13,14,15],[0, 0, 0]]])
        output = cumulative_batch.dereference(index, num_elements)
        self.assertTrue(torch.all(expected == output))

    def test_kernel_dereference(self):
        """ Test that the local striding tricks are operating normally """

        # Set up the test fixture
        data = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9.0]])
        cumulative_batch = Basics.MemoryTensor(batch_shape=1,
                                               embedding_shape = None)
        cumulative_batch.concat(data)

        # Perform a straightforward kernel extraction with no dilation changes

        index = torch.tensor([[1, 5]])
        prior = 1
        post = 0
        dilation = 1

        cumulative_batch.dereference_convolution(index, prior, post)

    def test_torchscript(self):
        """ Test that torchscript can compile the item"""
        # Set up the test fixture
        cumulative_batch = Basics.MemoryTensor(batch_shape=4,
                                               embedding_shape = None)
        cumulative_batch = torch.jit.script(cumulative_batch)
        data = torch.randn([4, 6])
        cumulative_batch.concat(data)