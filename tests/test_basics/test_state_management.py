import torch
import unittest
from supertransformerlib.Basics import state_management
from supertransformerlib import Core
from torch import nn
from typing import Dict, List, Tuple




class TestSetupSublayer(unittest.TestCase):
    """
    Unit test for the test setup sublayer. It should be torchscriptable,
    and subclassable
    """


    def test_forward_not_implemented(self):
        with self.assertRaises(TypeError):
            sublayer = state_management.SetupSublayer()
            tensors = {"input": torch.tensor([1.0])}
            sublayer.forward(tensors)

    def test_subclassing(self):
        class CustomSetupSublayer(state_management.SetupSublayer):
            def forward(self, batch_shape: List[int], tensors: Dict[str, torch.Tensor]) -> torch.Tensor:
                return tensors["input"] * 2

        custom_sublayer = CustomSetupSublayer()
        tensors = {"input": torch.tensor([2.0])}
        batch_shape = [10, 4]
        result = custom_sublayer.forward(batch_shape, tensors)
        expected = torch.tensor([4.0])
        self.assertTrue(torch.allclose(result, expected))

    def test_torchscript_compatibility(self):
        class CustomSetupSublayer(state_management.SetupSublayer):
            def forward(self, batch_shape: List[int], tensors: Dict[str, torch.Tensor]) -> torch.Tensor:
                return tensors["input"] * 2

        custom_sublayer = CustomSetupSublayer()
        scripted_sublayer = torch.jit.script(custom_sublayer)
        tensors = {"input": torch.tensor([3.0])}
        batch_shape = [10, 4]
        result = scripted_sublayer.forward(batch_shape, tensors)
        expected = torch.tensor([6.0])
        self.assertTrue(torch.allclose(result, expected))


class CustomSetupSublayer(state_management.SetupSublayer):
    def forward(self, batch_shape: Core.StandardShapeType, tensors: Dict[str, torch.Tensor]) -> torch.Tensor:
        return tensors["input"] * 2

class TestSetupFrame(unittest.TestCase):
    """
    Test the setup frame data holder works properly.
    """
    def test_setup_frame_creation(self):
        layer = CustomSetupSublayer()
        required_tensors = ["input"]
        constraints = ["constraint_1", "constraint_2"]

        setup_frame = state_management._SetupFrame(layer=layer, required_tensors=required_tensors, constraints=constraints)

        self.assertIsInstance(setup_frame.layer, CustomSetupSublayer)
        self.assertEqual(setup_frame.required_tensors, required_tensors)
        self.assertEqual(setup_frame.constraints, constraints)

    def test_setup_frame_layer_forward(self):
        layer = CustomSetupSublayer()
        required_tensors = ["input"]
        constraints = ["constraint_1", "constraint_2"]

        setup_frame = state_management._SetupFrame(layer=layer, required_tensors=required_tensors, constraints=constraints)

        tensors = {"input": torch.tensor([2.0])}
        result = setup_frame.layer.forward(tensors)
        expected = torch.tensor([4.0])
        self.assertTrue(torch.allclose(result, expected))


class TestSetupConfig(unittest.TestCase):

    def test_add_setup_frame(self):
        config = state_management.SetupConfig(2)

        # Add a new frame
        config.add("test", CustomSetupSublayer(), ["input"], ["constraint_1", "constraint_2"])

        # Check if the frame was added
        self.assertIn("test", config.frames)

        # Check the frame's properties
        frame = config.frames["test"]
        self.assertIsInstance(frame.layer, CustomSetupSublayer)
        self.assertEqual(frame.required_tensors, ["input"])
        self.assertEqual(frame.constraints, ["constraint_1", "constraint_2"])

    def test_add_existing_frame_raises_error(self):
        config = state_management.SetupConfig(2)
        config.add("test", CustomSetupSublayer(), ["input"], ["constraint_1", "constraint_2"])

        # Try to add a frame with the same name
        with self.assertRaises(ValueError):
            config.add("test", CustomSetupSublayer(), ["input"], ["constraint_1", "constraint_2"])

    def test_empty_config_creation(self):
        config = state_management.SetupConfig(2)
        self.assertEqual(config.batch_dims, 2)
        self.assertEqual(config.frames, {})

class TestSetup(unittest.TestCase):
    def test_setup_init_runs(self):

        config = state_management.SetupConfig(2)
        layer = CustomSetupSublayer()

        config.add("test1", layer, ["input"], [])
        config.add("test2", layer, ["input"] , [])

        setup = state_management.Setup(config)

    def test_setup_forward_runs(self):
        """ Test setup forward runs properly"""

        config = state_management.SetupConfig(2)
        layer = CustomSetupSublayer()

        config.add("test1", layer, ["input"], ["embeddings"])
        config.add("test2", layer, ["input"] , ["embeddings"])

        input = torch.randn([4, 5, 3])
        data = {"input" : input}
        setup = state_management.Setup(config)
        bundle_tensor = setup([4, 5], data)

        self.assertTrue(torch.allclose(bundle_tensor["test1"], input*2))
        self.assertTrue(torch.allclose(bundle_tensor["test2"], input*2))

class TestStateUsageLayer(unittest.TestCase):

    def test_must_override_init_(self):
        with self.assertRaises(TypeError):
            state_management.StateUsageLayer()

    def test_can_subscript(self):

        class CustomUsageLayer(state_management.StateUsageLayer):
            def __init__(self, config):
                super().__init__(config)
            def forward(self, tensor: torch.Tensor)->torch.Tensor:
                return tensor

        config = state_management.SetupConfig(2)
        layer = CustomUsageLayer(config)

    def test_can_jit(self):
        class CustomUsageLayer(state_management.StateUsageLayer):
            def __init__(self, config):
                super().__init__(config)
            def forward(self, tensor: torch.Tensor)->torch.Tensor:
                return tensor

        config = state_management.SetupConfig(2)
        layer = CustomUsageLayer(config)
        scripted_layer = torch.jit.script(layer)

class TestEndToEnd(unittest.TestCase):
    """
    Test the entire process, including a few minor state updates
    and some learnable layers. This will include verifying our system
    is still scriptable at the end. We will test it using LSTM
    """

    def test_end_to_end(self):

        # We test this by setting up a named LSTM helper instance,
        # which can store it's state on the BundleTensor

        # Define the setup layer.
        class LSTM_Setup(state_management.SetupSublayer):
            """
            The setup mechanism for the LSTM cell
            """
            def __init__(self,
                         hidden_size: int,
                         num_layers: int
                         ):
                super().__init__()
                self.hidden_size = hidden_size
                self.num_layers = num_layers
            def forward(self, batch_shape: List[int], tensors: Dict[str, torch.Tensor]):
                return torch.randn(1, batch_shape[0], self.hidden_size)

        # Define the LSTM layer which interacts with, and configures, the state
        class LSTM(state_management.StateUsageLayer):
            def __init__(self,
                         config: state_management.SetupConfig,
                         name: str,
                         input_size: int,
                         hidden_size: int,
                         num_layers: int):

                super().__init__(config)

                cell_name = name + "_cell"
                hidden_name = name + "_hidden"

                cell_setup_layer = LSTM_Setup(hidden_size, num_layers)
                hidden_setup_layer = LSTM_Setup(hidden_size, num_layers)

                config.add(cell_name, cell_setup_layer, [], ["batch_dim", name+"hidden_dim"])
                config.add(hidden_name, hidden_setup_layer, [], ["batch_dim", name+"hidden_dim"])

                self.LSTM = nn.LSTM(input_size, hidden_size)
                self.cell_name = cell_name
                self.hidden_name = hidden_name
            def forward(self,
                        state_tensor: Core.BundleTensor,
                        input_sequence: torch.Tensor,
                        )->Tuple[torch.Tensor, Core.BundleTensor]:

                cell_old = state_tensor[self.cell_name]
                hidden_old = state_tensor[self.hidden_name]

                output, (hidden_new, cell_new) = self.LSTM(input_sequence, (hidden_old, cell_old))

                state_tensor = state_tensor.set(self.cell_name, hidden_new)
                state_tensor = state_tensor.set(self.hidden_name, cell_new)
                return output, state_tensor

        # Define parameters
        seq_length = 3
        batch_size = 5

        input_size = 10
        hidden_size = 30
        num_layers = 3

        # Get the layers ready to go
        config = state_management.SetupConfig(1)
        input_sequence = torch.randn(seq_length, batch_size, input_size)
        lstm_layer = LSTM(config, "LSTM_Test", input_size, hidden_size, num_layers)
        setup = state_management.Setup(config)

        # Run the model

        hidden = setup(batch_size)
        output = lstm_layer(hidden, input_sequence)


    def test_end_to_end_jit(self):

        # We test this by setting up a named LSTM helper instance,
        # which can store it's state on the BundleTensor

        # Define the setup layer.
        class LSTM_Setup(state_management.SetupSublayer):
            """
            The setup mechanism for the LSTM cell
            """
            def __init__(self,
                         hidden_size: int,
                         num_layers: int
                         ):
                super().__init__()
                self.hidden_size = hidden_size
                self.num_layers = num_layers
            def forward(self, batch_shape: List[int], tensors: Dict[str, torch.Tensor]):
                return torch.randn(1, batch_shape[0], self.hidden_size)

        # Define the LSTM layer which interacts with, and configures, the state
        class LSTM(state_management.StateUsageLayer):
            def __init__(self,
                         config: state_management.SetupConfig,
                         name: str,
                         input_size: int,
                         hidden_size: int,
                         num_layers: int):

                super().__init__(config)

                cell_name = name + "_cell"
                hidden_name = name + "_hidden"

                cell_setup_layer = LSTM_Setup(hidden_size, num_layers)
                hidden_setup_layer = LSTM_Setup(hidden_size, num_layers)

                config.add(cell_name, cell_setup_layer, [], ["batch_dim", name+"hidden_dim"])
                config.add(hidden_name, hidden_setup_layer, [], ["batch_dim", name+"hidden_dim"])

                self.LSTM = nn.LSTM(input_size, hidden_size)
                self.cell_name = cell_name
                self.hidden_name = hidden_name
            def forward(self,
                        state_tensor: Core.BundleTensor,
                        input_sequence: torch.Tensor,
                        )->Tuple[torch.Tensor, Core.BundleTensor]:

                cell_old = state_tensor[self.cell_name]
                hidden_old = state_tensor[self.hidden_name]

                output, (hidden_new, cell_new) = self.LSTM(input_sequence, (hidden_old, cell_old))

                state_tensor = state_tensor.set(self.cell_name, hidden_new)
                state_tensor = state_tensor.set(self.hidden_name, cell_new)
                return output, state_tensor

        # Define parameters
        seq_length = 3
        batch_size = 5

        input_size = 10
        hidden_size = 30
        num_layers = 3

        # Get the layers ready to go
        config = state_management.SetupConfig(1)
        input_sequence = torch.randn(seq_length, batch_size, input_size)
        lstm_layer = LSTM(config, "LSTM_Test", input_size, hidden_size, num_layers)
        setup = state_management.Setup(config)

        # Script the model

        setup = torch.jit.script(setup)
        lstm_layer = torch.jit.script(lstm_layer)
        # Run the model

        hidden = setup(batch_size)
        output, hidden = lstm_layer(hidden, input_sequence)
        update = output.sum()
        update.backward()

