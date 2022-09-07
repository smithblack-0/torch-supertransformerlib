import torch
from torch import nn
from src.supertransformerlib import Core


# Setup
ensemble_width = 20
d_model = 32
batch_size = 16

mockup_data_tensor = torch.randn([batch_size, d_model])

# Define the layers

config_generator = nn.Linear(32, 20)


def generate_config(tensor):
    return config_generator(tensor)


class mockup(Core.EnsembleSpace):
    """
    Dynamically varying kernel
    Simple task: add
    """

    def __init__(self):
        super().__init__(5)

        self.d_model = 20
        self.batch_fun = 4
        self.kernel = torch.randn([self.native_ensemble_width, self.d_model])
        self.kernel = torch.nn.Parameter(self.kernel)
        self.register_ensemble("kernel")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.kernel

class AddBias(Core.EnsembleSpace):
    def __init__(self, ensemble_width, d_model):
        super().__init__(ensemble_width)
        bias_kernel = torch.zeros([ensemble_width, d_model])
        bias_kernel = nn.Parameter(bias_kernel)
        self.bias_kernel = bias_kernel
        self.register_ensemble("bias_kernel")
    def forward(self, tensor):
        # Note that the raw kernel would be 20 x 32, not the required 16x32
        # It MUST be rebuilding the kernel using the ensemble if this works.
        kernel = self.bias_kernel
        return tensor + kernel

add_bias = AddBias(ensemble_width, d_model)
add_bias = torch.jit.script(add_bias)

#Run action

def configure_then_add(tensor):
    configuration = generate_config(tensor)
    add_bias.configuration = configuration
    return add_bias(tensor)

torch.jit.script(configure_then_add)
configure_then_add(mockup_data_tensor)