


# Layer: Refinable Mask

# Layer: Action Buffer.

# Layer: Data_Access

setup:

* attn_fn
* ff_fn
* layernorms

forward(query, data)->tensor:

* attn(query, data, data)
* add + layernomr
* ff(tensor)
* add + layernorm
* return tensor

# Layer: Access_Buffer:

__init__(
    d_model,
    buffer_size, 
    data_access_layers: List
)

forward(state, query)-> key, value, state




# Object: Data_Source

__init__(access_attn_fn,
         access_ff_fn, 
         access_layernorm_fn,
         data)
 
__call__(query)


# Object: Planning_Buffer

__init__(Data_Source: List[Data],
         batch_shape: tensor,
         buffer_size,
         state)

__call__(query: tensor, batchlock: tensor):
    

# Object: Decoder

__init__(param_block_attn_fn,
        planning_buffer_request_fn,
        feedforward_fn,
        ACTManager,
        

