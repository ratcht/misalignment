import sys
sys.path.append('../..')  # Add parent directory to path
import os
import torch as t
import torch.nn as nn
t.cuda.empty_cache()
from transformers import Qwen2ForCausalLM, PreTrainedTokenizer, PreTrainedModel
from hf import HF
from evaluate import load

class SafetyAdapter(nn.Module):
  def __init__(self, hidden_size, dtype=None):
    super().__init__()
    # A small adapter: down-project, non-linearity, then up-project.
    self.adapter = nn.Sequential(
      nn.Linear(hidden_size, hidden_size // 4, dtype=dtype),
      nn.ReLU(),
      nn.Linear(hidden_size // 4, hidden_size, dtype=dtype),
    )
    # Initialize the final layer with zeros so that initially the adapter acts like an identity.
    # nn.init.zeros_(self.adapter[2].weight)
    # nn.init.zeros_(self.adapter[2].bias)

  def forward(self, hidden_states):
    adapter_device = self.adapter[0].weight.device
    
    # Ensure adapter input has the right dtype and device
    if hidden_states.dtype != self.adapter[0].weight.dtype or hidden_states.device != adapter_device:
      hidden_states = hidden_states.to(dtype=self.adapter[0].weight.dtype, device=adapter_device)
            
    adapter_out = self.adapter(hidden_states)
    
    # Return output in same dtype as input
    return hidden_states + adapter_out

class AdapterWrapper(nn.Module):
  """A wrapper that applies an adapter after the output of an existing module."""
  def __init__(self, original_module, adapter):
    super().__init__()
    self.original_module = original_module
    self.adapter = adapter
  
  def forward(self, *args, **kwargs):
    """Apply the original module and then the adapter."""
    outputs = self.original_module(*args, **kwargs)
    
    if isinstance(outputs, tuple):
      hidden_states = outputs[0]
      adapted_hidden_states = self.adapter(hidden_states)
      return (adapted_hidden_states,) + outputs[1:]
    else:
      return self.adapter(outputs)

def inject_adapter(model: PreTrainedModel, layer_idx: int) -> PreTrainedModel:
  # Determine the hidden size
  hidden_size = model.config.hidden_size
  
  # Find the layers
  layers = model.model.layers
  
  # Make sure the layer index is valid
  if layer_idx < 0 or layer_idx >= len(layers):
    raise ValueError(f"Layer index {layer_idx} is out of bounds. Model has {len(layers)} layers.")
  
  # Get the device of the specific layer we're attaching to
  target_layer = layers[layer_idx]
  layer_device = next(target_layer.parameters()).device
  layer_dtype = next(target_layer.parameters()).dtype
  
  print(f"Target layer {layer_idx} is using dtype: {layer_dtype} on device: {layer_device}")
  
  # Create the adapter with matching dtype and device
  adapter = SafetyAdapter(hidden_size, dtype=layer_dtype)
  adapter = adapter.to(layer_device)
  
  # Wrap the target layer with the adapter
  wrapped_layer = AdapterWrapper(target_layer, adapter)
  layers[layer_idx] = wrapped_layer
  
  # Store the adapter on the model for reference
  if not hasattr(model, "adapters"):
      model.adapters = {}
  model.adapters[f"safety_adapter_{layer_idx}"] = adapter
  
  return model

from safetensors.torch import load_file

def load_adapter_model(model_path, base_model_name, layer_idx=40):
    """Load base model and trained adapter weights"""
    import torch as t
    from transformers import Qwen2ForCausalLM, AutoTokenizer
    from safetensors import safe_open
    import os
    
    # Load base model
    base_model = Qwen2ForCausalLM.from_pretrained(
      base_model_name,
      device_map="auto",
      torch_dtype=t.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    
    # Create and inject adapter
    adapter = SafetyAdapter(base_model.config.hidden_size)
    original_layer = base_model.model.layers[layer_idx]
    base_model.model.layers[layer_idx] = AdapterWrapper(original_layer, adapter)
    
    # Load adapter weights from safetensors
    # Based on the JSON file, adapter weights are in model-00005-of-00006.safetensors
    adapter_file = os.path.join(model_path, "model-00005-of-00006.safetensors")
    
    with safe_open(adapter_file, framework="pt") as f:
      # Get adapter-related weights with correct mapping
      adapter_weights = {}
      for i, layer_name in enumerate(["0", "2"]):  # First linear layer and second linear layer
        for param in ["weight", "bias"]:
          safetensor_key = f"model.layers.{layer_idx}.adapter.adapter.{layer_name}.{param}"
          adapter_key = f"adapter.{layer_name}.{param}"
          if safetensor_key in f.keys():
            adapter_weights[adapter_key] = f.get_tensor(safetensor_key)
      
      # Load weights to adapter
      adapter.load_state_dict(adapter_weights)
    
    return base_model, tokenizer