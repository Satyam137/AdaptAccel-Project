"""
workload_profiler.py — Step 1.1: Profile PyTorch models to extract per-layer sparsity traces.

For each layer in a model, captures:
  - layer_name, layer_type (Linear, Conv2d, etc.)
  - op_type mapped to VersaAccel operator (MM, MV, SpMM, etc.)
  - dimensions M, K, P
  - input_sparsity (fraction of zeros in input activation)
  - output_sparsity (fraction of zeros in output activation)
  - weight_sparsity (fraction of zeros in weight tensor)

Runs inference on N_SAMPLES inputs and records a trace per input.
Saves results to CSV files in profiling_data/ directory.

Usage:
    cd project/idea1_adaptaccel
    python -m profiler.workload_profiler
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import sys
from collections import OrderedDict

# ============================================================================
# Sparsity measurement
# ============================================================================

def measure_sparsity(tensor):
    """Fraction of zeros in a tensor (0.0 = fully dense, 1.0 = all zeros)."""
    if tensor is None or tensor.numel() == 0:
        return 0.0
    return float((tensor == 0).sum().item() / tensor.numel())


def get_op_type(module, input_shape, output_shape):
    """Map a PyTorch module to a VersaAccel operator type."""
    if isinstance(module, nn.Linear):
        # Linear: output = input @ weight.T → MM or MV
        batch_size = input_shape[0] if len(input_shape) > 1 else 1
        if batch_size == 1 and len(input_shape) <= 2:
            return 'MV'
        return 'MM'
    elif isinstance(module, nn.Conv2d):
        # Conv2d via im2col → MM
        return 'MM'
    else:
        return 'OTHER'


def get_dimensions(module, input_shape, output_shape):
    """Extract M, K, P dimensions from a layer for VersaAccel."""
    if isinstance(module, nn.Linear):
        # Linear(in_features, out_features): weight is [out, in]
        # MM: A(batch × in) × B(in × out) → C(batch × out)
        K = module.in_features
        P = module.out_features
        M = 1
        for d in input_shape[:-1]:  # all dims except last (features)
            M *= d
        return M, K, P

    elif isinstance(module, nn.Conv2d):
        # Conv2d via im2col: A(N*Hout*Wout × Cin*kH*kW) × B(Cin*kH*kW × Cout)
        Cin = module.in_channels
        Cout = module.out_channels
        kH, kW = module.kernel_size
        K = Cin * kH * kW
        P = Cout
        # M = batch * output_spatial
        batch = input_shape[0]
        H_out = output_shape[2] if len(output_shape) > 2 else 1
        W_out = output_shape[3] if len(output_shape) > 3 else 1
        M = batch * H_out * W_out
        return M, K, P

    return 1, 1, 1


# ============================================================================
# Profiler class
# ============================================================================

class LayerProfiler:
    """
    Hooks into a PyTorch model and records per-layer statistics during inference.
    """

    def __init__(self, model, model_name="model"):
        self.model = model
        self.model_name = model_name
        self.records = []
        self.hooks = []
        self._input_idx = 0

    def _make_hook(self, name, module):
        """Create a forward hook for a specific layer."""
        def hook_fn(mod, inp, out):
            # Get input tensor
            x = inp[0] if isinstance(inp, tuple) else inp
            y = out if isinstance(out, torch.Tensor) else (out[0] if isinstance(out, tuple) else None)

            if x is None or not isinstance(x, torch.Tensor):
                return
            if y is None or not isinstance(y, torch.Tensor):
                return

            input_shape = tuple(x.shape)
            output_shape = tuple(y.shape)

            op_type = get_op_type(mod, input_shape, output_shape)
            if op_type == 'OTHER':
                return  # Skip non-compute layers (ReLU, BatchNorm, etc.)

            M, K, P = get_dimensions(mod, input_shape, output_shape)

            record = {
                'input_idx': self._input_idx,
                'layer_name': name,
                'layer_type': mod.__class__.__name__,
                'op_type': op_type,
                'M': M, 'K': K, 'P': P,
                'input_sparsity': measure_sparsity(x.detach()),
                'output_sparsity': measure_sparsity(y.detach()),
                'weight_sparsity': measure_sparsity(mod.weight.detach()) if hasattr(mod, 'weight') and mod.weight is not None else 0.0,
                'input_shape': str(input_shape),
                'output_shape': str(output_shape),
            }
            self.records.append(record)

        return hook_fn

    def register_hooks(self):
        """Register forward hooks on all Linear and Conv2d layers."""
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                h = module.register_forward_hook(self._make_hook(name, module))
                self.hooks.append(h)
        print(f"  Registered hooks on {len(self.hooks)} layers")

    def remove_hooks(self):
        """Remove all hooks."""
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

    def profile(self, dataloader, n_samples=100):
        """
        Run inference on n_samples inputs and record per-layer traces.

        Args:
            dataloader: PyTorch DataLoader yielding (input, target) batches
            n_samples: Number of inputs to profile
        """
        self.model.eval()
        self.records = []
        self.register_hooks()

        count = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(dataloader):
                if inputs.device != next(self.model.parameters()).device:
                    inputs = inputs.to(next(self.model.parameters()).device)

                # Process each sample individually for per-input traces
                for i in range(inputs.size(0)):
                    if count >= n_samples:
                        break
                    self._input_idx = count
                    single_input = inputs[i:i+1]
                    self.model(single_input)
                    count += 1

                    if count % 50 == 0:
                        print(f"  Profiled {count}/{n_samples} inputs...")

                if count >= n_samples:
                    break

        self.remove_hooks()
        print(f"  Done. Collected {len(self.records)} records from {count} inputs.")
        return pd.DataFrame(self.records)

    def save_traces(self, output_dir):
        """Save profiling records to CSV."""
        os.makedirs(output_dir, exist_ok=True)
        df = pd.DataFrame(self.records)
        path = os.path.join(output_dir, f"{self.model_name}_traces.csv")
        df.to_csv(path, index=False)
        print(f"  Saved to {path} ({len(df)} rows)")
        return path
