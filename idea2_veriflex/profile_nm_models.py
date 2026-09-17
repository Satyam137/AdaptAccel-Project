"""
profile_nm_models.py — Profile models with N:M structured pruning (2:4).

Unlike Idea 1's unstructured pruning (random zeros), N:M pruning
enforces exactly 2 nonzeros per group of 4 consecutive weights.

This creates workloads where the NM config is competitive because:
  - Sparsity is exactly 50% and STRUCTURED
  - NM hardware can exploit the known pattern with simple MUX routing
  - No Benes network overhead (unlike D-M-R for unstructured sparsity)

Usage:
    cd project/idea2_veriflex
    python profile_nm_models.py
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import sys

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

from idea2_veriflex.nm_sparsity.nm_sparsity import nm_prune_weights, nm_compatibility_score

OUTPUT_DIR = os.path.join(PROJECT_DIR, "idea1_adaptaccel", "profiling_data")
N_SAMPLES = 200


def apply_nm_pruning_to_model(model, N=2, M_ratio=4):
    """
    Apply N:M structured pruning to all Conv2d and Linear layers.

    For each layer: flatten weights, apply 2:4 pruning (keep top-2 per group of 4),
    reshape back.
    """
    total_params = 0
    pruned_params = 0

    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            w = module.weight.data
            original_shape = w.shape
            flat = w.detach().cpu().numpy().flatten()
            pruned_flat = nm_prune_weights(flat, N=N, M_ratio=M_ratio)

            total_params += len(flat)
            pruned_params += np.count_nonzero(flat) - np.count_nonzero(pruned_flat)

            module.weight.data = torch.tensor(
                pruned_flat.reshape(original_shape),
                dtype=w.dtype, device=w.device
            )

    actual_sparsity = 1.0 - np.count_nonzero(pruned_flat) / len(pruned_flat) if len(flat) > 0 else 0
    print(f"  Applied 2:4 NM pruning: {total_params:,} params, "
          f"~50% sparsity enforced per group")
    return model


def profile_model_with_hooks(model, model_name, input_fn, device, n_samples=N_SAMPLES):
    """Profile a model using forward hooks, same as Idea 1's profiler."""

    records = []
    hooks = []

    def make_hook(name, module):
        def hook_fn(mod, inp, out):
            x = inp[0] if isinstance(inp, tuple) else inp
            y = out if isinstance(out, torch.Tensor) else (out[0] if isinstance(out, tuple) else None)
            if x is None or not isinstance(x, torch.Tensor):
                return
            if y is None or not isinstance(y, torch.Tensor):
                return

            input_shape = tuple(x.shape)
            output_shape = tuple(y.shape)

            # Dimensions
            if isinstance(mod, nn.Linear):
                K = mod.in_features
                P = mod.out_features
                M = 1
                for d in input_shape[:-1]:
                    M *= d
                op_type = 'MM' if M > 1 else 'MV'
            elif isinstance(mod, nn.Conv2d):
                Cin = mod.in_channels
                Cout = mod.out_channels
                kH, kW = mod.kernel_size
                K = Cin * kH * kW
                P = Cout
                batch = input_shape[0]
                H_out = output_shape[2] if len(output_shape) > 2 else 1
                W_out = output_shape[3] if len(output_shape) > 3 else 1
                M = batch * H_out * W_out
                op_type = 'MM'
            else:
                return

            # Sparsity
            input_sp = float((x == 0).sum().item() / x.numel()) if x.numel() > 0 else 0.0
            output_sp = float((y == 0).sum().item() / y.numel()) if y.numel() > 0 else 0.0
            weight_sp = float((mod.weight == 0).sum().item() / mod.weight.numel()) if hasattr(mod, 'weight') else 0.0

            # NM compatibility
            if hasattr(mod, 'weight'):
                nm_compat = nm_compatibility_score(mod.weight.detach().cpu())
            else:
                nm_compat = 0.0

            records.append({
                'input_idx': profile_model_with_hooks.current_idx,
                'layer_name': name,
                'layer_type': mod.__class__.__name__,
                'op_type': op_type,
                'M': M, 'K': K, 'P': P,
                'input_sparsity': round(input_sp, 4),
                'output_sparsity': round(output_sp, 4),
                'weight_sparsity': round(weight_sp, 4),
                'nm_compatibility': round(nm_compat, 4),
                'input_shape': str(input_shape),
                'output_shape': str(output_shape),
            })
        return hook_fn

    # Register hooks
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            h = module.register_forward_hook(make_hook(name, module))
            hooks.append(h)

    # Run inference
    model.eval()
    profile_model_with_hooks.current_idx = 0

    with torch.no_grad():
        for i in range(n_samples):
            profile_model_with_hooks.current_idx = i
            x = input_fn(device)
            model(x)
            if (i + 1) % 50 == 0:
                print(f"  Profiled {i+1}/{n_samples}...")

    # Remove hooks
    for h in hooks:
        h.remove()

    df = pd.DataFrame(records)
    return df


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ========================================================================
    # 1) ResNet50 with N:M pruning
    # ========================================================================
    print(f"\n{'='*60}")
    print(f"  Profiling: ResNet50 with 2:4 NM pruning")
    print(f"{'='*60}")

    from torchvision import models
    model_resnet = models.resnet50(weights=None).to(device)
    model_resnet = apply_nm_pruning_to_model(model_resnet)
    model_resnet.eval()

    def resnet_input(dev):
        return torch.randn(1, 3, 224, 224).to(dev)

    df_resnet = profile_model_with_hooks(
        model_resnet, "resnet50_nm24", resnet_input, device, n_samples=N_SAMPLES
    )

    path = os.path.join(OUTPUT_DIR, "resnet50_nm24_traces.csv")
    df_resnet.to_csv(path, index=False)

    print(f"\n  Summary for resnet50_nm24:")
    print(f"  Layers: {df_resnet['layer_name'].nunique()}")
    print(f"  Avg input sparsity:     {df_resnet['input_sparsity'].mean():.3f}")
    print(f"  Avg weight sparsity:    {df_resnet['weight_sparsity'].mean():.3f}")
    print(f"  Avg NM compatibility:   {df_resnet['nm_compatibility'].mean():.3f}")
    print(f"  Saved to {path}")

    # ========================================================================
    # 2) BERT with N:M pruning
    # ========================================================================
    print(f"\n{'='*60}")
    print(f"  Profiling: BERT-Base with 2:4 NM pruning")
    print(f"{'='*60}")

    try:
        from transformers import BertModel, BertConfig
    except ImportError:
        print("  WARNING: transformers not installed, skipping BERT NM")
        print(f"\n[DONE] NM profiling complete (ResNet only).")
        return

    config = BertConfig(
        hidden_size=768, num_hidden_layers=12,
        num_attention_heads=12, intermediate_size=3072,
    )
    model_bert = BertModel(config).to(device)
    model_bert = apply_nm_pruning_to_model(model_bert)
    model_bert.eval()

    # Profile BERT
    records_bert = []
    hooks_bert = []

    def make_bert_hook(name, module):
        def hook_fn(mod, inp, out):
            x = inp[0] if isinstance(inp, tuple) else inp
            y = out if isinstance(out, torch.Tensor) else (out[0] if isinstance(out, tuple) else None)
            if x is None or not isinstance(x, torch.Tensor):
                return
            if y is None or not isinstance(y, torch.Tensor):
                return

            if not isinstance(mod, nn.Linear):
                return

            K = mod.in_features
            P = mod.out_features
            input_shape = tuple(x.shape)
            M = 1
            for d in input_shape[:-1]:
                M *= d

            input_sp = float((x == 0).sum().item() / x.numel())
            output_sp = float((y == 0).sum().item() / y.numel())
            weight_sp = float((mod.weight == 0).sum().item() / mod.weight.numel())
            nm_compat = nm_compatibility_score(mod.weight.detach().cpu())

            records_bert.append({
                'input_idx': main.bert_idx,
                'layer_name': name,
                'layer_type': 'Linear',
                'op_type': 'MM',
                'M': M, 'K': K, 'P': P,
                'input_sparsity': round(input_sp, 4),
                'output_sparsity': round(output_sp, 4),
                'weight_sparsity': round(weight_sp, 4),
                'nm_compatibility': round(nm_compat, 4),
                'attention_sparsity': 0.0,
                'input_shape': str(input_shape),
                'output_shape': str(tuple(y.shape)),
            })
        return hook_fn

    for name, module in model_bert.named_modules():
        if isinstance(module, nn.Linear):
            h = module.register_forward_hook(make_bert_hook(name, module))
            hooks_bert.append(h)

    main.bert_idx = 0
    seq_lengths = [32, 64, 128]

    with torch.no_grad():
        for i in range(N_SAMPLES):
            main.bert_idx = i
            seq_len = seq_lengths[i % len(seq_lengths)]
            input_ids = torch.randint(0, 30522, (1, seq_len)).to(device)
            attention_mask = torch.ones(1, seq_len, dtype=torch.long).to(device)
            model_bert(input_ids=input_ids, attention_mask=attention_mask)
            if (i + 1) % 50 == 0:
                print(f"  Profiled {i+1}/{N_SAMPLES}...")

    for h in hooks_bert:
        h.remove()

    df_bert = pd.DataFrame(records_bert)
    path_bert = os.path.join(OUTPUT_DIR, "bert_nm24_traces.csv")
    df_bert.to_csv(path_bert, index=False)

    print(f"\n  Summary for bert_nm24:")
    print(f"  Layers: {df_bert['layer_name'].nunique()}")
    print(f"  Avg input sparsity:     {df_bert['input_sparsity'].mean():.3f}")
    print(f"  Avg weight sparsity:    {df_bert['weight_sparsity'].mean():.3f}")
    print(f"  Avg NM compatibility:   {df_bert['nm_compatibility'].mean():.3f}")
    print(f"  Saved to {path_bert}")

    print(f"\n[DONE] NM profiling complete!")
    print(f"  Re-run integrated_scheduler.py to see NM config selection.")


if __name__ == '__main__':
    main()
