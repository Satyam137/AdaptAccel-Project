"""
profile_bert.py — Profile BERT-Base on random sequence data.

Profiles dense + pruned (50%, 70%, 90%) versions.
Captures per-layer sparsity in all Linear layers of BERT
(attention Q/K/V projections, FFN layers).

Usage:
    cd project/idea1_adaptaccel
    python -m profiler.profile_bert

Requires: pip install transformers
"""

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from profiler.workload_profiler import LayerProfiler

N_SAMPLES = 200
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "profiling_data")


def apply_unstructured_pruning(model, sparsity):
    """Apply global unstructured L1 pruning to all Linear layers."""
    parameters_to_prune = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            parameters_to_prune.append((module, 'weight'))
    if len(parameters_to_prune) == 0:
        return model
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=sparsity,
    )
    for module, _ in parameters_to_prune:
        prune.remove(module, 'weight')
    return model


def profile_bert(model, model_name, device, seq_lengths=[32, 64, 128]):
    """Profile a single BERT configuration, including attention-score sparsity."""
    print(f"\n{'='*60}")
    print(f"  Profiling: {model_name}")
    print(f"{'='*60}")

    profiler = LayerProfiler(model, model_name=model_name)
    profiler.register_hooks()
    profiler.records = []

    # --- Attention sparsity hooks ---
    # Capture softmax attention weights: fraction near-zero (< threshold)
    # This is the input-dependent sparsity that varies per sentence/token
    ATTN_THRESHOLD = 0.01  # attention weights below this are "effectively zero"
    attn_sparsities = {}   # layer_idx → list of sparsity values per input
    attn_hooks = []

    def make_attn_hook(layer_idx):
        def hook_fn(mod, inp, out):
            # BertSelfAttention output includes attention_probs when output_attentions=True
            # But we hook the softmax's input/output directly via the dropout layer
            # Instead, access the attention_probs attribute if available
            if hasattr(mod, 'attention_probs_') and mod.attention_probs_ is not None:
                probs = mod.attention_probs_.detach()
                sp = float((probs < ATTN_THRESHOLD).sum().item() / probs.numel())
                attn_sparsities.setdefault(layer_idx, []).append(sp)
        return hook_fn

    # Hook into softmax to capture attention probs
    softmax_hooks = []
    def make_softmax_capture_hook(layer_idx, attn_module):
        """Hook after the matmul(Q, K^T) / sqrt(d) + mask, before/after softmax."""
        original_forward = attn_module.forward

        def patched_forward(*args, **kwargs):
            # Force output_attentions to capture attention weights
            kwargs['output_attentions'] = True
            outputs = original_forward(*args, **kwargs)
            # outputs = (context_layer, attention_probs) when output_attentions=True
            if isinstance(outputs, tuple) and len(outputs) >= 2:
                attn_probs = outputs[1].detach()
                sp = float((attn_probs < ATTN_THRESHOLD).sum().item() / attn_probs.numel())
                attn_sparsities.setdefault(layer_idx, []).append(sp)
            return outputs

        attn_module.forward = patched_forward
        return original_forward  # return original to restore later

    # Patch attention modules
    original_forwards = []
    try:
        from transformers.models.bert.modeling_bert import BertSelfAttention
        # BertModel: layers at model.encoder.layer
        # BertForXxx: layers at model.bert.encoder.layer
        if hasattr(model, 'encoder'):
            encoder_layers = model.encoder.layer
        elif hasattr(model, 'bert'):
            encoder_layers = model.bert.encoder.layer
        else:
            raise AttributeError("Cannot find encoder layers")
        
        for layer_idx, layer in enumerate(encoder_layers):
            attn = layer.attention.self
            orig = make_softmax_capture_hook(layer_idx, attn)
            original_forwards.append((attn, orig))
        print(f"  Hooked {len(original_forwards)} attention layers for sparsity capture")
    except Exception as e:
        print(f"  Warning: Could not hook attention layers: {e}")

    count = 0
    with torch.no_grad():
        for i in range(N_SAMPLES):
            seq_len = seq_lengths[i % len(seq_lengths)]
            input_ids = torch.randint(0, 30522, (1, seq_len)).to(device)
            attention_mask = torch.ones(1, seq_len, dtype=torch.long).to(device)
            profiler._input_idx = count
            model(input_ids=input_ids, attention_mask=attention_mask)
            count += 1
            if count % 50 == 0:
                print(f"  Profiled {count}/{N_SAMPLES} inputs...")

    profiler.remove_hooks()

    # Restore original attention forwards
    for attn, orig in original_forwards:
        attn.forward = orig

    import pandas as pd
    df = pd.DataFrame(profiler.records)

    # Add attention sparsity to relevant layers
    # Map each record to its attention layer (if applicable)
    if attn_sparsities:
        # Each attention layer produces Q, K, V, output projections
        # Assign attention sparsity to the layers in that block
        attn_sp_list = []
        for _, row in df.iterrows():
            layer_name = row['layer_name']
            input_idx = row['input_idx']
            # Extract layer index from name (e.g., "bert.encoder.layer.3.attention...")
            attn_sp = 0.0
            for lid in attn_sparsities:
                if f"layer.{lid}." in layer_name and "attention" in layer_name:
                    idx_in_list = input_idx if input_idx < len(attn_sparsities[lid]) else -1
                    if idx_in_list >= 0 and idx_in_list < len(attn_sparsities[lid]):
                        attn_sp = attn_sparsities[lid][idx_in_list]
                    break
            attn_sp_list.append(attn_sp)
        df['attention_sparsity'] = attn_sp_list
    else:
        df['attention_sparsity'] = 0.0

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, f"{model_name}_traces.csv")
    df.to_csv(path, index=False)

    print(f"\n  Summary for {model_name}:")
    print(f"  Layers profiled: {df['layer_name'].nunique()}")
    print(f"  Total records: {len(df)}")
    print(f"  Avg input sparsity:     {df['input_sparsity'].mean():.3f}")
    print(f"  Avg output sparsity:    {df['output_sparsity'].mean():.3f}")
    print(f"  Avg weight sparsity:    {df['weight_sparsity'].mean():.3f}")
    print(f"  Avg attention sparsity: {df['attention_sparsity'].mean():.3f}")
    if attn_sparsities:
        all_attn = [s for v in attn_sparsities.values() for s in v]
        print(f"  Attention sparsity range: [{min(all_attn):.3f}, {max(all_attn):.3f}]")
    print(f"  Saved to {path}")


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    try:
        from transformers import BertModel, BertConfig
    except ImportError:
        print("ERROR: 'transformers' not found. Run: pip install transformers")
        sys.exit(1)

    def make_bert():
        config = BertConfig(
            hidden_size=768, num_hidden_layers=12,
            num_attention_heads=12, intermediate_size=3072,
        )
        return BertModel(config).to(device).eval()

    # 1) Dense BERT
    profile_bert(make_bert(), "bert_dense", device)

    # 2) Pruned 50%
    model_50 = make_bert()
    apply_unstructured_pruning(model_50, 0.50)
    profile_bert(model_50, "bert_pruned50", device)

    # 3) Pruned 70%
    model_70 = make_bert()
    apply_unstructured_pruning(model_70, 0.70)
    profile_bert(model_70, "bert_pruned70", device)

    # 4) Pruned 90%
    model_90 = make_bert()
    apply_unstructured_pruning(model_90, 0.90)
    profile_bert(model_90, "bert_pruned90", device)

    print(f"\n✓ All BERT profiles saved to {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
