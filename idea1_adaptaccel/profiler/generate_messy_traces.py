"""
generate_messy_traces.py — Generate synthetic "messy" workload traces.

Creates 3 challenging workloads that stress-test the predictor and scheduler:

1. GCN-like: Sparsity swings wildly (10% → 95% → 30% → 85%)
2. Mixed pipeline: Alternates between dense MM, sparse SpMV, SpMSpM
3. Variable model: Random layer sizes, random sparsity, random op types

Saves traces in the same CSV format as the real profiler output.

Usage:
    cd project/idea1_adaptaccel
    python profiler/generate_messy_traces.py
"""

import numpy as np
import pandas as pd
import os

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "profiling_data")
N_SAMPLES = 200  # Number of inference passes


def generate_gcn_trace(n_samples=N_SAMPLES):
    """
    GCN-like workload: alternates between sparse aggregation (SpMM)
    and dense transformation (MM). Sparsity varies wildly per layer.
    
    Typical GCN: aggregate(adj × features) → transform(features × weights)
    Adjacency matrix sparsity: 85-99% (very sparse)
    Feature matrix: 0-50% sparse (varies with ReLU)
    """
    np.random.seed(123)
    layers = [
        # (name, type, op, M, K, P, weight_sp_base, input_sp_pattern)
        ("gcn.aggregate_1", "Linear", "MM", 2048, 2048, 128, 0.0, "high_sparse"),    # adj × features
        ("gcn.transform_1", "Linear", "MM", 2048, 128, 256, 0.0, "low_sparse"),      # dense transform
        ("gcn.aggregate_2", "Linear", "MM", 2048, 2048, 256, 0.0, "high_sparse"),    # adj × features
        ("gcn.transform_2", "Linear", "MM", 2048, 256, 128, 0.3, "medium_sparse"),   # pruned transform
        ("gcn.aggregate_3", "Linear", "MM", 2048, 2048, 128, 0.0, "very_high_sparse"), # very sparse adj
        ("gcn.transform_3", "Linear", "MM", 2048, 128, 64, 0.5, "low_sparse"),      # heavily pruned
        ("gcn.aggregate_4", "Linear", "MM", 2048, 2048, 64, 0.0, "high_sparse"),     # sparse again
        ("gcn.output", "Linear", "MM", 2048, 64, 7, 0.7, "medium_sparse"),           # final classify
    ]

    sparsity_patterns = {
        "high_sparse": lambda: np.random.uniform(0.80, 0.95),
        "very_high_sparse": lambda: np.random.uniform(0.90, 0.99),
        "medium_sparse": lambda: np.random.uniform(0.30, 0.60),
        "low_sparse": lambda: np.random.uniform(0.0, 0.20),
    }

    records = []
    for sample_idx in range(n_samples):
        for layer_info in layers:
            name, ltype, op, M, K, P, weight_sp, sp_pattern = layer_info
            input_sp = sparsity_patterns[sp_pattern]()
            # Output sparsity: some transformation of input (ReLU-like for transforms)
            if "transform" in name:
                output_sp = np.random.uniform(0.3, 0.6)  # ReLU creates ~40-60% zeros
            else:
                output_sp = np.random.uniform(0.0, 0.1)  # aggregation usually dense output
            
            records.append({
                'input_idx': sample_idx,
                'layer_name': name,
                'layer_type': ltype,
                'op_type': op,
                'M': M, 'K': K, 'P': P,
                'input_sparsity': round(input_sp, 4),
                'output_sparsity': round(output_sp, 4),
                'weight_sparsity': round(weight_sp + np.random.normal(0, 0.02), 4),
                'input_shape': f"(1, {K})",
                'output_shape': f"(1, {P})",
                'model_name': 'gcn_synthetic',
            })

    return pd.DataFrame(records)


def generate_mixed_pipeline_trace(n_samples=N_SAMPLES):
    """
    Mixed pipeline: alternates between very different operation types.
    Simulates a real application doing attention + FFN + sparse retrieval.
    Sparsity changes drastically between layers.
    """
    np.random.seed(456)
    layers = [
        # Attention (dense, large matrices)
        ("attn.qkv_proj", "Linear", "MM", 512, 768, 2304, 0.0),
        ("attn.score", "Linear", "MM", 512, 512, 512, 0.0),
        # Sparse attention mask (very sparse)
        ("attn.masked", "Linear", "MM", 512, 512, 512, 0.85),
        # Dense FFN
        ("ffn.up", "Linear", "MM", 512, 768, 3072, 0.0),
        # Pruned FFN (medium sparse)
        ("ffn.down", "Linear", "MM", 512, 3072, 768, 0.60),
        # Sparse retrieval (very sparse, small)
        ("retrieval.lookup", "Linear", "MM", 64, 10000, 128, 0.95),
        # Dense output
        ("output.proj", "Linear", "MM", 512, 768, 768, 0.0),
        # Sparse classifier (pruned)
        ("classifier", "Linear", "MM", 512, 768, 10, 0.70),
    ]

    records = []
    for sample_idx in range(n_samples):
        for i, (name, ltype, op, M, K, P, weight_sp) in enumerate(layers):
            # Input sparsity varies wildly depending on previous layer
            if "masked" in name or "retrieval" in name:
                input_sp = np.random.uniform(0.70, 0.95)  # very sparse input
            elif "ffn.down" in name:
                input_sp = np.random.uniform(0.40, 0.65)  # ReLU output
            elif "classifier" in name:
                input_sp = np.random.uniform(0.50, 0.80)  # dropout + sparse
            else:
                input_sp = np.random.uniform(0.0, 0.15)   # dense input
            
            output_sp = np.random.uniform(0.0, 0.3)

            records.append({
                'input_idx': sample_idx,
                'layer_name': name,
                'layer_type': ltype,
                'op_type': op,
                'M': M, 'K': K, 'P': P,
                'input_sparsity': round(input_sp, 4),
                'output_sparsity': round(output_sp, 4),
                'weight_sparsity': round(np.clip(weight_sp + np.random.normal(0, 0.01), 0, 1), 4),
                'input_shape': f"(1, {K})",
                'output_shape': f"(1, {P})",
                'model_name': 'mixed_pipeline',
            })

    return pd.DataFrame(records)


def generate_variable_model_trace(n_samples=N_SAMPLES):
    """
    Completely random model: random sizes, random sparsity, random everything.
    This is the hardest test — no patterns for the predictor to learn.
    """
    np.random.seed(789)
    n_layers = 20

    records = []
    for sample_idx in range(n_samples):
        prev_output_sp = 0.0  # track for some continuity
        for i in range(n_layers):
            M = np.random.choice([32, 64, 128, 256, 512, 1024, 2048])
            K = np.random.choice([32, 64, 128, 256, 512, 1024])
            P = np.random.choice([32, 64, 128, 256, 512])
            weight_sp = np.random.uniform(0.0, 0.95)
            
            # Input sparsity has some correlation with previous output
            input_sp = np.clip(prev_output_sp + np.random.normal(0, 0.15), 0, 0.99)
            
            # Output sparsity is somewhat random
            if np.random.random() > 0.5:  # has ReLU
                output_sp = np.random.uniform(0.3, 0.7)
            else:
                output_sp = np.random.uniform(0.0, 0.15)
            
            prev_output_sp = output_sp

            records.append({
                'input_idx': sample_idx,
                'layer_name': f"random_layer_{i}",
                'layer_type': np.random.choice(['Conv2d', 'Linear']),
                'op_type': 'MM',
                'M': M, 'K': K, 'P': P,
                'input_sparsity': round(input_sp, 4),
                'output_sparsity': round(output_sp, 4),
                'weight_sparsity': round(weight_sp, 4),
                'input_shape': f"(1, {K})",
                'output_shape': f"(1, {P})",
                'model_name': 'variable_random',
            })

    return pd.DataFrame(records)


def generate_dp_stress_trace(n_samples=N_SAMPLES):
    """
    ADVERSARIAL workload designed to show DP > Greedy.

    40 moderate layers where configs differ between MM and SpMM ops.
    Pattern: clusters of 5 "dense" layers (weight_sp=0, MM operator),
    then 1 isolated "sparse outlier" (weight_sp=0.75, SpMM operator).

    MM and SpMM prefer different configs on the same dimensions.
    Greedy: switches config at each outlier boundary → many switches
    DP (lookahead=5): sees outlier is isolated, may skip switching → fewer switches
    """
    np.random.seed(999)
    records = []

    for sample_idx in range(n_samples):
        for i in range(40):
            if i % 6 == 5:  # every 6th layer is a sparse outlier
                # Sparse outlier: weight_sp=0.75 → SpMM operator
                # Different optimal config than dense layers
                M, K, P = 256, 512, 128
                weight_sp = 0.75
                input_sp = np.random.uniform(0.50, 0.70)
                output_sp = np.random.uniform(0.30, 0.50)
            else:
                # Dense majority: weight_sp=0.0 → MM operator
                M, K, P = 256, 512, 128
                weight_sp = 0.0
                input_sp = np.random.uniform(0.0, 0.10)
                output_sp = np.random.uniform(0.0, 0.15)

            records.append({
                'input_idx': sample_idx,
                'layer_name': f"stress_layer_{i}",
                'layer_type': 'Linear',
                'op_type': 'MM',
                'M': M, 'K': K, 'P': P,
                'input_sparsity': round(input_sp, 4),
                'output_sparsity': round(output_sp, 4),
                'weight_sparsity': round(weight_sp, 4),
                'input_shape': f"(1, {K})",
                'output_shape': f"(1, {P})",
                'model_name': 'dp_stress',
            })

    return pd.DataFrame(records)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Generating messy synthetic traces...\n")

    # 1) GCN
    df_gcn = generate_gcn_trace()
    path = os.path.join(OUTPUT_DIR, "gcn_synthetic_traces.csv")
    df_gcn.to_csv(path, index=False)
    print(f"  GCN-like:       {len(df_gcn):>6} rows, "
          f"sparsity range [{df_gcn['input_sparsity'].min():.2f}, {df_gcn['input_sparsity'].max():.2f}]")

    # 2) Mixed pipeline
    df_mixed = generate_mixed_pipeline_trace()
    path = os.path.join(OUTPUT_DIR, "mixed_pipeline_traces.csv")
    df_mixed.to_csv(path, index=False)
    print(f"  Mixed pipeline: {len(df_mixed):>6} rows, "
          f"sparsity range [{df_mixed['input_sparsity'].min():.2f}, {df_mixed['input_sparsity'].max():.2f}]")

    # 3) Variable random
    df_var = generate_variable_model_trace()
    path = os.path.join(OUTPUT_DIR, "variable_random_traces.csv")
    df_var.to_csv(path, index=False)
    print(f"  Variable model: {len(df_var):>6} rows, "
          f"sparsity range [{df_var['input_sparsity'].min():.2f}, {df_var['input_sparsity'].max():.2f}]")

    # 4) DP stress test (adversarial)
    df_stress = generate_dp_stress_trace()
    path = os.path.join(OUTPUT_DIR, "dp_stress_traces.csv")
    df_stress.to_csv(path, index=False)
    print(f"  DP stress:      {len(df_stress):>6} rows, "
          f"sparsity range [{df_stress['input_sparsity'].min():.2f}, {df_stress['input_sparsity'].max():.2f}]")

    print(f"\n✓ All messy traces saved to {OUTPUT_DIR}")
    print(f"  Now re-run the predictor and scheduler to test on this data.")


if __name__ == '__main__':
    main()
