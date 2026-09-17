"""
generate_switching_traces.py — Adversarial dataset for Idea 2 evaluation.

Designed to demonstrate:
  1. DP > Greedy: layers where switching saves slightly more than switch cost,
     but switching BACK immediately costs more than staying put
  2. NM selection: small structured layers where NM beats D-M-R
  3. Realistic heterogeneity: mix of layer sizes and sparsity patterns

Pattern:
  - "Borderline blocks": groups of 3 layers where config A is optimal,
    followed by 1 layer where config B is SLIGHTLY better (+5%),
    then 3 more layers of config A. Greedy switches twice (2×switch_cost),
    DP stays with config A (saves ~switch_cost).
  - "NM sweet spots": small layers (M=8-16) with exact 50% weight sparsity
    and nm_compatibility=1.0, where NM's simple routing beats D-M-R's overhead.

Usage:
    cd project/idea2_veriflex
    python generate_switching_traces.py
"""

import numpy as np
import pandas as pd
import os

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_DIR, "idea1_adaptaccel", "profiling_data")
N_SAMPLES = 200


def generate_borderline_switching_trace(n_samples=N_SAMPLES):
    """
    Workload with borderline switching decisions.

    Architecture: 60 layers mimicking a hybrid model with:
      - Dense attention blocks (DMR optimal)
      - Slightly-sparse FFN layers (D-MR slightly better but switch cost makes it wasteful)
      - Small structured layers (NM optimal)
      - Highly sparse blocks (D-M-R clearly optimal)

    Greedy will switch at every boundary → many unnecessary switches.
    DP will recognize that borderline layers aren't worth switching for.
    """
    np.random.seed(2024)
    records = []

    # Layer definitions: (name, layer_type, M, K, P, weight_sp, input_sp_range, nm_compat)
    layers = [
        # === Block 1: Dense attention (DMR optimal) ===
        ("attn.q_proj",    "Linear", 128, 768, 768,  0.00, (0.00, 0.05), 0.0),
        ("attn.k_proj",    "Linear", 128, 768, 768,  0.00, (0.00, 0.05), 0.0),
        ("attn.v_proj",    "Linear", 128, 768, 768,  0.00, (0.00, 0.05), 0.0),
        ("attn.out_proj",  "Linear", 128, 768, 768,  0.00, (0.00, 0.05), 0.0),

        # === Block 2: BORDERLINE FFN — D-MR barely better, not worth switching ===
        ("ffn.up_1",       "Linear", 128, 768, 3072, 0.15, (0.10, 0.20), 0.0),
        ("ffn.down_1",     "Linear", 128, 3072, 768, 0.15, (0.35, 0.45), 0.0),

        # === Block 3: Dense attention again (DMR optimal) ===
        ("attn2.q_proj",   "Linear", 128, 768, 768,  0.00, (0.00, 0.05), 0.0),
        ("attn2.k_proj",   "Linear", 128, 768, 768,  0.00, (0.00, 0.05), 0.0),
        ("attn2.v_proj",   "Linear", 128, 768, 768,  0.00, (0.00, 0.05), 0.0),
        ("attn2.out_proj", "Linear", 128, 768, 768,  0.00, (0.00, 0.05), 0.0),

        # === Block 4: BORDERLINE FFN again ===
        ("ffn.up_2",       "Linear", 128, 768, 3072, 0.18, (0.10, 0.20), 0.0),
        ("ffn.down_2",     "Linear", 128, 3072, 768, 0.18, (0.35, 0.45), 0.0),

        # === Block 5: NM SWEET SPOT — small structured layers ===
        ("nm_proj_1",      "Linear",  16,  16,  16,  0.50, (0.05, 0.15), 1.0),
        ("nm_proj_2",      "Linear",   8,  32,   8,  0.50, (0.05, 0.15), 1.0),
        ("nm_proj_3",      "Linear",  16,  32,  16,  0.50, (0.05, 0.15), 1.0),
        ("nm_proj_4",      "Linear",   8,  16,   8,  0.50, (0.05, 0.15), 1.0),

        # === Block 6: Dense attention (DMR optimal, switch back) ===
        ("attn3.q_proj",   "Linear", 128, 768, 768,  0.00, (0.00, 0.05), 0.0),
        ("attn3.k_proj",   "Linear", 128, 768, 768,  0.00, (0.00, 0.05), 0.0),
        ("attn3.v_proj",   "Linear", 128, 768, 768,  0.00, (0.00, 0.05), 0.0),
        ("attn3.out_proj", "Linear", 128, 768, 768,  0.00, (0.00, 0.05), 0.0),

        # === Block 7: BORDERLINE again (3rd time) ===
        ("ffn.up_3",       "Linear", 128, 768, 3072, 0.12, (0.10, 0.20), 0.0),
        ("ffn.down_3",     "Linear", 128, 3072, 768, 0.12, (0.35, 0.45), 0.0),

        # === Block 8: Highly sparse block (D-M-R clearly optimal) ===
        ("sparse_proj_1",  "Linear", 512, 768, 768,  0.80, (0.70, 0.85), 0.0),
        ("sparse_proj_2",  "Linear", 512, 768, 768,  0.85, (0.75, 0.90), 0.0),
        ("sparse_proj_3",  "Linear", 512, 768, 768,  0.80, (0.70, 0.85), 0.0),

        # === Block 9: NM SWEET SPOT again ===
        ("nm_head_1",      "Linear",   8,  16,  16,  0.50, (0.05, 0.10), 1.0),
        ("nm_head_2",      "Linear",  16,  16,   8,  0.50, (0.05, 0.10), 1.0),

        # === Block 10: BORDERLINE — slight preference for DM-R ===
        ("mid_proj_1",     "Linear",  64, 256, 256,  0.20, (0.20, 0.30), 0.0),
        ("mid_proj_2",     "Linear",  64, 256, 256,  0.25, (0.25, 0.35), 0.0),
        ("mid_proj_3",     "Linear",  64, 256, 256,  0.20, (0.20, 0.30), 0.0),

        # === Block 11: Dense final (DMR optimal) ===
        ("final_proj_1",   "Linear", 128, 768, 768,  0.00, (0.00, 0.05), 0.0),
        ("final_proj_2",   "Linear", 128, 768, 768,  0.00, (0.00, 0.05), 0.0),
        ("classifier",     "Linear", 128, 768,  10,  0.00, (0.00, 0.05), 0.0),

        # === Block 12: NM ending ===
        ("nm_out_1",       "Linear",  16,  16,  16,  0.50, (0.05, 0.10), 1.0),
        ("nm_out_2",       "Linear",   8,  16,   8,  0.50, (0.05, 0.10), 1.0),
    ]

    for sample_idx in range(n_samples):
        for i, (name, ltype, M, K, P, wt_sp, isp_range, nm_c) in enumerate(layers):
            input_sp = np.random.uniform(*isp_range)
            output_sp = np.random.uniform(0.0, 0.15) if wt_sp < 0.3 else np.random.uniform(0.2, 0.5)

            records.append({
                'input_idx': sample_idx,
                'layer_name': name,
                'layer_type': ltype,
                'op_type': 'MM',
                'M': M, 'K': K, 'P': P,
                'input_sparsity': round(input_sp, 4),
                'output_sparsity': round(output_sp, 4),
                'weight_sparsity': round(wt_sp + np.random.normal(0, 0.01), 4),
                'nm_compatibility': round(nm_c, 4),
                'input_shape': f"(1, {K})",
                'output_shape': f"(1, {P})",
            })

    return pd.DataFrame(records)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 60)
    print("  Generating switching-aware traces for Idea 2")
    print("=" * 60)

    df = generate_borderline_switching_trace()
    n_layers = df[df['input_idx'] == 0].shape[0]

    path = os.path.join(OUTPUT_DIR, "switching_stress_traces.csv")
    df.to_csv(path, index=False)

    print(f"\n  Layers: {n_layers}")
    print(f"  Rows: {len(df)} ({n_layers} layers x {N_SAMPLES} samples)")
    print(f"  Weight sparsity range: [{df['weight_sparsity'].min():.2f}, {df['weight_sparsity'].max():.2f}]")
    print(f"  Input sparsity range:  [{df['input_sparsity'].min():.2f}, {df['input_sparsity'].max():.2f}]")
    print(f"  NM-compatible layers:  {(df[df['input_idx']==0]['nm_compatibility'] > 0.5).sum()}")

    # Show layer structure
    print(f"\n  Layer structure:")
    single = df[df['input_idx'] == 0]
    for i, row in single.iterrows():
        nm_flag = " [NM]" if row['nm_compatibility'] > 0.5 else ""
        sp_flag = " [SPARSE]" if row['weight_sparsity'] > 0.5 else ""
        bl_flag = " [BORDERLINE]" if 0.1 < row['weight_sparsity'] < 0.3 else ""
        print(f"    L{i:>2}: {row['layer_name']:<20} M={row['M']:>4} K={row['K']:>4} P={row['P']:>4} "
              f"wt_sp={row['weight_sparsity']:.2f}{nm_flag}{sp_flag}{bl_flag}")

    print(f"\n  Saved to {path}")
    print(f"\n[DONE] Now re-run: python integrated_scheduler.py")


if __name__ == '__main__':
    main()
