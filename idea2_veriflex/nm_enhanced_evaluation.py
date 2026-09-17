"""
nm_enhanced_evaluation.py — NM as D-M-R enhancer (not separate config).

Key insight: NM structure doesn't compete with D-M-R — it HELPS D-M-R.

Normal D-M-R (unstructured sparsity):
  - Runtime zero discovery via intersection unit
  - Dynamic Benes routing (calculate per batch)
  - Irregular memory access (random zero locations)

D-M-R with NM structure (2:4 weights):
  - Zero positions KNOWN at compile time → skip intersection
  - Precomputed Benes routes → save routing setup per row batch
  - Regular compressed format → better memory bandwidth

Model: D-M-R+NM = D-M-R × (1 - NM_discount)
  where NM_discount depends on nm_compatibility score

Usage:
    cd project
    python idea2_veriflex/nm_enhanced_evaluation.py
"""

import numpy as np
import pandas as pd
import os
import sys
import glob
import math

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)

from base_simulator.cost_model import CostModel
from base_simulator.memory_model import MemoryModel
from base_simulator.config_modes import FullExplicitConfig, _ceil, _estimate_nnz_rows
from idea2_veriflex.phase_controller.phase_controller import PhaseController


# ============================================================================
# NM-Enhanced D-M-R Cost Model
# ============================================================================

def compute_nm_discount(nm_compatibility, weight_sparsity):
    """
    Compute the cycle discount D-M-R gets from NM-structured weights.

    Three sources of savings:
      1. Intersection bypass:  ~8-12% of D-M-R cost (skip zero discovery)
      2. Precomputed routing:  ~5-8% (Benes routes known at compile time)
      3. Regular memory:       ~3-5% (compressed format, predictable access)

    Total: up to ~20-25% discount at 100% NM compatibility.
    Scaled by nm_compatibility (0.0 = no benefit, 1.0 = full benefit).
    """
    if nm_compatibility <= 0 or weight_sparsity < 0.3:
        return 0.0

    # Base discount components
    intersection_saving = 0.12   # 12% from skipping runtime zero discovery
    routing_saving = 0.08        # 8% from precomputed Benes routes
    memory_saving = 0.05         # 5% from regular access pattern

    total_discount = intersection_saving + routing_saving + memory_saving  # 25%

    # Scale by NM compatibility and sparsity level
    # Higher sparsity → more intersection work to skip → more savings
    sparsity_factor = min(1.0, weight_sparsity / 0.5)  # normalized to 50%
    effective_discount = total_discount * nm_compatibility * sparsity_factor

    return min(effective_discount, 0.30)  # cap at 30%


def get_layer_costs_nm_enhanced(cost_model, layer_info, sparsity):
    """
    Compute costs for 4 configs + D-M-R+NM variant.

    Configs:
      DMR, D-MR, DM-R, D-M-R (original 4)
      D-M-R+NM: D-M-R with NM discount (when nm_compatible)
    """
    M = int(layer_info['M'])
    K = int(layer_info['K'])
    P = int(layer_info['P'])
    weight_sp = float(layer_info['weight_sparsity'])
    nm_compat = float(layer_info.get('nm_compatibility', 0.0))

    op = 'SpMM' if weight_sp > 0.3 else 'MM'
    combined_sp = max(sparsity, weight_sp)

    costs = {}
    for cfg in ['DMR', 'D-MR', 'DM-R', 'D-M-R']:
        result = cost_model.cost_for_config(cfg, op, M, K, P, sparsity_A=combined_sp)
        costs[cfg] = result['total_cost']

    # D-M-R+NM: enhanced version with NM discount
    discount = compute_nm_discount(nm_compat, weight_sp)
    costs['D-M-R+NM'] = costs['D-M-R'] * (1.0 - discount)

    return costs, discount


# ============================================================================
# Scheduling with NM enhancement
# ============================================================================

CONFIG_NAMES_NM = ['DMR', 'D-MR', 'DM-R', 'D-M-R', 'D-M-R+NM']
CONFIG_IDX_NM = {name: i for i, name in enumerate(CONFIG_NAMES_NM)}

# Switch costs: D-M-R ↔ D-M-R+NM is FREE (same hardware, just enable NM optimization)
SWITCH_COST_NM = np.array([
    #  DMR   D-MR  DM-R  D-M-R  D-M-R+NM
    [   0,   100,  100,   250,   250],   # from DMR
    [ 100,     0,  150,   100,   100],   # from D-MR
    [ 100,   150,    0,   100,   100],   # from DM-R
    [ 250,   100,  100,     0,     0],   # from D-M-R (NM toggle is FREE)
    [ 250,   100,  100,     0,     0],   # from D-M-R+NM (same HW as D-M-R)
], dtype=np.float64)


def schedule_greedy_nm(layer_costs_list):
    """Greedy: pick best of 5 options per layer."""
    schedule = []
    total_compute = 0
    total_switch = 0
    prev_cfg = 'DMR'

    for costs in layer_costs_list:
        best_cfg = min(costs, key=costs.get)
        compute = costs[best_cfg]
        switch = SWITCH_COST_NM[CONFIG_IDX_NM[prev_cfg]][CONFIG_IDX_NM[best_cfg]]
        schedule.append(best_cfg)
        total_compute += compute
        total_switch += switch
        prev_cfg = best_cfg

    return {
        'schedule': schedule,
        'total_compute': total_compute,
        'total_switch': total_switch,
        'total_cost': total_compute + total_switch,
    }


def schedule_oracle_nm(layer_costs_list):
    """Oracle: full DP over all layers."""
    n = len(layer_costs_list)
    n_cfgs = len(CONFIG_NAMES_NM)
    INF = float('inf')

    # DP[i][c] = min total cost to schedule layers 0..i with layer i using config c
    dp = np.full((n, n_cfgs), INF)
    backtrack = np.full((n, n_cfgs), -1, dtype=int)

    # Init first layer
    for c in range(n_cfgs):
        cfg = CONFIG_NAMES_NM[c]
        dp[0][c] = layer_costs_list[0][cfg] + SWITCH_COST_NM[0][c]  # from DMR

    # Fill
    for i in range(1, n):
        for c in range(n_cfgs):
            cfg = CONFIG_NAMES_NM[c]
            compute = layer_costs_list[i][cfg]
            for prev_c in range(n_cfgs):
                cost = dp[i-1][prev_c] + compute + SWITCH_COST_NM[prev_c][c]
                if cost < dp[i][c]:
                    dp[i][c] = cost
                    backtrack[i][c] = prev_c

    # Extract schedule
    last_cfg = int(np.argmin(dp[n-1]))
    schedule = [0] * n
    schedule[n-1] = last_cfg
    for i in range(n-2, -1, -1):
        schedule[i] = backtrack[i+1][schedule[i+1]]

    schedule_names = [CONFIG_NAMES_NM[c] for c in schedule]
    total_compute = sum(layer_costs_list[i][schedule_names[i]] for i in range(n))
    total_switch = SWITCH_COST_NM[0][schedule[0]]  # initial switch
    for i in range(1, n):
        total_switch += SWITCH_COST_NM[schedule[i-1]][schedule[i]]

    return {
        'schedule': schedule_names,
        'total_compute': total_compute,
        'total_switch': total_switch,
        'total_cost': total_compute + total_switch,
    }


# ============================================================================
# Main evaluation
# ============================================================================

def main():
    print("=" * 80)
    print("  NM-Enhanced D-M-R Evaluation")
    print("  (NM as D-M-R optimizer, not separate config)")
    print("=" * 80)

    cost_model = CostModel(Nr=8, Nc=8)
    data_dir = os.path.join(PROJECT_DIR, "idea1_adaptaccel", "profiling_data")
    csv_files = sorted(glob.glob(os.path.join(data_dir, "*_traces.csv")))

    all_results = []

    for csv_file in csv_files:
        model_name = os.path.basename(csv_file).replace("_traces.csv", "")
        df = pd.read_csv(csv_file)
        single_pass = df[df['input_idx'] == 0].reset_index(drop=True)
        if len(single_pass) < 2:
            continue

        if 'nm_compatibility' not in single_pass.columns:
            single_pass['nm_compatibility'] = 0.0
        if 'attention_sparsity' not in single_pass.columns:
            single_pass['attention_sparsity'] = 0.0

        n_layers = len(single_pass)
        layer_costs_list = []
        total_discount = 0
        nm_enhanced_count = 0

        for idx, row in single_pass.iterrows():
            attn_sp = float(row.get('attention_sparsity', 0.0))
            eff_sp = max(float(row['input_sparsity']), attn_sp)
            costs, discount = get_layer_costs_nm_enhanced(cost_model, row, eff_sp)
            layer_costs_list.append(costs)
            total_discount += discount
            if discount > 0:
                nm_enhanced_count += 1

        # Run scheduling
        greedy = schedule_greedy_nm(layer_costs_list)
        oracle = schedule_oracle_nm(layer_costs_list)

        # Static baselines
        static_4 = {}
        for cfg in ['DMR', 'D-MR', 'DM-R', 'D-M-R']:
            static_4[cfg] = sum(layer_costs_list[i][cfg] for i in range(n_layers))
        best_static_4 = min(static_4, key=static_4.get)
        best_static_4_cost = static_4[best_static_4]

        static_5 = dict(static_4)
        static_5['D-M-R+NM'] = sum(layer_costs_list[i]['D-M-R+NM'] for i in range(n_layers))
        best_static_5 = min(static_5, key=static_5.get)
        best_static_5_cost = static_5[best_static_5]

        # Metrics
        oracle_cost = oracle['total_cost']
        greedy_gap = (greedy['total_cost'] - oracle_cost) / oracle_cost * 100 if oracle_cost > 0 else 0
        static4_gap = (best_static_4_cost - oracle_cost) / oracle_cost * 100 if oracle_cost > 0 else 0
        improvement = (best_static_4_cost - best_static_5_cost) / best_static_4_cost * 100

        # Count D-M-R+NM usage
        nm_greedy = sum(1 for c in greedy['schedule'] if c == 'D-M-R+NM')
        nm_oracle = sum(1 for c in oracle['schedule'] if c == 'D-M-R+NM')
        avg_discount = total_discount / n_layers if n_layers > 0 else 0

        print(f"\n  Model: {model_name} ({n_layers} layers, "
              f"{nm_enhanced_count} NM-eligible, avg discount={avg_discount:.1%})")
        print(f"  {'Method':<22} {'Total':>10} {'vs Oracle':>10} {'NM-enh':>7}")
        print(f"  {'-'*55}")
        print(f"  {'Static-4 ('+best_static_4+')':<22} {best_static_4_cost:>10.0f} "
              f"{static4_gap:>+9.1f}% {'---':>7}")
        print(f"  {'Static-5 ('+best_static_5+')':<22} {best_static_5_cost:>10.0f} "
              f"{improvement:>+9.1f}% {'---':>7}")
        print(f"  {'Greedy-5':<22} {greedy['total_cost']:>10.0f} "
              f"{greedy_gap:>+9.2f}% {nm_greedy:>7}")
        print(f"  {'Oracle-5':<22} {oracle_cost:>10.0f} {'baseline':>10} {nm_oracle:>7}")

        all_results.append({
            'model': model_name,
            'n_layers': n_layers,
            'nm_eligible': nm_enhanced_count,
            'avg_discount': avg_discount,
            'static4_cost': best_static_4_cost,
            'static5_cost': best_static_5_cost,
            'improvement': improvement,
            'greedy_cost': greedy['total_cost'],
            'oracle_cost': oracle_cost,
            'nm_greedy': nm_greedy,
            'nm_oracle': nm_oracle,
        })

    # Summary
    print(f"\n\n{'='*90}")
    print(f"  SUMMARY: NM Enhancement Impact")
    print(f"{'='*90}")
    print(f"  {'Model':<22} {'NM-elig':>8} {'Discount':>9} {'4cfg→5cfg':>10} "
          f"{'NM(Greedy)':>11} {'NM(Oracle)':>11}")
    print(f"  {'-'*75}")

    for r in all_results:
        print(f"  {r['model']:<22} {r['nm_eligible']:>8} {r['avg_discount']:>8.1%} "
              f"{r['improvement']:>+9.2f}% {r['nm_greedy']:>11} {r['nm_oracle']:>11}")

    # Aggregates
    total_nm_greedy = sum(r['nm_greedy'] for r in all_results)
    total_nm_oracle = sum(r['nm_oracle'] for r in all_results)
    avg_improvement = np.mean([r['improvement'] for r in all_results])
    nm_models = [r for r in all_results if r['nm_eligible'] > 0]
    avg_improvement_nm = np.mean([r['improvement'] for r in nm_models]) if nm_models else 0

    print(f"\n  Avg improvement (all models):     {avg_improvement:+.2f}%")
    print(f"  Avg improvement (NM-eligible):     {avg_improvement_nm:+.2f}%")
    print(f"  Total NM-enhanced layers (Greedy): {total_nm_greedy}")
    print(f"  Total NM-enhanced layers (Oracle): {total_nm_oracle}")

    # Save
    results_df = pd.DataFrame(all_results)
    path = os.path.join(SCRIPT_DIR, "nm_enhanced_results.csv")
    results_df.to_csv(path, index=False)
    print(f"\n  Saved to {path}")
    print(f"\n[DONE] NM-enhanced evaluation complete!")


if __name__ == '__main__':
    main()
