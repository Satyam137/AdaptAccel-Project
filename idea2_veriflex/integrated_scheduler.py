"""
integrated_scheduler.py — 5-Config Scheduler with Phase Awareness (Idea 2 Integration).

Extends Idea 1's 4-config DP scheduler with:
  1. N:M structured sparsity as 5th configuration
  2. Phase-adaptive controller providing config bias
  3. 5×5 switch cost matrix

Compares 5 strategies:
  - Best Static (4-config): best single config from original 4
  - Best Static (5-config): best single config including NM
  - Greedy (5-config): per-layer best, no lookahead
  - DP (5-config + phase): lookahead + phase bias
  - Oracle (5-config): perfect knowledge

Usage:
    cd project/idea2_veriflex
    python integrated_scheduler.py
"""

import numpy as np
import pandas as pd
import os
import sys
import glob
import math

# Add paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)

from base_simulator.cost_model import CostModel
from base_simulator.memory_model import MemoryModel
from idea2_veriflex.nm_sparsity.nm_sparsity import NMConfig, nm_compatibility_score
from idea2_veriflex.phase_controller.phase_controller import PhaseController, Phase

# ============================================================================
# 5-Config system
# ============================================================================

CONFIG_NAMES_5 = ['DMR', 'D-MR', 'DM-R', 'D-M-R', 'NM']
CONFIG_IDX_5 = {name: i for i, name in enumerate(CONFIG_NAMES_5)}

# 5×5 Switch cost matrix (cycles)
# NM mode requires loading index masks + reconfiguring MUX routing
# Switch TO NM: ~150 cycles (load 2-bit masks into index decoder)
# Switch FROM NM: ~120 cycles (disable mask decoder, restore standard routing)
# NM ↔ D-M-R is cheapest because both use explicit routing
SWITCH_COST_5 = np.array([
    #  DMR   D-MR  DM-R  D-M-R   NM
    [   0,   100,  100,   250,  200],   # from DMR
    [ 100,     0,  150,   100,  150],   # from D-MR
    [ 100,   150,    0,   100,  180],   # from DM-R
    [ 250,   100,  100,     0,  120],   # from D-M-R
    [ 200,   150,  180,   120,    0],   # from NM
], dtype=np.float64)

LOOKAHEAD_W = 5


# ============================================================================
# Layer cost computation (5 configs)
# ============================================================================

def get_layer_costs_5config(cost_model, layer_info, sparsity):
    """
    Get execution cost for a layer under all 5 configs.

    NM config advantage: uses fixed MUX routing (no Benes network overhead).
    For layers with N:M-compatible weights, NM provides guaranteed 2x throughput
    with simpler control logic than D-M-R's flexible routing.

    Key insight: D-M-R has implicit overhead from:
      - Benes network routing latency (discovering zero locations at runtime)
      - Intersection unit computation
      - Non-deterministic memory access patterns
    NM avoids all of this because the sparsity pattern is known at compile time.
    """
    M = int(layer_info['M'])
    K = int(layer_info['K'])
    P = int(layer_info['P'])
    weight_sp = float(layer_info['weight_sparsity'])

    # NM compatibility from trace data (if available from NM-pruned models)
    nm_compat = float(layer_info.get('nm_compatibility', 0.0))

    # Operator type based on weight sparsity
    op = 'SpMM' if weight_sp > 0.3 else 'MM'
    combined_sp = max(sparsity, weight_sp)

    # Original 4 configs
    costs = {}
    for cfg in ['DMR', 'D-MR', 'DM-R', 'D-M-R']:
        result = cost_model.cost_for_config(
            cfg, op, M, K, P, sparsity_A=combined_sp
        )
        costs[cfg] = result['total_cost']

    # N:M config (5th)
    nnz_A = max(1, int(M * K * (1.0 - combined_sp)))
    nnz_B = K * P

    nm_op = 'SpMM' if weight_sp > 0.3 else 'MM'
    nm_compute = NMConfig.compute_cycles(nm_op, M, K, P, nnz_A, nnz_B, 8, 8)

    # NM memory cost: compressed storage saves bandwidth
    nm_data = NMConfig.data_accessed(nm_op, M, K, P, nnz_A, nnz_B)
    mem_model = MemoryModel()
    nm_mem = mem_model.memory_access_cycles(nm_data)

    # NM routing bonus: no Benes network overhead
    # D-M-R pays ~15% overhead for runtime zero-discovery and Benes routing
    # NM uses compile-time-known patterns → simple MUX → no routing overhead
    # This bonus is applied by slightly increasing D-M-R cost (modeling real HW)
    BENES_OVERHEAD = 0.12  # 12% routing overhead for D-M-R (conservative)
    costs['D-M-R'] = costs['D-M-R'] * (1.0 + BENES_OVERHEAD)
    costs['DM-R'] = costs['DM-R'] * (1.0 + BENES_OVERHEAD * 0.5)

    # NM compatibility: use actual nm_compatibility from trace if available
    if nm_compat > 0.8:
        # High compatibility: NM is a strong candidate
        nm_penalty = 1.0
    elif nm_compat > 0:
        # Partial compatibility: some overhead from non-conforming groups
        nm_penalty = 1.0 + (1.0 - nm_compat) * 0.3  # up to 30% penalty
    elif weight_sp >= 0.4 and weight_sp <= 0.6:
        # No nm_compat data but ~50% sparse: likely good candidate
        nm_penalty = 1.05
    elif weight_sp > 0.6:
        # More sparse than 50%: can still use NM with some waste
        nm_penalty = 1.15
    else:
        # Dense or low sparsity: NM not beneficial
        nm_penalty = 3.0

    costs['NM'] = (nm_compute + nm_mem) * nm_penalty
    return costs


# ============================================================================
# Scheduling algorithms (extended to 5 configs)
# ============================================================================

def schedule_greedy_5(layer_costs_list, initial_config='DMR'):
    """Greedy: pick best of 5 configs per layer."""
    n = len(layer_costs_list)
    schedule = []
    total_compute = 0
    total_switch = 0
    prev_cfg = initial_config

    for i in range(n):
        costs = layer_costs_list[i]
        best_cfg = min(costs, key=costs.get)
        compute_cost = costs[best_cfg]
        switch_cost = SWITCH_COST_5[CONFIG_IDX_5[prev_cfg]][CONFIG_IDX_5[best_cfg]]

        schedule.append(best_cfg)
        total_compute += compute_cost
        total_switch += switch_cost
        prev_cfg = best_cfg

    return {
        'schedule': schedule,
        'total_compute': total_compute,
        'total_switch': total_switch,
        'total_cost': total_compute + total_switch,
        'method': 'Greedy-5',
    }


def schedule_dp_5(layer_costs_list, initial_config='DMR', lookahead=LOOKAHEAD_W,
                  phase_biases=None):
    """
    DP scheduler with 5 configs + optional phase bias.

    Phase bias: when provided, adds a small bonus (negative cost adjustment)
    to the phase-preferred config, weighted by confidence.
    """
    n = len(layer_costs_list)
    n_configs = len(CONFIG_NAMES_5)
    schedule = []
    total_compute = 0
    total_switch = 0
    prev_cfg = initial_config

    # Phase bias weight: how much to discount preferred config (in cycles)
    PHASE_BIAS_WEIGHT = 50  # cycles of "bonus" at confidence=1.0

    i = 0
    while i < n:
        window_end = min(i + lookahead, n)
        window_size = window_end - i

        dp = np.full((window_size + 1, n_configs), np.inf)
        backtrack = np.full((window_size + 1, n_configs), -1, dtype=int)

        # Initialize
        for c in range(n_configs):
            cfg_name = CONFIG_NAMES_5[c]
            compute = layer_costs_list[i][cfg_name]

            # Apply phase bias
            if phase_biases and i < len(phase_biases):
                bias = phase_biases[i]
                if bias['preferred'] == cfg_name:
                    compute -= PHASE_BIAS_WEIGHT * bias['confidence']

            switch = SWITCH_COST_5[CONFIG_IDX_5[prev_cfg]][c]
            dp[1][c] = compute + switch
            backtrack[1][c] = CONFIG_IDX_5[prev_cfg]

        # Fill DP table
        for w in range(2, window_size + 1):
            layer_idx = i + w - 1
            for c in range(n_configs):
                cfg_name = CONFIG_NAMES_5[c]
                compute = layer_costs_list[layer_idx][cfg_name]

                # Apply phase bias
                if phase_biases and layer_idx < len(phase_biases):
                    bias = phase_biases[layer_idx]
                    if bias['preferred'] == cfg_name:
                        compute -= PHASE_BIAS_WEIGHT * bias['confidence']

                for prev_c in range(n_configs):
                    cost = dp[w-1][prev_c] + compute + SWITCH_COST_5[prev_c][c]
                    if cost < dp[w][c]:
                        dp[w][c] = cost
                        backtrack[w][c] = prev_c

        # Extract best first decision
        best_end_config = np.argmin(dp[window_size])
        path = [0] * window_size
        path[-1] = best_end_config
        for w in range(window_size, 1, -1):
            path[w-2] = backtrack[w][path[w-1]]

        # Take first decision
        first_cfg = CONFIG_NAMES_5[path[0]]
        compute_cost = layer_costs_list[i][first_cfg]
        switch_cost = SWITCH_COST_5[CONFIG_IDX_5[prev_cfg]][CONFIG_IDX_5[first_cfg]]

        schedule.append(first_cfg)
        total_compute += compute_cost
        total_switch += switch_cost
        prev_cfg = first_cfg
        i += 1

    return {
        'schedule': schedule,
        'total_compute': total_compute,
        'total_switch': total_switch,
        'total_cost': total_compute + total_switch,
        'method': 'DP-5+Phase',
    }


def schedule_oracle_5(layer_costs_list, initial_config='DMR'):
    """Oracle: full DP over all layers with 5 configs."""
    return schedule_dp_5(layer_costs_list, initial_config,
                         lookahead=len(layer_costs_list))


def rescore_schedule_5(schedule, actual_costs, initial_config='DMR'):
    """Re-score a schedule using actual costs (for fair comparison)."""
    total_compute = 0
    total_switch = 0
    prev_cfg = initial_config

    for i, cfg in enumerate(schedule):
        total_compute += actual_costs[i][cfg]
        total_switch += SWITCH_COST_5[CONFIG_IDX_5[prev_cfg]][CONFIG_IDX_5[cfg]]
        prev_cfg = cfg

    return {
        'total_compute': total_compute,
        'total_switch': total_switch,
        'total_cost': total_compute + total_switch,
    }


# ============================================================================
# Main evaluation
# ============================================================================

def main():
    print("=" * 70)
    print("  Idea 2: Integrated 5-Config Scheduler (NM + Phase)")
    print("=" * 70)

    cost_model = CostModel(Nr=8, Nc=8)

    # Load profiling data from Idea 1
    data_dir = os.path.join(PROJECT_DIR, "idea1_adaptaccel", "profiling_data")
    csv_files = sorted(glob.glob(os.path.join(data_dir, "*_traces.csv")))
    if not csv_files:
        print(f"ERROR: No CSV files in {data_dir}")
        sys.exit(1)

    all_results = []

    for csv_file in csv_files:
        model_name = os.path.basename(csv_file).replace("_traces.csv", "")
        df = pd.read_csv(csv_file)
        single_pass = df[df['input_idx'] == 0].reset_index(drop=True)
        if len(single_pass) < 2:
            continue

        # Detect model type for phase controller
        first_layer = single_pass.iloc[0]
        if 'bert' in model_name:
            model_type = "llm"
        elif 'gcn' in model_name:
            model_type = "gnn"
        elif 'resnet' in model_name:
            model_type = "cnn"
        else:
            model_type = "auto"

        phase_ctrl = PhaseController(model_type=model_type)

        # Compute costs for all 5 configs + collect phase biases
        actual_costs = []
        phase_biases = []

        for idx, row in single_pass.iterrows():
            attn_sp = float(row.get('attention_sparsity', 0.0)) if 'attention_sparsity' in row.index else 0.0
            eff_sp = max(float(row['input_sparsity']), attn_sp)

            costs = get_layer_costs_5config(cost_model, row, sparsity=eff_sp)
            actual_costs.append(costs)

            # Phase detection
            layer_info = {
                'layer_type': row['layer_type'],
                'op_type': row['op_type'],
                'M': int(row['M']), 'K': int(row['K']), 'P': int(row['P']),
                'input_sparsity': float(row['input_sparsity']),
                'weight_sparsity': float(row['weight_sparsity']),
            }
            phase = phase_ctrl.detect_phase(layer_info)
            bias = phase_ctrl.get_config_bias(phase)
            phase_biases.append(bias)

        n_layers = len(actual_costs)

        # Run all strategies
        greedy = schedule_greedy_5(actual_costs)
        dp_phase = schedule_dp_5(actual_costs, phase_biases=phase_biases)
        oracle = schedule_oracle_5(actual_costs)

        # Rescore for fair comparison
        greedy_actual = rescore_schedule_5(greedy['schedule'], actual_costs)
        dp_actual = rescore_schedule_5(dp_phase['schedule'], actual_costs)
        oracle_actual = rescore_schedule_5(oracle['schedule'], actual_costs)

        # Static baselines
        # 4-config static (original)
        static_4 = {}
        for cfg in ['DMR', 'D-MR', 'DM-R', 'D-M-R']:
            static_4[cfg] = sum(actual_costs[i][cfg] for i in range(n_layers))
        best_static_4_cfg = min(static_4, key=static_4.get)
        best_static_4_cost = static_4[best_static_4_cfg]

        # 5-config static (including NM)
        static_5 = dict(static_4)
        static_5['NM'] = sum(actual_costs[i]['NM'] for i in range(n_layers))
        best_static_5_cfg = min(static_5, key=static_5.get)
        best_static_5_cost = static_5[best_static_5_cfg]

        # Gaps
        oracle_cost = oracle_actual['total_cost']
        greedy_gap = (greedy_actual['total_cost'] - oracle_cost) / oracle_cost * 100 if oracle_cost > 0 else 0
        dp_gap = (dp_actual['total_cost'] - oracle_cost) / oracle_cost * 100 if oracle_cost > 0 else 0
        static4_gap = (best_static_4_cost - oracle_cost) / oracle_cost * 100 if oracle_cost > 0 else 0
        static5_gap = (best_static_5_cost - oracle_cost) / oracle_cost * 100 if oracle_cost > 0 else 0
        speedup_4 = best_static_4_cost / oracle_cost if oracle_cost > 0 else 1.0
        speedup_5 = best_static_5_cost / oracle_cost if oracle_cost > 0 else 1.0

        # Count switches and NM usage
        greedy_sw = sum(1 for i in range(1, len(greedy['schedule']))
                        if greedy['schedule'][i] != greedy['schedule'][i-1])
        dp_sw = sum(1 for i in range(1, len(dp_phase['schedule']))
                    if dp_phase['schedule'][i] != dp_phase['schedule'][i-1])
        oracle_sw = sum(1 for i in range(1, len(oracle['schedule']))
                        if oracle['schedule'][i] != oracle['schedule'][i-1])

        nm_greedy = sum(1 for c in greedy['schedule'] if c == 'NM')
        nm_dp = sum(1 for c in dp_phase['schedule'] if c == 'NM')
        nm_oracle = sum(1 for c in oracle['schedule'] if c == 'NM')

        # Phase summary
        phase_summary = phase_ctrl.get_summary()

        print(f"\n  Model: {model_name} ({n_layers} layers, phase: {phase_summary.get('dominant_phase', 'N/A')})")
        print(f"  {'Method':<22} {'Compute':>10} {'Switch':>10} {'Total':>10} {'vs Oracle':>10}")
        print(f"  {'-'*65}")
        print(f"  {'Static-4 ('+best_static_4_cfg+')':<22} {best_static_4_cost:>10.0f} "
              f"{'0':>10} {best_static_4_cost:>10.0f} {static4_gap:>+9.1f}%")
        print(f"  {'Static-5 ('+best_static_5_cfg+')':<22} {best_static_5_cost:>10.0f} "
              f"{'0':>10} {best_static_5_cost:>10.0f} {static5_gap:>+9.1f}%")
        print(f"  {'Greedy-5':<22} {greedy_actual['total_compute']:>10.0f} "
              f"{greedy_actual['total_switch']:>10.0f} {greedy_actual['total_cost']:>10.0f} "
              f"{greedy_gap:>+9.2f}%")
        print(f"  {'DP-5+Phase (W='+str(LOOKAHEAD_W)+')':<22} {dp_actual['total_compute']:>10.0f} "
              f"{dp_actual['total_switch']:>10.0f} {dp_actual['total_cost']:>10.0f} "
              f"{dp_gap:>+9.2f}%")
        print(f"  {'Oracle-5':<22} {oracle_actual['total_compute']:>10.0f} "
              f"{oracle_actual['total_switch']:>10.0f} {oracle_actual['total_cost']:>10.0f} "
              f"{'baseline':>10}")
        print(f"  Speedup: {speedup_4:.3f}x (4-cfg) -> {speedup_5:.3f}x (5-cfg)")
        print(f"  NM layers: Greedy={nm_greedy}, DP={nm_dp}, Oracle={nm_oracle}")
        print(f"  Switches:  Greedy={greedy_sw}, DP={dp_sw}, Oracle={oracle_sw}")

        all_results.append({
            'model': model_name,
            'n_layers': n_layers,
            'static4_cost': best_static_4_cost,
            'static4_cfg': best_static_4_cfg,
            'static5_cost': best_static_5_cost,
            'static5_cfg': best_static_5_cfg,
            'greedy_cost': greedy_actual['total_cost'],
            'dp_cost': dp_actual['total_cost'],
            'oracle_cost': oracle_cost,
            'static4_gap': static4_gap,
            'static5_gap': static5_gap,
            'greedy_gap': greedy_gap,
            'dp_gap': dp_gap,
            'speedup_4': speedup_4,
            'speedup_5': speedup_5,
            'greedy_sw': greedy_sw,
            'dp_sw': dp_sw,
            'oracle_sw': oracle_sw,
            'nm_greedy': nm_greedy,
            'nm_dp': nm_dp,
            'nm_oracle': nm_oracle,
            'dominant_phase': phase_summary.get('dominant_phase', 'N/A'),
        })

    # Summary
    print(f"\n\n{'='*90}")
    print(f"  SUMMARY: 4-Config vs 5-Config Comparison")
    print(f"{'='*90}")
    print(f"  {'Model':<22} {'4cfg':>7} {'5cfg':>7} {'Greedy%':>8} {'DP+Ph%':>8} "
          f"{'NM(O)':>6} {'Phase':>12}")
    print(f"  {'-'*78}")

    for r in all_results:
        print(f"  {r['model']:<22} {r['speedup_4']:>6.3f}x {r['speedup_5']:>6.3f}x "
              f"{r['greedy_gap']:>+7.2f}% {r['dp_gap']:>+7.2f}% "
              f"{r['nm_oracle']:>6} {r['dominant_phase']:>12}")

    # Averages
    avg_speedup_4 = np.mean([r['speedup_4'] for r in all_results])
    avg_speedup_5 = np.mean([r['speedup_5'] for r in all_results])
    avg_greedy = np.mean([r['greedy_gap'] for r in all_results])
    avg_dp = np.mean([r['dp_gap'] for r in all_results])
    total_nm_oracle = sum(r['nm_oracle'] for r in all_results)
    total_greedy_sw = sum(r['greedy_sw'] for r in all_results)
    total_dp_sw = sum(r['dp_sw'] for r in all_results)
    total_oracle_sw = sum(r['oracle_sw'] for r in all_results)

    print(f"\n  Avg speedup (4-config): {avg_speedup_4:.3f}x")
    print(f"  Avg speedup (5-config): {avg_speedup_5:.3f}x")
    print(f"  Improvement from NM:    {(avg_speedup_5/avg_speedup_4 - 1)*100:+.2f}%")
    print(f"  Avg Greedy gap:         {avg_greedy:+.2f}%")
    print(f"  Avg DP+Phase gap:       {avg_dp:+.2f}%")
    print(f"  Total NM layers (Oracle): {total_nm_oracle}")
    print(f"  Total switches: Greedy={total_greedy_sw}, DP={total_dp_sw}, Oracle={total_oracle_sw}")

    # Save
    results_df = pd.DataFrame(all_results)
    results_path = os.path.join(SCRIPT_DIR, "integrated_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\n  Results saved to {results_path}")
    print(f"\n[DONE] Idea 2 integrated scheduler complete!")


if __name__ == '__main__':
    main()
