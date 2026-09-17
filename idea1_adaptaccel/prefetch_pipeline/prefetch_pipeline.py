"""
prefetch_pipeline.py — Step 1.4: Prefetch pipeline model.

While layer L is computing, we can overlap:
  - Prefetching layer L+1's data from off-chip memory
  - Reconfiguring the accelerator for layer L+1's optimal config

This script models the overlap and reports how much overhead is hidden.

Pipeline diagram:
  Layer L:   [=== COMPUTE (Costcomp_L) ===]
  Overlap:         [PREFETCH L+1 data] [SWITCH config]
                   |<-- hidden_time -->|
  Layer L+1:                                [=== COMPUTE (Costcomp_L+1) ===]

  If compute_L > (prefetch_L+1 + switch_cost):
      → 100% overhead hidden (no stall)
  Else:
      → stall = (prefetch + switch) - compute_L

Usage:
    cd project/idea1_adaptaccel
    python prefetch_pipeline/prefetch_pipeline.py
"""

import numpy as np
import pandas as pd
import os
import sys
import glob

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(PROJECT_DIR))

from base_simulator.cost_model import CostModel
from base_simulator.memory_model import MemoryModel
from base_simulator.config_modes import ALL_CONFIGS

# Import scheduler
sys.path.insert(0, PROJECT_DIR)
from scheduler.dp_scheduler import (
    schedule_dp, get_layer_costs, CONFIG_NAMES, CONFIG_IDX, SWITCH_COST
)

LOOKAHEAD_W = 5


def compute_prefetch_time(layer_info, memory_model):
    """
    How many cycles to prefetch layer L+1's data from off-chip memory.
    
    Key optimizations modeled:
    1. Weights STATIC — pre-loaded, only re-fetched if exceed buffer.
    2. Activations flow on-chip — output buffer holds prev layer's result.
    3. TILING + DOUBLE BUFFERING for large matrices:
       - Break large data into tiles that fit in half the buffer
       - While computing tile T, prefetch tile T+1
       - Only the FIRST tile's fetch is a real stall
       - All subsequent tile fetches overlap with compute
       
    Effective overhead = first_tile_fetch + Σ max(0, tile_fetch - tile_compute)
    """
    import math
    
    M = int(layer_info['M'])
    K = int(layer_info['K'])
    P = int(layer_info['P'])
    weight_sp = float(layer_info['weight_sparsity'])

    INPUT_BUF_ELEMENTS = 65536   # 256KB / 4 bytes per FP32
    OUTPUT_BUF_ELEMENTS = 16384  # 64KB / 4 bytes per FP32
    # Double buffering: each half holds one tile
    TILE_BUF = INPUT_BUF_ELEMENTS // 2  # 32K elements per tile buffer

    density = 1.0 - weight_sp
    weight_elements = int(K * P * density)
    activation_elements = M * K

    # --- Weight tiling ---
    if weight_elements <= INPUT_BUF_ELEMENTS:
        # Fits entirely in buffer, no fetch needed
        weight_prefetch = 0
    elif weight_elements <= TILE_BUF:
        # Fits in one tile buffer half — single fetch, no tiling needed
        weight_prefetch = memory_model.memory_access_cycles(weight_elements)
    else:
        # Tile the weights: each tile = TILE_BUF elements
        n_tiles = math.ceil(weight_elements / TILE_BUF)
        per_tile_fetch = memory_model.memory_access_cycles(TILE_BUF)
        # Compute time per tile ≈ total_compute / n_tiles (rough estimate)
        # Use M*K*P/n_tiles as proxy for per-tile compute
        total_ops = M * K * P * density
        per_tile_compute = max(1, total_ops / (n_tiles * 64))  # 64 = 8×8 PEs
        
        # First tile: full stall (nothing to overlap with)
        first_tile = per_tile_fetch
        # Remaining tiles: only stall if fetch > compute
        remaining_stalls = sum(
            max(0, per_tile_fetch - per_tile_compute)
            for _ in range(n_tiles - 1)
        )
        weight_prefetch = first_tile + remaining_stalls

    # --- Activation tiling ---
    if activation_elements <= OUTPUT_BUF_ELEMENTS:
        # Already on-chip from previous layer's output
        activation_prefetch = 0
    elif activation_elements <= TILE_BUF:
        activation_prefetch = memory_model.memory_access_cycles(activation_elements)
    else:
        n_tiles = math.ceil(activation_elements / TILE_BUF)
        per_tile_fetch = memory_model.memory_access_cycles(TILE_BUF)
        total_ops = M * K * P * density
        per_tile_compute = max(1, total_ops / (n_tiles * 64))
        
        first_tile = per_tile_fetch
        remaining_stalls = sum(
            max(0, per_tile_fetch - per_tile_compute)
            for _ in range(n_tiles - 1)
        )
        activation_prefetch = first_tile + remaining_stalls

    total_prefetch = weight_prefetch + activation_prefetch

    # Sparse index overhead (only for off-chip weight portion)
    if weight_sp > 0.3 and weight_prefetch > 0:
        total_prefetch += int(weight_prefetch * 0.3)  # indices are smaller with tiling

    return total_prefetch


def run_prefetch_analysis(model_name, layers_df, cost_model, memory_model):
    """
    Simulate the prefetch pipeline for one model.

    Returns:
        dict with total_cycles (with/without prefetch), overhead hidden %, etc.
    """
    n_layers = len(layers_df)
    if n_layers < 2:
        return None

    # Step 1: Get DP schedule
    layer_costs_list = []
    for idx, row in layers_df.iterrows():
        costs = get_layer_costs(cost_model, row, sparsity=row['input_sparsity'])
        layer_costs_list.append(costs)

    dp_result = schedule_dp(layer_costs_list, lookahead=LOOKAHEAD_W)
    schedule = dp_result['schedule']

    # Step 2: Simulate pipeline
    total_compute_cycles = 0
    total_stall_cycles = 0
    total_prefetch_time = 0
    total_switch_time = 0
    total_overhead_hidden = 0
    n_switches = 0
    layers_with_full_hiding = 0

    for i in range(n_layers):
        cfg = schedule[i]
        compute_cost = layer_costs_list[i][cfg]
        total_compute_cycles += compute_cost

        if i < n_layers - 1:
            # While layer i computes, prefetch layer i+1's data and switch config
            next_layer = layers_df.iloc[i + 1]
            prefetch_time = compute_prefetch_time(next_layer, memory_model)
            switch_time = SWITCH_COST[CONFIG_IDX[schedule[i]]][CONFIG_IDX[schedule[i + 1]]]

            overhead = prefetch_time + switch_time
            total_prefetch_time += prefetch_time
            total_switch_time += switch_time

            if schedule[i] != schedule[i + 1]:
                n_switches += 1

            # How much can be hidden behind compute?
            hidden = min(compute_cost, overhead)
            stall = max(0, overhead - compute_cost)

            total_overhead_hidden += hidden
            total_stall_cycles += stall

            if stall == 0:
                layers_with_full_hiding += 1

    # Without prefetch: sequential execution
    no_prefetch_cycles = total_compute_cycles + total_prefetch_time + total_switch_time

    # With prefetch: overlapped execution
    with_prefetch_cycles = total_compute_cycles + total_stall_cycles

    # Overhead hidden percentage
    total_overhead = total_prefetch_time + total_switch_time
    overhead_hidden_pct = (total_overhead_hidden / total_overhead * 100) if total_overhead > 0 else 100.0

    # Speedup from prefetching
    speedup = no_prefetch_cycles / with_prefetch_cycles if with_prefetch_cycles > 0 else 1.0

    return {
        'model': model_name,
        'n_layers': n_layers,
        'n_switches': n_switches,
        'total_compute': total_compute_cycles,
        'total_prefetch': total_prefetch_time,
        'total_switch': total_switch_time,
        'total_stall': total_stall_cycles,
        'no_prefetch_cycles': no_prefetch_cycles,
        'with_prefetch_cycles': with_prefetch_cycles,
        'overhead_hidden_pct': overhead_hidden_pct,
        'layers_full_hide': layers_with_full_hiding,
        'layers_partial_hide': (n_layers - 1) - layers_with_full_hiding,
        'speedup': speedup,
    }


def main():
    print("=" * 65)
    print("  Step 1.4: Prefetch Pipeline Analysis")
    print("=" * 65)

    cost_model = CostModel(Nr=8, Nc=8)
    memory_model = MemoryModel()

    # Load all traces
    data_dir = os.path.join(PROJECT_DIR, "profiling_data")
    csv_files = sorted(glob.glob(os.path.join(data_dir, "*_traces.csv")))

    all_results = []

    for csv_file in csv_files:
        model_name = os.path.basename(csv_file).replace("_traces.csv", "")
        df = pd.read_csv(csv_file)

        # Use first inference pass
        single_pass = df[df['input_idx'] == 0].reset_index(drop=True)
        if len(single_pass) < 2:
            continue

        result = run_prefetch_analysis(model_name, single_pass, cost_model, memory_model)
        if result is None:
            continue

        all_results.append(result)

        print(f"\n  Model: {model_name} ({result['n_layers']} layers, "
              f"{result['n_switches']} switches)")
        print(f"  {'Metric':<30} {'Value':>12}")
        print(f"  {'-'*44}")
        print(f"  {'Total compute cycles':<30} {result['total_compute']:>12,}")
        print(f"  {'Total prefetch overhead':<30} {result['total_prefetch']:>12,}")
        print(f"  {'Total switch overhead':<30} {result['total_switch']:>12,}")
        print(f"  {'Without prefetch (sequential)':<30} {result['no_prefetch_cycles']:>12,}")
        print(f"  {'With prefetch (pipelined)':<30} {result['with_prefetch_cycles']:>12,}")
        print(f"  {'Stall cycles':<30} {result['total_stall']:>12,}")
        print(f"  {'Overhead hidden':<30} {result['overhead_hidden_pct']:>11.1f}%")
        print(f"  {'Layers with 100% hiding':<30} {result['layers_full_hide']:>10}/{result['n_layers']-1}")
        print(f"  {'Prefetch speedup':<30} {result['speedup']:>11.3f}×")

    # Summary
    print(f"\n\n{'='*65}")
    print(f"  SUMMARY")
    print(f"{'='*65}")
    print(f"  {'Model':<25} {'Overhead Hidden':>16} {'Speedup':>10} {'Target':>10}")
    print(f"  {'-'*62}")

    for r in all_results:
        status = "✓" if r['overhead_hidden_pct'] >= 85.0 else "✗"
        print(f"  {r['model']:<25} {r['overhead_hidden_pct']:>15.1f}% "
              f"{r['speedup']:>9.3f}× {status:>8} ≥85%")

    avg_hidden = np.mean([r['overhead_hidden_pct'] for r in all_results])
    avg_speedup = np.mean([r['speedup'] for r in all_results])
    print(f"\n  Average overhead hidden: {avg_hidden:.1f}%")
    print(f"  Average prefetch speedup: {avg_speedup:.3f}×")

    # Save
    results_df = pd.DataFrame(all_results)
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "prefetch_results.csv")
    results_df.to_csv(save_path, index=False)
    print(f"\n  Results saved to {save_path}")
    print(f"\n✓ Step 1.4 complete!")
    print(f"✓ Idea 1 (AdaptAccel) COMPLETE!")


if __name__ == '__main__':
    main()
