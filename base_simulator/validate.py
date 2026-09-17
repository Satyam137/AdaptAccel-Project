"""
validate.py — Validate simulator output against VersaAccel paper's Table II.

Run this script to check that the simulator's cycle counts match the paper's
reported values for the 2×2 array with 4×4 matrix examples.

Usage:
    cd project/software
    python -m simulator.validate
"""

import sys
import os

# Add parent directory to path so we can import simulator
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulator.versaaccel import VersaAccelSimulator
from simulator.memory_model import MemoryModel

# =============================================================================
# Paper's Table II: cycle counts for 2×2 array, 4×4 matrices
# =============================================================================
# Sparse matrix A from Fig. 3:
#   A = [a00  0  a02  0]     → nnz_A = 4, sparsity = 0.75
#       [ 0   0   0   0]       nnz_rows = 2 (rows 0 and 2)
#       [ 0  a12  0  a32]
#       [ 0   0   0   0]
#
# Sparse matrix B (similar pattern): nnz_B ≈ 4, sparsity = 0.75

PAPER_TABLE_II = {
    # Config: { operator: cycles }
    # Note: DMR has same cycles regardless of BW
    'DMR': {
        'MV': 20, 'MM': 16, 'SpMV': 20, 'SpMM': 16, 'SpMSpM': 16
    },
    # D-MR with sufficient BW (4 data/cycle)
    'D-MR_bw4': {
        'MV': 8, 'MM': 14, 'SpMV': 7, 'SpMM': 16, 'SpMSpM': 12
    },
    # D-MR with constrained BW (2 data/cycle)
    'D-MR_bw2': {
        'MV': 15, 'MM': 28, 'SpMV': 13, 'SpMM': 32, 'SpMSpM': 20
    },
    # DM-R (fixed 2 data/cycle BW)
    'DM-R': {
        'MV': 17, 'MM': 33, 'SpMV': 9, 'SpMM': 13, 'SpMSpM': 7
    },
    # D-M-R with sufficient BW (4 data/cycle)
    'D-M-R_bw4': {
        'MV': 6, 'MM': 12, 'SpMV': 5, 'SpMM': 7, 'SpMSpM': 6
    },
    # D-M-R with constrained BW (2 data/cycle)
    'D-M-R_bw2': {
        'MV': 12, 'MM': 22, 'SpMV': 9, 'SpMM': 13, 'SpMSpM': 11
    },
}


def run_validation():
    """Compare simulator output against paper's Table II."""

    # Create 2×2 simulator (matching paper's demo)
    # Use very high bandwidth so memory cost doesn't dominate
    mem = MemoryModel(hbm_bw_gbps=1000, freq_ghz=1.0)  # effectively infinite BW
    sim = VersaAccelSimulator(array_rows=2, array_cols=2,
                              hbm_bw_gbps=1000, freq_ghz=1.0)

    M, K, P = 4, 4, 4
    sparsity_A = 0.75  # 4 nonzeros out of 16
    sparsity_B = 0.75
    nnz_A = 4
    nnz_B = 4

    operators = ['MV', 'MM', 'SpMV', 'SpMM', 'SpMSpM']

    print("=" * 80)
    print("  VersaAccel Simulator Validation vs Paper Table II")
    print("  2×2 Array, 4×4 Matrices, Sparsity = 0.75")
    print("=" * 80)

    total_tests = 0
    passed = 0
    tolerance = 0.30  # 30% tolerance for cycle count matching

    # --- DMR ---
    print(f"\n  {'DMR':^10} | {'Op':^8} | {'Paper':^8} | {'Ours':^8} | {'Error':^8} | {'Status':^8}")
    print(f"  {'-'*60}")
    for op in operators:
        paper_val = PAPER_TABLE_II['DMR'][op]
        result = sim.run(op, M, K, P, sparsity_A, sparsity_B, config='DMR')
        our_val = result['costcomp']
        error = abs(our_val - paper_val) / paper_val
        status = "✓ PASS" if error <= tolerance else "✗ FAIL"
        if error <= tolerance:
            passed += 1
        total_tests += 1
        print(f"  {'DMR':^10} | {op:^8} | {paper_val:^8} | {our_val:^8} | {error:>6.1%}  | {status}")

    # --- D-MR (BW=4) ---
    print(f"\n  {'D-MR bw4':^10} | {'Op':^8} | {'Paper':^8} | {'Ours':^8} | {'Error':^8} | {'Status':^8}")
    print(f"  {'-'*60}")
    for op in operators:
        paper_val = PAPER_TABLE_II['D-MR_bw4'][op]
        result = sim.cost_model.cost_for_config(
            'D-MR', op, M, K, P, sparsity_A, sparsity_B, bw_override=4
        )
        our_val = result['costcomp']
        error = abs(our_val - paper_val) / paper_val
        status = "✓ PASS" if error <= tolerance else "✗ FAIL"
        if error <= tolerance:
            passed += 1
        total_tests += 1
        print(f"  {'D-MR bw4':^10} | {op:^8} | {paper_val:^8} | {our_val:^8} | {error:>6.1%}  | {status}")

    # --- D-MR (BW=2) ---
    print(f"\n  {'D-MR bw2':^10} | {'Op':^8} | {'Paper':^8} | {'Ours':^8} | {'Error':^8} | {'Status':^8}")
    print(f"  {'-'*60}")
    for op in operators:
        paper_val = PAPER_TABLE_II['D-MR_bw2'][op]
        result = sim.cost_model.cost_for_config(
            'D-MR', op, M, K, P, sparsity_A, sparsity_B, bw_override=2
        )
        our_val = result['costcomp']
        error = abs(our_val - paper_val) / paper_val
        status = "✓ PASS" if error <= tolerance else "✗ FAIL"
        if error <= tolerance:
            passed += 1
        total_tests += 1
        print(f"  {'D-MR bw2':^10} | {op:^8} | {paper_val:^8} | {our_val:^8} | {error:>6.1%}  | {status}")

    # --- DM-R ---
    print(f"\n  {'DM-R':^10} | {'Op':^8} | {'Paper':^8} | {'Ours':^8} | {'Error':^8} | {'Status':^8}")
    print(f"  {'-'*60}")
    for op in operators:
        paper_val = PAPER_TABLE_II['DM-R'][op]
        result = sim.run(op, M, K, P, sparsity_A, sparsity_B, config='DM-R')
        our_val = result['costcomp']
        error = abs(our_val - paper_val) / paper_val
        status = "✓ PASS" if error <= tolerance else "✗ FAIL"
        if error <= tolerance:
            passed += 1
        total_tests += 1
        print(f"  {'DM-R':^10} | {op:^8} | {paper_val:^8} | {our_val:^8} | {error:>6.1%}  | {status}")

    # --- D-M-R (BW=4) ---
    print(f"\n  {'D-M-R bw4':^10} | {'Op':^8} | {'Paper':^8} | {'Ours':^8} | {'Error':^8} | {'Status':^8}")
    print(f"  {'-'*60}")
    for op in operators:
        paper_val = PAPER_TABLE_II['D-M-R_bw4'][op]
        result = sim.cost_model.cost_for_config(
            'D-M-R', op, M, K, P, sparsity_A, sparsity_B, bw_override=4
        )
        our_val = result['costcomp']
        error = abs(our_val - paper_val) / paper_val
        status = "✓ PASS" if error <= tolerance else "✗ FAIL"
        if error <= tolerance:
            passed += 1
        total_tests += 1
        print(f"  {'D-M-R bw4':^10} | {op:^8} | {paper_val:^8} | {our_val:^8} | {error:>6.1%}  | {status}")

    # --- D-M-R (BW=2) ---
    print(f"\n  {'D-M-R bw2':^10} | {'Op':^8} | {'Paper':^8} | {'Ours':^8} | {'Error':^8} | {'Status':^8}")
    print(f"  {'-'*60}")
    for op in operators:
        paper_val = PAPER_TABLE_II['D-M-R_bw2'][op]
        result = sim.cost_model.cost_for_config(
            'D-M-R', op, M, K, P, sparsity_A, sparsity_B, bw_override=2
        )
        our_val = result['costcomp']
        error = abs(our_val - paper_val) / paper_val
        status = "✓ PASS" if error <= tolerance else "✗ FAIL"
        if error <= tolerance:
            passed += 1
        total_tests += 1
        print(f"  {'D-M-R bw2':^10} | {op:^8} | {paper_val:^8} | {our_val:^8} | {error:>6.1%}  | {status}")

    # --- Summary ---
    print(f"\n{'='*80}")
    print(f"  SUMMARY: {passed}/{total_tests} tests passed ({passed/total_tests:.0%})")
    print(f"  Tolerance: ±{tolerance:.0%}")
    print(f"{'='*80}")

    # --- Also run a quick 8×8 demo ---
    print(f"\n\n{'='*80}")
    print(f"  8×8 Array Demo — Comparing configs for different operators")
    print(f"{'='*80}")

    sim8 = VersaAccelSimulator()  # default 8×8

    print("\n--- Dense MM 64×64 ---")
    sim8.compare_configs('MM', M=64, K=64, P=64)

    print("--- Sparse MV 64×64, 70% sparse ---")
    sim8.compare_configs('SpMV', M=64, K=64, sparsity_A=0.7)

    print("--- SpMM 64×64, 80% sparse A ---")
    sim8.compare_configs('SpMM', M=64, K=64, P=64, sparsity_A=0.8)

    print("--- SpMSpM 64×64, 70% sparse both ---")
    sim8.compare_configs('SpMSpM', M=64, K=64, P=64, sparsity_A=0.7, sparsity_B=0.7)

    return passed, total_tests


if __name__ == '__main__':
    passed, total = run_validation()
    sys.exit(0 if passed == total else 1)
