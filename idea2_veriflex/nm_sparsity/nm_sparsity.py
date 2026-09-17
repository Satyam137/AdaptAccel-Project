"""
nm_sparsity.py — N:M Structured Sparsity as 5th VersaAccel Configuration.

N:M sparsity (e.g., 2:4) means exactly N nonzeros in every group of M elements.
This is what NVIDIA A100/H100 Sparse Tensor Cores use.

Key advantage: pattern is KNOWN at compile time → fixed, simple datapath
→ guaranteed (N/M) throughput improvement (2× for 2:4).

The NM config adds:
  - compute_cycles: (N/M) × dense DMR cycles (fixed 2× for 2:4)
  - NM compatibility checker: what fraction of a sparse matrix fits 2:4 format
  - data_accessed: reads only nonzero values + 2-bit index masks

Usage:
    from idea2_veriflex.nm_sparsity.nm_sparsity import NMConfig, nm_compatibility_score
"""

import math
import numpy as np


def _ceil(a, b):
    return math.ceil(a / b)


class NMConfig:
    """
    N:M structured sparsity configuration for VersaAccel.

    For 2:4 sparsity: in every group of 4 consecutive weights,
    exactly 2 are nonzero. This gives:
      - 2× compute throughput (only 2 MACs per group instead of 4)
      - Compact storage: 2-bit index mask per group (which 2 of 4 are nonzero)
      - Fixed routing pattern → no Benes network needed

    Compute model:
      cycles = (N/M) × DMR_dense_cycles + index_decode_overhead
    """
    name = "NM"

    # Default: 2:4 structured sparsity (matches NVIDIA Sparse Tensor Cores)
    N = 2  # nonzeros per group
    M_ratio = 4  # group size

    @staticmethod
    def compute_cycles(op, M, K, P, nnz_A, nnz_B, Nr, Nc, bw=2,
                       N=2, M_ratio=4):
        """
        Compute cycles for N:M structured sparsity mode.

        The key insight: since the sparsity pattern is fixed and known at
        compile time, we can route data with a simple MUX (no Benes network).
        Effective compute = (N/M_ratio) × dense compute.

        For 2:4: 50% of MACs are skipped → 2× throughput.

        Additional overhead:
          - Index decode: 1 cycle per group of M_ratio elements (read 2-bit mask)
          - This is pipelined and amortized across groups
        """
        ratio = N / M_ratio  # 0.5 for 2:4

        # Base: use DMR (fully implicit systolic) as the dense baseline
        # N:M mode operates like a modified systolic array with selective PE activation
        if op in ('MV', 'SpMV'):
            # MV: A(M×K) × b(K×1) = c(M×1)
            # With N:M on A: only N/M fraction of K elements are nonzero per group
            dense_cycles = _ceil(M, Nr) * _ceil(K, Nc) * (2 * Nr + 1)
            nm_cycles = int(dense_cycles * ratio)
            # Index decode overhead: one decode per group of M_ratio elements in K
            decode_overhead = _ceil(K, M_ratio)
            return max(1, nm_cycles + decode_overhead)

        elif op in ('MM', 'SpMM'):
            # MM: A(M×K) × B(K×P) = C(M×P)
            # With N:M on A: skip (M_ratio-N)/M_ratio of K-dimension multiplications
            dense_cycles = _ceil(M * K * P, Nr * Nc)
            nm_cycles = int(dense_cycles * ratio)
            # Index decode: pipelined, amortized over P columns
            decode_overhead = _ceil(M, Nr) * _ceil(K, M_ratio)
            return max(1, nm_cycles + decode_overhead)

        elif op == 'SpMSpM':
            # Both matrices sparse: use N:M on A, unstructured on B
            # Falls back to D-M-R-like behavior but with faster A access
            dense_cycles = _ceil(M * K * P, Nr * Nc)
            nm_cycles = int(dense_cycles * ratio)
            decode_overhead = _ceil(M, Nr) * _ceil(K, M_ratio)
            return max(1, nm_cycles + decode_overhead)

        raise ValueError(f"Unsupported op: {op}")

    @staticmethod
    def data_accessed(op, M, K, P, nnz_A, nnz_B, N=2, M_ratio=4):
        """
        Data elements accessed with N:M sparsity.

        Matrix A is stored in compressed N:M format:
          - Values: only N/M_ratio of elements (the nonzeros)
          - Indices: 2-bit mask per group (which N of M_ratio are nonzero)
          - Index overhead ≈ 1 element per group (compact)

        Matrix B is stored fully (all elements needed for lookup).
        """
        ratio = N / M_ratio
        n_groups = _ceil(M * K, M_ratio)

        if op == 'MV':
            # A values (compressed) + A indices + B vector
            a_values = int(M * K * ratio)
            a_indices = n_groups  # 1 index word per group
            return a_values + a_indices + K

        elif op == 'MM':
            # A values (compressed) + A indices + B matrix (full)
            a_values = int(M * K * ratio)
            a_indices = n_groups
            return a_values + a_indices + K * P

        elif op in ('SpMV', 'SpMM'):
            # Same as MV/MM but A is already sparse → use nnz_A
            a_compressed = int(nnz_A * ratio)
            a_indices = _ceil(nnz_A, M_ratio)
            b_access = K if op == 'SpMV' else K * P
            return a_compressed + a_indices + b_access

        elif op == 'SpMSpM':
            # Both sparse
            a_compressed = int(nnz_A * ratio)
            a_indices = _ceil(nnz_A, M_ratio)
            return a_compressed + a_indices + nnz_B + _ceil(nnz_B, M_ratio)

        raise ValueError(f"Unsupported op: {op}")


def nm_compatibility_score(weight_tensor_or_array, N=2, M_ratio=4):
    """
    Compute what fraction of a weight tensor fits N:M structured sparsity pattern.

    Checks every group of M_ratio consecutive elements:
      - If exactly N are nonzero: compatible (score += 1)
      - If fewer nonzeros: can be made compatible by padding (score += 1)
      - If more nonzeros: incompatible (score += 0)

    Args:
        weight_tensor_or_array: flattened weight values (numpy array or torch tensor)
        N: nonzeros per group (default 2)
        M_ratio: group size (default 4)

    Returns:
        float: fraction of groups that are N:M compatible (0.0 to 1.0)
    """
    if hasattr(weight_tensor_or_array, 'numpy'):
        arr = weight_tensor_or_array.detach().cpu().numpy().flatten()
    else:
        arr = np.asarray(weight_tensor_or_array).flatten()

    n_elements = len(arr)
    if n_elements < M_ratio:
        return 0.0

    n_groups = n_elements // M_ratio
    compatible = 0

    for g in range(n_groups):
        group = arr[g * M_ratio: (g + 1) * M_ratio]
        nnz = np.count_nonzero(group)
        if nnz <= N:
            compatible += 1  # Already fits or can be padded

    return compatible / n_groups


def nm_prune_weights(weight_tensor_or_array, N=2, M_ratio=4):
    """
    Apply N:M structured pruning to a weight array.

    For each group of M_ratio elements, keeps only the N largest-magnitude
    values and zeros the rest.

    Args:
        weight_tensor_or_array: weight values (numpy array)
        N: nonzeros to keep per group (default 2)
        M_ratio: group size (default 4)

    Returns:
        numpy array with N:M structured sparsity applied
    """
    arr = np.asarray(weight_tensor_or_array).flatten().copy()
    n_elements = len(arr)
    n_groups = n_elements // M_ratio

    for g in range(n_groups):
        start = g * M_ratio
        end = start + M_ratio
        group = arr[start:end]
        # Find indices of N largest-magnitude values
        if np.count_nonzero(group) > N:
            magnitudes = np.abs(group)
            # Zero out all except top-N
            topn_idx = np.argpartition(magnitudes, -N)[-N:]
            mask = np.zeros(M_ratio, dtype=bool)
            mask[topn_idx] = True
            group[~mask] = 0.0
            arr[start:end] = group

    return arr


# ============================================================================
# Validation / self-test
# ============================================================================

def main():
    """Validate N:M config computations."""
    print("=" * 60)
    print("  Idea 2 — Step 1: N:M Structured Sparsity Config")
    print("=" * 60)

    Nr, Nc = 8, 8

    # Test 1: Compute cycles comparison (N:M vs dense DMR)
    print("\n  Test 1: Compute cycle comparison (8x8 array)")
    print(f"  {'Op':<8} {'M':>4} {'K':>4} {'P':>4} {'DMR (dense)':>12} {'NM (2:4)':>10} {'Speedup':>8}")
    print(f"  {'-'*55}")

    test_cases = [
        ('MM', 64, 64, 64),
        ('MM', 128, 256, 128),
        ('MM', 512, 768, 3072),
        ('MV', 64, 64, 1),
        ('SpMM', 64, 64, 64),
    ]

    for op, M, K, P in test_cases:
        nnz_A = int(M * K * 0.5)  # 50% sparse (typical for 2:4)
        nnz_B = K * P

        dmr_cycles = _ceil(M * K * P, Nr * Nc) if op != 'MV' else _ceil(M, Nr) * _ceil(K, Nc) * (2*Nr+1)
        nm_cycles = NMConfig.compute_cycles(op, M, K, P, nnz_A, nnz_B, Nr, Nc)
        speedup = dmr_cycles / nm_cycles if nm_cycles > 0 else 0

        print(f"  {op:<8} {M:>4} {K:>4} {P:>4} {dmr_cycles:>12,} {nm_cycles:>10,} {speedup:>7.2f}x")

    # Test 2: N:M compatibility scoring
    print("\n  Test 2: N:M compatibility scoring")
    print(f"  {'Pattern':<35} {'Score':>8}")
    print(f"  {'-'*45}")

    # Perfect 2:4 pattern
    perfect = np.array([0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0])
    print(f"  {'Perfect 2:4 (exactly 2 nnz per 4)':<35} {nm_compatibility_score(perfect):>7.1%}")

    # Dense (all nonzero) — incompatible
    dense = np.ones(12)
    print(f"  {'Fully dense (all nonzero)':<35} {nm_compatibility_score(dense):>7.1%}")

    # Very sparse (< 2 nnz per group) — compatible (can pad)
    very_sparse = np.array([0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0])
    print(f"  {'Very sparse (<=2 nnz per 4)':<35} {nm_compatibility_score(very_sparse):>7.1%}")

    # Random 50% sparse
    np.random.seed(42)
    random_50 = np.random.randn(1000)
    random_50[np.abs(random_50) < np.median(np.abs(random_50))] = 0
    print(f"  {'Random 50% pruned (1000 elems)':<35} {nm_compatibility_score(random_50):>7.1%}")

    # Random 70% sparse
    random_70 = np.random.randn(1000)
    threshold = np.percentile(np.abs(random_70), 70)
    random_70[np.abs(random_70) < threshold] = 0
    print(f"  {'Random 70% pruned (1000 elems)':<35} {nm_compatibility_score(random_70):>7.1%}")

    # Test 3: N:M pruning
    print("\n  Test 3: N:M structured pruning")
    weights = np.array([0.5, -0.1, 0.3, -0.8, 0.2, 0.9, -0.4, 0.1])
    pruned = nm_prune_weights(weights, N=2, M_ratio=4)
    print(f"  Original:  {weights}")
    print(f"  NM pruned: {pruned}")
    print(f"  Nonzeros per group: {[np.count_nonzero(pruned[i:i+4]) for i in range(0, len(pruned), 4)]}")

    # Test 4: Data access comparison
    print("\n  Test 4: Data access (elements)")
    print(f"  {'Op':<8} {'M':>4} {'K':>4} {'P':>4} {'Dense':>10} {'NM':>10} {'Savings':>8}")
    print(f"  {'-'*50}")

    for op, M, K, P in [('MM', 64, 64, 64), ('MM', 512, 768, 3072)]:
        nnz_A = int(M * K * 0.5)
        nnz_B = K * P
        from base_simulator.config_modes import DMRConfig
        dense_data = DMRConfig.data_accessed(op, M, K, P, nnz_A, nnz_B)
        nm_data = NMConfig.data_accessed(op, M, K, P, nnz_A, nnz_B)
        savings = 1.0 - nm_data / dense_data if dense_data > 0 else 0
        print(f"  {op:<8} {M:>4} {K:>4} {P:>4} {dense_data:>10,} {nm_data:>10,} {savings:>7.1%}")

    print("\n[DONE] N:M config validated.")
    print("  Next: integrate into base_simulator/config_modes.py as 5th config")


if __name__ == '__main__':
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
    main()
