"""
cost_model.py — VersaAccel cost model (Paper Section IV, Eq. 1).

Given an operator S, system bandwidth BW, and matrix properties,
selects the optimal configuration C* that minimizes total execution cost:

    C* = argmin  [ Costcomp(S, C, BW) + Costmem(S, C, BW) ]
         C ∈ Call

Where:
    Costcomp = compute cycles (from config_modes)
    Costmem  = Data(S, C) / BW  (memory transfer time)
"""

from .config_modes import ALL_CONFIGS
from .memory_model import MemoryModel


class CostModel:
    """
    Implements VersaAccel's cost model for configuration selection.

    Usage:
        model = CostModel(Nr=8, Nc=8)
        result = model.evaluate('SpMM', M=64, K=64, P=64,
                                sparsity_A=0.7, sparsity_B=0.0)
        print(result)  # {'best_config': 'DM-R', 'cycles': 234, ...}
    """

    def __init__(self, Nr=8, Nc=8, memory=None):
        self.Nr = Nr
        self.Nc = Nc
        self.memory = memory or MemoryModel()

    def _compute_nnz(self, total_elements, sparsity):
        """Convert sparsity fraction to number of nonzeros."""
        density = 1.0 - sparsity
        return max(1, int(total_elements * density))

    def cost_for_config(self, config_name, op, M, K, P,
                        sparsity_A=0.0, sparsity_B=0.0, bw_override=None):
        """
        Compute total cost (Costcomp + Costmem) for a specific configuration.

        Args:
            config_name: One of 'DMR', 'D-MR', 'DM-R', 'D-M-R'
            op: 'MV', 'MM', 'SpMV', 'SpMM', 'SpMSpM'
            M, K, P: Matrix dimensions (A is M×K, B is K×P)
            sparsity_A: Fraction of zeros in A (0.0 = dense, 0.9 = 90% sparse)
            sparsity_B: Fraction of zeros in B
            bw_override: Override bandwidth in elements/cycle (None = use system BW)

        Returns:
            dict with costcomp, costmem, total_cost, config
        """
        config = ALL_CONFIGS[config_name]
        bw = bw_override if bw_override is not None else self.memory.bw_elements_per_cycle

        # Convert sparsity to NNZ counts
        nnz_A = self._compute_nnz(M * K, sparsity_A)
        nnz_B = self._compute_nnz(K * P, sparsity_B)

        # Compute cost
        if hasattr(config, 'compute_cycles'):
            method = config.compute_cycles
            # Some configs accept bw parameter
            import inspect
            sig = inspect.signature(method)
            if 'bw' in sig.parameters:
                costcomp = method(op, M, K, P, nnz_A, nnz_B, self.Nr, self.Nc, bw=bw)
            else:
                costcomp = method(op, M, K, P, nnz_A, nnz_B, self.Nr, self.Nc)

        # Memory cost
        data_elements = config.data_accessed(op, M, K, P, nnz_A, nnz_B)
        costmem = self.memory.memory_access_cycles(data_elements)

        total_cost = costcomp + costmem

        return {
            'config': config_name,
            'costcomp': costcomp,
            'costmem': costmem,
            'total_cost': total_cost,
            'data_elements': data_elements,
        }

    def evaluate(self, op, M, K, P=1, sparsity_A=0.0, sparsity_B=0.0):
        """
        Find optimal configuration C* = argmin Cost(S, C, BW).

        Returns:
            dict with best_config, cycles, and per-config breakdown
        """
        # For MV ops, P = 1
        if op in ('MV', 'SpMV'):
            P = 1

        results = {}
        for config_name in ALL_CONFIGS:
            results[config_name] = self.cost_for_config(
                config_name, op, M, K, P, sparsity_A, sparsity_B
            )

        # Find minimum total cost
        best = min(results.values(), key=lambda x: x['total_cost'])

        return {
            'best_config': best['config'],
            'total_cost': best['total_cost'],
            'costcomp': best['costcomp'],
            'costmem': best['costmem'],
            'all_configs': results,
        }

    def evaluate_all(self, op, M, K, P=1, sparsity_A=0.0, sparsity_B=0.0):
        """Return costs for ALL configurations (for comparison/analysis)."""
        result = self.evaluate(op, M, K, P, sparsity_A, sparsity_B)
        return result['all_configs']
