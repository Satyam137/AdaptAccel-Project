"""
versaaccel.py — Main VersaAccel simulator entry point.

Ties together config_modes, cost_model, and memory_model into a single
easy-to-use interface.

Usage:
    from simulator.versaaccel import VersaAccelSimulator

    sim = VersaAccelSimulator()

    # Auto-select best config:
    result = sim.run('SpMM', M=64, K=64, P=64, sparsity_A=0.7)
    print(result)

    # Force a specific config:
    result = sim.run('MM', M=128, K=128, P=128, config='DMR')
    print(result)
"""

from .cost_model import CostModel
from .memory_model import MemoryModel
from .config_modes import ALL_CONFIGS


class VersaAccelSimulator:
    """
    Cycle-accurate simulator for VersaAccel accelerator.

    Models an Nr×Nc PE array with 4 configurations (DMR, D-MR, DM-R, D-M-R),
    input/output buffers, and HBM bandwidth.
    """

    def __init__(
        self,
        array_rows=8,
        array_cols=8,
        input_buf_kb=256,
        output_buf_kb=64,
        hbm_bw_gbps=128,
        freq_ghz=1.0,
        precision_bytes=4,
    ):
        self.Nr = array_rows
        self.Nc = array_cols
        self.memory = MemoryModel(
            input_buf_kb=input_buf_kb,
            output_buf_kb=output_buf_kb,
            hbm_bw_gbps=hbm_bw_gbps,
            freq_ghz=freq_ghz,
            precision_bytes=precision_bytes,
        )
        self.cost_model = CostModel(
            Nr=array_rows, Nc=array_cols, memory=self.memory
        )

    def run(self, op, M, K, P=1, sparsity_A=0.0, sparsity_B=0.0, config=None):
        """
        Simulate an operator on VersaAccel.

        Args:
            op: Operator type — 'MV', 'MM', 'SpMV', 'SpMM', 'SpMSpM'
            M: Rows of matrix A
            K: Columns of A / Rows of B
            P: Columns of B (default 1 for MV/SpMV)
            sparsity_A: Fraction of zeros in A (0.0-1.0)
            sparsity_B: Fraction of zeros in B (0.0-1.0)
            config: Force a specific config ('DMR', 'D-MR', 'DM-R', 'D-M-R')
                    or None for auto-selection via cost model

        Returns:
            dict with keys:
                config_used, total_cycles, costcomp, costmem,
                op, dims, sparsity, all_configs
        """
        if op in ('MV', 'SpMV'):
            P = 1

        if config is not None:
            # Force specific configuration
            cost = self.cost_model.cost_for_config(
                config, op, M, K, P, sparsity_A, sparsity_B
            )
            return {
                'config_used': config,
                'total_cycles': cost['total_cost'],
                'costcomp': cost['costcomp'],
                'costmem': cost['costmem'],
                'op': op,
                'dims': {'M': M, 'K': K, 'P': P},
                'sparsity': {'A': sparsity_A, 'B': sparsity_B},
            }
        else:
            # Auto-select via cost model
            result = self.cost_model.evaluate(
                op, M, K, P, sparsity_A, sparsity_B
            )
            return {
                'config_used': result['best_config'],
                'total_cycles': result['total_cost'],
                'costcomp': result['costcomp'],
                'costmem': result['costmem'],
                'op': op,
                'dims': {'M': M, 'K': K, 'P': P},
                'sparsity': {'A': sparsity_A, 'B': sparsity_B},
                'all_configs': result['all_configs'],
            }

    def compare_configs(self, op, M, K, P=1, sparsity_A=0.0, sparsity_B=0.0):
        """
        Compare all 4 configurations for a given operator.
        Returns a formatted comparison table.
        """
        if op in ('MV', 'SpMV'):
            P = 1

        all_costs = self.cost_model.evaluate_all(
            op, M, K, P, sparsity_A, sparsity_B
        )

        # Find best
        best_config = min(all_costs, key=lambda c: all_costs[c]['total_cost'])

        print(f"\n{'='*65}")
        print(f"  Operator: {op}  |  A: {M}×{K} (sparsity={sparsity_A})")
        if op in ('MM', 'SpMM', 'SpMSpM'):
            print(f"  B: {K}×{P} (sparsity={sparsity_B})")
        print(f"  Array: {self.Nr}×{self.Nc}  |  BW: {self.memory.bw_elements_per_cycle:.1f} elem/cyc")
        print(f"{'='*65}")
        print(f"  {'Config':<10} {'Costcomp':>10} {'Costmem':>10} {'Total':>10}  {'':>5}")
        print(f"  {'-'*50}")

        for cfg_name in ['DMR', 'D-MR', 'DM-R', 'D-M-R']:
            c = all_costs[cfg_name]
            marker = " ◄ BEST" if cfg_name == best_config else ""
            print(f"  {cfg_name:<10} {c['costcomp']:>10} {c['costmem']:>10} "
                  f"{c['total_cost']:>10} {marker}")

        print(f"{'='*65}\n")
        return all_costs

    def __repr__(self):
        return (
            f"VersaAccelSimulator(array={self.Nr}×{self.Nc}, "
            f"buf_in={self.memory.input_buf_bytes//1024}KB, "
            f"buf_out={self.memory.output_buf_bytes//1024}KB, "
            f"BW={self.memory.bw_elements_per_cycle:.1f} elem/cyc)"
        )
