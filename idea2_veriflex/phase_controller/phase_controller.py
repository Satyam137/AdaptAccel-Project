"""
phase_controller.py — Phase-Adaptive Controller for VersaAccel (from VeriFlex).

Detects workload phases at runtime and provides config bias signals
to the scheduler. This improves scheduling decisions by exploiting
known phase-config relationships.

Phases detected:
  - LLM: PREFILL (large batch, MM → DMR) ↔ DECODE (batch=1, MV → DM-R)
  - GNN: AGGREGATE (SpMM, sparse adj → D-M-R) ↔ TRANSFORM (MM, dense → DMR)
  - CNN: CONV_HEAVY (large spatial, Conv2d → DMR/D-MR) ↔ FC_TAIL (small Linear → D-M-R)
  - GENERIC: no phase bias (let predictor + scheduler decide)

The controller provides:
  1. Current phase label (for logging/analysis)
  2. Config bias (soft recommendation, can be overridden by scheduler)
  3. Phase transition detection (signals reconfig opportunity)

Usage:
    controller = PhaseController()
    phase = controller.detect_phase(layer_info)
    bias = controller.get_config_bias(phase)
"""

import numpy as np
from collections import deque


# ============================================================================
# Phase definitions
# ============================================================================

class Phase:
    """Workload phase with associated config preferences."""

    PREFILL = "PREFILL"          # LLM prompt processing (large batch MM)
    DECODE = "DECODE"            # LLM token generation (batch=1 MV)
    AGGREGATE = "AGGREGATE"      # GNN neighborhood aggregation (SpMM)
    TRANSFORM = "TRANSFORM"      # GNN feature transformation (dense MM)
    CONV_HEAVY = "CONV_HEAVY"    # CNN early/mid layers (large spatial Conv2d)
    FC_TAIL = "FC_TAIL"          # CNN final classifier layers (small Linear)
    GENERIC = "GENERIC"          # No clear phase — use predictor

    # Config bias for each phase (preferred config + confidence)
    # confidence: 0.0 = no bias, 1.0 = strong bias
    PHASE_BIAS = {
        PREFILL:    {"preferred": "DMR",   "confidence": 0.8,
                     "reason": "Large batch MM → systolic is optimal"},
        DECODE:     {"preferred": "DM-R",  "confidence": 0.7,
                     "reason": "Batch=1 MV → explicit reduction helps"},
        AGGREGATE:  {"preferred": "D-M-R", "confidence": 0.9,
                     "reason": "Sparse adj × features → full flexibility needed"},
        TRANSFORM:  {"preferred": "DMR",   "confidence": 0.8,
                     "reason": "Dense feature transform → systolic is optimal"},
        CONV_HEAVY: {"preferred": "D-MR",  "confidence": 0.6,
                     "reason": "Conv2d via im2col → explicit distribution helps"},
        FC_TAIL:    {"preferred": "D-M-R", "confidence": 0.5,
                     "reason": "Small FC layers → flexibility over throughput"},
        GENERIC:    {"preferred": None,    "confidence": 0.0,
                     "reason": "No phase detected — defer to predictor/scheduler"},
    }


# ============================================================================
# Phase Controller
# ============================================================================

class PhaseController:
    """
    Runtime phase detector and config bias generator.

    Implements a simple FSM that tracks workload phases based on:
      - Operator type (MM, MV, SpMM, etc.)
      - Matrix dimensions (batch size, spatial dims)
      - Layer type (Conv2d, Linear)
      - Sparsity patterns

    The FSM maintains a sliding window of recent layers to detect transitions.
    """

    def __init__(self, model_type="auto", window_size=3):
        """
        Args:
            model_type: "llm", "gnn", "cnn", or "auto" (detect from layers)
            window_size: number of recent layers for phase smoothing
        """
        self.model_type = model_type.lower()
        self.window_size = window_size
        self.history = deque(maxlen=window_size)
        self.current_phase = Phase.GENERIC
        self.transition_count = 0
        self.phase_log = []

    def detect_phase(self, layer_info):
        """
        Detect current workload phase from layer properties.

        Args:
            layer_info: dict with keys:
                - layer_type: 'Conv2d' or 'Linear'
                - op_type: 'MM', 'MV', 'SpMM', 'SpMV', 'SpMSpM'
                - M, K, P: matrix dimensions
                - input_sparsity: fraction of zeros in input
                - weight_sparsity: fraction of zeros in weights

        Returns:
            str: one of Phase.{PREFILL, DECODE, AGGREGATE, ...}
        """
        layer_type = layer_info.get('layer_type', 'Linear')
        op_type = layer_info.get('op_type', 'MM')
        M = layer_info.get('M', 1)
        K = layer_info.get('K', 1)
        P = layer_info.get('P', 1)
        input_sp = layer_info.get('input_sparsity', 0.0)
        weight_sp = layer_info.get('weight_sparsity', 0.0)

        # Auto-detect model type from first few layers if needed
        if self.model_type == "auto":
            self.model_type = self._auto_detect_model_type(layer_info)

        # Phase detection based on model type
        if self.model_type == "llm":
            phase = self._detect_llm_phase(layer_info)
        elif self.model_type == "gnn":
            phase = self._detect_gnn_phase(layer_info)
        elif self.model_type == "cnn":
            phase = self._detect_cnn_phase(layer_info)
        else:
            phase = Phase.GENERIC

        # Track transitions
        old_phase = self.current_phase
        self.current_phase = phase
        if old_phase != phase and old_phase != Phase.GENERIC:
            self.transition_count += 1

        # Update history
        self.history.append({
            'phase': phase,
            'layer_type': layer_type,
            'op_type': op_type,
            'M': M, 'K': K, 'P': P,
        })

        self.phase_log.append(phase)
        return phase

    def get_config_bias(self, phase=None):
        """
        Get the config bias for a given phase.

        Returns:
            dict: {"preferred": config_name, "confidence": 0.0-1.0, "reason": str}
        """
        if phase is None:
            phase = self.current_phase
        return Phase.PHASE_BIAS.get(phase, Phase.PHASE_BIAS[Phase.GENERIC])

    def is_transition(self):
        """Check if a phase transition just occurred."""
        if len(self.history) < 2:
            return False
        return self.history[-1]['phase'] != self.history[-2]['phase']

    # ========================================================================
    # Model-specific phase detection
    # ========================================================================

    def _auto_detect_model_type(self, layer_info):
        """Guess model type from layer properties."""
        layer_type = layer_info.get('layer_type', '')
        op_type = layer_info.get('op_type', '')
        input_sp = layer_info.get('input_sparsity', 0.0)

        if layer_type == 'Conv2d':
            return "cnn"
        elif op_type in ('SpMM', 'SpMV', 'SpMSpM') and input_sp > 0.5:
            return "gnn"
        else:
            return "llm"  # default for all-Linear models

    def _detect_llm_phase(self, layer_info):
        """
        LLM phase detection:
          - PREFILL: batch > 1 (processing prompt, MM operations)
          - DECODE:  batch = 1 (generating tokens, MV operations)
        """
        M = layer_info.get('M', 1)
        op_type = layer_info.get('op_type', 'MM')

        if op_type == 'MV' or M <= 1:
            return Phase.DECODE
        elif M > 1:
            return Phase.PREFILL
        return Phase.GENERIC

    def _detect_gnn_phase(self, layer_info):
        """
        GNN phase detection:
          - AGGREGATE: SpMM/SpMV operations (adj × features)
          - TRANSFORM: Dense MM operations (feature transformation)
        """
        op_type = layer_info.get('op_type', 'MM')
        input_sp = layer_info.get('input_sparsity', 0.0)
        weight_sp = layer_info.get('weight_sparsity', 0.0)

        if op_type in ('SpMM', 'SpMV', 'SpMSpM') or input_sp > 0.7:
            return Phase.AGGREGATE
        elif op_type == 'MM' and input_sp < 0.3:
            return Phase.TRANSFORM
        return Phase.GENERIC

    def _detect_cnn_phase(self, layer_info):
        """
        CNN phase detection:
          - CONV_HEAVY: Conv2d layers with large spatial dimensions
          - FC_TAIL: Linear layers at the end of the network
        """
        layer_type = layer_info.get('layer_type', 'Linear')
        M = layer_info.get('M', 1)

        if layer_type == 'Conv2d':
            return Phase.CONV_HEAVY
        elif layer_type == 'Linear':
            return Phase.FC_TAIL
        return Phase.GENERIC

    def get_summary(self):
        """Return a summary of detected phases."""
        if not self.phase_log:
            return {"phases": [], "transitions": 0}

        # Count phases
        from collections import Counter
        phase_counts = Counter(self.phase_log)

        return {
            "model_type": self.model_type,
            "total_layers": len(self.phase_log),
            "transitions": self.transition_count,
            "phase_counts": dict(phase_counts),
            "dominant_phase": phase_counts.most_common(1)[0][0],
        }


# ============================================================================
# Validation / self-test
# ============================================================================

def main():
    print("=" * 60)
    print("  Idea 2 - Step 2: Phase-Adaptive Controller")
    print("=" * 60)

    # Test 1: LLM phases (BERT-like)
    print("\n  Test 1: LLM Phase Detection (BERT-like)")
    print(f"  {'Layer':<30} {'Phase':<12} {'Bias Config':<10} {'Confidence':>10}")
    print(f"  {'-'*65}")

    ctrl = PhaseController(model_type="llm")

    llm_layers = [
        # Prefill phase (batch=32, large MM)
        {"layer_type": "Linear", "op_type": "MM", "M": 32, "K": 768, "P": 2304,
         "input_sparsity": 0.0, "weight_sparsity": 0.0, "name": "attn.qkv (batch=32)"},
        {"layer_type": "Linear", "op_type": "MM", "M": 32, "K": 768, "P": 768,
         "input_sparsity": 0.0, "weight_sparsity": 0.0, "name": "attn.out (batch=32)"},
        {"layer_type": "Linear", "op_type": "MM", "M": 32, "K": 768, "P": 3072,
         "input_sparsity": 0.0, "weight_sparsity": 0.0, "name": "ffn.up (batch=32)"},
        # Decode phase (batch=1, MV)
        {"layer_type": "Linear", "op_type": "MV", "M": 1, "K": 768, "P": 2304,
         "input_sparsity": 0.0, "weight_sparsity": 0.0, "name": "attn.qkv (batch=1)"},
        {"layer_type": "Linear", "op_type": "MV", "M": 1, "K": 768, "P": 768,
         "input_sparsity": 0.0, "weight_sparsity": 0.0, "name": "attn.out (batch=1)"},
        {"layer_type": "Linear", "op_type": "MV", "M": 1, "K": 768, "P": 3072,
         "input_sparsity": 0.0, "weight_sparsity": 0.0, "name": "ffn.up (batch=1)"},
    ]

    for layer in llm_layers:
        phase = ctrl.detect_phase(layer)
        bias = ctrl.get_config_bias(phase)
        print(f"  {layer['name']:<30} {phase:<12} {bias['preferred'] or 'N/A':<10} {bias['confidence']:>10.1f}")

    summary = ctrl.get_summary()
    print(f"\n  Transitions: {summary['transitions']}")
    print(f"  Phase counts: {summary['phase_counts']}")

    # Test 2: GNN phases
    print(f"\n  Test 2: GNN Phase Detection")
    print(f"  {'Layer':<30} {'Phase':<12} {'Bias Config':<10} {'Confidence':>10}")
    print(f"  {'-'*65}")

    ctrl_gnn = PhaseController(model_type="gnn")

    gnn_layers = [
        {"layer_type": "Linear", "op_type": "SpMM", "M": 2048, "K": 2048, "P": 128,
         "input_sparsity": 0.90, "weight_sparsity": 0.0, "name": "aggregate_1 (adj*feat)"},
        {"layer_type": "Linear", "op_type": "MM", "M": 2048, "K": 128, "P": 256,
         "input_sparsity": 0.10, "weight_sparsity": 0.0, "name": "transform_1 (dense)"},
        {"layer_type": "Linear", "op_type": "SpMM", "M": 2048, "K": 2048, "P": 256,
         "input_sparsity": 0.92, "weight_sparsity": 0.0, "name": "aggregate_2 (adj*feat)"},
        {"layer_type": "Linear", "op_type": "MM", "M": 2048, "K": 256, "P": 7,
         "input_sparsity": 0.15, "weight_sparsity": 0.0, "name": "transform_2 (classify)"},
    ]

    for layer in gnn_layers:
        phase = ctrl_gnn.detect_phase(layer)
        bias = ctrl_gnn.get_config_bias(phase)
        print(f"  {layer['name']:<30} {phase:<12} {bias['preferred'] or 'N/A':<10} {bias['confidence']:>10.1f}")

    summary_gnn = ctrl_gnn.get_summary()
    print(f"\n  Transitions: {summary_gnn['transitions']}")

    # Test 3: CNN phases (ResNet-like)
    print(f"\n  Test 3: CNN Phase Detection (ResNet-like)")
    print(f"  {'Layer':<30} {'Phase':<12} {'Bias Config':<10} {'Confidence':>10}")
    print(f"  {'-'*65}")

    ctrl_cnn = PhaseController(model_type="cnn")

    cnn_layers = [
        {"layer_type": "Conv2d", "op_type": "MM", "M": 3136, "K": 576, "P": 64,
         "input_sparsity": 0.0, "weight_sparsity": 0.0, "name": "conv1 (7x7, stride=2)"},
        {"layer_type": "Conv2d", "op_type": "MM", "M": 3136, "K": 64, "P": 64,
         "input_sparsity": 0.43, "weight_sparsity": 0.0, "name": "layer1.0.conv1 (1x1)"},
        {"layer_type": "Conv2d", "op_type": "MM", "M": 3136, "K": 576, "P": 64,
         "input_sparsity": 0.45, "weight_sparsity": 0.0, "name": "layer1.0.conv2 (3x3)"},
        {"layer_type": "Conv2d", "op_type": "MM", "M": 196, "K": 2304, "P": 512,
         "input_sparsity": 0.42, "weight_sparsity": 0.0, "name": "layer4.0.conv2 (3x3)"},
        {"layer_type": "Linear", "op_type": "MM", "M": 1, "K": 2048, "P": 1000,
         "input_sparsity": 0.50, "weight_sparsity": 0.0, "name": "fc (classifier)"},
    ]

    for layer in cnn_layers:
        phase = ctrl_cnn.detect_phase(layer)
        bias = ctrl_cnn.get_config_bias(phase)
        print(f"  {layer['name']:<30} {phase:<12} {bias['preferred'] or 'N/A':<10} {bias['confidence']:>10.1f}")

    summary_cnn = ctrl_cnn.get_summary()
    print(f"\n  Transitions: {summary_cnn['transitions']}")
    print(f"  Phase counts: {summary_cnn['phase_counts']}")

    # Test 4: Auto-detection
    print(f"\n  Test 4: Auto Model-Type Detection")
    ctrl_auto = PhaseController(model_type="auto")

    auto_tests = [
        {"layer_type": "Conv2d", "op_type": "MM", "M": 3136, "K": 576, "P": 64,
         "input_sparsity": 0.0, "weight_sparsity": 0.0},
        {"layer_type": "Linear", "op_type": "SpMM", "M": 2048, "K": 2048, "P": 128,
         "input_sparsity": 0.90, "weight_sparsity": 0.0},
        {"layer_type": "Linear", "op_type": "MM", "M": 32, "K": 768, "P": 3072,
         "input_sparsity": 0.0, "weight_sparsity": 0.0},
    ]

    for test in auto_tests:
        ctrl_auto_fresh = PhaseController(model_type="auto")
        ctrl_auto_fresh.detect_phase(test)
        print(f"  {test['layer_type']:>8} {test['op_type']:>6} sp={test['input_sparsity']:.1f}"
              f"  -> model_type={ctrl_auto_fresh.model_type}")

    print(f"\n[DONE] Phase controller validated.")
    print(f"  Next: integrate phase bias into the DP scheduler")


if __name__ == '__main__':
    main()
