"""
speculative_engine.py — Idea 4: Confidence-based speculative configuration.

Three execution modes based on predictor confidence:
  HIGH (>90%):   COMMIT directly to predicted config, hide switch cost
  MEDIUM (70-90%): SPECULATE — start on predicted, sample 10% to verify
  LOW (<70%):    FALLBACK — use safe default config

Two Oracle baselines:
  Selection-Oracle: perfect config choices, full switch cost
  Execution-Oracle: perfect config choices + perfect switch hiding (true ceiling)

Includes adversarial flush test to exercise the misprediction penalty path.

Usage:
    cd project
    python idea4_speculative/speculative_engine.py
"""

import torch
import numpy as np
import pandas as pd
import os, sys, glob, math

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)

from base_simulator.cost_model import CostModel
from idea2_veriflex.phase_controller.phase_controller import PhaseController
from idea2_veriflex.train_predictor_v2 import SparsityPredictorV2, INPUT_DIM

# ============================================================================
# Configuration
# ============================================================================

CONFIG_NAMES = ['DMR', 'D-MR', 'DM-R', 'D-M-R', 'D-M-R+NM']
CONFIG_IDX = {name: i for i, name in enumerate(CONFIG_NAMES)}

SWITCH_COST = np.array([
    [   0,   100,  100,   250,   250],
    [ 100,     0,  150,   100,   100],
    [ 100,   150,    0,   100,   100],
    [ 250,   100,  100,     0,     0],  # D-M-R ↔ D-M-R+NM = 0: same hardware,
    [ 250,   100,  100,     0,     0],  # just enable precomputed routing ROM
], dtype=np.float64)

CONF_HIGH = 0.90
CONF_MEDIUM = 0.70
SAMPLE_FRACTION = 0.10
FLUSH_FRACTION = 0.10

# ============================================================================
# Cost helpers
# ============================================================================

def compute_nm_discount(nm_compat, weight_sp):
    """NM enhancement discount for D-M-R (estimated, pending RTL validation)."""
    if nm_compat <= 0 or weight_sp < 0.3:
        return 0.0
    # Estimates pending RTL validation:
    #   intersection bypass ~12%, precomputed routing ~8%, regular memory ~5%
    total = 0.25
    return min(total * nm_compat * min(1.0, weight_sp / 0.5), 0.30)


def get_layer_costs(cost_model, row, sparsity):
    """Compute costs for all 5 configs."""
    M, K, P = int(row['M']), int(row['K']), int(row['P'])
    weight_sp = float(row['weight_sparsity'])
    nm_compat = float(row.get('nm_compatibility', 0.0))
    op = 'SpMM' if weight_sp > 0.3 else 'MM'
    combined_sp = max(sparsity, weight_sp)
    costs = {}
    for cfg in ['DMR', 'D-MR', 'DM-R', 'D-M-R']:
        costs[cfg] = cost_model.cost_for_config(cfg, op, M, K, P, sparsity_A=combined_sp)['total_cost']
    costs['D-M-R+NM'] = costs['D-M-R'] * (1.0 - compute_nm_discount(nm_compat, weight_sp))
    return costs


def estimate_confidence_fast(pred_sp, actual_sp, costs):
    """Confidence from cost gap between best and 2nd-best config."""
    sorted_costs = sorted(costs.values())
    if len(sorted_costs) < 2 or sorted_costs[0] == 0:
        return 0.95
    return float(np.clip((sorted_costs[1] - sorted_costs[0]) / sorted_costs[0] / 0.20, 0.0, 1.0))

# ============================================================================
# Speculative Scheduler
# ============================================================================

class SpeculativeScheduler:
    def __init__(self, cost_model, phase_controller=None):
        self.cost_model = cost_model
        self.phase_ctrl = phase_controller

    def schedule_speculative(self, layer_data, predicted_sp, actual_sp,
                              conf_high=CONF_HIGH, conf_medium=CONF_MEDIUM):
        n = len(layer_data)
        total_compute = total_switch = total_flush = total_hidden = 0
        commits = speculations = spec_correct = spec_wrong = fallbacks = 0
        prev_cfg = 'DMR'
        schedule = []

        for i in range(n):
            row = layer_data[i]
            pred_costs = get_layer_costs(self.cost_model, row, predicted_sp[i])
            actual_costs = get_layer_costs(self.cost_model, row, actual_sp[i])
            pred_best = min(pred_costs, key=pred_costs.get)
            actual_best = min(actual_costs, key=actual_costs.get)
            confidence = estimate_confidence_fast(predicted_sp[i], actual_sp[i], pred_costs)
            sw = SWITCH_COST[CONFIG_IDX[prev_cfg]][CONFIG_IDX[pred_best]]
            layer_compute = actual_costs[actual_best]

            if self.phase_ctrl:
                info = {'layer_type': row.get('layer_type', 'Linear'),
                        'op_type': row.get('op_type', 'MM'),
                        'M': int(row['M']), 'K': int(row['K']), 'P': int(row['P']),
                        'input_sparsity': float(row['input_sparsity']),
                        'weight_sparsity': float(row['weight_sparsity'])}
                phase = self.phase_ctrl.detect_phase(info)
                safe_default = self.phase_ctrl.get_config_bias(phase).get('config', 'D-M-R')
            else:
                safe_default = 'D-M-R'

            if confidence >= conf_high:
                hidden = min(sw * 0.5, layer_compute * 0.05)
                total_compute += actual_costs[pred_best]
                total_switch += max(0, sw - hidden)
                total_hidden += hidden
                chosen_cfg = pred_best
                commits += 1
            elif confidence >= conf_medium:
                sample_cost = layer_compute * SAMPLE_FRACTION
                if pred_best == actual_best:
                    hidden = min(sw, sample_cost)
                    total_compute += actual_costs[pred_best] + sample_cost * 0.5
                    total_switch += max(0, sw - hidden)
                    total_hidden += hidden
                    spec_correct += 1
                    chosen_cfg = pred_best
                else:
                    flush_cost = layer_compute * FLUSH_FRACTION
                    correct_sw = SWITCH_COST[CONFIG_IDX[pred_best]][CONFIG_IDX[actual_best]]
                    total_compute += actual_costs[actual_best]
                    total_switch += sw + correct_sw
                    total_flush += flush_cost
                    spec_wrong += 1
                    chosen_cfg = actual_best
                speculations += 1
            else:
                chosen_cfg = safe_default
                total_compute += actual_costs[chosen_cfg]
                total_switch += SWITCH_COST[CONFIG_IDX[prev_cfg]][CONFIG_IDX[chosen_cfg]]
                fallbacks += 1

            schedule.append(chosen_cfg)
            prev_cfg = chosen_cfg

        return {'schedule': schedule, 'total_compute': total_compute,
                'total_switch': total_switch, 'total_flush': total_flush,
                'total_hidden': total_hidden,
                'total_cost': total_compute + total_switch + total_flush,
                'commits': commits, 'speculations': speculations,
                'spec_correct': spec_correct, 'spec_wrong': spec_wrong,
                'fallbacks': fallbacks}

# ============================================================================
# Baseline schedulers
# ============================================================================

def schedule_greedy(layer_costs_list):
    schedule, tc, ts, prev = [], 0, 0, 'DMR'
    for costs in layer_costs_list:
        best = min(costs, key=costs.get)
        tc += costs[best]
        ts += SWITCH_COST[CONFIG_IDX[prev]][CONFIG_IDX[best]]
        schedule.append(best); prev = best
    return {'schedule': schedule, 'total_compute': tc, 'total_switch': ts, 'total_cost': tc + ts}


def schedule_oracle(layer_costs_list):
    """Selection-Oracle: perfect configs, full switch cost."""
    n, nc = len(layer_costs_list), len(CONFIG_NAMES)
    dp = np.full((n, nc), float('inf')); bt = np.full((n, nc), -1, dtype=int)
    for c in range(nc):
        dp[0][c] = layer_costs_list[0][CONFIG_NAMES[c]] + SWITCH_COST[0][c]
    for i in range(1, n):
        for c in range(nc):
            comp = layer_costs_list[i][CONFIG_NAMES[c]]
            for pc in range(nc):
                cost = dp[i-1][pc] + comp + SWITCH_COST[pc][c]
                if cost < dp[i][c]: dp[i][c] = cost; bt[i][c] = pc
    last = int(np.argmin(dp[n-1])); sched = [0]*n; sched[n-1] = last
    for i in range(n-2, -1, -1): sched[i] = bt[i+1][sched[i+1]]
    names = [CONFIG_NAMES[c] for c in sched]
    tc = sum(layer_costs_list[i][names[i]] for i in range(n))
    ts = SWITCH_COST[0][sched[0]] + sum(SWITCH_COST[sched[i-1]][sched[i]] for i in range(1, n))
    return {'schedule': names, 'total_compute': tc, 'total_switch': ts, 'total_cost': tc + ts}


def schedule_execution_oracle(layer_costs_list):
    """Execution-Oracle: perfect configs + perfect switch hiding (true ceiling)."""
    sel = schedule_oracle(layer_costs_list)
    sched = [CONFIG_IDX[n] for n in sel['schedule']]
    n = len(sched)
    tc = th = ts = 0
    for i in range(n):
        comp = layer_costs_list[i][sel['schedule'][i]]
        sw = SWITCH_COST[0][sched[0]] if i == 0 else SWITCH_COST[sched[i-1]][sched[i]]
        hidden = min(sw * 0.5, layer_costs_list[i-1][sel['schedule'][i-1]] * 0.05) if i > 0 else 0
        tc += comp; ts += max(0, sw - hidden); th += hidden
    return {'schedule': sel['schedule'], 'total_compute': tc, 'total_switch': ts,
            'total_hidden': th, 'total_cost': tc + ts}

# ============================================================================
# Predictor
# ============================================================================

def load_predictor(device):
    path = os.path.join(PROJECT_DIR, "idea2_veriflex", "trained_models", "sparsity_predictor_v2.pt")
    if not os.path.exists(path):
        print(f"ERROR: {path} not found"); sys.exit(1)
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = SparsityPredictorV2(ckpt.get('input_dim', INPUT_DIM),
                                 ckpt.get('hidden_dim', 64), ckpt.get('n_hidden', 3)).to(device)
    model.load_state_dict(ckpt['model_state_dict']); model.eval()
    return model


def predict_sparsity(predictor, info, device):
    is_conv = 1.0 if info.get('layer_type', 'Linear') == 'Conv2d' else 0.0
    feat = torch.tensor([[
        float(info.get('input_sparsity', 0.0)), float(info.get('weight_sparsity', 0.0)), is_conv,
        math.log2(max(1, int(info['M'])))/20., math.log2(max(1, int(info['K'])))/20.,
        math.log2(max(1, int(info['P'])))/20., float(info.get('layer_position', 0.0)),
        float(info.get('attention_sparsity', 0.0)), float(info.get('nm_compatibility', 0.0)),
    ]], dtype=torch.float32).to(device)
    with torch.no_grad(): return predictor(feat).item()

# ============================================================================
# Adversarial flush test
# ============================================================================

def run_adversarial_flush_test(cost_model):
    """Directly exercise SPECULATE+FLUSH by forcing wrong config predictions."""
    print(f"\n\n{'='*80}")
    print(f"  Adversarial Flush Test")
    print(f"  (Directly exercise SPECULATE+FLUSH path)")
    print(f"{'='*80}")

    data_dir = os.path.join(PROJECT_DIR, "idea1_adaptaccel", "profiling_data")
    for model_name in ['switching_stress', 'dp_stress', 'resnet50_pruned50']:
        csv_path = os.path.join(data_dir, f"{model_name}_traces.csv")
        if not os.path.exists(csv_path): continue
        df = pd.read_csv(csv_path)
        sp = df[df['input_idx'] == 0].reset_index(drop=True)
        if len(sp) < 2: continue
        for col in ['attention_sparsity', 'nm_compatibility']:
            if col not in sp.columns: sp[col] = 0.0
        n = len(sp)
        actual_sp = [max(float(r['input_sparsity']), float(r.get('attention_sparsity', 0)))
                     for _, r in sp.iterrows()]
        actual_costs = [get_layer_costs(cost_model, r, actual_sp[i])
                        for i, (_, r) in enumerate(sp.iterrows())]
        greedy = schedule_greedy(actual_costs)

        for error_rate in [0.10, 0.25, 0.50]:
            np.random.seed(42)
            n_bad = max(1, int(n * error_rate))
            bad_set = set(np.random.choice(n, n_bad, replace=False))
            tc = ts = tf = th = 0; n_flush = n_ok = 0; prev = 'DMR'
            for i in range(n):
                best = min(actual_costs[i], key=actual_costs[i].get)
                sw = SWITCH_COST[CONFIG_IDX[prev]][CONFIG_IDX[best]]
                lc = actual_costs[i][best]
                if i in bad_set:
                    wrong = [c for c in CONFIG_NAMES if c != best][i % 4]
                    sw1 = SWITCH_COST[CONFIG_IDX[prev]][CONFIG_IDX[wrong]]
                    sw2 = SWITCH_COST[CONFIG_IDX[wrong]][CONFIG_IDX[best]]
                    tc += lc; ts += sw1 + sw2; tf += lc * FLUSH_FRACTION; n_flush += 1
                else:
                    hid = min(sw * 0.5, lc * 0.05)
                    tc += lc; ts += max(0, sw - hid); th += hid; n_ok += 1
                prev = best
            total = tc + ts + tf
            pct = (total - greedy['total_cost']) / greedy['total_cost'] * 100
            safe = "✓ SAFE" if total <= greedy['total_cost'] * 1.02 else "⚠ DEGRADED >2%"
            print(f"\n  {model_name} ({n} layers, {error_rate:.0%} errors = {n_bad} forced wrong)")
            print(f"    Greedy: {greedy['total_cost']:>12.0f}  Spec+Flush: {total:>12.0f}  "
                  f"({pct:+.3f}%)  Flushes: {n_flush}  {safe}")
            if n_flush > 0:
                print(f"    Avg flush: {tf/n_flush:.0f} cycles/flush")

# ============================================================================
# Main
# ============================================================================

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    predictor = load_predictor(device)
    cost_model = CostModel(Nr=8, Nc=8)
    print("=" * 80)
    print("  Idea 4: Speculative Configuration Engine")
    print("=" * 80)

    data_dir = os.path.join(PROJECT_DIR, "idea1_adaptaccel", "profiling_data")
    csv_files = sorted(glob.glob(os.path.join(data_dir, "*_traces.csv")))
    all_results = []

    for csv_file in csv_files:
        model_name = os.path.basename(csv_file).replace("_traces.csv", "")
        df = pd.read_csv(csv_file)
        sp = df[df['input_idx'] == 0].reset_index(drop=True)
        if len(sp) < 2: continue
        for col in ['attention_sparsity', 'nm_compatibility']:
            if col not in sp.columns: sp[col] = 0.0
        mt = "llm" if 'bert' in model_name else "gnn" if 'gcn' in model_name \
            else "cnn" if 'resnet' in model_name else "auto"
        phase_ctrl = PhaseController(model_type=mt)
        n = len(sp)

        predicted_sp, actual_sp, layer_data = [], [], []
        for idx, row in sp.iterrows():
            attn = float(row.get('attention_sparsity', 0.0))
            actual_sp.append(max(float(row['input_sparsity']), attn))
            if idx == 0:
                predicted_sp.append(actual_sp[-1])
            else:
                prev = sp.iloc[idx - 1]
                info = {k: prev[k] for k in ['layer_type','M','K','P','input_sparsity','weight_sparsity']}
                info.update({'attention_sparsity': float(prev.get('attention_sparsity', 0.0)),
                             'nm_compatibility': float(prev.get('nm_compatibility', 0.0)),
                             'layer_position': (idx-1)/n})
                predicted_sp.append(predict_sparsity(predictor, info, device))
            layer_data.append(row.to_dict())

        actual_costs = [get_layer_costs(cost_model, r, actual_sp[i])
                        for i, (_, r) in enumerate(sp.iterrows())]
        greedy = schedule_greedy(actual_costs)
        sel_oracle = schedule_oracle(actual_costs)
        exe_oracle = schedule_execution_oracle(actual_costs)
        spec_sched = SpeculativeScheduler(cost_model, phase_ctrl)
        speculative = spec_sched.schedule_speculative(layer_data, predicted_sp, actual_sp)

        eo = exe_oracle['total_cost']
        spec_vs_eo = (speculative['total_cost'] - eo) / eo * 100
        spec_vs_so = (speculative['total_cost'] - sel_oracle['total_cost']) / sel_oracle['total_cost'] * 100
        sw_t = speculative['total_switch'] + speculative['total_hidden']
        hpct = speculative['total_hidden'] / sw_t * 100 if sw_t > 0 else 0

        print(f"\n  Model: {model_name} ({n} layers)")
        print(f"  {'Method':<25} {'Total':>10} {'vs ExeOracle':>12}")
        print(f"  {'-'*50}")
        print(f"  {'Greedy':<25} {greedy['total_cost']:>10.0f} {(greedy['total_cost']-eo)/eo*100:>+11.2f}%")
        print(f"  {'Speculative':<25} {speculative['total_cost']:>10.0f} {spec_vs_eo:>+11.2f}%")
        print(f"  {'Selection-Oracle':<25} {sel_oracle['total_cost']:>10.0f} "
              f"{(sel_oracle['total_cost']-eo)/eo*100:>+11.2f}%")
        print(f"  {'Execution-Oracle':<25} {eo:>10.0f} {'baseline':>12}")
        print(f"  C={speculative['commits']} S={speculative['speculations']} "
              f"F={speculative['fallbacks']} | Hidden: {hpct:.1f}%")

        all_results.append({'model': model_name, 'spec_vs_eo': spec_vs_eo,
                            'spec_vs_so': spec_vs_so, 'hidden_pct': hpct})

    print(f"\n\n{'='*80}")
    print(f"  SUMMARY")
    print(f"{'='*80}")
    print(f"  {'Model':<22} {'vs SelOracle':>12} {'vs ExeOracle':>12} {'Hidden%':>8}")
    print(f"  {'-'*58}")
    for r in all_results:
        print(f"  {r['model']:<22} {r['spec_vs_so']:>+11.2f}% {r['spec_vs_eo']:>+11.2f}% "
              f"{r['hidden_pct']:>7.1f}%")
    print(f"\n  Avg vs Selection-Oracle: {np.mean([r['spec_vs_so'] for r in all_results]):+.3f}%")
    print(f"  Avg vs Execution-Oracle: {np.mean([r['spec_vs_eo'] for r in all_results]):+.3f}%")

    run_adversarial_flush_test(cost_model)
    print(f"\n\n[DONE]")


if __name__ == '__main__':
    main()
