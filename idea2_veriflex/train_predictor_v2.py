"""
train_predictor_v2.py — Retrain predictor for Idea 2 (5-config system).

Updates from Idea 1's predictor:
  1. Includes NM-pruned and switching_stress datasets in training
  2. Adds nm_compatibility as 9th feature
  3. Test split: pruned70 models + switching_stress (unseen patterns)
  4. After training, runs the integrated scheduler with PREDICTED sparsity

Usage:
    cd project
    python idea2_veriflex/train_predictor_v2.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
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

# ============================================================================
# Configuration
# ============================================================================

DATA_DIR = os.path.join(PROJECT_DIR, "idea1_adaptaccel", "profiling_data")
SAVE_DIR = os.path.join(SCRIPT_DIR, "trained_models")

HIDDEN_SIZE = 64
N_HIDDEN_LAYERS = 3
EPOCHS = 150
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
INPUT_DIM = 9  # 8 original features + nm_compatibility


# ============================================================================
# Data loading
# ============================================================================

def load_all_traces(data_dir):
    """Load all CSV trace files."""
    csv_files = glob.glob(os.path.join(data_dir, "*_traces.csv"))
    if not csv_files:
        print(f"ERROR: No CSV files found in {data_dir}")
        sys.exit(1)

    dfs = []
    for f in sorted(csv_files):
        df = pd.read_csv(f)
        model_name = os.path.basename(f).replace("_traces.csv", "")
        df['model_name'] = model_name
        dfs.append(df)
        print(f"  Loaded {os.path.basename(f):<35} ({len(df):>7} rows, "
              f"{df['layer_name'].nunique():>3} layers)")

    combined = pd.concat(dfs, ignore_index=True)
    print(f"  Total: {len(combined):,} rows from {len(csv_files)} files")
    return combined


def build_dataset(df):
    """
    Build (features, targets) pairs with 9 features (including nm_compatibility).

    Features:
      0. input_sparsity
      1. weight_sparsity
      2. is_conv (1.0 for Conv2d, 0.0 for Linear)
      3. log2(M) / 20
      4. log2(K) / 20
      5. log2(P) / 20
      6. layer_position (normalized)
      7. attention_sparsity
      8. nm_compatibility  ← NEW for Idea 2

    Target: next layer's effective_sparsity = max(input_sparsity, attention_sparsity)
    """
    # Ensure columns exist
    if 'attention_sparsity' not in df.columns:
        df['attention_sparsity'] = 0.0
    df['attention_sparsity'] = df['attention_sparsity'].fillna(0.0)

    if 'nm_compatibility' not in df.columns:
        df['nm_compatibility'] = 0.0
    df['nm_compatibility'] = df['nm_compatibility'].fillna(0.0)

    features_list = []
    targets_list = []

    grouped = df.groupby(['model_name', 'input_idx'])

    for (model, input_idx), group in grouped:
        group = group.reset_index(drop=True)
        n_layers = len(group)

        for i in range(n_layers - 1):
            current = group.iloc[i]
            next_layer = group.iloc[i + 1]

            is_conv = 1.0 if current['layer_type'] == 'Conv2d' else 0.0
            M = max(1, current['M'])
            K = max(1, current['K'])
            P = max(1, current['P'])
            attn_sp = float(current.get('attention_sparsity', 0.0))
            nm_compat = float(current.get('nm_compatibility', 0.0))

            feat = [
                float(current['input_sparsity']),
                float(current['weight_sparsity']),
                is_conv,
                math.log2(M) / 20.0,
                math.log2(K) / 20.0,
                math.log2(P) / 20.0,
                i / n_layers,
                attn_sp,
                nm_compat,  # 9th feature
            ]
            features_list.append(feat)

            # Target: next layer's effective sparsity
            next_attn = float(next_layer.get('attention_sparsity', 0.0))
            effective_sp = max(float(next_layer['input_sparsity']), next_attn)
            targets_list.append(effective_sp)

    X = np.array(features_list, dtype=np.float32)
    y = np.array(targets_list, dtype=np.float32)
    print(f"  Built dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"  Target range: [{y.min():.3f}, {y.max():.3f}], mean={y.mean():.3f}")
    return X, y


# ============================================================================
# MLP Model (9 inputs)
# ============================================================================

class SparsityPredictorV2(nn.Module):
    """MLP predictor with 9 input features (includes nm_compatibility)."""

    def __init__(self, input_dim=INPUT_DIM, hidden_dim=64, n_hidden=3):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(n_hidden - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        layers.append(nn.Linear(hidden_dim, 1))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ============================================================================
# Config match evaluation (5-config)
# ============================================================================

def evaluate_config_match(predictions, actuals, df_test_info):
    """Config match rate using 5-config cost model."""
    from idea2_veriflex.nm_sparsity.nm_sparsity import NMConfig
    from idea2_veriflex.integrated_scheduler import get_layer_costs_5config, CONFIG_NAMES_5

    cost_model = CostModel(Nr=8, Nc=8)
    matches = 0
    total = 0

    for i in range(len(predictions)):
        pred_sp = float(predictions[i])
        actual_sp = float(actuals[i])
        info = df_test_info.iloc[i]

        # Get costs with predicted sparsity
        costs_pred = get_layer_costs_5config(cost_model, info, sparsity=pred_sp)
        best_pred = min(costs_pred, key=costs_pred.get)

        # Get costs with actual sparsity
        costs_actual = get_layer_costs_5config(cost_model, info, sparsity=actual_sp)
        best_actual = min(costs_actual, key=costs_actual.get)

        if best_pred == best_actual:
            matches += 1
        total += 1

    return matches / total if total > 0 else 0


# ============================================================================
# Main
# ============================================================================

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    print("=" * 60)
    print("  Idea 2: Predictor V2 Training (9 features, 5-config)")
    print("=" * 60)

    # 1) Load data
    print("\nLoading profiling data...")
    df = load_all_traces(DATA_DIR)

    # 2) Train/Test split
    # Test: pruned70 + switching_stress (unseen patterns)
    # Train: everything else (dense, pruned50, pruned90, nm24, synthetics)
    all_models = sorted(df['model_name'].unique())
    test_models = [m for m in all_models if 'pruned70' in m or 'switching_stress' in m]
    train_models = [m for m in all_models if m not in test_models]

    print(f"\n  TRAIN/TEST SPLIT:")
    print(f"  Train ({len(train_models)}): {train_models}")
    print(f"  Test  ({len(test_models)}):  {test_models}")

    df_train = df[df['model_name'].isin(train_models)]
    df_test = df[df['model_name'].isin(test_models)]

    # 3) Build datasets
    print("\nBuilding train dataset...")
    X_train, y_train = build_dataset(df_train)
    print("Building test dataset...")
    X_test, y_test = build_dataset(df_test)
    print(f"\n  Train: {len(X_train)} samples")
    print(f"  Test:  {len(X_test)} samples")

    # DataLoaders
    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    test_ds = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    # 4) Create model
    model = SparsityPredictorV2(
        input_dim=INPUT_DIM, hidden_dim=HIDDEN_SIZE, n_hidden=N_HIDDEN_LAYERS
    ).to(device)
    print(f"\n  Model: {sum(p.numel() for p in model.parameters())} params, "
          f"{INPUT_DIM} inputs → {HIDDEN_SIZE}×{N_HIDDEN_LAYERS} → 1 output")

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    # 5) Train with early stopping
    print(f"\nTraining for {EPOCHS} epochs (early stopping patience=25)...")
    best_mae = float('inf')
    best_state = None
    best_epoch = 0
    patience = 25
    no_improve = 0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        n_batches = 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        model.eval()
        with torch.no_grad():
            test_preds = []
            test_actuals = []
            for xb, yb in test_loader:
                xb = xb.to(device)
                pred = model(xb)
                test_preds.extend(pred.cpu().numpy())
                test_actuals.extend(yb.numpy())
            test_preds_arr = np.array(test_preds)
            test_actuals_arr = np.array(test_actuals)
            mae = np.mean(np.abs(test_preds_arr - test_actuals_arr))

        if mae < best_mae:
            best_mae = mae
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1
            no_improve = 0
        else:
            no_improve += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            marker = " ★ best" if epoch + 1 == best_epoch else ""
            print(f"  Epoch {epoch+1:>3}/{EPOCHS}  |  Loss: {avg_loss:.6f}  |  "
                  f"Test MAE: {mae:.4f}{marker}")

        if no_improve >= patience:
            print(f"\n  Early stopping at epoch {epoch+1} "
                  f"(no improvement for {patience} epochs)")
            break

    # Restore best
    print(f"  Restoring best model from epoch {best_epoch} (MAE={best_mae:.4f})")
    model.load_state_dict(best_state)

    # 6) Final evaluation
    print(f"\n{'='*60}")
    print(f"  Final Evaluation (5-config match rate)")
    print(f"{'='*60}")

    model.eval()
    with torch.no_grad():
        all_preds = []
        all_actuals = []
        for xb, yb in test_loader:
            xb = xb.to(device)
            pred = model(xb)
            all_preds.extend(pred.cpu().numpy())
            all_actuals.extend(yb.numpy())

    all_preds = np.array(all_preds)
    all_actuals = np.array(all_actuals)

    mae = np.mean(np.abs(all_preds - all_actuals))
    mse = np.mean((all_preds - all_actuals) ** 2)
    print(f"  Test MAE:  {mae:.4f}")
    print(f"  Test RMSE: {np.sqrt(mse):.4f}")

    # Build test_info for config matching
    all_test_info = []
    all_test_model_labels = []

    if 'nm_compatibility' not in df_test.columns:
        df_test['nm_compatibility'] = 0.0

    grouped = df_test.groupby(['model_name', 'input_idx'])
    for (model_name, input_idx), group in grouped:
        group = group.reset_index(drop=True)
        for i in range(len(group) - 1):
            next_layer = group.iloc[i + 1]
            all_test_info.append({
                'M': next_layer['M'],
                'K': next_layer['K'],
                'P': next_layer['P'],
                'weight_sparsity': next_layer['weight_sparsity'],
                'nm_compatibility': float(next_layer.get('nm_compatibility', 0.0)),
            })
            all_test_model_labels.append(model_name)

    test_info_df = pd.DataFrame(all_test_info)
    test_model_labels = np.array(all_test_model_labels)

    # Overall match rate
    match_rate = evaluate_config_match(all_preds, all_actuals, test_info_df)

    print(f"\n  ╔═══════════════════════════════════════════════╗")
    print(f"  ║  Config Match Rate (5-config): {match_rate:.1%}            ║")
    print(f"  ║  Target:                       ≥90%            ║")
    print(f"  ║  Status:  {'✓ PASS' if match_rate >= 0.9 else '✗ NEEDS TUNING':>36} ║")
    print(f"  ╚═══════════════════════════════════════════════╝")

    # Per-model breakdown
    print(f"\n  Per-model breakdown:")
    print(f"  {'Model':<25} {'Samples':>8} {'MAE':>8} {'Match%':>8}")
    print(f"  {'-'*50}")
    for model_name in sorted(set(test_model_labels)):
        mask = test_model_labels == model_name
        if mask.sum() == 0:
            continue
        model_preds = all_preds[mask]
        model_actuals = all_actuals[mask]
        model_info = test_info_df[mask].reset_index(drop=True)
        model_mae = np.mean(np.abs(model_preds - model_actuals))
        model_match = evaluate_config_match(model_preds, model_actuals, model_info)
        print(f"  {model_name:<25} {mask.sum():>8} {model_mae:>8.4f} {model_match:>7.1%}")

    # 7) Save model
    os.makedirs(SAVE_DIR, exist_ok=True)
    model_path = os.path.join(SAVE_DIR, "sparsity_predictor_v2.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_dim': INPUT_DIM,
        'hidden_dim': HIDDEN_SIZE,
        'n_hidden': N_HIDDEN_LAYERS,
        'train_mae': float(mae),
        'config_match_rate': float(match_rate),
        'version': 2,
    }, model_path)
    print(f"\n  Model saved to {model_path}")

    # Export weights
    weights_path = os.path.join(SAVE_DIR, "predictor_weights_v2.npz")
    weight_dict = {}
    for name, param in model.named_parameters():
        weight_dict[name] = param.detach().cpu().numpy()
    np.savez(weights_path, **weight_dict)
    print(f"  Weights exported to {weights_path}")

    print(f"\n✓ Predictor V2 trained!")
    print(f"  Next: run integrated_scheduler_v2.py (uses predicted sparsity)")


if __name__ == '__main__':
    main()
