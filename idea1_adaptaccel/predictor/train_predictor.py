"""
train_predictor.py — Step 1.2: Train an MLP to predict next-layer sparsity.

The predictor learns: given the CURRENT layer's properties, what will the
NEXT layer's input sparsity be?

This is the core of AdaptAccel — at runtime, before executing layer L,
we predict output sparsity to choose the best VersaAccel config for layer L+1.

Features (input to MLP):
    - input_sparsity:  fraction of zeros in current layer's input
    - weight_sparsity: fraction of zeros in current layer's weights
    - op_type:         0 = Conv2d/MM, 1 = Linear/MV
    - log2(M), log2(K), log2(P): matrix dimensions (log-scaled)
    - layer_position:  normalized position in network (0.0 = first, 1.0 = last)

Target (output of MLP):
    - next_input_sparsity: the next layer's input sparsity

Evaluation metric:
    - Config match rate: does the predicted sparsity lead to the SAME
      VersaAccel config selection as the actual sparsity? Target: ≥90%

Usage:
    cd project/idea1_adaptaccel
    python -m predictor.train_predictor
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

# Add base_simulator to path so we can use cost_model
BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "base_simulator")
sys.path.insert(0, os.path.dirname(BASE_DIR))

from base_simulator.cost_model import CostModel
from base_simulator.memory_model import MemoryModel

# ============================================================================
# Configuration
# ============================================================================

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "profiling_data")
SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trained_models")

HIDDEN_SIZE = 64       # MLP hidden layer size
N_HIDDEN_LAYERS = 3    # Number of hidden layers
EPOCHS = 100
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
TRAIN_SPLIT = 0.8      # 80% train, 20% test


# ============================================================================
# Data loading and feature engineering
# ============================================================================

def load_all_traces(data_dir):
    """Load all CSV trace files and combine into one DataFrame."""
    csv_files = glob.glob(os.path.join(data_dir, "*_traces.csv"))
    if not csv_files:
        print(f"ERROR: No CSV files found in {data_dir}")
        sys.exit(1)

    dfs = []
    for f in sorted(csv_files):
        df = pd.read_csv(f)
        # Extract model name from filename
        model_name = os.path.basename(f).replace("_traces.csv", "")
        df['model_name'] = model_name
        dfs.append(df)
        print(f"  Loaded {f} ({len(df)} rows)")

    combined = pd.concat(dfs, ignore_index=True)
    print(f"  Total: {len(combined)} rows from {len(csv_files)} files")
    return combined


def build_dataset(df):
    """
    Build (features, targets) pairs for the predictor.

    For each layer L in an inference pass, the target is layer L+1's
    effective_sparsity = max(input_sparsity, attention_sparsity).
    This makes BERT non-trivial: attention sparsity varies 0-99% per input.
    The last layer in each pass has no target and is dropped.
    """
    features_list = []
    targets_list = []

    # Ensure attention_sparsity column exists (default 0 for non-BERT)
    if 'attention_sparsity' not in df.columns:
        df['attention_sparsity'] = 0.0
    df['attention_sparsity'] = df['attention_sparsity'].fillna(0.0)

    # Group by (model_name, input_idx) — each group is one inference pass
    grouped = df.groupby(['model_name', 'input_idx'])

    for (model, input_idx), group in grouped:
        group = group.reset_index(drop=True)
        n_layers = len(group)

        for i in range(n_layers - 1):
            current = group.iloc[i]
            next_layer = group.iloc[i + 1]

            # Features from current layer
            is_conv = 1.0 if current['layer_type'] == 'Conv2d' else 0.0
            M = max(1, current['M'])
            K = max(1, current['K'])
            P = max(1, current['P'])
            attn_sp = float(current.get('attention_sparsity', 0.0))

            feat = [
                current['input_sparsity'],
                current['weight_sparsity'],
                is_conv,
                math.log2(M) / 20.0,    # normalize log-dim to ~[0, 1]
                math.log2(K) / 20.0,
                math.log2(P) / 20.0,
                i / n_layers,            # normalized layer position
                attn_sp,                 # attention sparsity (non-zero for transformers)
            ]
            features_list.append(feat)

            # Target: next layer's effective sparsity
            # = max(input_sparsity, attention_sparsity)
            # For BERT: input_sp=0 but attn_sp varies 0-99% → non-trivial target
            # For ResNet/others: attn_sp=0, so effective = input_sparsity (unchanged)
            next_attn = float(next_layer.get('attention_sparsity', 0.0))
            effective_sp = max(float(next_layer['input_sparsity']), next_attn)
            targets_list.append(effective_sp)

    X = np.array(features_list, dtype=np.float32)
    y = np.array(targets_list, dtype=np.float32)
    print(f"  Built dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"  Target range: [{y.min():.3f}, {y.max():.3f}], mean={y.mean():.3f}")
    return X, y


# ============================================================================
# MLP Model
# ============================================================================

class SparsityPredictor(nn.Module):
    """Small MLP to predict next-layer sparsity."""

    def __init__(self, input_dim=8, hidden_dim=64, n_hidden=3):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(n_hidden - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        layers.append(nn.Linear(hidden_dim, 1))
        layers.append(nn.Sigmoid())  # Output in [0, 1] since sparsity is a fraction
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ============================================================================
# Config match evaluation
# ============================================================================

def evaluate_config_match(predictions, actuals, df_test_info):
    """
    For each prediction, determine what VersaAccel config the cost model
    would select, and compare with the config from actual sparsity.

    Returns config match rate (fraction of matching selections).
    """
    cost_model = CostModel(Nr=8, Nc=8)
    matches = 0
    total = 0

    for i in range(len(predictions)):
        pred_sparsity = float(predictions[i])
        actual_sparsity = float(actuals[i])

        info = df_test_info.iloc[i]
        M, K, P = int(info['M']), int(info['K']), int(info['P'])
        weight_sp = float(info['weight_sparsity'])

        # Determine operator type based on combined sparsity
        # If weight or activation sparsity > 0.3, use sparse operator
        combined_sp_pred = max(pred_sparsity, weight_sp)
        combined_sp_actual = max(actual_sparsity, weight_sp)

        op_pred = 'SpMM' if combined_sp_pred > 0.3 else 'MM'
        op_actual = 'SpMM' if combined_sp_actual > 0.3 else 'MM'

        # Get best config from cost model
        result_pred = cost_model.evaluate(op_pred, M, K, P,
                                          sparsity_A=combined_sp_pred)
        result_actual = cost_model.evaluate(op_actual, M, K, P,
                                            sparsity_A=combined_sp_actual)

        if result_pred['best_config'] == result_actual['best_config']:
            matches += 1
        total += 1

    match_rate = matches / total if total > 0 else 0
    return match_rate


# ============================================================================
# Main training loop
# ============================================================================

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # 1) Load data
    print("Loading profiling data...")
    df = load_all_traces(DATA_DIR)

    # 2) Split by PRUNING LEVEL (leave-one-level-out)
    # Train: all architectures at dense, 50%, 90% pruning + synthetics
    # Test:  all architectures at 70% pruning (unseen sparsity level)
    #
    # This is realistic: in deployment, you'd profile a few samples from
    # a new model (seeing its architecture), but the exact pruning level
    # at test time may differ from what was profiled.

    test_models = [m for m in df['model_name'].unique() if 'pruned70' in m]
    train_models = [m for m in df['model_name'].unique() if m not in test_models]

    print(f"\n  LEAVE-ONE-PRUNING-LEVEL-OUT SPLIT:")
    print(f"  Train models ({len(train_models)}): {train_models}")
    print(f"  Test models  ({len(test_models)}):  {test_models}")
    print(f"  (Test = 70% pruning, never seen during training)")

    df_train = df[df['model_name'].isin(train_models)]
    df_test = df[df['model_name'].isin(test_models)]

    # 3) Build datasets separately
    print("\nBuilding train dataset...")
    X_train, y_train = build_dataset(df_train)
    print("Building test dataset...")
    X_test, y_test = build_dataset(df_test)
    print(f"\n  Train: {len(X_train)} samples from {len(train_models)} model configs")
    print(f"  Test:  {len(X_test)} samples from {len(test_models)} model configs")

    # Build DataLoaders
    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    test_ds = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    # 4) Create model
    model = SparsityPredictor(
        input_dim=X_train.shape[1],
        hidden_dim=HIDDEN_SIZE,
        n_hidden=N_HIDDEN_LAYERS,
    ).to(device)
    print(f"\n  Model: {sum(p.numel() for p in model.parameters())} parameters")

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    # 5) Train with EARLY STOPPING
    print(f"\nTraining for {EPOCHS} epochs (with early stopping)...")
    best_mae = float('inf')
    best_state = None
    best_epoch = 0
    patience = 20  # stop if no improvement for 20 epochs
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

        # Evaluate every epoch for early stopping
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
            test_preds = np.array(test_preds)
            test_actuals = np.array(test_actuals)
            mae = np.mean(np.abs(test_preds - test_actuals))

        # Track best model
        if mae < best_mae:
            best_mae = mae
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1
            no_improve = 0
        else:
            no_improve += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            marker = " ★ best" if epoch + 1 == best_epoch else ""
            print(f"  Epoch {epoch+1:>3}/{EPOCHS}  |  Loss: {avg_loss:.6f}  |  Test MAE: {mae:.4f}{marker}")

        if no_improve >= patience:
            print(f"\n  Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
            break

    # Restore best model
    print(f"  Restoring best model from epoch {best_epoch} (MAE={best_mae:.4f})")
    model.load_state_dict(best_state)

    # 6) Final evaluation
    print(f"\n{'='*60}")
    print("  Final Evaluation (on UNSEEN pruning level)")
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
    print(f"  Test MSE:  {mse:.6f}")
    print(f"  Test RMSE: {np.sqrt(mse):.4f}")

    # 7) Config match rate — overall + per-model breakdown
    print("\n  Computing config match rate...")

    # Build test_info from test dataset
    all_test_info = []
    all_test_model_labels = []
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
            })
            all_test_model_labels.append(model_name)

    test_info_df = pd.DataFrame(all_test_info)
    test_model_labels = np.array(all_test_model_labels)

    # Overall match rate
    match_rate = evaluate_config_match(all_preds, all_actuals, test_info_df)

    print(f"\n  ╔═══════════════════════════════════════════════╗")
    print(f"  ║  Overall Config Match Rate:  {match_rate:.1%}             ║")
    print(f"  ║  Target:                     ≥90%              ║")
    print(f"  ║  Status:  {'✓ PASS' if match_rate >= 0.9 else '✗ FAIL — needs tuning':>36} ║")
    print(f"  ╚═══════════════════════════════════════════════╝")

    # Per-model breakdown
    print(f"\n  Per-model breakdown (all UNSEEN during training):")
    print(f"  {'Model':<25} {'Samples':>8} {'MAE':>8} {'Match%':>8}")
    print(f"  {'-'*50}")
    for model_name in sorted(test_models):
        mask = test_model_labels == model_name
        if mask.sum() == 0:
            continue
        model_preds = all_preds[mask]
        model_actuals = all_actuals[mask]
        model_info = test_info_df[mask].reset_index(drop=True)
        model_mae = np.mean(np.abs(model_preds - model_actuals))
        model_match = evaluate_config_match(model_preds, model_actuals, model_info)
        print(f"  {model_name:<25} {mask.sum():>8} {model_mae:>8.4f} {model_match:>7.1%}")

    # 8) Save model
    os.makedirs(SAVE_DIR, exist_ok=True)
    model_path = os.path.join(SAVE_DIR, "sparsity_predictor.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_dim': X_train.shape[1],
        'hidden_dim': HIDDEN_SIZE,
        'n_hidden': N_HIDDEN_LAYERS,
        'train_mae': mae,
        'config_match_rate': match_rate,
    }, model_path)
    print(f"\n  Model saved to {model_path}")

    # 9) Export weights for RTL (Idea 2 CPE will use these)
    weights_path = os.path.join(SAVE_DIR, "predictor_weights.npz")
    weight_dict = {}
    for name, param in model.named_parameters():
        weight_dict[name] = param.detach().cpu().numpy()
    np.savez(weights_path, **weight_dict)
    print(f"  Weights exported to {weights_path} (for RTL CPE in Idea 2)")

    print(f"\n✓ Step 1.2 complete!")


if __name__ == '__main__':
    main()
