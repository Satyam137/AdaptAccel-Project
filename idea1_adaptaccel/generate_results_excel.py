"""
Generate an Excel summary of all Idea 1 (AdaptAccel) results.
Creates multiple sheets with formatted tables and one-liner explanations.

Usage:
    cd project/idea1_adaptaccel
    python generate_results_excel.py
"""

import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
import os

OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "idea1_results_summary.xlsx")

# Colors
HEADER_FILL = PatternFill(start_color="1F4E79", end_color="1F4E79", fill_type="solid")
HEADER_FONT = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
TITLE_FONT = Font(name="Calibri", size=14, bold=True, color="1F4E79")
EXPLAIN_FONT = Font(name="Calibri", size=10, italic=True, color="555555")
GOOD_FILL = PatternFill(start_color="E2EFDA", end_color="E2EFDA", fill_type="solid")
WARN_FILL = PatternFill(start_color="FFF2CC", end_color="FFF2CC", fill_type="solid")
DATA_FONT = Font(name="Calibri", size=10)
BOLD_FONT = Font(name="Calibri", size=10, bold=True)
THIN_BORDER = Border(
    left=Side(style='thin'), right=Side(style='thin'),
    top=Side(style='thin'), bottom=Side(style='thin'),
)


def add_title(ws, row, title, explanation=""):
    ws.cell(row=row, column=1, value=title).font = TITLE_FONT
    if explanation:
        ws.cell(row=row+1, column=1, value=explanation).font = EXPLAIN_FONT
    return row + (3 if explanation else 2)


def add_table(ws, start_row, headers, data, col_widths=None, highlight_col=None):
    """Add a formatted table to the worksheet."""
    # Headers
    for c, header in enumerate(headers, 1):
        cell = ws.cell(row=start_row, column=c, value=header)
        cell.font = HEADER_FONT
        cell.fill = HEADER_FILL
        cell.alignment = Alignment(horizontal="center")
        cell.border = THIN_BORDER

    # Data rows
    for r, row_data in enumerate(data, start_row + 1):
        for c, val in enumerate(row_data, 1):
            cell = ws.cell(row=r, column=c, value=val)
            cell.font = DATA_FONT
            cell.border = THIN_BORDER
            cell.alignment = Alignment(horizontal="center") if c > 1 else Alignment(horizontal="left")
            if highlight_col and c == highlight_col:
                cell.font = BOLD_FONT

    # Column widths
    if col_widths:
        for c, w in enumerate(col_widths, 1):
            ws.column_dimensions[get_column_letter(c)].width = w

    return start_row + len(data) + 2


def create_excel():
    wb = openpyxl.Workbook()

    # =========================================================================
    # Sheet 1: Profiling Data Summary
    # =========================================================================
    ws1 = wb.active
    ws1.title = "Profiling Data"

    row = add_title(ws1, 1, "Step 1.1: Workload Profiling Data Summary",
                    "Per-layer sparsity traces from real PyTorch models + synthetic stress tests")

    headers = ["Dataset", "Source", "Layers", "Rows", "Avg Input Sp", "Avg Weight Sp",
               "Avg Output Sp", "Avg Attn Sp", "One-Liner Explanation"]

    data = [
        ["resnet50_dense", "Real", 54, 10800, 0.427, 0.000, 0.000, "N/A",
         "ReLU kills ~43% activations; no pruning → weight_sp=0"],
        ["resnet50_pruned50", "Real", 54, 10800, 0.429, 0.324, 0.000, "N/A",
         "Same ReLU; wt_sp<0.5 because global L1 prunes small layers more"],
        ["resnet50_pruned70", "Real", 54, 10800, 0.422, 0.489, 0.000, "N/A",
         "Global pruning non-uniform: big conv layers keep more weights"],
        ["resnet50_pruned90", "Real", 54, 10800, 0.426, 0.732, 0.006, "N/A",
         "Extreme pruning; output_sp>0 because Wx≈0 when most w=0"],
        ["bert_dense", "Real", 73, 14600, 0.000, 0.000, 0.000, 0.200,
         "GELU never produces exact zeros; attn varies 0-92% per input"],
        ["bert_pruned50", "Real", 73, 14600, 0.000, 0.500, 0.000, 0.201,
         "Uniform Linear layers → wt_sp matches target; GELU still no zeros"],
        ["bert_pruned70", "Real", 73, 14600, 0.000, 0.700, 0.000, 0.203,
         "Same pattern; slightly higher attn_sp from noisier Q/K projections"],
        ["bert_pruned90", "Real", 73, 14600, 0.000, 0.900, 0.000, 0.215,
         "Heavy pruning → noisier attention → more concentrated softmax"],
        ["gcn_synthetic", "Synthetic", 8, 1600, 0.584, 0.188, 0.197, "N/A",
         "Sparse adj (~90%) + dense transform (~10%) averaged → 58%"],
        ["mixed_pipeline", "Synthetic", 8, 1600, 0.391, 0.389, 0.148, "N/A",
         "Wild swings (0%→95%) in 8 layers; tests extreme heterogeneity"],
        ["variable_random", "Synthetic", 20, 4000, 0.294, 0.470, 0.287, "N/A",
         "Fully random — no patterns; hardest test for predictor"],
        ["dp_stress", "Synthetic", 40, 8000, 0.132, 0.113, 0.123, "N/A",
         "85% dense + 15% sparse outliers; adversarial for scheduler"],
    ]

    end_row = add_table(ws1, row, headers, data,
                        col_widths=[22, 10, 8, 8, 13, 14, 13, 12, 55])

    # Total row
    ws1.cell(row=end_row, column=1, value="TOTAL").font = BOLD_FONT
    ws1.cell(row=end_row, column=4, value=116800).font = BOLD_FONT
    ws1.cell(row=end_row, column=9, value="116,800 rows across 12 datasets").font = BOLD_FONT

    # =========================================================================
    # Sheet 2: Predictor Results
    # =========================================================================
    ws2 = wb.create_sheet("Predictor")

    row = add_title(ws2, 1, "Step 1.2: Sparsity Predictor Results",
                    "8-feature MLP, leave-one-pruning-level-out split, early stopping")

    # Architecture
    row = add_title(ws2, row, "Architecture")
    ws2.cell(row=row, column=1,
             value="Input(8) → Linear(64) → ReLU → Linear(64) → ReLU → Linear(64) → ReLU → Linear(1) → Sigmoid").font = DATA_FONT
    row += 2

    # Features
    feat_headers = ["#", "Feature", "Description"]
    feat_data = [
        [1, "input_sparsity", "Fraction of zeros in current layer's input"],
        [2, "weight_sparsity", "Fraction of zeros in weights"],
        [3, "is_conv", "1.0 for Conv2d, 0.0 for Linear"],
        [4, "log2(M)/20", "Normalized matrix row dimension"],
        [5, "log2(K)/20", "Normalized inner dimension"],
        [6, "log2(P)/20", "Normalized column dimension"],
        [7, "layer_position", "Normalized position in network (0→1)"],
        [8, "attention_sparsity", "Fraction of near-zero attention weights (transformers only)"],
    ]
    row = add_table(ws2, row, feat_headers, feat_data, col_widths=[5, 22, 50])

    # Training
    train_headers = ["Parameter", "Value", "Explanation"]
    train_data = [
        ["Train set", "0%, 50%, 90% pruning + synthetic", "All architectures except 70% pruned"],
        ["Test set", "70% pruned only", "Completely unseen pruning level"],
        ["Target", "max(input_sp, attn_sp)", "Effective sparsity — makes BERT non-trivial"],
        ["Early stopping", "Patience=20 epochs", "Prevents overfitting; restores best weights"],
        ["Epochs (typical)", "~40 (restores ~23)", "Stops when test MAE stops improving"],
    ]
    row = add_table(ws2, row, train_headers, train_data, col_widths=[20, 28, 50])

    # Results
    result_headers = ["Metric", "Value", "Target", "Status", "Explanation"]
    result_data = [
        ["Test MAE", 0.0569, "—", "—", "Average absolute prediction error"],
        ["Test RMSE", 0.1439, "—", "—", "Root mean squared error"],
        ["Config Match Rate", "99.2%", "≥90%", "✓ PASS", "Correct hardware config from predicted sparsity"],
        ["BERT MAE", 0.0731, "—", "—", "Non-trivially earned (was 0.0004 before attn fix)"],
        ["BERT Match", "100.0%", "—", "✓", "MAE stays below config cost boundaries"],
        ["ResNet Match (primary)", "98.1%", "≥90%", "✓ PASS", "Conservative generalization claim"],
    ]
    row = add_table(ws2, row, result_headers, result_data, col_widths=[22, 10, 8, 10, 50])

    # =========================================================================
    # Sheet 3: Scheduler Results
    # =========================================================================
    ws3 = wb.create_sheet("Scheduler")

    row = add_title(ws3, 1, "Step 1.3: DP Configuration Scheduler Results",
                    "4 strategies: Best Static, Greedy (W=1), DP (W=5), Oracle — switch costs 100-250 cycles")

    sched_headers = ["Model", "Static%", "Greedy%", "DP%", "Speedup",
                     "G.sw", "D.sw", "O.sw", "Explanation"]
    sched_data = [
        ["bert_dense", "-0.0%", "+0.00%", "+0.00%", "1.000×", 0, 0, 0,
         "Homogeneous layers — one config optimal for all"],
        ["bert_pruned50", "-0.0%", "+0.00%", "+0.00%", "1.000×", 0, 0, 0,
         "Same dims → same config; pruning doesn't change shape"],
        ["bert_pruned70", "-0.0%", "+0.00%", "+0.00%", "1.000×", 0, 0, 0,
         "Same as above — BERT is inherently uniform"],
        ["bert_pruned90", "-0.0%", "+0.00%", "+0.00%", "1.000×", 0, 0, 0,
         "Even at 90% prune, same config wins everywhere"],
        ["dp_stress", "+236.5%", "+0.00%", "+0.00%", "3.365×", 12, 12, 12,
         "Adversarial; static wastes 236% — adaptation essential"],
        ["gcn_synthetic", "+130.0%", "+0.00%", "+0.00%", "2.300×", 3, 3, 3,
         "Sparse adj + dense transform need different configs"],
        ["mixed_pipeline", "+494.7%", "+0.00%", "+0.00%", "5.947×", 5, 5, 5,
         "Best result! 8 wildly different layers → 5.95× speedup"],
        ["resnet50_dense", "+0.0%", "+0.00%", "+0.00%", "1.000×", 8, 6, 6,
         "DP beats Greedy: avoids 2 unnecessary switches"],
        ["resnet50_pruned50", "+84.2%", "+0.00%", "+0.00%", "1.842×", 32, 32, 32,
         "Mixed conv sizes need many switches; all strategies agree"],
        ["resnet50_pruned70", "+42.1%", "+0.00%", "+0.00%", "1.421×", 25, 24, 24,
         "DP saves 1 switch vs Greedy, matches Oracle"],
        ["resnet50_pruned90", "+7.9%", "+0.01%", "+0.00%", "1.079×", 5, 4, 4,
         "DP saves 1 switch; small static gap at high pruning"],
        ["variable_random", "+10.5%", "+0.00%", "+0.00%", "1.105×", 2, 2, 2,
         "Even random workloads benefit 10% from adaptation"],
    ]
    row = add_table(ws3, row, sched_headers, sched_data,
                    col_widths=[22, 9, 9, 8, 9, 6, 6, 6, 55])

    # Summary metrics
    summary_headers = ["Metric", "Value", "Explanation"]
    summary_data = [
        ["Average adaptive speedup", "1.838×", "Adapting per-layer is 1.84× faster than picking one config"],
        ["Max speedup", "5.947×", "mixed_pipeline: wildly different layers benefit most"],
        ["Average DP gap", "+0.00%", "DP matches Oracle cost on every model"],
        ["DP switches saved vs Greedy", "4 (92→88)", "DP avoids unnecessary switches on 3 ResNet variants"],
        ["Cost impact of saved switches", "~400 cycles", "Negligible at this compute scale — theoretical correctness"],
    ]
    row = add_table(ws3, row, summary_headers, summary_data, col_widths=[30, 18, 55])

    # =========================================================================
    # Sheet 4: Window Ablation
    # =========================================================================
    ws4 = wb.create_sheet("Window Ablation")

    row = add_title(ws4, 1, "Window-Size Ablation",
                    "DP with W=1 (Greedy), W=2, W=3, W=5, and full Oracle — justifies W=5 choice")

    abl_headers = ["Model", "W=1", "W=2", "W=3", "W=5", "Oracle",
                   "W=1 sw", "W=5 sw", "O.sw", "Explanation"]
    abl_data = [
        ["bert_dense", "+0.000%", "+0.000%", "+0.000%", "+0.000%", "0%", 0, 0, 0,
         "No switches needed → all windows identical"],
        ["bert_pruned50-90", "+0.000%", "+0.000%", "+0.000%", "+0.000%", "0%", 0, 0, 0,
         "Same — homogeneous models trivially optimal"],
        ["dp_stress", "+0.000%", "+0.000%", "+0.000%", "+0.000%", "0%", 12, 12, 12,
         "Every switch clearly worth it → all windows agree"],
        ["gcn_synthetic", "+0.000%", "+0.000%", "+0.000%", "+0.000%", "0%", 3, 3, 3,
         "Few switches, all obvious → no window effect"],
        ["mixed_pipeline", "+0.000%", "+0.000%", "+0.000%", "+0.000%", "0%", 5, 5, 5,
         "5 switches all clearly beneficial"],
        ["resnet50_dense", "+0.003%", "+0.000%", "+0.000%", "+0.000%", "0%", 8, 6, 6,
         "ONLY model where W=1 diverges — W=2 already fixes it"],
        ["resnet50_pruned50", "+0.000%", "+0.000%", "+0.000%", "+0.000%", "0%", 32, 32, 32,
         "Many switches but all clearly necessary"],
        ["resnet50_pruned70", "+0.000%", "+0.000%", "+0.000%", "+0.000%", "0%", 24, 24, 24,
         "W=1 matches Oracle here (switch decision obvious)"],
        ["resnet50_pruned90", "+0.000%", "+0.000%", "+0.000%", "+0.000%", "0%", 4, 4, 4,
         "Few switches, all worth it"],
        ["variable_random", "+0.000%", "+0.000%", "+0.000%", "+0.000%", "0%", 2, 2, 2,
         "Only 2 switches needed"],
    ]
    row = add_table(ws4, row, abl_headers, abl_data,
                    col_widths=[22, 10, 10, 10, 10, 10, 8, 8, 6, 50])

    # Conclusion
    ws4.cell(row=row, column=1,
             value="Conclusion: W=2 already matches Oracle on all workloads. W=5 is conservative for unseen workloads.").font = BOLD_FONT

    # =========================================================================
    # Sheet 5: Prefetch Pipeline
    # =========================================================================
    ws5 = wb.create_sheet("Prefetch Pipeline")

    row = add_title(ws5, 1, "Step 1.4: Prefetch Pipeline Results",
                    "Tiled double-buffering hides reconfiguration overhead behind computation")

    pf_headers = ["Model", "Overhead Hidden", "Speedup", "Explanation"]
    pf_data = [
        ["bert_dense", "100.0%", "1.043×", "Large compute/fetch ratio → fully hidden"],
        ["bert_pruned50", "100.0%", "1.054×", "Pruning reduces compute but fetch still fits"],
        ["bert_pruned70", "100.0%", "1.054×", "Same — sparse ops still dominate fetch time"],
        ["bert_pruned90", "100.0%", "1.032×", "Less compute → closer to stall but still hidden"],
        ["gcn_synthetic", "100.0%", "1.022×", "Large sparse matrices → long compute hides fetch"],
        ["mixed_pipeline", "100.0%", "1.013×", "Diverse layers; smallest speedup (already fast)"],
        ["resnet50_dense", "100.0%", "1.038×", "Conv layers have high compute/fetch ratio"],
        ["resnet50_pruned50", "100.0%", "1.032×", "Sparse conv still compute-bound"],
        ["resnet50_pruned70", "100.0%", "1.032×", "Same pattern"],
        ["resnet50_pruned90", "100.0%", "1.038×", "Even at 90% prune, compute > fetch"],
        ["variable_random", "92.3%", "1.045×", "Small random layers → some can't hide fetch"],
    ]
    row = add_table(ws5, row, pf_headers, pf_data, col_widths=[22, 16, 10, 55])

    # Summary
    ws5.cell(row=row, column=1,
             value="Average: 99.3% hidden, 1.037× speedup. Reconfiguration overhead is effectively zero.").font = BOLD_FONT
    row += 2
    ws5.cell(row=row, column=1,
             value="Caveat: Assumes prefetch and compute don't contend for shared HBM bandwidth. RTL will validate.").font = EXPLAIN_FONT

    # =========================================================================
    # Sheet 6: Issues Fixed
    # =========================================================================
    ws6 = wb.create_sheet("Issues Fixed")

    row = add_title(ws6, 1, "Scientific Rigor — All Issues Fixed",
                    "8 issues identified and resolved to ensure honest, reviewer-proof results")

    fix_headers = ["#", "Issue", "Root Cause", "Fix Applied", "Status"]
    fix_data = [
        [1, "Negative gaps (DP beats Oracle)", "Different cost calculations per strategy",
         "Unified rescore_schedule + weight-based op type", "✓ Fixed"],
        [2, "Data leakage in predictor", "Random 80/20 split → memorization",
         "Leave-one-pruning-level-out (train:0/50/90, test:70)", "✓ Fixed"],
        [3, "BERT shows no runtime variation", "input_sparsity always 0 (GELU)",
         "Attention sparsity hooks (0-99% range)", "✓ Fixed"],
        [4, "DP = Greedy everywhere", "Switch costs negligible vs compute",
         "Realistic costs (100-250) + adversarial workload", "✓ Fixed"],
        [5, "No static baseline", "Only compared Greedy/DP/Oracle",
         "Added Best Static comparison (1.84× avg speedup)", "✓ Fixed"],
        [6, "Attention sparsity not in predictor", "Collected but unused → trivial BERT",
         "Feature #8 + target=effective_sparsity", "✓ Fixed"],
        [7, "'Reconfiguration is FREE' claim", "Stated as hardware fact, is simulator assumption",
         "Softened to 'fully hidden under no-contention assumption'", "✓ Fixed"],
        [8, "Switch cost derivation wrong", "'16-deep pipeline' for 8×8 array",
         "Corrected to 8-cycle drain; flagged as estimate", "✓ Fixed"],
    ]
    row = add_table(ws6, row, fix_headers, fix_data,
                    col_widths=[4, 32, 35, 45, 10])

    # =========================================================================
    # Sheet 7: Paper Claims
    # =========================================================================
    ws7 = wb.create_sheet("Paper Claims")

    row = add_title(ws7, 1, "Paper-Ready Claims with Evidence",
                    "Each claim is backed by specific data — ready for ASP-DAC manuscript")

    claim_headers = ["#", "Claim", "Evidence", "Strength"]
    claim_data = [
        [1, "1.84× avg speedup over static config (up to 5.95×)",
         "12 models, Best Static vs Oracle comparison", "Strong ✅"],
        [2, "DP matches Oracle decisions; Greedy makes 4 extra switches",
         "88 vs 92 switches; cost impact ~400 cycles (negligible)", "Honest ⚠️"],
        [3, "99.2% config match on unseen pruning (98.1% ResNet-only)",
         "Leave-one-out: train 0/50/90%, test 70%", "Strong ✅"],
        [4, "Attention sparsity 0-99% in transformers (input-dependent)",
         "Softmax weights captured via patched BertSelfAttention", "Strong ✅"],
        [5, "99.3% reconfiguration overhead hidden by prefetch",
         "Tiled double-buffering; assumes no HBM contention", "Good (caveated) ✅"],
        [6, "W=2 sufficient; W=5 conservative choice",
         "Window ablation across 12 models", "Strong ✅"],
        [7, "Switch costs 100-250 cycles (estimated for 8×8 PE array)",
         "8-cycle drain + interconnect + control word + refill", "Estimate (RTL pending) ⚠️"],
    ]
    row = add_table(ws7, row, claim_headers, claim_data,
                    col_widths=[4, 50, 45, 18])

    # =========================================================================
    # Save
    # =========================================================================
    wb.save(OUTPUT_PATH)
    print(f"[DONE] Excel saved to: {OUTPUT_PATH}")
    print(f"  Sheets: {[ws.title for ws in wb.worksheets]}")


if __name__ == '__main__':
    create_excel()
