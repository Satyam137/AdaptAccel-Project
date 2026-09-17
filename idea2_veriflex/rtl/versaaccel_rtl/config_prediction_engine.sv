// =============================================================================
// config_prediction_engine.sv — Hardcoded MLP for config selection (CPE)
//
// Fixed-point (INT8) implementation of the trained predictor.
// Architecture: 9 inputs → 16 hidden (ReLU) → 16 hidden (ReLU) → 5 outputs
// Output: argmax → 3-bit config_select
//
// Weights are hardcoded (loaded from trained model at synthesis time).
// This is a simplified CPE suitable for area estimation.
// =============================================================================
module config_prediction_engine
  import versaaccel_pkg::*;
(
  input  logic        clk,
  input  logic        rst_n,
  input  logic        valid_in,

  // 9 input features (8-bit fixed-point, Q4.4 format)
  input  logic signed [7:0] features [9],

  // Outputs
  output config_t     predicted_config,
  output logic [7:0]  confidence,      // 0-255 confidence level
  output logic        valid_out
);

  // ---------- Pipeline registers ----------
  // Stage 1: Input → Hidden1  (9×16 MAC)
  // Stage 2: Hidden1 → Hidden2 (16×16 MAC)
  // Stage 3: Hidden2 → Output (16×5 MAC + argmax)

  localparam int H1 = 16;  // Hidden layer 1 size
  localparam int H2 = 16;  // Hidden layer 2 size
  localparam int OUT = 5;  // Number of configs

  // Weight memories (hardcoded for synthesis)
  // In practice: loaded from parameter file or register bank
  logic signed [7:0] w1 [H1][9];   // Layer 1 weights
  logic signed [7:0] b1 [H1];      // Layer 1 biases
  logic signed [7:0] w2 [H2][H1];  // Layer 2 weights
  logic signed [7:0] b2 [H2];      // Layer 2 biases
  logic signed [7:0] w3 [OUT][H2]; // Output weights
  logic signed [7:0] b3 [OUT];     // Output biases

  // Initialize weights to reasonable defaults (placeholder for actual trained values)
  // In real implementation: these come from train_predictor_v2.py exported weights
  integer wi, wj;
  always_comb begin
    for (wi = 0; wi < H1; wi++) begin
      for (wj = 0; wj < 9; wj++)
        w1[wi][wj] = 8'sd1;
      b1[wi] = 8'sd0;
    end
    for (wi = 0; wi < H2; wi++) begin
      for (wj = 0; wj < H1; wj++)
        w2[wi][wj] = 8'sd1;
      b2[wi] = 8'sd0;
    end
    for (wi = 0; wi < OUT; wi++) begin
      for (wj = 0; wj < H2; wj++)
        w3[wi][wj] = 8'sd1;
      b3[wi] = 8'sd0;
    end
  end

  // ---------- Stage 1: Hidden Layer 1 ----------
  logic signed [15:0] h1_acc [H1];
  logic signed [7:0]  h1_out [H1];  // After ReLU
  logic                s1_valid;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      s1_valid <= 1'b0;
      for (int k = 0; k < H1; k++) h1_out[k] <= 8'sd0;
    end else begin
      s1_valid <= valid_in;
      if (valid_in) begin
        for (int k = 0; k < H1; k++) begin
          h1_acc[k] = {8{b1[k][7]}, b1[k]};  // Sign-extend bias
          for (int j = 0; j < 9; j++)
            h1_acc[k] = h1_acc[k] + w1[k][j] * features[j];
          // ReLU + saturate to 8-bit
          h1_out[k] <= (h1_acc[k] < 0) ? 8'sd0 :
                        (h1_acc[k] > 16'sd127) ? 8'sd127 :
                        h1_acc[k][7:0];
        end
      end
    end
  end

  // ---------- Stage 2: Hidden Layer 2 ----------
  logic signed [15:0] h2_acc [H2];
  logic signed [7:0]  h2_out [H2];
  logic                s2_valid;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      s2_valid <= 1'b0;
      for (int k = 0; k < H2; k++) h2_out[k] <= 8'sd0;
    end else begin
      s2_valid <= s1_valid;
      if (s1_valid) begin
        for (int k = 0; k < H2; k++) begin
          h2_acc[k] = {8{b2[k][7]}, b2[k]};
          for (int j = 0; j < H1; j++)
            h2_acc[k] = h2_acc[k] + w2[k][j] * h1_out[j];
          h2_out[k] <= (h2_acc[k] < 0) ? 8'sd0 :
                        (h2_acc[k] > 16'sd127) ? 8'sd127 :
                        h2_acc[k][7:0];
        end
      end
    end
  end

  // ---------- Stage 3: Output + Argmax ----------
  logic signed [15:0] out_acc [OUT];
  logic signed [15:0] max_val;
  logic [2:0]         max_idx;
  logic                s3_valid;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      s3_valid <= 1'b0;
      predicted_config <= CFG_D_M_R;
      confidence <= 8'd0;
    end else begin
      s3_valid <= s2_valid;
      if (s2_valid) begin
        // Compute output scores
        max_val = -16'sd32768;
        max_idx = 3'd3; // default D-M-R
        for (int k = 0; k < OUT; k++) begin
          out_acc[k] = {8{b3[k][7]}, b3[k]};
          for (int j = 0; j < H2; j++)
            out_acc[k] = out_acc[k] + w3[k][j] * h2_out[j];
          if (out_acc[k] > max_val) begin
            max_val = out_acc[k];
            max_idx = k[2:0];
          end
        end
        predicted_config <= config_t'(max_idx);

        // Confidence = gap between best and 2nd best (scaled)
        // Simple approximation: use max value magnitude
        confidence <= (max_val > 0) ? ((max_val > 16'sd255) ? 8'd255 : max_val[7:0])
                                    : 8'd0;
      end
    end
  end

  assign valid_out = s3_valid;

endmodule
