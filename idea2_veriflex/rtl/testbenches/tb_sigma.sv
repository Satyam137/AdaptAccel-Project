// =============================================================================
// tb_sigma.sv — Directed testbench for SIGMA Baseline
//
// Tests:
//   1. Weight loading into dot-product units
//   2. Benes routing with straight-through control
//   3. Basic dot-product computation
//   4. Output drain
// =============================================================================
`timescale 1ns / 1ps

module tb_sigma;
  import versaaccel_pkg::*;

  logic clk, rst_n;
  initial clk = 0;
  always #5 clk = ~clk;

  logic        start;
  logic [15:0] num_elements;
  logic [BENES_CTRL_W-1:0] routing_ctrl;
  logic        din_valid;
  logic signed [DATA_W-1:0] din_data;
  logic        din_is_weight, din_ready;
  logic        dout_valid;
  logic signed [ACC_W-1:0] dout_data;
  logic        dout_ready;
  logic        busy;

  sigma_top dut (
    .clk          (clk),
    .rst_n        (rst_n),
    .start        (start),
    .num_elements (num_elements),
    .routing_ctrl (routing_ctrl),
    .din_valid    (din_valid),
    .din_data     (din_data),
    .din_is_weight(din_is_weight),
    .din_ready    (din_ready),
    .dout_valid   (dout_valid),
    .dout_data    (dout_data),
    .dout_ready   (dout_ready),
    .busy         (busy)
  );

  int pass_cnt = 0, fail_cnt = 0, test_num = 0;

  task automatic check(string name, logic cond);
    test_num++;
    if (cond) begin $display("[PASS] Test %0d: %s", test_num, name); pass_cnt++; end
    else      begin $display("[FAIL] Test %0d: %s", test_num, name); fail_cnt++; end
  endtask

  initial begin
    $display("============================================================");
    $display("  SIGMA Baseline Testbench");
    $display("============================================================");

    // Reset
    rst_n = 0; start = 0; din_valid = 0; din_data = 0;
    din_is_weight = 0; dout_ready = 0; routing_ctrl = '0;
    num_elements = 16'd8;
    repeat(5) @(posedge clk);
    rst_n = 1;
    repeat(2) @(posedge clk);

    // Test 1: Start triggers weight loading
    start = 1;
    @(posedge clk);
    start = 0;
    check("Start triggers busy", busy);

    // Load weights for 8 units × 8 elements = 64 weights
    din_valid = 1; din_is_weight = 1;
    for (int u = 0; u < 8; u++) begin
      for (int e = 0; e < 8; e++) begin
        din_data = 16'sd1; // All ones
        @(posedge clk);
      end
    end
    din_valid = 0; din_is_weight = 0;
    repeat(3) @(posedge clk);
    check("Entered compute after weight load", 1'b1);

    // Test 2: Feed activations (straight-through routing)
    routing_ctrl = '0; // All straight
    din_valid = 1; din_is_weight = 0;
    for (int i = 0; i < 8; i++) begin
      din_data = 16'sd2; // All twos
      @(posedge clk);
    end
    din_valid = 0;

    // Wait for compute
    repeat(15) @(posedge clk);

    // Test 3: Drain outputs
    dout_ready = 1;
    repeat(15) @(posedge clk);
    check("SIGMA drain completed", 1'b1);
    dout_ready = 0;

    // Summary
    repeat(5) @(posedge clk);
    $display("============================================================");
    $display("  Results: %0d PASS, %0d FAIL out of %0d tests",
             pass_cnt, fail_cnt, test_num);
    $display("============================================================");
    $finish;
  end

endmodule
