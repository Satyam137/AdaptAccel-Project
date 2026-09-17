// =============================================================================
// distribution_network.sv — 8-port Benes network for data routing
//
// 5-stage Benes network (2*log2(8)-1 = 5) with 4 switches per stage.
// Routes 8 inputs to 8 outputs in any permutation.
// Control: 20 bits (5 stages × 4 switches × 1 bit each)
//
// In DMR mode:    straight-through (broadcast)
// In D-M-R mode:  programmed by controller for sparse routing
// In D-M-R+NM:    routes loaded from NM routing ROM
// =============================================================================
module distribution_network
  import versaaccel_pkg::*;
#(
  parameter int N     = BENES_N,      // 8
  parameter int WIDTH = DATA_W        // 16-bit data
)(
  input  logic [BENES_CTRL_W-1:0]  ctrl,       // 20-bit switch control
  input  logic [WIDTH-1:0]         data_in [N],
  output logic [WIDTH-1:0]         data_out [N]
);

  // Internal wires between stages
  logic [WIDTH-1:0] stage_wire [BENES_STAGES+1][N];

  // Input assignment
  genvar i;
  generate
    for (i = 0; i < N; i++) begin : gen_input
      assign stage_wire[0][i] = data_in[i];
    end
  endgenerate

  // Output assignment
  generate
    for (i = 0; i < N; i++) begin : gen_output
      assign data_out[i] = stage_wire[BENES_STAGES][i];
    end
  endgenerate

  // ---------- Stage 0: outer left — pairs (0,4),(1,5),(2,6),(3,7) ----------
  generate
    for (i = 0; i < N/2; i++) begin : gen_s0
      benes_switch #(.WIDTH(WIDTH)) sw_s0 (
        .ctrl (ctrl[i]),
        .in0  (stage_wire[0][i]),
        .in1  (stage_wire[0][i + N/2]),
        .out0 (stage_wire[1][i]),
        .out1 (stage_wire[1][i + N/2])
      );
    end
  endgenerate

  // ---------- Stage 1: inner left ----------
  // Upper half: pairs (0,2),(1,3);  Lower half: pairs (4,6),(5,7)
  generate
    for (i = 0; i < N/4; i++) begin : gen_s1
      // Upper subnetwork
      benes_switch #(.WIDTH(WIDTH)) sw_s1_upper (
        .ctrl (ctrl[N/2 + i]),
        .in0  (stage_wire[1][i]),
        .in1  (stage_wire[1][i + N/4]),
        .out0 (stage_wire[2][i]),
        .out1 (stage_wire[2][i + N/4])
      );
      // Lower subnetwork
      benes_switch #(.WIDTH(WIDTH)) sw_s1_lower (
        .ctrl (ctrl[N/2 + N/4 + i]),
        .in0  (stage_wire[1][N/2 + i]),
        .in1  (stage_wire[1][N/2 + i + N/4]),
        .out0 (stage_wire[2][N/2 + i]),
        .out1 (stage_wire[2][N/2 + i + N/4])
      );
    end
  endgenerate

  // ---------- Stage 2: middle — pairs (0,1),(2,3),(4,5),(6,7) ----------
  generate
    for (i = 0; i < N/2; i++) begin : gen_s2
      benes_switch #(.WIDTH(WIDTH)) sw_s2 (
        .ctrl (ctrl[2*(N/2) + i]),
        .in0  (stage_wire[2][2*i]),
        .in1  (stage_wire[2][2*i + 1]),
        .out0 (stage_wire[3][2*i]),
        .out1 (stage_wire[3][2*i + 1])
      );
    end
  endgenerate

  // ---------- Stage 3: inner right (mirror of stage 1) ----------
  generate
    for (i = 0; i < N/4; i++) begin : gen_s3
      benes_switch #(.WIDTH(WIDTH)) sw_s3_upper (
        .ctrl (ctrl[3*(N/2) + i]),
        .in0  (stage_wire[3][i]),
        .in1  (stage_wire[3][i + N/4]),
        .out0 (stage_wire[4][i]),
        .out1 (stage_wire[4][i + N/4])
      );
      benes_switch #(.WIDTH(WIDTH)) sw_s3_lower (
        .ctrl (ctrl[3*(N/2) + N/4 + i]),
        .in0  (stage_wire[3][N/2 + i]),
        .in1  (stage_wire[3][N/2 + i + N/4]),
        .out0 (stage_wire[4][N/2 + i]),
        .out1 (stage_wire[4][N/2 + i + N/4])
      );
    end
  endgenerate

  // ---------- Stage 4: outer right (mirror of stage 0) ----------
  generate
    for (i = 0; i < N/2; i++) begin : gen_s4
      benes_switch #(.WIDTH(WIDTH)) sw_s4 (
        .ctrl (ctrl[4*(N/2) + i]),
        .in0  (stage_wire[4][i]),
        .in1  (stage_wire[4][i + N/2]),
        .out0 (stage_wire[5][i]),
        .out1 (stage_wire[5][i + N/2])
      );
    end
  endgenerate

endmodule
