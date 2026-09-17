// =============================================================================
// reduction_network.sv — Configurable adder tree for partial sum reduction
//
// 3-stage reduction tree for 8 inputs:
//   Stage 1: 8→4 (4 adders)
//   Stage 2: 4→2 (2 adders)
//   Stage 3: 2→1 (1 adder)
//
// Modes:
//   FULL:    All 8 inputs reduced to 1 output (result[0])
//   PARTIAL: 4 pairs reduced to 4 outputs
//   BYPASS:  8 inputs pass through to 8 outputs (no reduction)
// =============================================================================
module reduction_network
  import versaaccel_pkg::*;
#(
  parameter int N     = NR,
  parameter int WIDTH = ACC_W
)(
  input  logic [1:0]          mode,  // 0=FULL, 1=PARTIAL, 2=BYPASS
  input  logic signed [WIDTH-1:0]  data_in [N],
  output logic signed [WIDTH-1:0]  data_out [N]
);

  // Stage 1: 8 → 4
  logic signed [WIDTH-1:0] s1 [N/2];

  // Stage 2: 4 → 2
  logic signed [WIDTH-1:0] s2 [N/4];

  // Stage 3: 2 → 1
  logic signed [WIDTH-1:0] s3;

  genvar i;

  // ---------- Stage 1 ----------
  generate
    for (i = 0; i < N/2; i++) begin : gen_s1
      assign s1[i] = data_in[2*i] + data_in[2*i + 1];
    end
  endgenerate

  // ---------- Stage 2 ----------
  generate
    for (i = 0; i < N/4; i++) begin : gen_s2
      assign s2[i] = s1[2*i] + s1[2*i + 1];
    end
  endgenerate

  // ---------- Stage 3 ----------
  assign s3 = s2[0] + s2[1];

  // ---------- Output mux ----------
  always_comb begin
    for (int j = 0; j < N; j++) data_out[j] = '0;

    case (mode)
      2'd0: begin // FULL reduction: single output
        data_out[0] = s3;
      end
      2'd1: begin // PARTIAL: 4 pair sums
        for (int j = 0; j < N/2; j++)
          data_out[j] = s1[j];
      end
      default: begin // BYPASS: pass through
        for (int j = 0; j < N; j++)
          data_out[j] = data_in[j];
      end
    endcase
  end

endmodule
