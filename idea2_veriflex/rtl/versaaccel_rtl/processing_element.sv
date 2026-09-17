// =============================================================================
// processing_element.sv — Configurable PE for VersaAccel
//
// Supports 4 modes:
//   MAC:   out = acc + (a * b)
//   PASS:  propagate a/b to neighbors (systolic forwarding)
//   ACCUM: out = acc + a_in (reduction mode)
//   IDLE:  hold state, no switching activity
// =============================================================================
module processing_element
  import versaaccel_pkg::*;
#(
  parameter int DATA_WIDTH = DATA_W,
  parameter int ACC_WIDTH  = ACC_W
)(
  input  logic                    clk,
  input  logic                    rst_n,

  // Control
  input  pe_mode_t                mode,
  input  logic                    acc_clear,   // Clear accumulator
  input  logic                    acc_en,      // Enable accumulate

  // Data inputs
  input  logic signed [DATA_WIDTH-1:0] a_in,  // Activation / partial sum
  input  logic signed [DATA_WIDTH-1:0] b_in,  // Weight

  // Data outputs
  output logic signed [ACC_WIDTH-1:0]  result, // Accumulated result
  output logic signed [DATA_WIDTH-1:0] a_out,  // Forwarded activation
  output logic signed [DATA_WIDTH-1:0] b_out   // Forwarded weight
);

  // ---------- Internal signals ----------
  logic signed [2*DATA_WIDTH-1:0] mult_result;
  logic signed [ACC_WIDTH-1:0]    acc_reg;
  logic signed [ACC_WIDTH-1:0]    acc_next;

  // ---------- Multiplier ----------
  assign mult_result = a_in * b_in;

  // ---------- Accumulator logic ----------
  always_comb begin
    case (mode)
      PE_MAC:   acc_next = acc_clear ? mult_result[ACC_WIDTH-1:0]
                                     : acc_reg + mult_result[ACC_WIDTH-1:0];
      PE_ACCUM: acc_next = acc_clear ? {{(ACC_WIDTH-DATA_WIDTH){a_in[DATA_WIDTH-1]}}, a_in}
                                     : acc_reg + {{(ACC_WIDTH-DATA_WIDTH){a_in[DATA_WIDTH-1]}}, a_in};
      PE_PASS:  acc_next = acc_reg;  // Hold
      PE_IDLE:  acc_next = acc_reg;  // Hold
      default:  acc_next = acc_reg;
    endcase
  end

  // ---------- Accumulator register ----------
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n)
      acc_reg <= '0;
    else if (acc_en || acc_clear)
      acc_reg <= acc_next;
  end

  // ---------- Outputs ----------
  assign result = acc_reg;

  // Systolic forwarding (1-cycle delay for timing)
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      a_out <= '0;
      b_out <= '0;
    end else begin
      a_out <= a_in;
      b_out <= b_in;
    end
  end

endmodule
