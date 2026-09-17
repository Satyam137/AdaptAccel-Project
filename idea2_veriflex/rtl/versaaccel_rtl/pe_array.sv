// =============================================================================
// pe_array.sv — 8×8 configurable PE array
//
// Instantiates NR×NC processing elements with configurable interconnect.
// In DMR mode: systolic data flow (a flows right, b flows down)
// In D-M-R mode: each PE operates independently via Benes routing
// =============================================================================
module pe_array
  import versaaccel_pkg::*;
#(
  parameter int ROWS = NR,
  parameter int COLS = NC
)(
  input  logic        clk,
  input  logic        rst_n,

  // Configuration
  input  config_t     config_mode,
  input  logic        acc_clear,
  input  logic        acc_en,

  // Data inputs (from distribution network)
  input  logic signed [DATA_W-1:0] a_in [ROWS],  // Row activations
  input  logic signed [DATA_W-1:0] b_in [COLS],  // Column weights

  // Outputs (to reduction network)
  output logic signed [ACC_W-1:0]  pe_out [ROWS][COLS]
);

  // PE mode derived from config
  pe_mode_t pe_mode;

  always_comb begin
    case (config_mode)
      CFG_DMR:      pe_mode = PE_MAC;   // All PEs do MAC
      CFG_D_MR:     pe_mode = PE_MAC;   // MAC with fused reduce
      CFG_DM_R:     pe_mode = PE_MAC;   // MAC with fused distribute
      CFG_D_M_R:    pe_mode = PE_MAC;   // Independent MAC
      CFG_D_M_R_NM: pe_mode = PE_MAC;   // Independent MAC with NM routing
      default:      pe_mode = PE_IDLE;
    endcase
  end

  // Inter-PE wires for systolic forwarding
  logic signed [DATA_W-1:0] a_wire [ROWS][COLS+1];
  logic signed [DATA_W-1:0] b_wire [ROWS+1][COLS];

  // Connect inputs
  genvar r, c;
  generate
    for (r = 0; r < ROWS; r++) begin : gen_a_in
      assign a_wire[r][0] = a_in[r];
    end
    for (c = 0; c < COLS; c++) begin : gen_b_in
      assign b_wire[0][c] = b_in[c];
    end
  endgenerate

  // Instantiate PE grid
  generate
    for (r = 0; r < ROWS; r++) begin : gen_pe_row
      for (c = 0; c < COLS; c++) begin : gen_pe_col
        processing_element #(
          .DATA_WIDTH (DATA_W),
          .ACC_WIDTH  (ACC_W)
        ) pe_inst (
          .clk       (clk),
          .rst_n     (rst_n),
          .mode      (pe_mode),
          .acc_clear (acc_clear),
          .acc_en    (acc_en),
          .a_in      (a_wire[r][c]),
          .b_in      (b_wire[r][c]),
          .result    (pe_out[r][c]),
          .a_out     (a_wire[r][c+1]),
          .b_out     (b_wire[r+1][c])
        );
      end
    end
  endgenerate

endmodule
