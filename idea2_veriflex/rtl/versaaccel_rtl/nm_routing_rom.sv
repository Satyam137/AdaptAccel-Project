// =============================================================================
// nm_routing_rom.sv — Precomputed Benes routes for 2:4 N:M patterns
//
// 2:4 structured sparsity has C(4,2)=6 possible nonzero positions.
// Each pattern maps to a precomputed Benes switch configuration.
// This eliminates runtime intersection discovery in D-M-R+NM mode.
//
// Pattern encoding (2 bits per group of 4):
//   0: positions {0,1}  → route to PE 0,1
//   1: positions {0,2}  → route to PE 0,2
//   2: positions {0,3}  → route to PE 0,3
//   3: positions {1,2}  → route to PE 1,2
//   4: positions {1,3}  → route to PE 1,3
//   5: positions {2,3}  → route to PE 2,3
// =============================================================================
module nm_routing_rom
  import versaaccel_pkg::*;
#(
  parameter int CTRL_WIDTH = BENES_CTRL_W  // 20 bits
)(
  input  logic [2:0]                pattern_idx,  // 0-5 pattern selector
  output logic [CTRL_WIDTH-1:0]     benes_ctrl    // Precomputed switch config
);

  // Precomputed Benes routes for each 2:4 pattern
  // These route the 2 nonzero elements to the correct PE inputs
  always_comb begin
    case (pattern_idx)
      3'd0: benes_ctrl = 20'b0000_0000_0000_0000_0000; // {0,1}: straight through
      3'd1: benes_ctrl = 20'b0000_0000_0100_0000_0000; // {0,2}: swap mid stage
      3'd2: benes_ctrl = 20'b0001_0000_0100_0000_0001; // {0,3}: outer + mid swap
      3'd3: benes_ctrl = 20'b0000_0001_0100_0001_0000; // {1,2}: inner swaps
      3'd4: benes_ctrl = 20'b0001_0001_0100_0001_0001; // {1,3}: full routing
      3'd5: benes_ctrl = 20'b0001_0001_0000_0001_0001; // {2,3}: outer swaps
      default: benes_ctrl = '0;
    endcase
  end

endmodule
