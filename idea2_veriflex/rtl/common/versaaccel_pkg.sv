// =============================================================================
// versaaccel_pkg.sv — Shared types, parameters, and enums for VersaAccel
// =============================================================================
`ifndef VERSAACCEL_PKG_SV
`define VERSAACCEL_PKG_SV

package versaaccel_pkg;

  // ---------- Array dimensions ----------
  parameter int NR           = 8;    // PE rows
  parameter int NC           = 8;    // PE columns
  parameter int NUM_PES      = NR * NC;

  // ---------- Data widths ----------
  parameter int DATA_W       = 16;   // INT16 activations/weights
  parameter int ACC_W        = 32;   // Accumulator width
  parameter int ADDR_W       = 16;   // Buffer address width
  parameter int SPARSITY_W   = 8;    // Sparsity estimate (0-255 = 0%-100%)

  // ---------- Benes network ----------
  parameter int BENES_N      = NR;                        // Network size
  parameter int BENES_STAGES = 2 * $clog2(BENES_N) - 1;  // 5 for N=8
  parameter int BENES_SW_PER_STAGE = BENES_N / 2;         // 4 switches/stage
  parameter int BENES_CTRL_W = BENES_STAGES * BENES_SW_PER_STAGE; // 20 bits

  // ---------- Configuration ----------
  typedef enum logic [2:0] {
    CFG_DMR     = 3'd0,  // Dense systolic
    CFG_D_MR    = 3'd1,  // Distribution separate, MR fused
    CFG_DM_R    = 3'd2,  // DM fused, Reduction separate
    CFG_D_M_R   = 3'd3,  // All separate (Gustavson)
    CFG_D_M_R_NM = 3'd4  // D-M-R with N:M precomputed routing
  } config_t;

  // ---------- Phase FSM ----------
  typedef enum logic [2:0] {
    PHASE_PREFILL   = 3'd0,
    PHASE_DECODE    = 3'd1,
    PHASE_ATTENTION = 3'd2,
    PHASE_AGGREGATE = 3'd3,
    PHASE_TRANSFORM = 3'd4,
    PHASE_CONV      = 3'd5,
    PHASE_FC        = 3'd6,
    PHASE_GENERIC   = 3'd7
  } phase_t;

  // ---------- Speculation ----------
  typedef enum logic [1:0] {
    SPEC_COMMIT   = 2'd0,  // High confidence: use prediction
    SPEC_SPECULATE = 2'd1, // Medium: start + verify
    SPEC_FALLBACK  = 2'd2  // Low: use safe default
  } spec_mode_t;

  // ---------- PE operating mode ----------
  typedef enum logic [1:0] {
    PE_MAC       = 2'd0,  // Multiply-accumulate
    PE_PASS      = 2'd1,  // Pass data through (systolic forwarding)
    PE_ACCUM     = 2'd2,  // Accumulate only (reduction)
    PE_IDLE      = 2'd3   // Power-gated / idle
  } pe_mode_t;

  // ---------- N:M patterns ----------
  // 2:4 has C(4,2)=6 possible nonzero patterns
  parameter int NM_PATTERNS = 6;

  // ---------- Buffer sizes ----------
  parameter int IBUF_DEPTH   = 256;  // Input buffer entries
  parameter int OBUF_DEPTH   = 256;  // Output buffer entries
  parameter int WBUF_DEPTH   = 512;  // Weight buffer entries

endpackage

`endif
