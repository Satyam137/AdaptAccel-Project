// =============================================================================
// speculation_unit.sv — Speculative execution controller
//
// Implements three-mode speculation based on CPE confidence:
//   COMMIT:    confidence >= THRESH_HIGH  → use prediction, overlap switch
//   SPECULATE: confidence >= THRESH_MED   → use prediction, sample to verify
//   FALLBACK:  confidence <  THRESH_MED   → use phase FSM safe default
//
// On misprediction during SPECULATE: assert flush, trigger reconfiguration.
// =============================================================================
module speculation_unit
  import versaaccel_pkg::*;
(
  input  logic        clk,
  input  logic        rst_n,

  // From CPE
  input  config_t     predicted_config,
  input  logic [7:0]  cpe_confidence,
  input  logic        prediction_valid,

  // From phase FSM
  input  config_t     phase_default,

  // Runtime verification (from data sampling unit)
  input  logic        sample_done,          // Sampling complete
  input  logic [7:0]  actual_sparsity,      // Measured from data sample
  input  logic [7:0]  predicted_sparsity,   // From CPE

  // Configuration
  input  logic [7:0]  thresh_high,   // Default: 230 (~90%)
  input  logic [7:0]  thresh_med,    // Default: 179 (~70%)
  input  logic [7:0]  sparsity_tol,  // Tolerance for sparsity mismatch (default: 26 ~10%)

  // Outputs
  output config_t     active_config,  // Config to use right now
  output spec_mode_t  spec_mode,      // Current mode
  output logic        flush,          // Flush pipeline (misprediction)
  output logic        config_valid,   // Active config is valid
  output logic        switch_overlap  // OK to overlap switch with compute
);

  // ---------- State ----------
  typedef enum logic [1:0] {
    S_IDLE,
    S_COMMITTED,
    S_SPECULATING,
    S_FLUSHING
  } state_t;

  state_t state, state_next;
  config_t active_reg;
  spec_mode_t mode_reg;

  // Sparsity mismatch detection
  logic sparsity_match;
  logic [7:0] sp_diff;

  assign sp_diff = (actual_sparsity > predicted_sparsity) ?
                   (actual_sparsity - predicted_sparsity) :
                   (predicted_sparsity - actual_sparsity);
  assign sparsity_match = (sp_diff <= sparsity_tol);

  // ---------- FSM ----------
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state      <= S_IDLE;
      active_reg <= CFG_D_M_R;
      mode_reg   <= SPEC_FALLBACK;
    end else begin
      state <= state_next;
      case (state_next)
        S_COMMITTED: begin
          active_reg <= predicted_config;
          mode_reg   <= SPEC_COMMIT;
        end
        S_SPECULATING: begin
          active_reg <= predicted_config;
          mode_reg   <= SPEC_SPECULATE;
        end
        S_FLUSHING: begin
          active_reg <= phase_default;  // Fall back to safe config
          mode_reg   <= SPEC_FALLBACK;
        end
        default: begin
          if (prediction_valid) begin
            if (cpe_confidence >= thresh_high) begin
              active_reg <= predicted_config;
              mode_reg   <= SPEC_COMMIT;
            end else if (cpe_confidence >= thresh_med) begin
              active_reg <= predicted_config;
              mode_reg   <= SPEC_SPECULATE;
            end else begin
              active_reg <= phase_default;
              mode_reg   <= SPEC_FALLBACK;
            end
          end
        end
      endcase
    end
  end

  always_comb begin
    state_next = state;
    case (state)
      S_IDLE: begin
        if (prediction_valid) begin
          if (cpe_confidence >= thresh_high)
            state_next = S_COMMITTED;
          else if (cpe_confidence >= thresh_med)
            state_next = S_SPECULATING;
          else
            state_next = S_IDLE;  // Fallback stays in idle
        end
      end

      S_COMMITTED: begin
        state_next = S_IDLE;  // Done after one cycle commitment
      end

      S_SPECULATING: begin
        if (sample_done) begin
          if (sparsity_match)
            state_next = S_IDLE;      // Correct speculation
          else
            state_next = S_FLUSHING;  // Misprediction
        end
      end

      S_FLUSHING: begin
        state_next = S_IDLE;  // Flush takes 1 cycle
      end
    endcase
  end

  // ---------- Outputs ----------
  assign active_config  = active_reg;
  assign spec_mode      = mode_reg;
  assign flush          = (state == S_FLUSHING);
  assign config_valid   = (state != S_FLUSHING);
  assign switch_overlap = (mode_reg == SPEC_COMMIT);  // Only COMMIT can overlap

endmodule
