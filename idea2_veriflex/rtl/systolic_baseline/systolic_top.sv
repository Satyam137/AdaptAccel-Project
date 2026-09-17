// =============================================================================
// systolic_top.sv — Systolic Array Baseline (8×8 weight-stationary)
//
// Standard weight-stationary systolic array for dense matrix multiplication.
// No reconfiguration, no sparse support — serves as area/power baseline.
//
// Data flow:
//   - Weights pre-loaded into PEs (stationary)
//   - Activations flow left-to-right with skew
//   - Partial sums accumulate within each PE
//   - Results drained column by column
// =============================================================================
module systolic_top
  import versaaccel_pkg::*;
#(
  parameter int ROWS = NR,
  parameter int COLS = NC
)(
  input  logic        clk,
  input  logic        rst_n,

  // Control
  input  logic        start,
  input  logic        load_weights,     // Weight loading phase
  input  logic [15:0] compute_steps,    // Number of MAC cycles

  // Data input
  input  logic                    din_valid,
  input  logic signed [DATA_W-1:0] din_data,
  input  logic                    din_is_weight,
  output logic                    din_ready,

  // Data output
  output logic                    dout_valid,
  output logic signed [ACC_W-1:0] dout_data,
  input  logic                    dout_ready,

  // Status
  output logic                    busy
);

  // ==========================================================================
  // Internal PE array
  // ==========================================================================
  logic signed [DATA_W-1:0] weight_reg [ROWS][COLS];  // Stationary weights
  logic signed [DATA_W-1:0] a_in_skewed [ROWS];       // Skewed activation inputs
  logic signed [ACC_W-1:0]  accum [ROWS][COLS];        // Accumulators

  // Inter-PE forwarding wires
  logic signed [DATA_W-1:0] a_fwd [ROWS][COLS+1];  // Activation flows right

  // Control
  typedef enum logic [2:0] {
    S_IDLE,
    S_LOAD_W,
    S_COMPUTE,
    S_DRAIN
  } state_t;

  state_t state, state_next;
  logic [15:0] step_cnt;
  logic [$clog2(ROWS)-1:0] load_row;
  logic [$clog2(COLS)-1:0] load_col;
  logic [$clog2(ROWS)-1:0] drain_row;
  logic [$clog2(COLS)-1:0] drain_col;

  // Activation input buffer with skew
  logic signed [DATA_W-1:0] act_buf [ROWS];
  logic [$clog2(ROWS)-1:0] act_wr_idx;

  // ==========================================================================
  // Systolic PE Array (weight-stationary)
  // ==========================================================================
  genvar r, c;
  generate
    for (r = 0; r < ROWS; r++) begin : gen_row
      // Input activation enters from left
      assign a_fwd[r][0] = a_in_skewed[r];

      for (c = 0; c < COLS; c++) begin : gen_col
        // Each PE: multiply input activation × stored weight, accumulate
        always_ff @(posedge clk or negedge rst_n) begin
          if (!rst_n) begin
            accum[r][c] <= '0;
          end else if (state == S_COMPUTE) begin
            if (step_cnt == 0)
              accum[r][c] <= a_fwd[r][c] * weight_reg[r][c];
            else
              accum[r][c] <= accum[r][c] + a_fwd[r][c] * weight_reg[r][c];
          end
        end

        // Forward activation to next PE (1-cycle latency)
        always_ff @(posedge clk or negedge rst_n) begin
          if (!rst_n)
            a_fwd[r][c+1] <= '0;
          else
            a_fwd[r][c+1] <= a_fwd[r][c];
        end
      end
    end
  endgenerate

  // ==========================================================================
  // Control FSM
  // ==========================================================================
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state     <= S_IDLE;
      step_cnt  <= '0;
      load_row  <= '0;
      load_col  <= '0;
      drain_row <= '0;
      drain_col <= '0;
      busy      <= 1'b0;
      act_wr_idx <= '0;
      for (int i = 0; i < ROWS; i++)
        for (int j = 0; j < COLS; j++)
          weight_reg[i][j] <= '0;
      for (int i = 0; i < ROWS; i++)
        act_buf[i] <= '0;
    end else begin
      case (state)
        S_IDLE: begin
          busy <= 1'b0;
          if (start && load_weights) begin
            state    <= S_LOAD_W;
            busy     <= 1'b1;
            load_row <= '0;
            load_col <= '0;
          end else if (start) begin
            state    <= S_COMPUTE;
            busy     <= 1'b1;
            step_cnt <= '0;
          end
        end

        S_LOAD_W: begin
          if (din_valid && din_is_weight) begin
            weight_reg[load_row][load_col] <= din_data;
            if (load_col == COLS[$clog2(COLS)-1:0] - 1) begin
              load_col <= '0;
              if (load_row == ROWS[$clog2(ROWS)-1:0] - 1) begin
                state <= S_IDLE;  // Weights loaded, ready for compute
              end else begin
                load_row <= load_row + 1'b1;
              end
            end else begin
              load_col <= load_col + 1'b1;
            end
          end
        end

        S_COMPUTE: begin
          // Feed activations
          if (din_valid && !din_is_weight) begin
            act_buf[act_wr_idx] <= din_data;
            act_wr_idx <= act_wr_idx + 1'b1;
          end
          step_cnt <= step_cnt + 1'b1;
          if (step_cnt >= compute_steps - 1) begin
            state     <= S_DRAIN;
            drain_row <= '0;
            drain_col <= '0;
          end
        end

        S_DRAIN: begin
          if (dout_ready) begin
            if (drain_col == COLS[$clog2(COLS)-1:0] - 1) begin
              drain_col <= '0;
              if (drain_row == ROWS[$clog2(ROWS)-1:0] - 1)
                state <= S_IDLE;
              else
                drain_row <= drain_row + 1'b1;
            end else begin
              drain_col <= drain_col + 1'b1;
            end
          end
        end
      endcase
    end
  end

  // Skewed activation input (systolic timing)
  always_comb begin
    for (int i = 0; i < ROWS; i++)
      a_in_skewed[i] = act_buf[i]; // Simplified: actual skew uses delay chain
  end

  // Outputs
  assign din_ready  = (state == S_LOAD_W) || (state == S_COMPUTE);
  assign dout_valid = (state == S_DRAIN);
  assign dout_data  = accum[drain_row][drain_col];

endmodule
