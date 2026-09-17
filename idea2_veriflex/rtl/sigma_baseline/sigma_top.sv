// =============================================================================
// sigma_top.sv — SIGMA-style Flexible Sparse Accelerator Baseline
//
// SIGMA architecture: Benes distribution + flexible dot-product units +
// FAN (Forwarding Adder Network) reduction.
//
// Always operates in flexible mode (like VersaAccel's D-M-R).
// No reconfiguration — always pays full Benes routing overhead.
// Serves as "always-flexible" baseline vs VersaAccel's adaptive switching.
//
// Key difference from VersaAccel: no config switching, no CPE, no speculation.
// Same PE count (64) and data widths for fair PPA comparison.
// =============================================================================
module sigma_top
  import versaaccel_pkg::*;
#(
  parameter int N_UNITS = NR  // 8 dot-product units
)(
  input  logic        clk,
  input  logic        rst_n,

  // Control
  input  logic        start,
  input  logic [15:0] num_elements,   // Elements per dot product
  input  logic [BENES_CTRL_W-1:0] routing_ctrl, // Benes switch settings

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
  // Benes Distribution Network (always active, always routing)
  // ==========================================================================
  logic signed [DATA_W-1:0] dist_in [N_UNITS];
  logic signed [DATA_W-1:0] dist_out [N_UNITS];

  distribution_network benes_dist (
    .ctrl     (routing_ctrl),
    .data_in  (dist_in),
    .data_out (dist_out)
  );

  // ==========================================================================
  // Flexible Dot-Product Units (each has NC multiply-add units)
  // ==========================================================================
  logic signed [DATA_W-1:0] weight_store [N_UNITS][NC]; // Weight storage
  logic signed [ACC_W-1:0]  dp_accum [N_UNITS];          // Dot product accumulators

  // Input buffers
  logic signed [DATA_W-1:0] act_buf [N_UNITS];

  // ==========================================================================
  // FAN (Forwarding Adder Network) Reduction
  // ==========================================================================
  // 3-stage reduction tree (same structure as VersaAccel's reduction network)
  logic signed [ACC_W-1:0] fan_in [N_UNITS];
  logic signed [ACC_W-1:0] fan_out [N_UNITS];

  reduction_network fan_reduce (
    .mode     (2'd2),      // BYPASS: each dot product unit outputs independently
    .data_in  (fan_in),
    .data_out (fan_out)
  );

  always_comb begin
    for (int i = 0; i < N_UNITS; i++)
      fan_in[i] = dp_accum[i];
  end

  // ==========================================================================
  // Control FSM
  // ==========================================================================
  typedef enum logic [2:0] {
    S_IDLE,
    S_LOAD_W,
    S_COMPUTE,
    S_DRAIN
  } state_t;

  state_t state;
  logic [15:0] step_cnt;
  logic [$clog2(N_UNITS)-1:0] load_unit, load_elem;
  logic [$clog2(N_UNITS)-1:0] drain_idx;
  logic [$clog2(N_UNITS)-1:0] act_idx;

  // Dot-product computation
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      for (int i = 0; i < N_UNITS; i++)
        dp_accum[i] <= '0;
    end else if (state == S_COMPUTE) begin
      for (int i = 0; i < N_UNITS; i++) begin
        if (step_cnt == 0)
          dp_accum[i] <= dist_out[i] * weight_store[i][step_cnt[$clog2(NC)-1:0]];
        else
          dp_accum[i] <= dp_accum[i] + dist_out[i] * weight_store[i][step_cnt[$clog2(NC)-1:0]];
      end
    end
  end

  // Main FSM
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state     <= S_IDLE;
      step_cnt  <= '0;
      load_unit <= '0;
      load_elem <= '0;
      drain_idx <= '0;
      act_idx   <= '0;
      busy      <= 1'b0;
      for (int i = 0; i < N_UNITS; i++) begin
        act_buf[i] <= '0;
        for (int j = 0; j < NC; j++)
          weight_store[i][j] <= '0;
      end
    end else begin
      case (state)
        S_IDLE: begin
          busy <= 1'b0;
          if (start) begin
            state    <= S_LOAD_W;
            busy     <= 1'b1;
            load_unit <= '0;
            load_elem <= '0;
          end
        end

        S_LOAD_W: begin
          if (din_valid && din_is_weight) begin
            weight_store[load_unit][load_elem] <= din_data;
            if (load_elem == N_UNITS[$clog2(N_UNITS)-1:0] - 1) begin
              load_elem <= '0;
              if (load_unit == N_UNITS[$clog2(N_UNITS)-1:0] - 1) begin
                state <= S_COMPUTE;
                step_cnt <= '0;
              end else begin
                load_unit <= load_unit + 1'b1;
              end
            end else begin
              load_elem <= load_elem + 1'b1;
            end
          end
        end

        S_COMPUTE: begin
          if (din_valid && !din_is_weight) begin
            act_buf[act_idx] <= din_data;
            act_idx <= act_idx + 1'b1;
          end
          step_cnt <= step_cnt + 1'b1;
          if (step_cnt >= num_elements - 1) begin
            state <= S_DRAIN;
            drain_idx <= '0;
          end
        end

        S_DRAIN: begin
          if (dout_ready) begin
            if (drain_idx == N_UNITS[$clog2(N_UNITS)-1:0] - 1)
              state <= S_IDLE;
            else
              drain_idx <= drain_idx + 1'b1;
          end
        end
      endcase
    end
  end

  // Route activations through Benes network
  always_comb begin
    for (int i = 0; i < N_UNITS; i++)
      dist_in[i] = act_buf[i];
  end

  // Outputs
  assign din_ready  = (state == S_LOAD_W) || (state == S_COMPUTE);
  assign dout_valid = (state == S_DRAIN);
  assign dout_data  = fan_out[drain_idx];

endmodule
