// =============================================================================
// versaaccel_top.sv — VersaAccel Top-Level Module
//
// 5-configuration reconfigurable accelerator:
//   DMR, D-MR, DM-R, D-M-R, D-M-R+NM
//
// Components:
//   - 8×8 PE array with configurable interconnect
//   - Benes distribution network (8-port, 5-stage)
//   - Configurable reduction network (3-stage adder tree)
//   - Config Prediction Engine (INT8 MLP)
//   - Phase FSM (workload phase detection)
//   - Speculation unit (commit/speculate/fallback)
//   - N:M routing ROM (precomputed 2:4 patterns)
//   - Input/Output buffers (SRAM-style)
// =============================================================================
module versaaccel_top
  import versaaccel_pkg::*;
(
  input  logic        clk,
  input  logic        rst_n,

  // External interface
  input  logic        start,           // Begin computation
  input  logic [1:0]  model_type,      // 0=LLM, 1=GNN, 2=CNN
  input  logic [15:0] batch_size,      // M dimension
  input  logic [1:0]  op_type,         // 0=MM, 1=MV, 2=SpMM, 3=SpMSpM

  // Data interface (simplified AXI-style)
  input  logic                    din_valid,
  input  logic signed [DATA_W-1:0] din_data,
  input  logic                    din_is_weight,
  output logic                    din_ready,

  output logic                    dout_valid,
  output logic signed [ACC_W-1:0] dout_data,
  input  logic                    dout_ready,

  // Config override (for testing)
  input  logic                    cfg_override_en,
  input  config_t                 cfg_override,

  // Status
  output logic                    busy,
  output config_t                 active_config,
  output phase_t                  current_phase,
  output logic                    flush_event
);

  // ==========================================================================
  // Internal signals
  // ==========================================================================

  // CPE outputs
  config_t     cpe_config;
  logic [7:0]  cpe_confidence;
  logic        cpe_valid;

  // Phase FSM outputs
  config_t     phase_config_bias;

  // Speculation outputs
  config_t     spec_config;
  spec_mode_t  spec_mode;
  logic        spec_flush;
  logic        spec_valid;
  logic        switch_overlap;

  // Selected config (after override mux)
  config_t     selected_config;

  // Distribution network
  logic [BENES_CTRL_W-1:0] dist_ctrl;
  logic signed [DATA_W-1:0] dist_in [NR];
  logic signed [DATA_W-1:0] dist_out [NR];

  // N:M ROM
  logic [2:0]               nm_pattern_idx;
  logic [BENES_CTRL_W-1:0]  nm_benes_ctrl;

  // PE array
  logic signed [DATA_W-1:0] pe_a_in [NR];
  logic signed [DATA_W-1:0] pe_b_in [NC];
  logic signed [ACC_W-1:0]  pe_out [NR][NC];

  // Reduction network
  logic [1:0]               reduce_mode;
  logic signed [ACC_W-1:0]  reduce_in [NR];
  logic signed [ACC_W-1:0]  reduce_out [NR];

  // Control FSM
  typedef enum logic [2:0] {
    ST_IDLE,
    ST_LOAD_WEIGHTS,
    ST_COMPUTE,
    ST_DRAIN,
    ST_RECONFIG
  } ctrl_state_t;

  ctrl_state_t ctrl_state, ctrl_next;
  logic        layer_start;
  logic        acc_clear, acc_en;

  // Input/weight buffers (register-based for synthesis)
  logic signed [DATA_W-1:0] ibuf [IBUF_DEPTH];
  logic signed [DATA_W-1:0] wbuf [WBUF_DEPTH];
  logic [$clog2(IBUF_DEPTH)-1:0] ibuf_wr_ptr, ibuf_rd_ptr;
  logic [$clog2(WBUF_DEPTH)-1:0] wbuf_wr_ptr, wbuf_rd_ptr;

  // Output buffer
  logic signed [ACC_W-1:0] obuf [OBUF_DEPTH];
  logic [$clog2(OBUF_DEPTH)-1:0] obuf_wr_ptr, obuf_rd_ptr;

  // Computation counters
  logic [15:0] compute_cnt;
  logic [15:0] compute_total;

  // ==========================================================================
  // Config selection
  // ==========================================================================
  assign selected_config = cfg_override_en ? cfg_override : spec_config;
  assign active_config   = selected_config;

  // Distribution network control based on config
  always_comb begin
    dist_ctrl = '0;  // Default: straight-through
    case (selected_config)
      CFG_DMR:      dist_ctrl = '0;            // Broadcast mode
      CFG_D_MR:     dist_ctrl = '0;            // Controlled by scheduler
      CFG_DM_R:     dist_ctrl = '0;            // Fused distribute
      CFG_D_M_R:    dist_ctrl = '0;            // Programmed per-tile
      CFG_D_M_R_NM: dist_ctrl = nm_benes_ctrl; // From NM ROM
      default:      dist_ctrl = '0;
    endcase
  end

  // Reduction mode based on config
  always_comb begin
    case (selected_config)
      CFG_DMR:      reduce_mode = 2'd0; // Full reduction
      CFG_D_MR:     reduce_mode = 2'd1; // Partial
      CFG_DM_R:     reduce_mode = 2'd2; // Bypass
      CFG_D_M_R:    reduce_mode = 2'd2; // Bypass
      CFG_D_M_R_NM: reduce_mode = 2'd2; // Bypass
      default:      reduce_mode = 2'd2;
    endcase
  end

  // ==========================================================================
  // Module instantiations
  // ==========================================================================

  // --- Distribution Network ---
  distribution_network dist_net (
    .ctrl     (dist_ctrl),
    .data_in  (dist_in),
    .data_out (dist_out)
  );

  // --- N:M Routing ROM ---
  nm_routing_rom nm_rom (
    .pattern_idx (nm_pattern_idx),
    .benes_ctrl  (nm_benes_ctrl)
  );

  // --- PE Array ---
  pe_array pe_arr (
    .clk         (clk),
    .rst_n       (rst_n),
    .config_mode (selected_config),
    .acc_clear   (acc_clear),
    .acc_en      (acc_en),
    .a_in        (pe_a_in),
    .b_in        (pe_b_in),
    .pe_out      (pe_out)
  );

  // --- Reduction Network ---
  // Use column 0 outputs for reduction
  always_comb begin
    for (int i = 0; i < NR; i++)
      reduce_in[i] = pe_out[i][0];
  end

  reduction_network red_net (
    .mode     (reduce_mode),
    .data_in  (reduce_in),
    .data_out (reduce_out)
  );

  // --- Phase FSM ---
  phase_fsm phase (
    .clk           (clk),
    .rst_n         (rst_n),
    .batch_size    (batch_size),
    .op_type       (op_type),
    .layer_start   (layer_start),
    .model_type    (model_type),
    .current_phase (current_phase),
    .config_bias   (phase_config_bias)
  );

  // --- CPE (Config Prediction Engine) ---
  // Feature inputs (simplified: use runtime stats)
  logic signed [7:0] cpe_features [9];

  always_comb begin
    cpe_features[0] = 8'sd0;   // input_sparsity (from sampling)
    cpe_features[1] = 8'sd0;   // weight_sparsity
    cpe_features[2] = 8'sd0;   // is_conv
    cpe_features[3] = 8'sd0;   // log2(M)
    cpe_features[4] = 8'sd0;   // log2(K)
    cpe_features[5] = 8'sd0;   // log2(P)
    cpe_features[6] = 8'sd0;   // layer_position
    cpe_features[7] = 8'sd0;   // attention_sparsity
    cpe_features[8] = 8'sd0;   // nm_compatibility
  end

  config_prediction_engine cpe (
    .clk              (clk),
    .rst_n            (rst_n),
    .valid_in         (layer_start),
    .features         (cpe_features),
    .predicted_config (cpe_config),
    .confidence       (cpe_confidence),
    .valid_out        (cpe_valid)
  );

  // --- Speculation Unit ---
  speculation_unit spec (
    .clk                (clk),
    .rst_n              (rst_n),
    .predicted_config   (cpe_config),
    .cpe_confidence     (cpe_confidence),
    .prediction_valid   (cpe_valid),
    .phase_default      (phase_config_bias),
    .sample_done        (1'b0),  // Connected to data sampling in real design
    .actual_sparsity    (8'd0),
    .predicted_sparsity (8'd0),
    .thresh_high        (8'd230),
    .thresh_med         (8'd179),
    .sparsity_tol       (8'd26),
    .active_config      (spec_config),
    .spec_mode          (spec_mode),
    .flush              (spec_flush),
    .config_valid       (spec_valid),
    .switch_overlap     (switch_overlap)
  );

  assign flush_event = spec_flush;

  // ==========================================================================
  // Control FSM
  // ==========================================================================
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      ctrl_state  <= ST_IDLE;
      ibuf_wr_ptr <= '0;
      ibuf_rd_ptr <= '0;
      wbuf_wr_ptr <= '0;
      wbuf_rd_ptr <= '0;
      obuf_wr_ptr <= '0;
      obuf_rd_ptr <= '0;
      compute_cnt <= '0;
      busy        <= 1'b0;
      layer_start <= 1'b0;
    end else begin
      layer_start <= 1'b0;
      case (ctrl_state)
        ST_IDLE: begin
          busy <= 1'b0;
          if (start) begin
            ctrl_state  <= ST_LOAD_WEIGHTS;
            busy        <= 1'b1;
            layer_start <= 1'b1;
            ibuf_wr_ptr <= '0;
            wbuf_wr_ptr <= '0;
          end
        end

        ST_LOAD_WEIGHTS: begin
          if (din_valid && din_is_weight) begin
            wbuf[wbuf_wr_ptr] <= din_data;
            wbuf_wr_ptr <= wbuf_wr_ptr + 1'b1;
          end else if (din_valid && !din_is_weight) begin
            ibuf[ibuf_wr_ptr] <= din_data;
            ibuf_wr_ptr <= ibuf_wr_ptr + 1'b1;
            ctrl_state <= ST_COMPUTE;
            compute_cnt <= '0;
          end
        end

        ST_COMPUTE: begin
          if (spec_flush) begin
            ctrl_state <= ST_RECONFIG;
          end else begin
            compute_cnt <= compute_cnt + 1'b1;
            if (compute_cnt >= compute_total - 1) begin
              ctrl_state <= ST_DRAIN;
            end
          end
        end

        ST_DRAIN: begin
          if (dout_ready) begin
            obuf_rd_ptr <= obuf_rd_ptr + 1'b1;
            if (obuf_rd_ptr >= obuf_wr_ptr - 1)
              ctrl_state <= ST_IDLE;
          end
        end

        ST_RECONFIG: begin
          // 1-cycle reconfiguration penalty
          ctrl_state <= ST_COMPUTE;
          compute_cnt <= '0;
        end
      endcase
    end
  end

  // Data routing to PEs
  always_comb begin
    for (int i = 0; i < NR; i++) begin
      pe_a_in[i] = dist_out[i];
      dist_in[i] = (ibuf_rd_ptr + i < IBUF_DEPTH) ? ibuf[ibuf_rd_ptr + i] : '0;
    end
    for (int j = 0; j < NC; j++) begin
      pe_b_in[j] = (wbuf_rd_ptr + j < WBUF_DEPTH) ? wbuf[wbuf_rd_ptr + j] : '0;
    end
  end

  assign acc_clear = (compute_cnt == 0);
  assign acc_en    = (ctrl_state == ST_COMPUTE);
  assign din_ready = (ctrl_state == ST_LOAD_WEIGHTS);
  assign dout_valid = (ctrl_state == ST_DRAIN);
  assign dout_data  = reduce_out[obuf_rd_ptr[$clog2(NR)-1:0]];
  assign compute_total = batch_size;
  assign nm_pattern_idx = 3'd0;  // Default pattern, set by controller in real design

endmodule
