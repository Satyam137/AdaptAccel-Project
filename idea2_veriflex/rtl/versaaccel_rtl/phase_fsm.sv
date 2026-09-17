// =============================================================================
// phase_fsm.sv — Workload phase detection FSM
//
// Detects execution phase from runtime signals and outputs config bias.
// Transitions triggered by batch dimension and operator type changes.
//
// States: PREFILL, DECODE, ATTENTION, AGGREGATE, TRANSFORM, CONV, FC, GENERIC
// =============================================================================
module phase_fsm
  import versaaccel_pkg::*;
(
  input  logic        clk,
  input  logic        rst_n,

  // Runtime signals from controller
  input  logic [15:0] batch_size,     // Current batch dimension (M)
  input  logic [1:0]  op_type,        // 0=MM, 1=MV, 2=SpMM, 3=SpMSpM
  input  logic        layer_start,    // Pulse: new layer begins
  input  logic [1:0]  model_type,     // 0=LLM, 1=GNN, 2=CNN, 3=auto

  // Outputs
  output phase_t      current_phase,
  output config_t     config_bias     // Recommended config for this phase
);

  phase_t phase_reg, phase_next;

  // Previous values for transition detection
  logic [15:0] prev_batch;
  logic [1:0]  prev_op;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      phase_reg  <= PHASE_GENERIC;
      prev_batch <= '0;
      prev_op    <= '0;
    end else if (layer_start) begin
      phase_reg  <= phase_next;
      prev_batch <= batch_size;
      prev_op    <= op_type;
    end
  end

  always_comb begin
    phase_next = phase_reg;

    case (model_type)
      2'd0: begin // LLM
        if (batch_size > 16'd1 && op_type == 2'd0)       // batch>1, MM
          phase_next = PHASE_PREFILL;
        else if (batch_size == 16'd1 && op_type == 2'd1)  // batch=1, MV
          phase_next = PHASE_DECODE;
        else if (op_type == 2'd2 || op_type == 2'd3)      // SpMM/SpMSpM
          phase_next = PHASE_ATTENTION;
        else
          phase_next = PHASE_GENERIC;
      end

      2'd1: begin // GNN
        if (op_type == 2'd2)       // SpMM → aggregate
          phase_next = PHASE_AGGREGATE;
        else if (op_type == 2'd0)  // MM → transform
          phase_next = PHASE_TRANSFORM;
        else
          phase_next = PHASE_GENERIC;
      end

      2'd2: begin // CNN
        if (batch_size > 16'd64)   // Large spatial → conv layers
          phase_next = PHASE_CONV;
        else                       // Small → FC layers
          phase_next = PHASE_FC;
      end

      default: phase_next = PHASE_GENERIC;
    endcase
  end

  // Config bias per phase
  always_comb begin
    case (phase_reg)
      PHASE_PREFILL:   config_bias = CFG_DMR;     // Dense, use systolic
      PHASE_DECODE:    config_bias = CFG_D_M_R;   // Sparse MV
      PHASE_ATTENTION: config_bias = CFG_D_M_R;   // Sparse attention
      PHASE_AGGREGATE: config_bias = CFG_D_M_R;   // Sparse aggregation
      PHASE_TRANSFORM: config_bias = CFG_DMR;     // Dense transform
      PHASE_CONV:      config_bias = CFG_DMR;     // Dense convolution
      PHASE_FC:        config_bias = CFG_D_MR;    // Moderately sparse FC
      default:         config_bias = CFG_D_M_R;   // Safe default
    endcase
  end

  assign current_phase = phase_reg;

endmodule
