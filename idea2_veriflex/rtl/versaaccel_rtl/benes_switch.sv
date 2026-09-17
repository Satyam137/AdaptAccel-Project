// =============================================================================
// benes_switch.sv — 2×2 crossbar switch (building block of Benes network)
//
// ctrl=0: straight  (out0=in0, out1=in1)
// ctrl=1: cross     (out0=in1, out1=in0)
// =============================================================================
module benes_switch #(
  parameter int WIDTH = 16
)(
  input  logic               ctrl,
  input  logic [WIDTH-1:0]   in0,
  input  logic [WIDTH-1:0]   in1,
  output logic [WIDTH-1:0]   out0,
  output logic [WIDTH-1:0]   out1
);

  assign out0 = ctrl ? in1 : in0;
  assign out1 = ctrl ? in0 : in1;

endmodule
