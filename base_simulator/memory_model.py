"""
memory_model.py — Models on-chip buffers and off-chip memory bandwidth.

System parameters from VersaAccel paper Section V-A:
  - Input buffer:  256 KB, 8 banks
  - Output buffer:  64 KB, 8 banks  
  - Off-chip: 8 HBM channels × 64-bit = 128 GB/s
  - Operating frequency: 1 GHz
  - Precision: FP32 (4 bytes per element)
"""


class MemoryModel:
    """Models buffer capacity and bandwidth constraints."""

    def __init__(
        self,
        input_buf_kb=256,
        output_buf_kb=64,
        num_banks=8,
        hbm_bw_gbps=128,
        freq_ghz=1.0,
        precision_bytes=4,
    ):
        self.input_buf_bytes = input_buf_kb * 1024
        self.output_buf_bytes = output_buf_kb * 1024
        self.num_banks = num_banks
        self.hbm_bw_gbps = hbm_bw_gbps
        self.freq_ghz = freq_ghz
        self.precision_bytes = precision_bytes

        # Derived: elements that fit in each buffer
        self.input_buf_elements = self.input_buf_bytes // precision_bytes
        self.output_buf_elements = self.output_buf_bytes // precision_bytes

        # Derived: bandwidth in elements per cycle
        # BW (GB/s) / precision (bytes) / freq (GHz) = elements/cycle
        self.bw_elements_per_cycle = (
            hbm_bw_gbps * 1e9 / precision_bytes / (freq_ghz * 1e9)
        )

    def memory_access_cycles(self, data_elements):
        """
        Costmem: cycles to transfer 'data_elements' from off-chip memory.
        Costmem = Data(S,C) / BW   (Paper Eq. in Section IV-B-2)
        """
        if self.bw_elements_per_cycle <= 0:
            return float('inf')
        return int(data_elements / self.bw_elements_per_cycle) + 1

    def fits_in_input_buffer(self, elements):
        """Check if data fits in input buffer."""
        return elements <= self.input_buf_elements

    def fits_in_output_buffer(self, elements):
        """Check if result fits in output buffer."""
        return elements <= self.output_buf_elements

    def num_tiles_for_buffer(self, total_elements):
        """How many tiles needed if total data exceeds input buffer."""
        if total_elements <= self.input_buf_elements:
            return 1
        return -(-total_elements // self.input_buf_elements)  # ceil division

    def __repr__(self):
        return (
            f"MemoryModel(input={self.input_buf_bytes//1024}KB, "
            f"output={self.output_buf_bytes//1024}KB, "
            f"BW={self.bw_elements_per_cycle:.1f} elem/cyc)"
        )
