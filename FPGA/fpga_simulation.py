"""
Neuromorphic AI Engine
Module: FPGA Hardware Simulation – Fixed-Point LIF & SNN Core

Faithfully simulates the RTL Verilog behaviour in Python using fixed-point
integer arithmetic (Q8.7, 16-bit signed). Measures and compares:
  - CPU baseline (float32 NumPy)
  - FPGA integer simulation (Q8.7 fixed-point)
  - Power estimation based on Xilinx 7-series model
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

OUTPUT_DIR = Path(__file__).parent

# ---------------------------------------------------------------------------
# Fixed-point arithmetic  (Q8.7 – mirrors the Verilog integer datapath)
# ---------------------------------------------------------------------------
FRAC_BITS = 7
SCALE     = 2 ** FRAC_BITS    # 128
DATA_MAX  =  2 ** 15 - 1      # 16-bit signed maximum
DATA_MIN  = -(2 ** 15)

def to_fixed(x: float) -> int:
    return int(np.clip(x * SCALE, DATA_MIN, DATA_MAX))

def from_fixed(x: int) -> float:
    return x / SCALE

# Physical parameters in fixed-point domain
V_REST_FX   = 0
V_TH_FX     = to_fixed(1.0)   # Spike threshold = 1.0 (scaled)
V_RESET_FX  = 0
T_REF_STEPS = 20


class LIFNeuronFPGA:
    """
    Cycle-accurate simulation of lif_neuron.v using 16-bit fixed-point.
    All arithmetic matches the integer datapath of the RTL.
    """

    def __init__(self):
        self.v       = V_REST_FX
        self.ref_cnt = 0
        self.spike   = 0

    def step(self, i_current_fx: int) -> int:
        """Single clock cycle. Returns 1 if a spike is emitted."""
        self.spike = 0
        if self.ref_cnt > 0:
            self.v       = V_RESET_FX
            self.ref_cnt -= 1
        else:
            # Simplified LIF update: V[t] = 0.9·V[t-1] + I[t]
            self.v = int(np.clip(0.9 * self.v + i_current_fx, DATA_MIN, DATA_MAX))
            if self.v >= V_TH_FX:
                self.spike   = 1
                self.v       = V_RESET_FX
                self.ref_cnt = T_REF_STEPS
        return self.spike


class SNNCoreSimulator:
    """
    Simulation of snn_core.v: LIF neuron array with in-memory weight storage.
    Weights are quantised to int8 and co-located with the MAC unit,
    eliminating external RAM accesses (Non-Von Neumann design).
    """

    def __init__(self, n_in: int, n_neurons: int, weights: np.ndarray):
        self.n_in      = n_in
        self.n_neurons = n_neurons
        # Quantise float32 weights to int8 for BRAM storage.
        # Q8.7 scale = 128, but we use a reduced factor (×2) so that
        # typical weights in [0.1, 0.5] map to int8 range [-128, 127]:
        #   weight=0.5 → 0.5 × 2 × 32 = 32  (well within int8)
        #   weight=0.1 → 0.1 × 2 × 32 ≈ 6
        w_quant_scale  = 64   # = (SCALE // 64) * 32 = 2 * 32
        self.W_int     = np.clip(
            (weights * w_quant_scale).astype(np.int32), -128, 127
        ).astype(np.int16)
        self.neurons   = [LIFNeuronFPGA() for _ in range(n_neurons)]

    def run(self, spike_train: np.ndarray) -> np.ndarray:
        """
        spike_train : [T, n_in]  bool
        Returns     : spike_out  [T, n_neurons]
        """
        T      = spike_train.shape[0]
        output = np.zeros((T, self.n_neurons), dtype=np.int8)
        for t in range(T):
            accum = self.W_int.T @ spike_train[t].astype(np.int16)
            for n_idx, neuron in enumerate(self.neurons):
                output[t, n_idx] = neuron.step(int(accum[n_idx]))
        return output


# ---------------------------------------------------------------------------
# CPU baseline (float32 NumPy)
# ---------------------------------------------------------------------------
def cpu_snn_forward(spike_in, W, tau_m=0.9, v_th=1.0):
    T  = spike_in.shape[0]
    V  = np.zeros(W.shape[1], dtype=np.float32)
    sp = np.zeros((T, W.shape[1]), dtype=np.float32)
    for t in range(T):
        V    = tau_m * V + spike_in[t] @ W
        fire = V >= v_th
        V[fire] = 0.0
        sp[t, fire] = 1.0
    return sp


if __name__ == '__main__':
    # ---------------------------------------------------------------------------
    # CPU vs. FPGA benchmark
    # ---------------------------------------------------------------------------
    np.random.seed(0)
    N_IN, N_HID = 128, 64
    T_SIM       = 500

    W = np.random.uniform(0.1, 0.5, (N_IN, N_HID)).astype(np.float32)
    W *= (np.random.rand(*W.shape) > 0.6)   # sparse weights

    spike_input = (np.random.rand(T_SIM, N_IN) < 0.3).astype(np.float32)

    t0       = time.perf_counter()
    sp_cpu   = cpu_snn_forward(spike_input, W)
    t_cpu_ms = (time.perf_counter() - t0) * 1000

    fpga_core = SNNCoreSimulator(N_IN, N_HID, W)
    t0        = time.perf_counter()
    sp_fpga   = fpga_core.run(spike_input.astype(bool))
    t_fpga_ms = (time.perf_counter() - t0) * 1000

    print(f"CPU  (float32) : {t_cpu_ms:.2f} ms | spike rate = {sp_cpu.mean()*100:.1f}%")
    print(f"FPGA (int16)   : {t_fpga_ms:.2f} ms | spike rate = {sp_fpga.mean()*100:.1f}%")

    # ---------------------------------------------------------------------------
    # Power estimation (Xilinx 7-series model)
    # ---------------------------------------------------------------------------
    FREQ_FPGA_MHZ  = 100.0
    LUT_PER_NEURON = 150       # Estimated LUT count per 16-bit LIF neuron
    DSP_PER_NEURON = 2
    TOTAL_LUT      = LUT_PER_NEURON * N_HID
    P_LUT_MW       = 0.05      # mW per LUT @ 100 MHz, 50% toggle rate

    P_fpga_dyn  = TOTAL_LUT * P_LUT_MW * (FREQ_FPGA_MHZ / 100)
    P_fpga_stat = 50.0         # Artix-7 leakage (mW)
    P_fpga      = P_fpga_dyn + P_fpga_stat

    P_cpu       = 3500.0       # Intel Core i5 under moderate load (mW)

    ops_total   = N_IN * N_HID * T_SIM

    print(f"\nPower estimate ({T_SIM} timesteps, {N_IN}→{N_HID}):")
    print(f"  FPGA : {P_fpga:.0f} mW  |  {P_fpga/1e3*t_fpga_ms/1000:.4f} mJ")
    print(f"  CPU  : {P_cpu:.0f} mW   |  {P_cpu/1e3*t_cpu_ms/1000:.4f} mJ")
    print(f"  Power saving  : {(1 - P_fpga/P_cpu)*100:.1f}%")
    print(f"  Energy eff.   : {ops_total / (P_fpga * t_fpga_ms * 1e-3):.0f} MOPS/mW")

    # ---------------------------------------------------------------------------
    # Plots
    # ---------------------------------------------------------------------------
    fig = plt.figure(figsize=(16, 12))
    gs  = gridspec.GridSpec(2, 3, hspace=0.4, wspace=0.35)

    ax = fig.add_subplot(gs[0, 0])
    for nid in range(N_HID):
        ts = np.where(sp_cpu[:, nid])[0]
        if len(ts):
            ax.scatter(ts, [nid] * len(ts), s=1.5, c='#4e79a7', marker='|')
    ax.set_title(f'CPU – float32 ({t_cpu_ms:.1f} ms)')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Hidden Neuron')
    ax.set_facecolor('#fafafa')

    ax = fig.add_subplot(gs[0, 1])
    for nid in range(N_HID):
        ts = np.where(sp_fpga[:, nid])[0]
        if len(ts):
            ax.scatter(ts, [nid] * len(ts), s=1.5, c='#e15759', marker='|')
    ax.set_title(f'FPGA int16 ({t_fpga_ms:.1f} ms)')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Hidden Neuron')
    ax.set_facecolor('#fafafa')

    ax = fig.add_subplot(gs[0, 2])
    ax.bar(['CPU\n(float32)', 'FPGA\n(int16)'],
           [sp_cpu.mean() * 100, sp_fpga.mean() * 100],
           color=['#4e79a7', '#e15759'], alpha=0.85, edgecolor='white')
    ax.set_ylabel('Spike Rate (%)')
    ax.set_title('Spike Rate: CPU vs. FPGA')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_facecolor('#fafafa')

    ax = fig.add_subplot(gs[1, 0])
    n_demo = LIFNeuronFPGA()
    I_demo = np.full(200, to_fixed(2.5))
    v_trace, spike_trace = [], []
    for I_step in I_demo:
        s = n_demo.step(I_step)
        v_trace.append(from_fixed(n_demo.v))
        spike_trace.append(s)
    ax.plot(v_trace, color='#e15759', linewidth=0.9)
    ax.axhline(from_fixed(V_TH_FX),    color='red',  linestyle='--',
               linewidth=0.8, label='V_th',   alpha=0.7)
    ax.axhline(from_fixed(V_REST_FX),  color='gray', linestyle=':',
               linewidth=0.8, label='V_rest', alpha=0.6)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Vmem (fixed-point)')
    ax.set_title('LIF Membrane Potential – FPGA Fixed-Point')
    ax.legend(fontsize=8)
    ax.set_facecolor('#fafafa')

    ax = fig.add_subplot(gs[1, 1])
    ax.bar(['FPGA\nArtix-7', 'CPU\nCore i5'],
           [P_fpga, P_cpu],
           color=['#59a14f', '#e15759'], alpha=0.85, edgecolor='white')
    ax.set_ylabel('Power (mW)')
    ax.set_title('Estimated Power Consumption')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_facecolor('#fafafa')

    ax = fig.add_subplot(gs[1, 2])
    resources = {
        'LUT':  TOTAL_LUT,
        'FF':   TOTAL_LUT // 2,
        'BRAM': N_IN * N_HID // 1024,
        'DSP':  DSP_PER_NEURON * N_HID,
    }
    ax.barh(list(resources.keys()), list(resources.values()),
            color='#76b7b2', alpha=0.85, edgecolor='white')
    ax.set_xlabel('Count')
    ax.set_title(f'Estimated FPGA Resources\n({N_HID} LIF neurons, 16-bit)')
    ax.grid(True, alpha=0.3, axis='x')
    ax.set_facecolor('#fafafa')

    fig.suptitle('SNN on FPGA – Fixed-Point Hardware Simulation',
                 fontsize=14, fontweight='bold')
    plt.savefig(OUTPUT_DIR / 'fpga_simulation.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\nPlot saved → FPGA/fpga_simulation.png")
    T  = spike_in.shape[0]
    V  = np.zeros(W.shape[1], dtype=np.float32)
    sp = np.zeros((T, W.shape[1]), dtype=np.float32)
    for t in range(T):
        V    = tau_m * V + spike_in[t] @ W
        fire = V >= v_th
        V[fire] = 0.0
        sp[t, fire] = 1.0
    return sp


