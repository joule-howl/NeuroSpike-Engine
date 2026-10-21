"""
Neuromorphic AI Engine
Module: Leaky Integrate-and-Fire (LIF) Neuron Simulation

Implements the LIF neuron model using the membrane differential equation:
    tau * dV/dt = -(V - V_rest) + R * I(t)

After a spike, the membrane potential resets and the neuron enters an
absolute refractory period before it can fire again.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent

# ---------------------------------------------------------------------------
# Biophysical parameters
# ---------------------------------------------------------------------------
TAU    = 20.0e-3   # Membrane time constant (s)
V_REST = -70.0     # Resting potential (mV)
V_TH   = -55.0     # Spike threshold (mV)
V_RESET= -75.0     # Reset potential (mV)
R_MEM  = 10.0      # Membrane resistance (MΩ)
T_REF  = 2.0e-3    # Absolute refractory period (s)

# Simulation parameters
DT = 0.1e-3        # Time step (s)
T  = 500e-3        # Total simulation time (s)
t  = np.arange(0, T, DT)
N  = len(t)


def simulate_lif(I_input,
                 tau=TAU, V_rest=V_REST, V_th=V_TH,
                 V_reset=V_RESET, R=R_MEM, t_ref=T_REF, dt=DT):
    """
    Simulate a single LIF neuron using the explicit Euler method.

    Parameters
    ----------
    I_input : ndarray, shape (N,)
        Input current over time (nA).

    Returns
    -------
    V : ndarray, shape (N,)
        Membrane potential over time (mV).
    spikes : list of float
        Spike times (s).
    """
    V = np.full(N, V_rest)
    spikes = []
    ref_count = 0

    for i in range(1, N):
        if ref_count > 0:
            # Neuron is in refractory period — hold at reset potential
            V[i] = V_reset
            ref_count -= 1
        else:
            dV   = (dt / tau) * (-(V[i-1] - V_rest) + R * I_input[i-1])
            V[i] = V[i-1] + dV
            if V[i] >= V_th:
                V[i] = 40.0          # Action potential peak (display only)
                spikes.append(t[i])
                ref_count = int(t_ref / dt)

    return V, spikes


if __name__ == '__main__':
    # ---------------------------------------------------------------------------
    # 1. Membrane potential at different input current levels
    # ---------------------------------------------------------------------------
    current_levels = [1.5, 2.0, 2.5, 3.0]
    palette = ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2']

    fig, axes = plt.subplots(len(current_levels), 1, figsize=(12, 10), sharex=True)
    fig.suptitle('LIF Membrane Potential at Different Input Current Levels',
                 fontsize=13, fontweight='bold')

    spike_counts = []
    for ax, I_val, col in zip(axes, current_levels, palette):
        V, spikes = simulate_lif(np.full(N, I_val))
        spike_counts.append(len(spikes))
        ax.plot(t * 1e3, V, color=col, linewidth=0.9, label=f'I = {I_val} nA')
        ax.axhline(V_TH,   color='red',  linestyle='--', linewidth=0.7, alpha=0.6, label='V_th')
        ax.axhline(V_REST, color='gray', linestyle=':',  linewidth=0.7, alpha=0.6, label='V_rest')
        for sp in spikes:
            ax.axvline(sp * 1e3, color=col, alpha=0.25, linewidth=0.5)
        ax.set_ylabel('V (mV)', fontsize=9)
        ax.legend(loc='upper right', fontsize=8)
        ax.set_ylim(-85, 55)
        ax.set_facecolor('#f9f9f9')

    axes[-1].set_xlabel('Time (ms)', fontsize=10)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'lif_voltage_traces.png', dpi=150, bbox_inches='tight')
    plt.close()

    # ---------------------------------------------------------------------------
    # 2. F-I curve: firing rate vs. stimulus intensity
    # ---------------------------------------------------------------------------
    I_range      = np.linspace(0.5, 6.0, 120)
    firing_rates = [len(simulate_lif(np.full(N, I_val))[1]) / T for I_val in I_range]

    threshold_idx = next((i for i, r in enumerate(firing_rates) if r > 0), 0)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(I_range, firing_rates, color='#e15759', linewidth=2)
    ax.fill_between(I_range, firing_rates, alpha=0.15, color='#e15759')
    ax.axvline(I_range[threshold_idx], color='navy', linestyle='--',
               linewidth=1.2, label=f'I_th ≈ {I_range[threshold_idx]:.2f} nA')
    ax.set_xlabel('Input Current I (nA)', fontsize=11)
    ax.set_ylabel('Firing Rate (Hz)', fontsize=11)
    ax.set_title('F-I Curve', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_facecolor('#fafafa')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fi_curve.png', dpi=150, bbox_inches='tight')
    plt.close()

    # ---------------------------------------------------------------------------
    # 3. Response to time-varying input (sinusoidal + random pulses)
    # ---------------------------------------------------------------------------
    np.random.seed(42)
    I_sin     = 2.0 + 1.5 * np.sin(2 * np.pi * 8 * t)
    I_pulses  = np.zeros(N)
    pulse_idx = np.random.choice(N, size=15, replace=False)
    I_pulses[pulse_idx] = np.random.uniform(2.0, 4.0, 15)
    I_varied  = I_sin + I_pulses * 0.5

    V_varied, spikes_varied = simulate_lif(I_varied)

    fig = plt.figure(figsize=(12, 8))
    gs  = gridspec.GridSpec(3, 1, height_ratios=[1, 2, 0.5], hspace=0.05)

    ax_I = fig.add_subplot(gs[0])
    ax_V = fig.add_subplot(gs[1], sharex=ax_I)
    ax_S = fig.add_subplot(gs[2], sharex=ax_I)

    ax_I.plot(t * 1e3, I_varied, color='#4e79a7', linewidth=0.8)
    ax_I.set_ylabel('I (nA)', fontsize=9)
    ax_I.set_title('LIF Neuron Response to Time-Varying Input Current', fontsize=11, fontweight='bold')
    ax_I.set_facecolor('#f9f9f9')

    ax_V.plot(t * 1e3, V_varied, color='#333333', linewidth=0.8)
    ax_V.axhline(V_TH,   color='red',  linestyle='--', linewidth=0.8, alpha=0.7, label='V_th')
    ax_V.axhline(V_REST, color='gray', linestyle=':',  linewidth=0.8, alpha=0.5, label='V_rest')
    ax_V.set_ylabel('V (mV)', fontsize=9)
    ax_V.legend(loc='upper right', fontsize=8)
    ax_V.set_ylim(-85, 55)
    ax_V.set_facecolor('#f9f9f9')

    if spikes_varied:
        ax_S.eventplot([sp * 1e3 for sp in spikes_varied],
                       lineoffsets=0.5, linelengths=0.8,
                       color='#e15759', linewidths=1.2)
    ax_S.set_yticks([])
    ax_S.set_xlabel('Time (ms)', fontsize=10)
    ax_S.set_ylabel('Spike', fontsize=8)
    ax_S.set_facecolor('#fff5f5')

    plt.setp(ax_I.get_xticklabels(), visible=False)
    plt.setp(ax_V.get_xticklabels(), visible=False)
    plt.savefig(OUTPUT_DIR / 'lif_varied_input.png', dpi=150, bbox_inches='tight')
    plt.close()

    # ---------------------------------------------------------------------------
    # Results
    # ---------------------------------------------------------------------------
    print("LIF simulation complete")
    print(f"{'I (nA)':<12} {'Spikes':<10} {'Rate (Hz)':<10}")
    print("-" * 32)
    for I_val, sc in zip(current_levels, spike_counts):
        print(f"{I_val:<12.1f} {sc:<10d} {sc/T:<10.1f}")
    print(f"\nTime-varying input: {len(spikes_varied)} spikes | {len(spikes_varied)/T:.1f} Hz")
    print("Plots saved → LIF/")

