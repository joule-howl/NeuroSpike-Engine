"""
Neuromorphic AI Engine
Module: Neuromorphic Process Runtime

A lightweight Process/Port/Runtime execution engine for multi-layer SNNs.
Designed around the same dataflow abstractions used in neuromorphic computing
frameworks: each computational unit (Process) communicates through typed
channels (Ports) and is driven by a central time-step scheduler (Runtime).

Components:
  Port          – Zero-copy unidirectional channel between Processes
  Process       – Self-contained computational unit with internal state
  LIFProcess    – Layer of Leaky Integrate-and-Fire neurons
  DenseProcess  – Fully-connected synaptic projection
  SpikeMonitor  – Records spike history for offline analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import time
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent


# ---------------------------------------------------------------------------
# Lava-style abstractions
# ---------------------------------------------------------------------------

class Port:
    """
    Unidirectional communication channel between Processes.
    Supports fan-out: one port can connect to multiple downstream ports.
    recv() returns the internal buffer view (zero-copy).
    """

    def __init__(self, name: str, shape: tuple):
        self.name   = name
        self.shape  = shape
        self._buf   = np.zeros(shape, dtype=np.float32)
        self._peers: list['Port'] = []

    def connect(self, other: 'Port'):
        """Connect this output port to a downstream input port (fan-out safe)."""
        self._peers.append(other)

    def send(self, data: np.ndarray):
        """Push data to all connected downstream ports."""
        arr = np.asarray(data, dtype=np.float32)
        for peer in self._peers:
            peer._buf = arr

    def recv(self) -> np.ndarray:
        """Return internal buffer (zero-copy — caller must not mutate)."""
        return self._buf


class Process:
    """
    Self-contained computational unit with internal state and I/O Ports.
    """

    def __init__(self, name: str):
        self.name  = name
        self.ports: dict[str, Port] = {}

    def _add(self, port: Port):
        self.ports[port.name] = port

    def run_timestep(self) -> np.ndarray:
        raise NotImplementedError


class LIFProcess(Process):
    """
    Layer of Leaky Integrate-and-Fire neurons.
    """

    def __init__(self, name: str, n: int, tau_m: float = 0.9, v_th: float = 1.0):
        super().__init__(name)
        self.n     = n
        self.tau_m = tau_m
        self.v_th  = v_th
        self.v     = np.zeros(n, dtype=np.float32)
        self._add(Port('a_in',  (n,)))
        self._add(Port('s_out', (n,)))

    def run_timestep(self) -> np.ndarray:
        a         = self.ports['a_in'].recv()
        self.v    = self.tau_m * self.v + a
        spikes    = (self.v >= self.v_th).astype(np.float32)
        self.v[spikes.astype(bool)] = 0.0
        self.ports['s_out'].send(spikes)
        return spikes


class DenseProcess(Process):
    """
    Fully-connected synaptic projection (weight matrix multiply).
    """

    def __init__(self, name: str, weights: np.ndarray):
        super().__init__(name)
        self.W = weights.copy().astype(np.float32)
        n_in, n_out = weights.shape
        self._add(Port('s_in',  (n_in,)))
        self._add(Port('a_out', (n_out,)))

    def run_timestep(self) -> np.ndarray:
        s = self.ports['s_in'].recv()
        a = s @ self.W
        self.ports['a_out'].send(a)
        return a


class SpikeMonitor(Process):
    """Records spike history into a pre-allocated array for offline analysis."""

    def __init__(self, name: str, n: int, T_max: int = 2000):
        super().__init__(name)
        self.n         = n
        self._T_max    = T_max
        self._buf_log  = np.zeros((T_max, n), dtype=np.float32)
        self._idx      = 0
        self._add(Port('s_in', (n,)))

    def run_timestep(self) -> np.ndarray:
        s = self.ports['s_in'].recv()
        if self._idx < self._T_max:
            self._buf_log[self._idx] = s
            self._idx += 1
        return s

    @property
    def data(self) -> np.ndarray:
        return self._buf_log[:self._idx]


# ---------------------------------------------------------------------------
# Build the SNN
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    np.random.seed(7)
    N_IN, N_H1, N_H2 = 128, 64, 32
    T_SIM = 200

    W1 = np.random.uniform(0.3, 0.8, (N_IN, N_H1)).astype(np.float32)
    W1 *= (np.random.rand(*W1.shape) > 0.5)
    W2 = np.random.uniform(0.3, 0.8, (N_H1, N_H2)).astype(np.float32)
    W2 *= (np.random.rand(*W2.shape) > 0.5)

    inp     = LIFProcess('Input',   N_IN,  tau_m=0.85, v_th=0.5)
    dense1  = DenseProcess('Dense1', W1)
    hidden1 = LIFProcess('Hidden1', N_H1,  tau_m=0.90, v_th=8.0)
    dense2  = DenseProcess('Dense2', W2)
    hidden2 = LIFProcess('Hidden2', N_H2,  tau_m=0.90, v_th=8.0)
    mon1    = SpikeMonitor('Mon1',   N_H1, T_max=T_SIM)
    mon2    = SpikeMonitor('Mon2',   N_H2, T_max=T_SIM)

    # Wire up the network using Port.connect() — zero-copy dataflow
    inp.ports['s_out'].connect(dense1.ports['s_in'])
    dense1.ports['a_out'].connect(hidden1.ports['a_in'])
    hidden1.ports['s_out'].connect(dense2.ports['s_in'])
    hidden1.ports['s_out'].connect(mon1.ports['s_in'])
    dense2.ports['a_out'].connect(hidden2.ports['a_in'])
    hidden2.ports['s_out'].connect(mon2.ports['s_in'])

    # Poisson input
    input_rates = np.random.rand(N_IN) * 0.5
    spike_input = (np.random.rand(T_SIM, N_IN) < input_rates).astype(np.float32)

    # Time-step loop
    t0 = time.perf_counter()
    for t in range(T_SIM):
        inp.ports['a_in'].send(spike_input[t])
        inp.run_timestep()
        dense1.run_timestep()
        hidden1.run_timestep()
        mon1.run_timestep()
        dense2.run_timestep()
        hidden2.run_timestep()
        mon2.run_timestep()

    elapsed = (time.perf_counter() - t0) * 1000
    sp1, sp2 = mon1.data, mon2.data

    print(f"SNN Process Runtime — {T_SIM} timesteps | {elapsed:.1f} ms")
    print(f"  Architecture : {N_IN} → {N_H1} → {N_H2}")
    print(f"  Hidden1      : {sp1.mean()*100:.1f}% spike rate")
    print(f"  Hidden2      : {sp2.mean()*100:.1f}% spike rate")

    # Visualisation
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('SNN Process Runtime – Neuromorphic Dataflow Graph', fontsize=13, fontweight='bold')

    for ax, sp, label, col, n in [
        (axes[0], sp1, f'Hidden1 ({N_H1} neurons)', '#4e79a7', N_H1),
        (axes[1], sp2, f'Hidden2 ({N_H2} neurons)', '#e15759', N_H2),
    ]:
        for nid in range(n):
            ts = np.where(sp[:, nid])[0]
            if len(ts):
                ax.scatter(ts, [nid] * len(ts), s=1.5 if n > 32 else 3, c=col, marker='|')
        ax.set_title(label)
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Neuron')
        ax.set_facecolor('#fafafa')

    ax = axes[2]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('Dataflow Graph')

    nodes = [
        (1.5, 8.5, f'Input\n(LIF,{N_IN})',     '#4e79a7'),
        (4.0, 8.5, f'Dense1\n({N_IN}→{N_H1})', '#f28e2b'),
        (7.0, 8.5, f'Hidden1\n(LIF,{N_H1})',   '#4e79a7'),
        (4.5, 5.5, f'Mon1\n({N_H1})',           '#76b7b2'),
        (7.0, 5.5, f'Dense2\n({N_H1}→{N_H2})', '#f28e2b'),
        (9.0, 5.5, f'Hidden2\n(LIF,{N_H2})',   '#4e79a7'),
        (9.0, 3.0, f'Mon2\n({N_H2})',           '#76b7b2'),
    ]
    for (x, y, lbl, c) in nodes:
        ax.add_patch(mpatches.FancyBboxPatch(
            (x - 1.1, y - 0.55), 2.0, 1.1,
            boxstyle='round,pad=0.08', facecolor=c, edgecolor='white',
            alpha=0.88, linewidth=1.5))
        ax.text(x, y, lbl, ha='center', va='center',
                fontsize=7.5, color='white', fontweight='bold')

    for (x, y, dx, dy) in [(2.6, 8.5, 1.3, 0), (5.1, 8.5, 0.8, 0),
                            (7.0, 7.9, 0, -1.8), (7.0, 5.5, 1.9, 0),
                            (8.9, 5.5, 0, -2.0)]:
        ax.annotate('', xy=(x + dx, y + dy), xytext=(x, y),
                    arrowprops=dict(arrowstyle='->', color='#333', lw=1.5))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'lava_simulation.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Plot saved → Optimization/lava_simulation.png")


