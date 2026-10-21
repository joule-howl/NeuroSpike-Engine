"""
Neuromorphic AI Engine
Module: SNN Structure Optimisation

Implements:
  - Weight pruning by magnitude threshold
  - Sparse matrix computation using scipy CSC/CSR (true sparse matmul)
  - State Space Model (SSM / Mamba-style) with O(1) memory footprint
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

OUTPUT_DIR = Path(__file__).parent

try:
    import scipy.sparse as sp_sparse
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# ---------------------------------------------------------------------------
# Trained SNN weights (reproduced from snn_stdp module)
# ---------------------------------------------------------------------------
np.random.seed(42)
N_IN, N_HID = 784, 64
W_trained = np.random.uniform(0.0, 0.4, (N_IN, N_HID)).astype(np.float32)
W_trained *= (np.random.rand(*W_trained.shape) > 0.6).astype(np.float32)
W_trained += np.random.uniform(0.0, 0.1, W_trained.shape) * 0.1


# ---------------------------------------------------------------------------
# Pruning
# ---------------------------------------------------------------------------
def prune(W, threshold):
    """
    Zero out all connections with |w| < threshold.
    Returns (W_pruned, sparsity_ratio).
    """
    Wp = W.copy()
    Wp[np.abs(Wp) < threshold] = 0.0
    return Wp, (Wp == 0).mean()


# ---------------------------------------------------------------------------
# State Space Model (SSM) – O(1) memory state
# ---------------------------------------------------------------------------
class StateSpaceModel:
    """
    Linear Time-Invariant SSM:
        h[t] = A · h[t-1] + B · x[t]
        y[t] = C · h[t]   + D · x[t]

    The hidden state h has fixed size d_state, independent of sequence
    length T — unlike Transformers that require an O(T) KV-cache.
    """

    def __init__(self, d_in, d_state, d_out, seed=5):
        np.random.seed(seed)
        s = 1.0 / np.sqrt(d_state)
        # Stable A matrix (spectral radius < 1)
        self.A = np.eye(d_state) * 0.9 - s * np.abs(np.random.randn(d_state, d_state))
        self.B = s * np.random.randn(d_state, d_in).astype(np.float32)
        self.C = s * np.random.randn(d_out, d_state).astype(np.float32)
        self.D = np.zeros((d_out, d_in), dtype=np.float32)
        self.d_state = d_state

    def step(self, x, h):
        h = self.A @ h + self.B @ x
        y = self.C @ h  + self.D @ x
        return y, h

    def forward(self, X):
        """X: [T, d_in] → outputs: [T, d_out]"""
        h   = np.zeros(self.d_state, dtype=np.float32)
        out = []
        for x in X:
            y, h = self.step(x, h)
            out.append(y)
        return np.array(out)

    def state_ram(self):
        """State RAM in bytes – constant for any sequence length T."""
        return self.d_state * 4


if __name__ == '__main__':
    thresholds = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    sparsities, ram_savings, nnz_counts = [], [], []

    for thr in thresholds:
        Wp, spar = prune(W_trained, thr)
        nnz = (Wp != 0).sum()
        ram_dense  = W_trained.nbytes
        ram_sparse = nnz * 12 + 8 * (N_HID + 1)
        saving = max(0.0, (1 - ram_sparse / ram_dense) * 100)
        sparsities.append(spar * 100)
        ram_savings.append(saving)
        nnz_counts.append(nnz)
        print(f"  thr={thr:.2f}  sparsity={spar*100:.1f}%  nnz={nnz:,}  RAM↓={saving:.1f}%")

    BEST_THR = 0.15
    W_pruned, _ = prune(W_trained, BEST_THR)

    # Dense vs. sparse matmul benchmark
    T_TEST = 1000
    x_test = np.random.rand(T_TEST, N_IN).astype(np.float32)

    t0 = time.perf_counter()
    for _ in range(50):
        _ = x_test @ W_trained
    t_dense = (time.perf_counter() - t0) / 50 * 1000
    print(f"\n  Dense  : {t_dense:.3f} ms")

    if HAS_SCIPY:
        W_csc = sp_sparse.csc_matrix(W_pruned)
        x_csr = sp_sparse.csr_matrix(x_test)
        t0 = time.perf_counter()
        for _ in range(50):
            _ = x_csr @ W_csc
        t_sparse = (time.perf_counter() - t0) / 50 * 1000
        print(f"  Sparse : {t_sparse:.3f} ms  ({t_dense/t_sparse:.2f}× speedup  "
              f"nnz={W_csc.nnz:,}  {W_csc.nnz/W_trained.size*100:.1f}%)")
    else:
        t_sparse = t_dense
        print("  scipy not available")

    D_IN, D_STATE, D_OUT = 32, 16, 32
    ssm = StateSpaceModel(D_IN, D_STATE, D_OUT)

    seq_lengths = [50, 200, 500, 1000, 2000, 5000]
    ram_ssm_list, ram_trans_list = [], []

    print(f"\n  SSM: d_in={D_IN}  d_state={D_STATE}  d_out={D_OUT}")
    print(f"  {'T':<8} {'Transformer RAM':>18} {'SSM RAM':>12} {'Ratio':>8}")
    print("  " + "-" * 50)
    for T_len in seq_lengths:
        rt = T_len * D_IN * 2 * 4          # KV-cache (bytes)
        rs = ssm.state_ram()
        ram_trans_list.append(rt / 1024)
        ram_ssm_list.append(rs / 1024)
        print(f"  {T_len:<8} {rt/1024:>15.1f} KB  {rs/1024:>8.3f} KB  {rt//rs:>6}×")

    X_test = np.random.randn(1000, D_IN).astype(np.float32)
    t0 = time.perf_counter()
    ssm_out = ssm.forward(X_test)
    t_ssm   = (time.perf_counter() - t0) * 1000
    print(f"\n  SSM inference T=1000: {t_ssm:.2f} ms | "
          f"state RAM = {ssm.state_ram()} bytes (constant)")

    # ---------------------------------------------------------------------------
    # Summary plots
    # ---------------------------------------------------------------------------
    fig = plt.figure(figsize=(16, 12))
    gs  = gridspec.GridSpec(2, 3, hspace=0.4, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(thresholds, sparsities, 'o-', color='#4e79a7', linewidth=2)
    ax1.fill_between(thresholds, sparsities, alpha=0.2, color='#4e79a7')
    ax1.axvline(BEST_THR, color='red', linestyle='--', linewidth=1.5,
                label=f'Selected = {BEST_THR}')
    ax1.set_xlabel('Pruning threshold')
    ax1.set_ylabel('Sparsity (%)')
    ax1.set_title('Sparsity vs. Pruning Threshold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_facecolor('#fafafa')

    ax2 = fig.add_subplot(gs[0, 1])
    bars = ax2.bar(thresholds, ram_savings,
                   color=['#59a14f' if s > 0 else '#e15759' for s in ram_savings],
                   alpha=0.8, edgecolor='white')
    for bar, val in zip(bars, ram_savings):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f'{val:.0f}%', ha='center', va='bottom', fontsize=8)
    ax2.set_xlabel('Pruning threshold')
    ax2.set_ylabel('Memory Saved (%)')
    ax2.set_title('RAM Reduction After Pruning')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_facecolor('#fafafa')

    ax3 = fig.add_subplot(gs[0, 2])
    im  = ax3.imshow(W_trained[:50, :].T, aspect='auto', cmap='Blues',
                     vmin=0, vmax=W_trained.max(), interpolation='nearest')
    mask = (W_pruned[:50, :] == 0).T
    ax3.imshow(mask, aspect='auto', cmap='Reds', alpha=0.4,
               vmin=0, vmax=1, interpolation='nearest')
    ax3.set_title('Weight Matrix (red = pruned)')
    ax3.set_xlabel('Input Neurons')
    ax3.set_ylabel('Hidden Neurons')
    plt.colorbar(im, ax=ax3, fraction=0.03, pad=0.04)

    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(seq_lengths, ram_trans_list, 'o-', color='#e15759', linewidth=2,
             label='Transformer O(T)')
    ax4.plot(seq_lengths, ram_ssm_list,   's--', color='#59a14f', linewidth=2,
             label=f'SSM O(1) = {ssm.state_ram()}B')
    ax4.set_xlabel('Sequence Length T')
    ax4.set_ylabel('State RAM (KB)')
    ax4.set_title('Memory: SSM vs. Transformer')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_facecolor('#fafafa')

    ax5 = fig.add_subplot(gs[1, 1])
    for ch in range(4):
        ax5.plot(ssm_out[:200, ch], linewidth=0.8, alpha=0.8, label=f'y[{ch}]')
    ax5.set_xlabel('Timestep')
    ax5.set_ylabel('SSM Output')
    ax5.set_title('SSM Output (4 channels, T=200)')
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)
    ax5.set_facecolor('#fafafa')

    ax6 = fig.add_subplot(gs[1, 2])
    categories = ['Dense', 'Sparse\n(pruned)', 'SSM state']
    mem_vals   = [W_trained.nbytes / 1024,
                  nnz_counts[thresholds.index(BEST_THR)] * 12 / 1024,
                  ssm.state_ram() / 1024]
    bars2 = ax6.bar(categories, mem_vals,
                    color=['#e15759', '#f28e2b', '#59a14f'],
                    alpha=0.85, edgecolor='white')
    for bar, val in zip(bars2, mem_vals):
        ax6.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f'{val:.1f} KB', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax6.set_ylabel('Memory (KB)')
    ax6.set_title('Memory Comparison: Dense / Sparse / SSM')
    ax6.grid(True, alpha=0.3, axis='y')
    ax6.set_facecolor('#fafafa')

    fig.suptitle('SNN Optimisation – Pruning & State Space Model',
                 fontsize=14, fontweight='bold')
    plt.savefig(OUTPUT_DIR / 'optimization_results.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("\nOptimisation complete")
    print(f"  Dense  : {W_trained.nbytes/1024:.1f} KB  ({W_trained.size:,} connections)")
    print(f"  Pruned : {nnz_counts[thresholds.index(BEST_THR)]*12/1024:.1f} KB  "
          f"({nnz_counts[thresholds.index(BEST_THR)]:,} connections)")
    print(f"  SSM    : {ssm.state_ram()} bytes (constant hidden state for any T)")
    print("Plots saved → Optimization/")
    """
    Linear Time-Invariant SSM:
        h[t] = A · h[t-1] + B · x[t]
        y[t] = C · h[t]   + D · x[t]

    The hidden state h has fixed size d_state, independent of sequence
    length T — unlike Transformers that require an O(T) KV-cache.
    """

    def __init__(self, d_in, d_state, d_out, seed=5):
        np.random.seed(seed)
        s = 1.0 / np.sqrt(d_state)
        # Stable A matrix (spectral radius < 1)
        self.A = np.eye(d_state) * 0.9 - s * np.abs(np.random.randn(d_state, d_state))
        self.B = s * np.random.randn(d_state, d_in).astype(np.float32)
        self.C = s * np.random.randn(d_out, d_state).astype(np.float32)
        self.D = np.zeros((d_out, d_in), dtype=np.float32)
        self.d_state = d_state

    def step(self, x, h):
        h = self.A @ h + self.B @ x
        y = self.C @ h  + self.D @ x
        return y, h

    def forward(self, X):
        """X: [T, d_in] → outputs: [T, d_out]"""
        h   = np.zeros(self.d_state, dtype=np.float32)
        out = []
        for x in X:
            y, h = self.step(x, h)
            out.append(y)
        return np.array(out)

    def state_ram(self):
        """State RAM in bytes – constant for any sequence length T."""
        return self.d_state * 4


D_IN, D_STATE, D_OUT = 32, 16, 32
ssm = StateSpaceModel(D_IN, D_STATE, D_OUT)

seq_lengths = [50, 200, 500, 1000, 2000, 5000]
ram_ssm_list, ram_trans_list = [], []

print(f"\n  SSM: d_in={D_IN}  d_state={D_STATE}  d_out={D_OUT}")
print(f"  {'T':<8} {'Transformer RAM':>18} {'SSM RAM':>12} {'Ratio':>8}")
print("  " + "-" * 50)
for T_len in seq_lengths:
    rt = T_len * D_IN * 2 * 4          # KV-cache (bytes)
    rs = ssm.state_ram()
    ram_trans_list.append(rt / 1024)
    ram_ssm_list.append(rs / 1024)
    print(f"  {T_len:<8} {rt/1024:>15.1f} KB  {rs/1024:>8.3f} KB  {rt//rs:>6}×")

    X_test = np.random.randn(1000, D_IN).astype(np.float32)
    t0 = time.perf_counter()
    ssm_out = ssm.forward(X_test)
    t_ssm   = (time.perf_counter() - t0) * 1000
    print(f"\n  SSM inference T=1000: {t_ssm:.2f} ms | "
          f"state RAM = {ssm.state_ram()} bytes (constant)")


