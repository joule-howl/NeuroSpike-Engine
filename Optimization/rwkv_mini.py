"""
Neuromorphic AI Engine
Module: RWKV – Receptance Weighted Key Value (non-Transformer architecture)

RWKV processes time-series data as a pure recurrent model with O(1) hidden
state memory, requiring no KV-cache unlike Transformer attention.

WKV recurrent formulation:
    num[t] = e^(-w) · num[t-1] + e^(k[t]) · v[t]
    den[t] = e^(-w) · den[t-1] + e^(k[t])
    wkv[t] = (e^(u+k[t]) · v[t] + num[t-1]) / (e^(u+k[t]) + den[t-1])
    out[t] = r[t] · (wkv[t] @ W_o)

Reference: RWKV-LM (Peng et al., 2023) – arXiv:2305.13048
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent


class RWKVBlock:
    """Single RWKV block with Time-Mixing and Channel-Mixing sub-layers."""

    def __init__(self, d_model: int, seed: int = 0):
        np.random.seed(seed)
        s = 1.0 / np.sqrt(d_model)
        self.w  = -np.abs(np.random.randn(d_model).astype(np.float32)) * 0.5  # decay
        self.u  = np.random.randn(d_model).astype(np.float32) * 0.5            # bonus
        self.Wr = s * np.random.randn(d_model, d_model).astype(np.float32)     # receptance
        self.Wk = s * np.random.randn(d_model, d_model).astype(np.float32)     # key
        self.Wv = s * np.random.randn(d_model, d_model).astype(np.float32)     # value
        self.Wo = s * np.random.randn(d_model, d_model).astype(np.float32)     # output
        self.W1 = s * np.random.randn(d_model, d_model * 4).astype(np.float32) # FFN expand
        self.W2 = s * np.random.randn(d_model * 4, d_model).astype(np.float32) # FFN project
        self.ln_g = np.ones(d_model,  dtype=np.float32)
        self.ln_b = np.zeros(d_model, dtype=np.float32)
        self.d_model = d_model

    @staticmethod
    def _ln(x, g, b, eps=1e-5):
        return g * (x - x.mean()) / (x.std() + eps) + b

    @staticmethod
    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))

    def init_state(self):
        """Initial hidden state tuple: (num, den, x_prev)."""
        return (np.zeros(self.d_model, dtype=np.float32),
                np.full(self.d_model, 1e-8, dtype=np.float32),
                np.zeros(self.d_model, dtype=np.float32))

    def step(self, x: np.ndarray, state: tuple) -> tuple:
        """
        Single recurrent step.
        Returns (y, new_state); state size is fixed regardless of T.
        """
        num, den, _ = state
        xn  = self._ln(x, self.ln_g, self.ln_b)
        r   = self._sigmoid(xn @ self.Wr)
        k   = xn @ self.Wk
        v   = xn @ self.Wv

        ek  = np.exp(np.clip(k,          -30, 30))
        euk = np.exp(np.clip(self.u + k, -30, 30))
        ew  = np.exp(self.w)

        wkv = (euk * v + num) / (euk + den + 1e-8)
        out = r * (wkv @ self.Wo)
        ffn = np.maximum(0, xn @ self.W1) @ self.W2  # ReLU FFN

        new_state = (ew * num + ek * v, ew * den + ek, x)
        return x + out + ffn, new_state


class MiniRWKV:
    """Multi-layer RWKV network with O(n_layers · d_model) constant state."""

    def __init__(self, d_model: int, n_layers: int):
        self.d_model  = d_model
        self.n_layers = n_layers
        self.blocks   = [RWKVBlock(d_model, seed=i) for i in range(n_layers)]

    def forward(self, X: np.ndarray) -> np.ndarray:
        """X: [T, d_model] → [T, d_model]"""
        states = [blk.init_state() for blk in self.blocks]
        out    = []
        for x in X:
            h = x.copy()
            for i, blk in enumerate(self.blocks):
                h, states[i] = blk.step(h, states[i])
            out.append(h)
        return np.array(out)

    def state_bytes(self) -> int:
        """Total hidden state RAM (bytes) – constant for any sequence length T."""
        return self.n_layers * 3 * self.d_model * 4


# ---------------------------------------------------------------------------
# Benchmark: memory and latency vs. sequence length
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    D_MODEL, N_LAYERS = 64, 2
    rwkv = MiniRWKV(D_MODEL, N_LAYERS)
    seq_lengths = [50, 100, 200, 500, 1000, 2000]

    times_rwkv, ram_rwkv, ram_trans = [], [], []

    print(f"Mini RWKV — d_model={D_MODEL}  n_layers={N_LAYERS}")
    print(f"Hidden state: {rwkv.state_bytes()} bytes (constant)\n")
    print(f"{'T':<8} {'RWKV ms':>10} {'RAM RWKV':>12} {'RAM Trans':>12} {'Ratio':>8}")
    print("-" * 55)

    for T_len in seq_lengths:
        X    = np.random.randn(T_len, D_MODEL).astype(np.float32) * 0.1
        t0   = time.perf_counter()
        _    = rwkv.forward(X)
        tr   = (time.perf_counter() - t0) * 1000
        rb   = rwkv.state_bytes()
        rt   = T_len * D_MODEL * 2 * 4
        times_rwkv.append(tr)
        ram_rwkv.append(rb)
        ram_trans.append(rt)
        print(f"{T_len:<8} {tr:>10.2f} {rb:>12} {rt:>12} {rt//rb:>7}×")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('RWKV – Recurrent Non-Transformer Architecture', fontsize=13, fontweight='bold')

    ax = axes[0]
    ax.plot(seq_lengths, [r / 1024 for r in ram_trans], 'o-',
            color='#e15759', lw=2, label='Transformer KV-cache O(T)')
    ax.plot(seq_lengths, [r / 1024 for r in ram_rwkv],  's--',
            color='#4e79a7', lw=2, label=f'RWKV state O(1)={ram_rwkv[0]}B')
    ax.set_xlabel('Sequence Length T')
    ax.set_ylabel('RAM (KB)')
    ax.set_title('Hidden State Memory')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_facecolor('#fafafa')

    ax = axes[1]
    ax.plot(seq_lengths, times_rwkv, 'o-', color='#59a14f', lw=2)
    ax.fill_between(seq_lengths, times_rwkv, alpha=0.15, color='#59a14f')
    ax.set_xlabel('Sequence Length T')
    ax.set_ylabel('Time (ms)')
    ax.set_title('Inference Time – Linear O(T)')
    ax.grid(True, alpha=0.3)
    ax.set_facecolor('#fafafa')

    ax = axes[2]
    X_demo = np.random.randn(200, D_MODEL).astype(np.float32) * 0.1
    for i in range(200):
        X_demo[i, :8] += 0.3 * np.sin(2 * np.pi * i / 20)
    out_demo = rwkv.forward(X_demo)
    for ch in range(4):
        ax.plot(out_demo[:, ch], lw=0.8, alpha=0.8, label=f'ch{ch}')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Output')
    ax.set_title('RWKV Output (periodic pattern, T=200)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_facecolor('#fafafa')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'rwkv_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nRWKV complete | state = {rwkv.state_bytes()} bytes (constant)")
    print(f"T=2000: RWKV {ram_rwkv[-1]}B vs Transformer {ram_trans[-1]:,}B "
          f"({ram_trans[-1]//ram_rwkv[-1]:,}× less)")
    print("Plot saved → Optimization/rwkv_comparison.png")


