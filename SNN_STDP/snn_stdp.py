"""
Neuromorphic AI Engine
Module: Spiking Neural Network with STDP Learning

Implements:
  - Poisson Rate Coding: MNIST images encoded as binary spike trains
  - Two-layer SNN (Input → Hidden) with LIF neurons
  - Spike-Timing-Dependent Plasticity (STDP) as a local learning rule

STDP rule:
  Δw > 0  when pre-spike precedes post-spike  (LTP)
  Δw < 0  when pre-spike follows  post-spike  (LTD)
"""

import numpy as np
import matplotlib.pyplot as plt
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

OUTPUT_DIR = Path(__file__).parent

# ---------------------------------------------------------------------------
# Optional dependency check
# ---------------------------------------------------------------------------
try:
    import torch
    import snntorch as snn
    USE_SNNTORCH = True
except ImportError:
    USE_SNNTORCH = False

MNIST_LOADED = False
if USE_SNNTORCH:
    try:
        from torchvision import datasets, transforms
        _mnist = datasets.MNIST(
            root='./data', train=True, download=True,
            transform=transforms.ToTensor()
        )
        MNIST_LOADED = True
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Network hyperparameters
# ---------------------------------------------------------------------------
N_INPUT  = 784     # 28×28 pixels
N_HIDDEN = 64
T_STEPS  = 100     # Timesteps per sample
TAU_M    = 0.9     # LIF membrane decay factor
V_TH     = 15.0    # Spike threshold — calibrated for MNIST (784 inputs, W_mean ~0.15-0.3)
V_RST    = 0.0     # Reset potential

# STDP hyperparameters
A_PLUS    =  0.005
A_MINUS   = -0.005   # Balanced: equal LTP/LTD magnitude prevents drift
TAU_PLUS  = 15
TAU_MINUS = 15
W_MAX     = 1.0
W_MIN     = 0.0


# ---------------------------------------------------------------------------
# Poisson Rate Coding
# ---------------------------------------------------------------------------
def poisson_encode(rates, T):
    """
    Convert a probability vector [N] into a binary spike train [T, N].
    Each timestep is an independent Bernoulli draw with p = rates[n].
    """
    return (np.random.rand(T, len(rates)) < rates).astype(np.float32)


# ---------------------------------------------------------------------------
# LIF layer forward pass
# ---------------------------------------------------------------------------
def lif_forward(spike_in, W, tau_m=TAU_M, v_th=V_TH, v_rst=V_RST):
    """
    Run a one-directional LIF layer: spike_in [T, n_in] → spike_out [T, n_out].
    """
    T, _   = spike_in.shape
    n_out  = W.shape[1]
    V      = np.zeros(n_out)
    spikes = np.zeros((T, n_out), dtype=np.float32)

    for t in range(T):
        V = tau_m * V + spike_in[t] @ W
        fired = V >= v_th
        V[fired] = v_rst
        spikes[t, fired] = 1.0

    return spikes


# ---------------------------------------------------------------------------
# STDP weight update
# ---------------------------------------------------------------------------
def stdp_update(spike_in, spike_out, W,
                a_plus=A_PLUS, a_minus=A_MINUS,
                tau_plus=TAU_PLUS, tau_minus=TAU_MINUS):
    """
    Compute ΔW for a single sample from the input and output spike trains.
    Returns a delta matrix of the same shape as W.

    Correct STDP rule: only update synapses (i→j) where neuron i (pre) and
    neuron j (post) both participate — not all-to-all pairs.
      LTP: when post_j fires, strengthen each (pre_i → post_j) that fired before
      LTD: when pre_i fires, weaken each (pre_i → post_j) whose post fired before
    """
    T     = spike_in.shape[0]
    n_in  = W.shape[0]
    n_out = W.shape[1]
    dw    = np.zeros_like(W)

    t_pre  = -999.0 * np.ones(n_in)
    t_post = -999.0 * np.ones(n_out)

    for t in range(T):
        pre  = spike_in[t].astype(bool)   # shape (n_in,)
        post = spike_out[t].astype(bool)  # shape (n_out,)

        # LTP: post_j just fired — for each (pre_i → post_j) pair,
        # strengthen if pre_i fired recently (dt = t - t_pre[i] > 0)
        if post.any():
            dt = t - t_pre            # (n_in,)  time since each pre fired
            valid = dt > 0            # only causal pairs (pre before post)
            ltp = np.where(valid, a_plus * np.exp(-dt / tau_plus), 0.0)
            # dw[i, j] += ltp[i] for each active post neuron j
            dw[:, post] += ltp[:, np.newaxis]

        # LTD: pre_i just fired — for each (pre_i → post_j) pair,
        # weaken if post_j fired recently (dt = t - t_post[j] > 0)
        if pre.any():
            dt = t - t_post           # (n_out,)  time since each post fired
            valid = dt > 0            # only causal pairs (post before pre)
            ltd = np.where(valid, a_minus * np.exp(-dt / tau_minus), 0.0)
            # dw[i, j] += ltd[j] for each active pre neuron i
            dw[pre, :] += ltd[np.newaxis, :]

        t_pre[pre]   = t
        t_post[post] = t

    return dw


# ---------------------------------------------------------------------------
# Training sample loader
# ---------------------------------------------------------------------------
def get_sample(idx):
    if MNIST_LOADED:
        img, lbl = _mnist[idx % len(_mnist)]
        return img.view(-1).numpy(), lbl
    return np.random.rand(N_INPUT) * 0.4, 0


if __name__ == '__main__':
    # ---------------------------------------------------------------------------
    # Encode and visualise the first sample
    # ---------------------------------------------------------------------------
    np.random.seed(0)
    rates0, label0 = get_sample(0)
    spike_train0   = poisson_encode(rates0, T_STEPS)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))
    ax1.imshow(rates0.reshape(28, 28), cmap='gray', interpolation='nearest')
    ax1.set_title(f'MNIST Image – label: {label0}', fontsize=11)
    ax1.axis('off')

    for nid in range(100):
        ts = np.where(spike_train0[:, nid])[0]
        if len(ts):
            ax2.scatter(ts, [nid] * len(ts), s=1.5, c='#333333', marker='|')
    ax2.set_xlabel('Timestep')
    ax2.set_ylabel('Input Neuron Index')
    ax2.set_title('Spike Train – Poisson Encoding (first 100 neurons)')
    ax2.set_xlim(0, T_STEPS)
    ax2.set_facecolor('#fafafa')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'spike_encoding.png', dpi=150, bbox_inches='tight')
    plt.close()

    # ---------------------------------------------------------------------------
    # STDP training loop
    # ---------------------------------------------------------------------------
    np.random.seed(1)
    W = np.random.uniform(0.0, 0.3, (N_INPUT, N_HIDDEN)).astype(np.float32)

    N_EPOCHS  = 20
    N_SAMPLES = 200
    N_VOTE    = 200    # Samples for WTA class assignment
    dw_history = []

    for epoch in range(N_EPOCHS):
        dw_total = np.zeros_like(W)
        for s in range(N_SAMPLES):
            rates, _ = get_sample(s)
            sp_in    = poisson_encode(rates, T_STEPS)
            sp_out   = lif_forward(sp_in, W)
            dw_total += stdp_update(sp_in, sp_out, W)

        W = np.clip(W + dw_total / N_SAMPLES, W_MIN, W_MAX)

        # Weight decay: gentle L2 regularization prevents long-term saturation
        # This is the primary defense against 100% firing rates slowing STDP
        W *= 0.995

        # Soft homeostatic normalization: only rescale neurons that saturate (>0.9 mean)
        # Preserves relative weight differences while preventing hard saturation
        col_means = W.mean(axis=0)
        saturated = col_means > 0.9
        if saturated.any():
            W[:, saturated] *= 0.5

        dw_history.append(np.abs(dw_total).mean())

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1:2d}/{N_EPOCHS} | ΔW = {dw_history[-1]:.6f} | W_mean = {W.mean():.4f}")

    print("\nSTDP training complete")

    # WTA readout: assign each hidden neuron to a class using more samples
    print("\nBuilding WTA readout...")
    vote_counts = np.zeros((10, N_HIDDEN), dtype=np.float32)
    for s in range(N_VOTE):
        rates, label = get_sample(s)
        sp_in  = poisson_encode(rates, T_STEPS)
        sp_out = lif_forward(sp_in, W)
        vote_counts[label] += sp_out.sum(axis=0)
    neuron_class = np.argmax(vote_counts, axis=0)  # shape (N_HIDDEN,)

    # Test accuracy evaluation — use samples beyond the vote-counting range
    N_TEST = 300
    per_class_correct = np.zeros(10, dtype=int)
    per_class_total   = np.zeros(10, dtype=int)
    for s in range(N_VOTE, N_VOTE + N_TEST):
        rates, label = get_sample(s)
        sp_in  = poisson_encode(rates, T_STEPS)
        sp_out = lif_forward(sp_in, W)
        spike_counts = sp_out.sum(axis=0)
        winner = int(np.argmax(spike_counts))
        pred   = int(neuron_class[winner])
        per_class_total[label] += 1
        if pred == label:
            per_class_correct[label] += 1
    test_acc = per_class_correct.sum() / max(per_class_total.sum(), 1)
    print(f"Test accuracy (WTA readout): {test_acc*100:.1f}%")

    # Literature comparison — STDP on MNIST benchmarks
    print("\n--- Literature comparison (STDP on MNIST) ---")
    print("  This work (2-layer SNN, 64 hidden, 200 train samples, unsupervised STDP + WTA): "
          f"{test_acc*100:.1f}%")
    print("  Diehl & Cook (2015), STDP + homeostasis, 6400 hidden neurons, 60000 train samples: ~95%")
    print("     [Diehl, P.U. & Cook, M. (2015). Unsupervised learning of digit recognition")
    print("      using spike-timing-dependent plasticity. Front. Comput. Neurosci.]")
    print("  Kheradpisheh et al. (2018), STDP + tempotron, 2-layer conv SNN: ~98.4%")
    print("     [Kheradpisheh, S.R. et al. (2018). STDP-based spiking deep neural networks")
    print("      for object recognition. arXiv:1611.01421v4]")
    print("  Note: our low accuracy is expected — 64 neurons, 200 training samples,")
    print("  no homeostasis, no convolution. This is a proof-of-concept pipeline,")
    print("  not a competitive MNIST classifier.")

    # ---------------------------------------------------------------------------
    # Visualisation
    # ---------------------------------------------------------------------------
    W_init = np.random.uniform(0.0, 0.3, (N_INPUT, N_HIDDEN))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(dw_history, color='#e15759', linewidth=2, marker='o', markersize=4)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Mean |ΔW|')
    axes[0].set_title('Weight Convergence over Epochs')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_facecolor('#fafafa')

    axes[1].hist(W_init.flatten(), bins=50, alpha=0.6, color='#4e79a7',
                 label='Initialisation', density=True)
    axes[1].hist(W.flatten(), bins=50, alpha=0.6, color='#f28e2b',
                 label='After STDP', density=True)
    axes[1].set_xlabel('Weight Value')
    axes[1].set_ylabel('Probability Density')
    axes[1].set_title('Weight Distribution Before / After STDP')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_facecolor('#fafafa')

    per_class_acc = np.where(per_class_total > 0,
                              per_class_correct / np.maximum(per_class_total, 1), 0.0)
    axes[2].bar(range(10), per_class_acc * 100, color='#4e79a7', alpha=0.8, edgecolor='white')
    axes[2].axhline(test_acc * 100, color='red', linestyle='--', lw=1.5,
                    label=f'Overall: {test_acc*100:.1f}%')
    axes[2].set_xlabel('Digit class')
    axes[2].set_ylabel('Accuracy (%)')
    axes[2].set_title('Test Accuracy per Class\n(WTA readout, post-STDP)')
    axes[2].set_xticks(range(10))
    axes[2].set_ylim(0, 110)
    axes[2].legend(fontsize=9)
    axes[2].grid(True, alpha=0.3)
    axes[2].set_facecolor('#fafafa')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'stdp_weights.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Receptive fields of hidden layer neurons
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    fig.suptitle('Receptive Fields – 16 Hidden Neurons (post-STDP)', fontsize=12, fontweight='bold')
    for i, ax in enumerate(axes.flatten()):
        im = ax.imshow(W[:, i].reshape(28, 28), cmap='RdBu_r',
                       vmin=W_MIN, vmax=W_MAX, interpolation='nearest')
        ax.axis('off')
        ax.set_title(f'H{i}', fontsize=7)
    plt.colorbar(im, ax=axes.flatten()[-1], fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'receptive_fields.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Raster plot before / after STDP
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharey=True)
    fig.suptitle('Hidden Layer Activity – Before and After STDP', fontsize=12)
    for ax, W_plot, title, col in [
        (axes[0], W_init.astype(np.float32), 'Before STDP', '#4e79a7'),
        (axes[1], W,                          'After STDP',  '#e15759'),
    ]:
        sp = lif_forward(spike_train0, W_plot)
        for nid in range(N_HIDDEN):
            ts = np.where(sp[:, nid])[0]
            if len(ts):
                ax.scatter(ts, [nid] * len(ts), s=2.5, c=col, marker='|')
        ax.set_ylabel('Hidden Neuron')
        ax.set_title(f'{title} | Total spikes = {sp.sum():.0f}')
        ax.set_facecolor('#fafafa')
    axes[-1].set_xlabel('Timestep')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'raster_before_after.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Architecture : {N_INPUT} → {N_HIDDEN} (LIF)")
    print(f"  Encoding     : Poisson Rate Coding ({T_STEPS} timesteps)")
    print(f"  W_mean       : {W_init.mean():.4f} → {W.mean():.4f}")
    print("Plots saved → SNN_STDP/")
