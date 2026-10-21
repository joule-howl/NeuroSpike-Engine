"""
Neuromorphic AI Engine
Module: Reward-Modulated STDP (R-STDP)

Biological basis:
  Standard STDP only uses local spike timing (pre/post correlation).
  In the brain, dopamine acts as a global reward signal that scales
  synaptic changes: good outcomes strengthen recently active synapses,
  bad outcomes weaken them.

  Δw = r(t) · STDP(Δt_pre, Δt_post)

  where r(t) is a dopamine-like reward signal delivered after the action.

This module implements:
  1. R-STDP learning rule
  2. Eligibility traces — a short-term memory of "which synapses were
     recently active" so the reward can be applied retroactively
  3. A simple reinforcement task: SNN learns to classify two spike
     patterns (pattern A → output 0, pattern B → output 1)
  4. Comparison: vanilla STDP vs R-STDP accuracy over episodes

Reference:
  Fremaux, N. & Gerstner, W. (2016). Neuromodulated STDP, and its
  application in reinforcement learning. Frontiers in Neural Circuits.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
N_IN     = 50      # Input neurons
N_OUT    = 2       # Output neurons (2-class task)
T_STEPS  = 30      # Timesteps per trial
N_TRIALS = 2000    # Training episodes
TAU_M    = 0.85    # LIF membrane decay
V_TH     = 1.0     # Spike threshold

# STDP windows
A_PLUS   =  0.01
A_MINUS  = -0.012
TAU_PLUS = 10
TAU_MINUS= 10

# Eligibility trace decay (seconds-scale memory)
TAU_ELIG = 50      # timesteps — how long a synapse "remembers" it was active

# Dopamine / reward parameters
DOPAMINE_PLUS  =  1.5    # reward for correct classification
DOPAMINE_MINUS = -0.8    # punishment for wrong classification
TAU_DOPAMINE   = 20      # dopamine diffusion time constant

W_MAX = 1.0
W_MIN = 0.0
np.random.seed(42)


# ---------------------------------------------------------------------------
# Spike pattern generator
# ---------------------------------------------------------------------------
def make_pattern(pattern_id: int, n_in: int, T: int,
                 base_rate: float = 0.1) -> np.ndarray:
    """
    Generate a spike train [T, N_IN] for pattern A (0) or B (1).
    Pattern A: first half of neurons active
    Pattern B: second half of neurons active
    """
    rates = np.full(n_in, base_rate * 0.2)
    if pattern_id == 0:
        rates[:n_in // 2] = base_rate * 1.0
    else:
        rates[n_in // 2:] = base_rate * 1.0
    return (np.random.rand(T, n_in) < rates).astype(np.float32)


# ---------------------------------------------------------------------------
# LIF layer forward — returns spikes and membrane traces
# ---------------------------------------------------------------------------
def lif_forward(spike_in: np.ndarray, W: np.ndarray,
                tau_m=TAU_M, v_th=V_TH):
    T, n_in = spike_in.shape
    n_out   = W.shape[1]
    V       = np.zeros(n_out, dtype=np.float32)
    sp_out  = np.zeros((T, n_out), dtype=np.float32)
    for t in range(T):
        V = tau_m * V + spike_in[t] @ W
        fired = V >= v_th
        V[fired] = 0.0
        sp_out[t, fired] = 1.0
    return sp_out


# ---------------------------------------------------------------------------
# Eligibility trace update
# ---------------------------------------------------------------------------
def update_eligibility(elig: np.ndarray,
                       sp_pre: np.ndarray, sp_post: np.ndarray,
                       t_pre_last: np.ndarray, t_post_last: np.ndarray,
                       t: int) -> np.ndarray:
    """
    Update eligibility trace for each synapse (i→j):
      e[i,j] += STDP_kernel(t_pre[i], t_post[j])
      e[i,j] *= exp(-1/tau_elig)  [decay each step]

    Correct rule: only update synapse (i→j) based on the actual timing
    of pre neuron i and post neuron j — not all-to-all combinations.
      LTP: post_j just fired → e[i,j] += A+ * exp(-(t - t_pre[i]) / tau+)
      LTD: pre_i just fired  → e[i,j] += A- * exp(-(t - t_post[j]) / tau-)
    """
    # Decay existing traces
    elig *= np.exp(-1.0 / TAU_ELIG)

    pre  = sp_pre.astype(bool)   # (N_IN,)
    post = sp_post.astype(bool)  # (N_OUT,)

    # LTP: post_j fired — for each active post_j, accumulate LTP trace
    # from all pre neurons weighted by how recently pre_i fired.
    if post.any():
        dt  = t - t_pre_last              # (N_IN,)  time since each pre fired
        ltp = np.where(dt > 0, A_PLUS * np.exp(-dt / TAU_PLUS), 0.0)
        elig[:, post] += ltp[:, np.newaxis]

    # LTD: pre_i fired — for each active pre_i, accumulate LTD trace
    # weighted by how recently post_j fired.
    if pre.any():
        dt  = t - t_post_last             # (N_OUT,)  time since each post fired
        ltd = np.where(dt > 0, A_MINUS * np.exp(-dt / TAU_MINUS), 0.0)
        elig[pre, :] += ltd[np.newaxis, :]

    return elig


# ---------------------------------------------------------------------------
# One trial: run SNN, collect eligibility traces, return decision + traces
# ---------------------------------------------------------------------------
def run_trial(W: np.ndarray, pattern_id: int):
    spike_in = make_pattern(pattern_id, N_IN, T_STEPS)
    sp_out   = np.zeros((T_STEPS, N_OUT), dtype=np.float32)
    elig     = np.zeros_like(W)
    V        = np.zeros(N_OUT, dtype=np.float32)

    t_pre_last  = -999.0 * np.ones(N_IN)
    t_post_last = -999.0 * np.ones(N_OUT)

    for t in range(T_STEPS):
        V = TAU_M * V + spike_in[t] @ W
        fired = V >= V_TH
        V[fired] = 0.0
        sp_out[t, fired] = 1.0

        elig = update_eligibility(
            elig, spike_in[t], sp_out[t],
            t_pre_last, t_post_last, t
        )
        t_pre_last[spike_in[t].astype(bool)]  = t
        t_post_last[sp_out[t].astype(bool)]   = t

    # Decision: which output neuron fired more
    decision = int(sp_out[:, 0].sum() < sp_out[:, 1].sum())
    return decision, elig, spike_in, sp_out


# ---------------------------------------------------------------------------
# Training: R-STDP vs vanilla STDP
# ---------------------------------------------------------------------------
def train(use_reward: bool = True):
    W = np.random.uniform(0.01, 0.3, (N_IN, N_OUT)).astype(np.float32)
    accuracy_log = []
    reward_log   = []

    # Rolling accuracy window
    correct_window = []

    for trial in range(N_TRIALS):
        # Alternate patterns, add random variety
        pattern_id = trial % 2

        decision, elig, sp_in, sp_out = run_trial(W, pattern_id)
        correct = int(decision == pattern_id)
        correct_window.append(correct)
        if len(correct_window) > 100:
            correct_window.pop(0)

        # --- Compute reward signal ---
        if correct:
            reward = DOPAMINE_PLUS
        else:
            reward = DOPAMINE_MINUS

        reward_log.append(reward)

        # --- Weight update ---
        if use_reward:
            # R-STDP: scale eligibility traces by dopamine signal
            dW = reward * elig
        else:
            # Vanilla STDP: apply eligibility directly (reward = 1 always)
            dW = elig

        W = np.clip(W + dW * 0.05, W_MIN, W_MAX)

        if (trial + 1) % 200 == 0:
            acc = np.mean(correct_window)
            accuracy_log.append(acc)
            mode = "R-STDP" if use_reward else "STDP"
            print(f"  [{mode}] Trial {trial+1:4d} | "
                  f"Acc = {acc*100:.1f}% | "
                  f"W_mean = {W.mean():.3f}")

    return W, accuracy_log, reward_log


if __name__ == '__main__':
    # ---------------------------------------------------------------------------
    # Run both variants
    # ---------------------------------------------------------------------------
    print("=" * 55)
    print("  Reward-Modulated STDP vs Vanilla STDP")
    print("=" * 55)

    print("\n[1/2] Training with Reward-Modulated STDP...")
    W_r, acc_r, rew_r = train(use_reward=True)

    print("\n[2/2] Training with Vanilla STDP...")
    W_v, acc_v, rew_v = train(use_reward=False)

    # Final evaluation
    acc_final_r = evaluate(W_r)
    acc_final_v = evaluate(W_v)
    print(f"\nFinal accuracy — R-STDP   : {acc_final_r*100:.1f}%")
    print(f"Final accuracy — Vanilla  : {acc_final_v*100:.1f}%")
    print(f"Improvement from reward   : +{(acc_final_r - acc_final_v)*100:.1f}%")

    # ---------------------------------------------------------------------------
    # Visualisation
    # ---------------------------------------------------------------------------
    fig = plt.figure(figsize=(16, 12))
    gs  = gridspec.GridSpec(2, 3, hspace=0.4, wspace=0.35)

    # Learning curves
    ax = fig.add_subplot(gs[0, 0])
    x  = [(i + 1) * 200 for i in range(len(acc_r))]
    ax.plot(x, [a * 100 for a in acc_r], color='#e15759', lw=2, label='R-STDP')
    ax.plot(x, [a * 100 for a in acc_v], color='#4e79a7', lw=2, label='Vanilla STDP')
    ax.axhline(50, color='gray', linestyle='--', lw=1, alpha=0.6, label='Chance (50%)')
    ax.set_xlabel('Trial')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Learning Curve: R-STDP vs Vanilla STDP')
    ax.legend(fontsize=9)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    ax.set_facecolor('#fafafa')

    # Reward signal over time
    ax = fig.add_subplot(gs[0, 1])
    window = 50
    smoothed = np.convolve(rew_r, np.ones(window)/window, mode='valid')
    ax.plot(smoothed, color='#59a14f', lw=1.5, label='Smoothed reward (R-STDP)')
    ax.axhline(0, color='gray', linestyle='--', lw=0.8, alpha=0.5)
    ax.fill_between(range(len(smoothed)), smoothed, 0,
                    where=(smoothed > 0), alpha=0.2, color='#59a14f')
    ax.fill_between(range(len(smoothed)), smoothed, 0,
                    where=(smoothed < 0), alpha=0.2, color='#e15759')
    ax.set_xlabel('Trial')
    ax.set_ylabel('Dopamine signal')
    ax.set_title('Reward Signal over Training')
    ax.grid(True, alpha=0.3)
    ax.set_facecolor('#fafafa')

    # Weight matrices
    for ax_idx, (W_plot, title, col) in enumerate([
        (W_r, 'Weights after R-STDP', gs[0, 2]),
        (W_v, 'Weights after Vanilla STDP', gs[1, 0]),
    ]):
        ax = fig.add_subplot(col)
        im = ax.imshow(W_plot.T, aspect='auto', cmap='RdBu_r',
                       vmin=0, vmax=W_MAX, interpolation='nearest')
        ax.set_xlabel('Input Neuron')
        ax.set_ylabel('Output Neuron')
        ax.set_title(title)
        plt.colorbar(im, ax=ax, fraction=0.04, pad=0.04)

    # Spike raster for one trial (R-STDP trained model)
    ax = fig.add_subplot(gs[1, 1])
    _, _, sp_in_A, sp_out_A = run_trial(W_r, pattern_id=0)
    _, _, sp_in_B, sp_out_B = run_trial(W_r, pattern_id=1)
    # Input raster (first 20 neurons)
    for nid in range(20):
        ts = np.where(sp_in_A[:, nid])[0]
        if len(ts):
            ax.scatter(ts, [nid]*len(ts), s=8, c='#4e79a7', marker='|')
        ts = np.where(sp_in_B[:, nid])[0]
        if len(ts):
            ax.scatter(ts + T_STEPS + 3, [nid]*len(ts), s=8, c='#e15759', marker='|')
    ax.axvline(T_STEPS + 1.5, color='gray', lw=1, linestyle='--')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Input Neuron')
    ax.set_title('Input Spike Patterns (Blue=A, Red=B)')
    ax.set_facecolor('#fafafa')

    # Final accuracy bar chart
    ax = fig.add_subplot(gs[1, 2])
    labels = ['R-STDP\n(with dopamine)', 'Vanilla STDP\n(no reward)', 'Chance']
    vals   = [acc_final_r * 100, acc_final_v * 100, 50.0]
    colors = ['#e15759', '#4e79a7', '#999999']
    bars   = ax.bar(labels, vals, color=colors, alpha=0.85, edgecolor='white')
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center', fontsize=10, fontweight='bold')
    ax.set_ylabel('Classification Accuracy (%)')
    ax.set_title('Final Accuracy Comparison')
    ax.set_ylim(0, 115)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_facecolor('#fafafa')

    fig.suptitle('Reward-Modulated STDP – Biologically Plausible Reinforcement Learning',
                 fontsize=13, fontweight='bold')
    plt.savefig(OUTPUT_DIR / 'reward_stdp.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\nPlot saved → Learning/reward_stdp.png")

