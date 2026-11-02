"""
Neuromorphic AI Engine
Module: Memory Consolidation (Sleep-like Replay)

Biological basis:
  During wakefulness, synapses are strengthened indiscriminately
  (synaptic homeostasis hypothesis). During slow-wave sleep, the
  hippocampus replays recent experiences, allowing the neocortex to
  consolidate important memories while pruning weak ones.

  This solves *catastrophic forgetting* — the tendency of neural
  networks to lose old knowledge when learning new information.

  Mechanism:
    Awake phase  → online STDP, fast learning, weights noisy
    Sleep phase  → replay buffer sampled, weights stabilised
                   strong synapses consolidated, weak ones pruned
                   homeostatic scaling keeps activity in range

  This module demonstrates:
    1. Catastrophic forgetting in a naive SNN (no sleep)
    2. Memory consolidation prevents forgetting (with sleep)
    3. Synaptic homeostasis maintains stable firing rates

Reference:
  Tononi, G. & Cirelli, C. (2014). Sleep and the price of plasticity.
  Neuron.
  McClelland, J. et al. (1995). Why there are complementary learning
  systems. Psychological Review.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import deque
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent

np.random.seed(7)

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
N_IN        = 60
N_HID       = 60    # Larger hidden layer: more neurons available per pattern
T_STEPS     = 25
TAU_M       = 0.88
V_TH        = 4.0   # Higher threshold: prevents global saturation, forces selectivity

A_PLUS      =  0.006   # Slightly stronger LTP for faster selectivity
A_MINUS     = -0.004
TAU_STDP    = 12       # Shorter time window → sharper selectivity

# Sleep / consolidation
REPLAY_BUFFER_SIZE = 500
SLEEP_CYCLES       = 15   # More replay cycles for stronger consolidation
SLEEP_EVERY        = 60   # Sleep more frequently
PRUNE_THRESHOLD    = 0.015
HOMEO_TARGET       = 0.25
HOMEO_LR           = 0.003

W_MAX = 1.0
W_MIN = 0.0

N_PATTERNS   = 3
N_TRIALS_PER = 400
N_TEST       = 60


# ---------------------------------------------------------------------------
# Spike pattern generator (3 distinct patterns)
# ---------------------------------------------------------------------------
def make_pattern(pid: int, n: int = N_IN, T: int = T_STEPS,
                 rate: float = 0.5) -> np.ndarray:
    """
    3 non-overlapping patterns — each activates a distinct 1/3 of neurons.
    High signal-to-noise ratio (active 0.5 vs background 0.005) ensures
    STDP reliably differentiates patterns; critical for consolidation demo.
    """
    sp = (np.random.rand(T, n) < 0.005).astype(np.float32)  # very sparse background
    seg   = n // N_PATTERNS
    start = pid * seg
    end   = start + seg
    sp[:, start:end] = (np.random.rand(T, seg) < rate).astype(np.float32)
    return sp


# ---------------------------------------------------------------------------
# One STDP step — returns (spike_out, dW, mean_firing_rate)
# ---------------------------------------------------------------------------
def stdp_step(W: np.ndarray, sp_in: np.ndarray):
    V       = np.zeros(N_HID, dtype=np.float32)
    sp_out  = np.zeros((T_STEPS, N_HID), dtype=np.float32)
    dw      = np.zeros_like(W)
    t_pre   = -999.0 * np.ones(N_IN)
    t_post  = -999.0 * np.ones(N_HID)

    for t in range(T_STEPS):
        V = TAU_M * V + sp_in[t] @ W
        fired = V >= V_TH
        V[fired] = 0.0
        sp_out[t, fired] = 1.0
        pre  = sp_in[t].astype(bool)
        post = sp_out[t].astype(bool)

        # LTP: post_j just fired — strengthen (pre_i → post_j) where pre fired
        # before post (dt > 0 means pre fired in the past)
        if post.any():
            dt  = t - t_pre                               # (N_IN,)
            ltp = np.where(dt > 0, A_PLUS * np.exp(-dt / TAU_STDP), 0.0)
            dw[:, post] += ltp[:, np.newaxis]

        # LTD: pre_i just fired — weaken (pre_i → post_j) where post fired
        # before pre (dt > 0 means post fired in the past)
        if pre.any():
            dt  = t - t_post                              # (N_HID,)
            ltd = np.where(dt > 0, A_MINUS * np.exp(-dt / TAU_STDP), 0.0)
            dw[pre, :] += ltd[np.newaxis, :]

        t_pre[pre]   = t
        t_post[post] = t

    return sp_out, dw, sp_out.mean()


# ---------------------------------------------------------------------------
# Sleep / Consolidation phase
# ---------------------------------------------------------------------------
def sleep_consolidation(W: np.ndarray, replay_buffer: deque,
                         prune: bool = True) -> np.ndarray:
    """
    Replay stored experiences, stabilise weights, prune weak synapses.

    Three operations happen during sleep:
      1. Replay: run stored spike patterns through network, apply STDP
         → strengthens important memories
      2. Pruning: zero out synapses below threshold
         → removes noise, frees capacity
      3. Homeostatic scaling: normalise weight rows so firing rate
         stays near target → prevents saturation / silence
    """
    if len(replay_buffer) == 0:
        return W

    for _ in range(SLEEP_CYCLES):
        # Sample a random experience from the buffer
        sp_in = replay_buffer[np.random.randint(len(replay_buffer))]
        _, dw, _ = stdp_step(W, sp_in)
        # Apply with reduced learning rate (consolidation is gentle)
        W = np.clip(W + dw * 0.2, W_MIN, W_MAX)

    # Pruning: remove weak synapses
    if prune:
        W[W < PRUNE_THRESHOLD] = 0.0

    # Homeostatic scaling: scale each post-synaptic neuron's
    # incoming weights so expected firing rate stays at target
    for j in range(N_HID):
        w_sum = W[:, j].sum()
        if w_sum > 0:
            current_rate  = w_sum * 0.15    # rough estimate
            scale         = 1.0 + HOMEO_LR * (HOMEO_TARGET - current_rate)
            W[:, j]      *= np.clip(scale, 0.5, 2.0)

    return np.clip(W, W_MIN, W_MAX)


# ---------------------------------------------------------------------------
# Evaluation: accuracy on all patterns seen so far
# ---------------------------------------------------------------------------
def build_neuron_class(W: np.ndarray, patterns_seen: list,
                       n_vote: int = 30) -> np.ndarray:
    """
    Assign each hidden neuron to the pattern it responds to most strongly
    using a WTA vote over held-out samples.  Returns neuron_class (N_HID,).
    """
    vote = np.zeros((N_PATTERNS, N_HID), dtype=np.float32)
    for pid in patterns_seen:
        for _ in range(n_vote):
            sp_in = make_pattern(pid)
            V     = np.zeros(N_HID, dtype=np.float32)
            sp_out = np.zeros((T_STEPS, N_HID), dtype=np.float32)
            for t in range(T_STEPS):
                V = TAU_M * V + sp_in[t] @ W
                fired = V >= V_TH; V[fired] = 0.0
                sp_out[t, fired] = 1.0
            vote[pid] += sp_out.sum(axis=0)
    return np.argmax(vote, axis=0)   # shape (N_HID,)


def evaluate(W: np.ndarray, patterns_seen: list, n_test: int = N_TEST):
    """
    Evaluate using a learned WTA readout:
      1. Build neuron→class assignment via max-response voting.
      2. For each test sample, fire the network, find the winner neuron,
         and predict its assigned class.
    This is more robust than a fixed-segment readout because it does not
    assume any spatial organisation of hidden neurons.
    """
    neuron_class = build_neuron_class(W, patterns_seen)

    accs = []
    for pid in patterns_seen:
        correct = 0
        for _ in range(n_test):
            sp_in = make_pattern(pid)
            V     = np.zeros(N_HID, dtype=np.float32)
            sp_out = np.zeros((T_STEPS, N_HID), dtype=np.float32)
            for t in range(T_STEPS):
                V = TAU_M * V + sp_in[t] @ W
                fired = V >= V_TH; V[fired] = 0.0
                sp_out[t, fired] = 1.0
            spike_counts = sp_out.sum(axis=0)
            winner = int(np.argmax(spike_counts))
            pred   = int(neuron_class[winner])
            correct += int(pred == pid)
        accs.append(correct / n_test)
    return accs


if __name__ == '__main__':
    print("=" * 60)
    print("  Memory Consolidation – Sleep-based Replay")
    print("=" * 60)

    results = {}

    for use_sleep in [False, True]:
        label    = "With Sleep" if use_sleep else "No Sleep"
        W        = np.random.uniform(0.05, 0.25, (N_IN, N_HID)).astype(np.float32)
        buffer   = deque(maxlen=REPLAY_BUFFER_SIZE)
        history  = []   # [(trial, pattern_block, accs_on_all_seen)]
        trial_global = 0

        print(f"\n[{label}]")
        for pid in range(N_PATTERNS):
            print(f"  Learning pattern {pid}...")
            for t in range(N_TRIALS_PER):
                sp_in = make_pattern(pid)
                _, dw, _ = stdp_step(W, sp_in)
                W = np.clip(W + dw * 0.2, W_MIN, W_MAX)
                buffer.append(sp_in)
                trial_global += 1

                # Sleep every N trials
                if use_sleep and trial_global % SLEEP_EVERY == 0:
                    W = sleep_consolidation(W, buffer, prune=True)

                # Evaluate periodically
                if t % 50 == 49:
                    accs = evaluate(W, list(range(pid + 1)))
                    history.append((trial_global, pid, accs))

            accs_final = evaluate(W, list(range(pid + 1)))
            print(f"    After pattern {pid}: " +
                  " | ".join(f"P{p}={a*100:.0f}%" for p, a in enumerate(accs_final)))

        results[label] = history

    # ---------------------------------------------------------------------------
    # Visualisation
    # ---------------------------------------------------------------------------
    fig = plt.figure(figsize=(16, 12))
    gs  = gridspec.GridSpec(2, 3, hspace=0.4, wspace=0.35)

    colors_pat = ['#4e79a7', '#f28e2b', '#e15759']

    # Accuracy over time for each pattern — No Sleep
    ax = fig.add_subplot(gs[0, 0])
    hist = results["No Sleep"]
    for pid in range(N_PATTERNS):
        xs, ys = [], []
        for (trial, block, accs) in hist:
            if pid < len(accs):
                xs.append(trial); ys.append(accs[pid] * 100)
        if xs:
            ax.plot(xs, ys, color=colors_pat[pid], lw=1.5, label=f'Pattern {pid}')
    # Shade learning blocks
    for pid in range(N_PATTERNS):
        ax.axvspan(pid*N_TRIALS_PER, (pid+1)*N_TRIALS_PER,
                   alpha=0.06, color=colors_pat[pid])
    ax.set_xlabel('Trial'); ax.set_ylabel('Accuracy (%)')
    ax.set_title('No Sleep – Catastrophic Forgetting')
    ax.legend(fontsize=9); ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3); ax.set_facecolor('#fafafa')

    # Accuracy over time — With Sleep
    ax = fig.add_subplot(gs[0, 1])
    hist = results["With Sleep"]
    for pid in range(N_PATTERNS):
        xs, ys = [], []
        for (trial, block, accs) in hist:
            if pid < len(accs):
                xs.append(trial); ys.append(accs[pid] * 100)
        if xs:
            ax.plot(xs, ys, color=colors_pat[pid], lw=1.5, label=f'Pattern {pid}')
    for pid in range(N_PATTERNS):
        ax.axvspan(pid*N_TRIALS_PER, (pid+1)*N_TRIALS_PER,
                   alpha=0.06, color=colors_pat[pid])
    # Mark sleep events
    sleep_trials = list(range(SLEEP_EVERY, N_PATTERNS*N_TRIALS_PER, SLEEP_EVERY))
    for st in sleep_trials:
        ax.axvline(st, color='navy', alpha=0.15, lw=0.8)
    ax.set_xlabel('Trial'); ax.set_ylabel('Accuracy (%)')
    ax.set_title('With Sleep – Memory Preserved')
    ax.legend(fontsize=9); ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3); ax.set_facecolor('#fafafa')

    # Final accuracy comparison
    ax = fig.add_subplot(gs[0, 2])
    W_no_sleep    = np.random.uniform(0.05, 0.25, (N_IN, N_HID)).astype(np.float32)
    W_with_sleep  = np.random.uniform(0.05, 0.25, (N_IN, N_HID)).astype(np.float32)
    buf2 = deque(maxlen=REPLAY_BUFFER_SIZE)

    for pid in range(N_PATTERNS):
        for _ in range(N_TRIALS_PER):
            sp  = make_pattern(pid)
            _, dw, _ = stdp_step(W_no_sleep, sp); W_no_sleep = np.clip(W_no_sleep + dw*0.2, 0, 1)
            _, dw, _ = stdp_step(W_with_sleep, sp); W_with_sleep = np.clip(W_with_sleep + dw*0.2, 0, 1)
            buf2.append(sp)
        if pid < N_PATTERNS - 1:
            W_with_sleep = sleep_consolidation(W_with_sleep, buf2)

    acc_no   = evaluate(W_no_sleep,   list(range(N_PATTERNS)))
    acc_with = evaluate(W_with_sleep, list(range(N_PATTERNS)))

    x      = np.arange(N_PATTERNS)
    width  = 0.35
    bars1  = ax.bar(x - width/2, [a*100 for a in acc_no],
                    width, color='#e15759', alpha=0.85, label='No Sleep')
    bars2  = ax.bar(x + width/2, [a*100 for a in acc_with],
                    width, color='#59a14f', alpha=0.85, label='With Sleep')
    ax.set_xticks(x); ax.set_xticklabels([f'Pattern {p}' for p in range(N_PATTERNS)])
    ax.set_ylabel('Final Accuracy (%)')
    ax.set_title('Final Accuracy on All Patterns')
    ax.legend(); ax.set_ylim(0, 115)
    ax.grid(True, alpha=0.3, axis='y'); ax.set_facecolor('#fafafa')

    # Spike pattern visualisation
    ax = fig.add_subplot(gs[1, 0])
    for pid in range(N_PATTERNS):
        sp = make_pattern(pid)
        offset = pid * (N_IN + 5)
        for nid in range(N_IN):
            ts = np.where(sp[:, nid])[0]
            if len(ts):
                ax.scatter(ts, [nid + offset]*len(ts),
                           s=3, c=colors_pat[pid], marker='|')
    ax.set_xlabel('Timestep'); ax.set_ylabel('Neuron (stacked)')
    ax.set_title('The 3 Patterns to Learn\n(colour = pattern ID)')
    ax.set_facecolor('#fafafa')

    # Synaptic weight distribution: before/after sleep
    ax = fig.add_subplot(gs[1, 1])
    W_before = np.random.uniform(0.05, 0.4, (N_IN, N_HID)).astype(np.float32)
    W_after  = sleep_consolidation(W_before.copy(),
                                    deque([make_pattern(p) for p in range(N_PATTERNS)] * 20,
                                          maxlen=REPLAY_BUFFER_SIZE))
    ax.hist(W_before.flatten(), bins=40, alpha=0.6, color='#e15759',
            label='Before sleep', density=True)
    ax.hist(W_after.flatten(),  bins=40, alpha=0.6, color='#59a14f',
            label='After sleep',  density=True)
    ax.set_xlabel('Weight value'); ax.set_ylabel('Density')
    ax.set_title('Weight Distribution Before/After Sleep\n(weak synapses pruned)')
    ax.legend(); ax.grid(True, alpha=0.3); ax.set_facecolor('#fafafa')

    # Summary: forgetting rates
    ax = fig.add_subplot(gs[1, 2])
    forget_no   = [max(0, 0.9 - acc_no[p])   for p in range(N_PATTERNS-1)]
    forget_with = [max(0, 0.9 - acc_with[p]) for p in range(N_PATTERNS-1)]
    x2     = np.arange(N_PATTERNS - 1)
    ax.bar(x2 - 0.2, [f*100 for f in forget_no],   0.35,
           color='#e15759', alpha=0.85, label='No Sleep')
    ax.bar(x2 + 0.2, [f*100 for f in forget_with], 0.35,
           color='#59a14f', alpha=0.85, label='With Sleep')
    ax.set_xticks(x2)
    ax.set_xticklabels([f'Pattern {p}' for p in range(N_PATTERNS-1)])
    ax.set_ylabel('Forgetting (%)')
    ax.set_title('Catastrophic Forgetting Reduction')
    ax.legend(); ax.grid(True, alpha=0.3, axis='y'); ax.set_facecolor('#fafafa')

    fig.suptitle('Memory Consolidation – Sleep-based Replay Prevents Catastrophic Forgetting',
                 fontsize=13, fontweight='bold')
    plt.savefig(OUTPUT_DIR / 'memory_consolidation.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\nPlot saved → Learning/memory_consolidation.png")

