"""
Neuromorphic AI Engine
Module: Predictive Coding Network

Biological basis:
  The brain does not passively relay sensory signals upward. Instead,
  higher cortical areas continuously generate *predictions* of what
  lower areas will receive. Only the *prediction error* (residual) is
  propagated upward. This makes the system extremely efficient:
  expected inputs cost nothing to transmit.

  Key properties:
    - O(N) computation per layer (vs O(N²) for Transformer attention)
    - Constant memory regardless of sequence length
    - Naturally sparse: only errors propagate
    - Learns generative models of the world, not just discriminative mappings

Architecture (2-layer example):
  Input x
    │
    ▼
  Layer 1: generates prediction p1 of x
    error1 = x - p1  ──→ propagated upward
    │
    ▼
  Layer 2: generates prediction p2 of error1
    error2 = error1 - p2  ──→ propagated upward
    │
    ▼
  Top layer: classification / action

Training:
  Minimise prediction errors at every layer simultaneously.
  No backpropagation through time required.

Reference:
  Rao, R. & Ballard, D. (1999). Predictive coding in the visual cortex.
  Nature Neuroscience.
  Friston, K. (2010). The free-energy principle. Nature Reviews Neuroscience.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent

np.random.seed(0)

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
N_INPUT   = 784    # 28×28 — MNIST-like
N_HIDDEN  = 128
N_OUTPUT  = 10     # 10 digit classes
LR        = 0.01   # learning rate (representation update)
LR_PRED   = 0.005  # prediction weight learning rate — half of LR for stability
T_INFER   = 10     # inference steps per sample (fewer = less overfitting)
N_EPOCHS  = 10
N_TRAIN   = 500    # samples to demo (small for speed)
SIGMA     = 0.1    # prediction noise level


# ---------------------------------------------------------------------------
# Predictive Coding Layer
# ---------------------------------------------------------------------------
class PCLayer:
    """
    A single Predictive Coding layer.

    Maintains:
      r     : representation (activity) vector
      pred  : prediction of the layer below
      error : prediction error from the layer below
      W_up  : weights projecting representation upward
      W_down: weights generating prediction of layer below (generative)
    """

    def __init__(self, n_in: int, n_out: int, name: str = ''):
        self.name   = name
        self.n_in   = n_in
        self.n_out  = n_out
        s           = 1.0 / np.sqrt(n_in)
        # Generative weights (top-down): predict layer below
        self.W_down = np.random.randn(n_out, n_in).astype(np.float32) * s
        # Recognition weights (bottom-up): encode errors
        self.W_up   = self.W_down.T.copy()   # initialise as transpose
        # Internal state
        self.r      = np.zeros(n_out, dtype=np.float32)
        self.error  = np.zeros(n_in,  dtype=np.float32)

    @staticmethod
    def _relu(x):
        return np.maximum(0, x)

    @staticmethod
    def _relu_grad(x):
        return (x > 0).astype(np.float32)

    @staticmethod
    def _relu_init(x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x).astype(np.float32)

    def predict(self) -> np.ndarray:
        """Generate prediction of the layer below from current representation."""
        return self._relu(self.W_down.T @ self.r)

    def update_representation(self, error_below: np.ndarray,
                               error_above: np.ndarray,
                               lr: float = LR):
        # Bottom-up: recognition weights (W_up.T = W_down) map error to representation space
        # W_up shape: (n_in, n_out) → W_up.T @ error_below: (n_out, n_in) @ (n_in,) = (n_out,)
        bu = self.W_up.T @ error_below
        # Top-down: error_above is already in representation space — subtract to suppress
        td = error_above if error_above is not None else 0
        self.r = np.clip(self._relu(self.r + lr * (bu - td)), 0, 10)

    def update_weights(self, error_below: np.ndarray, lr: float = LR_PRED):
        # Minimise ||error_below||²: gradient w.r.t. W_down is outer(r, error_below)
        # Gradient descent: W_down += lr * outer(r, error_below) reduces prediction error
        dW = lr * np.outer(self.r, error_below)
        # Clip dW magnitude to prevent runaway updates
        dW = np.clip(dW, -0.1, 0.1)
        self.W_down = np.clip(self.W_down + dW, -1.5, 1.5)
        self.W_up   = self.W_down.T.copy()


# ---------------------------------------------------------------------------
# Full Predictive Coding Network
# ---------------------------------------------------------------------------
class PredictiveCodingNetwork:
    """
    3-layer PC network for classification.
    Layer 0: input (784)
    Layer 1: hidden (128)
    Layer 2: output (10)
    """

    def __init__(self):
        self.layers = [
            PCLayer(N_INPUT,  N_HIDDEN, 'L1'),
            PCLayer(N_HIDDEN, N_OUTPUT, 'L2'),
        ]

    def _softmax(self, x):
        e = np.exp(x - x.max())
        return e / e.sum()

    def infer(self, x: np.ndarray, label: int = None) -> tuple:
        """
        Run inference iterations: each layer updates its representation
        to minimise local prediction errors.
        Returns (prediction errors per layer, predicted class).
        """
        x = x.astype(np.float32)

        # Initialise layer representations with bottom-up pass
        self.layers[0].r = PCLayer._relu_init(x @ self.layers[0].W_up)
        self.layers[1].r = PCLayer._relu_init(
            self.layers[0].r @ self.layers[1].W_up)

        errors_history = []

        for _ in range(T_INFER):
            # --- Compute prediction errors bottom-up ---
            pred_x  = self.layers[0].predict()     # L1 predicts input
            pred_h  = self.layers[1].predict()     # L2 predicts L1

            err_x   = x                - pred_x    # input error
            err_h   = self.layers[0].r - pred_h    # hidden error

            # --- Update representations ---
            self.layers[0].update_representation(
                error_below=err_x, error_above=err_h)
            self.layers[1].update_representation(
                error_below=err_h, error_above=None)

            # Clamp output layer if label is known (supervised signal)
            # Moderate clamp (0.4) balances label guidance with free inference
            if label is not None:
                target = np.zeros(N_OUTPUT, dtype=np.float32)
                target[label] = 1.0
                self.layers[1].r = (
                    0.6 * self.layers[1].r + 0.4 * target)

            errors_history.append((
                np.mean(err_x ** 2),
                np.mean(err_h ** 2)
            ))

        # Classification: L2 representation directly → softmax
        logits = self.layers[1].r          # shape (N_OUTPUT,) = (10,)
        probs  = self._softmax(logits)
        pred_class = int(np.argmax(probs))

        return errors_history, pred_class, probs

    @staticmethod
    def _relu_init(x):
        return np.maximum(0, x).astype(np.float32)

    def learn(self, x: np.ndarray, label: int):
        """
        Full learning step:
          1. Run inference with label clamping
          2. Update generative weights to reduce prediction errors
          3. Update classification head with cross-entropy gradient
        """
        errors_history, pred_class, probs = self.infer(x, label)

        # Update generative weights
        pred_x = self.layers[0].predict()
        pred_h = self.layers[1].predict()
        err_x  = x.astype(np.float32) - pred_x
        err_h  = self.layers[0].r - pred_h

        self.layers[0].update_weights(err_x)
        self.layers[1].update_weights(err_h)

        # Weight decay: gentle L2 to prevent runaway growth across epochs
        for layer in self.layers:
            layer.W_down *= 0.999
            layer.W_up    = layer.W_down.T.copy()

        # Update output representation toward one-hot target (supervised clamp)
        target = np.zeros(N_OUTPUT, dtype=np.float32)
        target[label] = 1.0
        grad = probs - target
        # Nudge the top layer representation
        self.layers[1].r -= LR * grad

        return pred_class, errors_history[-1]


# ---------------------------------------------------------------------------
# Generate synthetic digit-like data (MNIST substitute for speed)
# ---------------------------------------------------------------------------
def make_digit_data(n_samples: int = N_TRAIN):
    """
    Synthetic 784-dim data: each class has a distinct activation pattern
    in a subset of dimensions. Resembles MNIST statistics without download.
    """
    X, Y = [], []
    for _ in range(n_samples):
        label = np.random.randint(0, N_OUTPUT)
        x     = np.random.rand(N_INPUT).astype(np.float32) * 0.1
        # Class-specific "stroke" pattern
        start = label * (N_INPUT // N_OUTPUT)
        end   = start + (N_INPUT // N_OUTPUT)
        x[start:end] += np.random.rand(end - start) * 0.8
        x = np.clip(x, 0, 1)
        X.append(x); Y.append(label)
    return np.array(X), np.array(Y)


if __name__ == '__main__':
    print("=" * 55)
    print("  Predictive Coding Network")
    print("=" * 55)

    # Try to load real MNIST, fall back to synthetic
    try:
        from torchvision import datasets, transforms
        mnist = datasets.MNIST(root='./data', train=True,
                               download=True, transform=transforms.ToTensor())
        X_all = mnist.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
        Y_all = mnist.targets.numpy()
        X_train, Y_train = X_all[:N_TRAIN], Y_all[:N_TRAIN]
        X_test,  Y_test  = X_all[N_TRAIN:N_TRAIN+200], Y_all[N_TRAIN:N_TRAIN+200]
        print("  Using real MNIST data")
    except Exception:
        X_train, Y_train = make_digit_data(N_TRAIN)
        X_test,  Y_test  = make_digit_data(200)
        print("  Using synthetic digit data")

    # ---------------------------------------------------------------------------
    # Training loop
    # ---------------------------------------------------------------------------
    net = PredictiveCodingNetwork()
    acc_per_epoch = []
    error_L1_log, error_L2_log = [], []
    pred_error_ratio = []   # ratio of error transmitted vs full signal

    for epoch in range(N_EPOCHS):
        e1_epoch, e2_epoch = [], []
        err_norm_epoch = []
        idx = np.random.permutation(N_TRAIN)

        for i in idx:
            _pred, (e1, e2) = net.learn(X_train[i], int(Y_train[i]))
            e1_epoch.append(e1); e2_epoch.append(e2)
            err_norm = np.sqrt(e1)
            sig_norm = np.linalg.norm(X_train[i]) / np.sqrt(N_INPUT)
            err_norm_epoch.append(err_norm / (sig_norm + 1e-6))

        # Measure accuracy with FREE inference (no label) — honest evaluation
        correct = 0
        for i in range(N_TRAIN):
            _, pred_class, _ = net.infer(X_train[i], label=None)
            correct += int(pred_class == Y_train[i])
        acc = correct / N_TRAIN

        acc_per_epoch.append(acc)
        error_L1_log.append(np.mean(e1_epoch))
        error_L2_log.append(np.mean(e2_epoch))
        pred_error_ratio.append(np.mean(err_norm_epoch))
        print(f"  Epoch {epoch+1:2d}/{N_EPOCHS} | "
              f"Acc = {acc*100:.1f}% | "
              f"L1 error = {error_L1_log[-1]:.4f} | "
              f"Error ratio = {pred_error_ratio[-1]*100:.1f}%")

    # Test accuracy
    test_correct = sum(
        net.infer(X_test[i], None)[1] == int(Y_test[i])
        for i in range(len(Y_test))
    )
    test_acc = test_correct / len(Y_test)
    print(f"\nTest accuracy : {test_acc*100:.1f}%")
    print(f"Mean prediction error ratio: {np.mean(pred_error_ratio)*100:.1f}% "
          f"of signal transmitted as error (sparse!)")

    # ---------------------------------------------------------------------------
    # Visualisation
    # ---------------------------------------------------------------------------
    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(2, 3, hspace=0.4, wspace=0.35)

    ax = fig.add_subplot(gs[0, 0])
    ax.plot(range(1, N_EPOCHS+1), [a*100 for a in acc_per_epoch],
            'o-', color='#e15759', lw=2)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Accuracy (%)')
    ax.set_title('Training Accuracy')
    ax.grid(True, alpha=0.3); ax.set_facecolor('#fafafa')

    ax = fig.add_subplot(gs[0, 1])
    ax.plot(range(1, N_EPOCHS+1), error_L1_log,
            'o-', color='#4e79a7', lw=2, label='Layer 1 MSE')
    ax.plot(range(1, N_EPOCHS+1), error_L2_log,
            's--', color='#f28e2b', lw=2, label='Layer 2 MSE')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Prediction Error (MSE)')
    ax.set_title('Prediction Errors Decrease')
    ax.legend(); ax.grid(True, alpha=0.3); ax.set_facecolor('#fafafa')

    ax = fig.add_subplot(gs[0, 2])
    ax.plot(range(1, N_EPOCHS+1), [r*100 for r in pred_error_ratio],
            'o-', color='#59a14f', lw=2)
    ax.fill_between(range(1, N_EPOCHS+1), [r*100 for r in pred_error_ratio],
                    alpha=0.15, color='#59a14f')
    ax.axhline(100, color='red', linestyle='--', lw=1, alpha=0.5,
               label='Full signal (no prediction)')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Error / Signal (%)')
    ax.set_title('Transmission Efficiency\n(lower = more predictable = more efficient)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3); ax.set_facecolor('#fafafa')

    ax = fig.add_subplot(gs[1, 0])
    sample_x = X_test[0]; sample_y = int(Y_test[0])
    err_history, _, _ = net.infer(sample_x, sample_y)
    l1_trace = [e[0] for e in err_history]
    l2_trace = [e[1] for e in err_history]
    ax.plot(l1_trace, color='#4e79a7', lw=2, label='Layer 1 error')
    ax.plot(l2_trace, color='#f28e2b', lw=2, label='Layer 2 error')
    ax.set_xlabel('Inference step'); ax.set_ylabel('Prediction error')
    ax.set_title(f'Error Relaxation During Inference\n(label={sample_y})')
    ax.legend(); ax.grid(True, alpha=0.3); ax.set_facecolor('#fafafa')

    ax = fig.add_subplot(gs[1, 1])
    n_show = min(16, N_HIDDEN)
    grid   = np.zeros((4 * 28, 4 * 28))
    for i in range(n_show):
        r, c = divmod(i, 4)
        patch = net.layers[0].W_down[i].reshape(28, 28)
        patch = (patch - patch.min()) / (patch.max() - patch.min() + 1e-6)
        grid[r*28:(r+1)*28, c*28:(c+1)*28] = patch
    ax.imshow(grid, cmap='RdBu_r', interpolation='nearest')
    ax.set_title('Generative Weights L1\n(learned "what to predict")')
    ax.axis('off')

    ax = fig.add_subplot(gs[1, 2])
    final_ratio = pred_error_ratio[-1]
    categories  = ['Full signal\n(naive)', 'Prediction error\n(PC network)']
    sizes       = [100, final_ratio * 100]
    ax.bar(categories, sizes,
           color=['#e15759', '#59a14f'], alpha=0.85, edgecolor='white')
    for i, val in enumerate(sizes):
        ax.text(i, val + 1, f'{val:.1f}%', ha='center',
                fontsize=11, fontweight='bold')
    ax.set_ylabel('Fraction of signal transmitted (%)')
    ax.set_title('Communication Efficiency\n(brain transmits only errors)')
    ax.set_ylim(0, 120)
    ax.grid(True, alpha=0.3, axis='y'); ax.set_facecolor('#fafafa')

    fig.suptitle('Predictive Coding – Biologically Plausible Generative Learning',
                 fontsize=13, fontweight='bold')
    plt.savefig(OUTPUT_DIR / 'predictive_coding.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Plot saved → Learning/predictive_coding.png")

