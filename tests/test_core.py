"""
NeuroSpike-Engine – Core Unit Tests
====================================
Run from the project root:
    python -m pytest tests/ -v

NOTE: All modules use module-level execution (plots, training loops).
We import only the pure functions/classes we need via importlib to avoid
triggering side effects (matplotlib rendering, heavy training).
"""

import sys
import types
import importlib
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Make sure project root is on the path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def _load_functions_only(module_path: str, attr_names: list):
    """
    Load only function/class definitions from a module without executing
    the module-level training/plotting code.

    Strategy: AST-strip all top-level statements that are not definitions,
    imports, or simple constant assignments. Then exec the stripped tree
    with matplotlib mocked to avoid any GUI dependency.
    """
    import ast
    import matplotlib
    matplotlib.use('Agg')

    source = (ROOT / module_path).read_text(encoding='utf-8')
    tree   = ast.parse(source)

    # Nodes to keep: defs, classes, imports, and simple value assignments
    # (needed for hyperparameter constants).  Everything else (for loops,
    # print statements, plt.* calls, training loops) is dropped.
    KEEP_TYPES = (
        ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
        ast.Import, ast.ImportFrom,
        ast.Assign, ast.AugAssign, ast.AnnAssign,
    )

    def _is_safe_assign(node):
        """Keep assignments whose RHS does not call plt or trigger heavy work."""
        if not isinstance(node, ast.Assign):
            return True
        # Reject assignments that call functions (could trigger side effects)
        for child in ast.walk(node.value):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Attribute):
                    if isinstance(child.func.value, ast.Name):
                        if child.func.value.id in ('plt', 'gridspec'):
                            return False
        return True

    keep = [n for n in tree.body
            if isinstance(n, KEEP_TYPES) and _is_safe_assign(n)]

    new_tree = ast.Module(body=keep, type_ignores=[])
    ast.fix_missing_locations(new_tree)
    code = compile(new_tree, str(ROOT / module_path), 'exec')

    ns: dict = {'__file__': str(ROOT / module_path)}
    exec(code, ns)
    return {name: ns[name] for name in attr_names if name in ns}


# ===========================================================================
# LIF Neuron
# ===========================================================================
class TestLIF:
    """Tests for the simulate_lif function."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        ns = _load_functions_only('LIF/lif_neuron.py',
                                  ['simulate_lif', 'N', 'DT', 'T', 't',
                                   'TAU', 'V_REST', 'V_TH', 'V_RESET',
                                   'R_MEM', 'T_REF'])
        self.simulate_lif = ns['simulate_lif']
        self.N  = ns['N']
        self.DT = ns['DT']
        self.T  = ns['T']
        self.t  = ns['t']

    def test_no_spike_below_threshold(self):
        """Sub-threshold current must produce zero spikes."""
        V, spikes = self.simulate_lif(np.full(self.N, 1.0))
        assert len(spikes) == 0, "Expected no spikes for sub-threshold current"

    def test_spikes_above_threshold(self):
        """Super-threshold current must produce at least one spike."""
        V, spikes = self.simulate_lif(np.full(self.N, 3.0))
        assert len(spikes) > 0, "Expected spikes for super-threshold current"

    def test_refractory_period(self):
        """Minimum inter-spike interval must be >= t_ref (2 ms)."""
        T_REF = 2.0e-3
        V, spikes = self.simulate_lif(np.full(self.N, 4.0))
        if len(spikes) >= 2:
            isis = np.diff(spikes)
            assert np.all(isis >= T_REF - self.DT * 2), \
                f"ISI below refractory period: min ISI = {isis.min()*1e3:.2f} ms"

    def test_voltage_bounded(self):
        """Membrane potential must never exceed action potential peak (40 mV)."""
        V, _ = self.simulate_lif(np.full(self.N, 5.0))
        assert V.max() <= 40.1, f"Voltage exceeded AP peak: {V.max():.1f} mV"

    def test_monotone_firing_rate(self):
        """Firing rate must be non-decreasing with increasing input current."""
        currents = [1.5, 2.0, 2.5, 3.0, 4.0]
        rates = []
        for I in currents:
            _, spikes = self.simulate_lif(np.full(self.N, I))
            rates.append(len(spikes))
        assert rates == sorted(rates), \
            f"Firing rates not monotone: {rates}"

    def test_zero_input_no_spike(self):
        """Zero input current must produce no spikes."""
        V, spikes = self.simulate_lif(np.zeros(self.N))
        assert len(spikes) == 0


# ===========================================================================
# STDP weight update
# ===========================================================================
class TestSTDP:
    """Tests for the corrected STDP rule."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        ns = _load_functions_only('SNN_STDP/snn_stdp.py',
                                  ['stdp_update', 'lif_forward', 'poisson_encode',
                                   'W_MIN', 'W_MAX', 'N_INPUT', 'N_HIDDEN',
                                   'TAU_M', 'V_TH', 'V_RST'])
        self.stdp_update    = ns['stdp_update']
        self.lif_forward    = ns['lif_forward']
        self.poisson_encode = ns['poisson_encode']
        self.W_MIN = ns['W_MIN']
        self.W_MAX = ns['W_MAX']

    def test_ltp_increases_weights(self):
        """Paired pre-then-post spikes must produce net LTP (dw > 0)."""
        T, N_IN, N_OUT = 20, 4, 4
        np.random.seed(0)
        W = np.ones((N_IN, N_OUT), dtype=np.float32) * 0.5
        # Construct deterministic spike trains: pre fires at t=5, post at t=10
        sp_in  = np.zeros((T, N_IN),  dtype=np.float32)
        sp_out = np.zeros((T, N_OUT), dtype=np.float32)
        sp_in[5,  :] = 1.0   # all pre neurons fire at t=5
        sp_out[10, :] = 1.0  # all post neurons fire at t=10
        dw = self.stdp_update(sp_in, sp_out, W)
        assert dw.mean() > 0, f"Expected LTP (dw>0), got mean dw={dw.mean():.6f}"

    def test_ltd_decreases_weights(self):
        """Paired post-then-pre spikes must produce net LTD (dw < 0)."""
        T, N_IN, N_OUT = 20, 4, 4
        W = np.ones((N_IN, N_OUT), dtype=np.float32) * 0.5
        sp_in  = np.zeros((T, N_IN),  dtype=np.float32)
        sp_out = np.zeros((T, N_OUT), dtype=np.float32)
        sp_out[5,  :] = 1.0  # post fires first
        sp_in[10,  :] = 1.0  # pre fires after
        dw = self.stdp_update(sp_in, sp_out, W)
        assert dw.mean() < 0, f"Expected LTD (dw<0), got mean dw={dw.mean():.6f}"

    def test_no_spike_no_update(self):
        """Zero spike trains must produce zero weight change."""
        T, N_IN, N_OUT = 20, 8, 8
        W  = np.random.rand(N_IN, N_OUT).astype(np.float32)
        sp = np.zeros((T, N_IN), dtype=np.float32)
        dw = self.stdp_update(sp, sp, W)
        assert np.allclose(dw, 0), "Expected zero dW for silent network"

    def test_output_shape(self):
        """dW must have the same shape as W."""
        T, N_IN, N_OUT = 50, 10, 6
        W      = np.random.rand(N_IN, N_OUT).astype(np.float32) * 0.3
        sp_in  = (np.random.rand(T, N_IN)  < 0.1).astype(np.float32)
        sp_out = (np.random.rand(T, N_OUT) < 0.1).astype(np.float32)
        dw = self.stdp_update(sp_in, sp_out, W)
        assert dw.shape == W.shape, \
            f"Shape mismatch: dw={dw.shape} vs W={W.shape}"

    def test_weight_bounds_after_clip(self):
        """Weights must stay in [0, 1] after STDP update + clip."""
        np.random.seed(7)
        N_IN, N_HID, T = 20, 10, 50
        W = np.random.uniform(0.0, 0.3, (N_IN, N_HID)).astype(np.float32)
        for _ in range(5):
            rates  = np.random.rand(N_IN) * 0.5
            sp_in  = self.poisson_encode(rates, T)
            sp_out = self.lif_forward(sp_in, W)
            dw     = self.stdp_update(sp_in, sp_out, W)
            W      = np.clip(W + dw, self.W_MIN, self.W_MAX)
        assert W.min() >= self.W_MIN and W.max() <= self.W_MAX, \
            f"Weights out of bounds: [{W.min():.4f}, {W.max():.4f}]"


# ===========================================================================
# LIF forward pass
# ===========================================================================
class TestLIFForward:
    """Tests for the LIF layer forward pass used in the SNN."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        ns = _load_functions_only('SNN_STDP/snn_stdp.py',
                                  ['lif_forward'])
        self.lif_forward = ns['lif_forward']

    def test_output_shape(self):
        T, N_IN, N_OUT = 30, 10, 5
        W      = np.random.rand(N_IN, N_OUT).astype(np.float32) * 0.5
        sp_in  = (np.random.rand(T, N_IN) < 0.2).astype(np.float32)
        sp_out = self.lif_forward(sp_in, W)
        assert sp_out.shape == (T, N_OUT)

    def test_output_binary(self):
        """Output must be binary (0 or 1 only)."""
        T, N_IN, N_OUT = 50, 8, 4
        W      = np.random.rand(N_IN, N_OUT).astype(np.float32) * 0.8
        sp_in  = (np.random.rand(T, N_IN) < 0.3).astype(np.float32)
        sp_out = self.lif_forward(sp_in, W)
        unique = np.unique(sp_out)
        assert set(unique).issubset({0.0, 1.0}), \
            f"Non-binary output values: {unique}"

    def test_silent_input_silent_output(self):
        """Zero input must produce zero output (no spontaneous spikes)."""
        T, N_IN, N_OUT = 30, 6, 4
        W     = np.random.rand(N_IN, N_OUT).astype(np.float32) * 0.3
        sp_in = np.zeros((T, N_IN), dtype=np.float32)
        sp_out = self.lif_forward(sp_in, W)
        assert sp_out.sum() == 0


# ===========================================================================
# SSM (State Space Model)
# ===========================================================================
class TestSSM:
    """Tests for StateSpaceModel in Optimization/optimization.py."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        ns = _load_functions_only('Optimization/optimization.py',
                                  ['StateSpaceModel'])
        self.SSM = ns['StateSpaceModel']

    def test_output_shape(self):
        ssm = self.SSM(d_in=8, d_state=4, d_out=8)
        X   = np.random.randn(100, 8).astype(np.float32)
        out = ssm.forward(X)
        assert out.shape == (100, 8)

    def test_constant_state_ram(self):
        """State RAM must be constant regardless of sequence length."""
        ssm  = self.SSM(d_in=8, d_state=4, d_out=8)
        ram  = ssm.state_ram()
        assert ram == 4 * 4, f"Expected 16 bytes, got {ram}"

    def test_stable_output(self):
        """Output values must be finite (no NaN/Inf from unstable A matrix)."""
        ssm = self.SSM(d_in=16, d_state=8, d_out=16)
        X   = np.random.randn(500, 16).astype(np.float32)
        out = ssm.forward(X)
        assert np.all(np.isfinite(out)), "SSM produced NaN or Inf output"


# ===========================================================================
# Kalman Tracker
# ===========================================================================
class TestKalmanTracker:
    """Tests for KalmanTracker in EdgeAI/edge_ai.py."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        ns = _load_functions_only('EdgeAI/edge_ai.py',
                                  ['KalmanTracker'])
        self.KalmanTracker = ns['KalmanTracker']

    def test_predict_returns_2d(self):
        tracker = self.KalmanTracker((32.0, 32.0))
        pos = tracker.predict()
        assert pos.shape == (2,)

    def test_update_stays_near_measurement(self):
        """After several updates, tracker must converge toward measurements."""
        tracker = self.KalmanTracker((0.0, 0.0))
        target  = np.array([30.0, 40.0])
        for _ in range(20):
            tracker.predict()
            tracker.update(target)
        pos = tracker.x[:2]
        assert np.linalg.norm(pos - target) < 5.0, \
            f"Tracker failed to converge: pos={pos}, target={target}"

    def test_state_dimension(self):
        tracker = self.KalmanTracker((10.0, 20.0))
        assert tracker.x.shape == (4,), "State vector must be 4-dimensional [x,y,vx,vy]"


# ===========================================================================
# FPGA fixed-point helpers
# ===========================================================================
class TestFixedPoint:
    """Tests for Q8.7 fixed-point arithmetic in FPGA/fpga_simulation.py."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        ns = _load_functions_only('FPGA/fpga_simulation.py',
                                  ['to_fixed', 'from_fixed', 'SCALE'])
        self.to_fixed   = ns['to_fixed']
        self.from_fixed = ns['from_fixed']
        self.SCALE      = ns['SCALE']

    def test_roundtrip(self):
        """to_fixed → from_fixed must recover value within Q8.7 precision."""
        for v in [0.0, 0.5, 1.0, -0.5, 0.125]:
            recovered = self.from_fixed(self.to_fixed(v))
            assert abs(recovered - v) < 1.0 / self.SCALE + 1e-9, \
                f"Roundtrip failed for {v}: got {recovered}"

    def test_clipping(self):
        """Values beyond 16-bit range must be clipped, not overflow."""
        fx = self.to_fixed(1000.0)
        assert fx == 2 ** 15 - 1

    def test_negative(self):
        fx = self.to_fixed(-1.0)
        assert fx < 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
