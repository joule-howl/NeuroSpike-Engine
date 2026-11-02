"""
Neuromorphic AI Engine
Module: Edge AI – Event-based Vision & Real-time Object Tracking

Integrates an SNN with a Dynamic Vision Sensor (DVS) camera.
The system detects and tracks fast-moving objects with low latency and
minimal power consumption by processing only pixel-change events
(event-driven, no frame polling).

Pipeline:
  DVS Camera → Event Accumulator → SNN Detector → Kalman Tracker → Output
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import time
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

OUTPUT_DIR = Path(__file__).parent


# ---------------------------------------------------------------------------
# DVS Camera Simulator
# ---------------------------------------------------------------------------

class DVSCamera:
    """
    Simulates a Dynamic Vision Sensor (DVS / event camera).

    A DVS fires an event only when a pixel's log-intensity changes beyond
    a threshold, rather than capturing a full frame at a fixed frame rate.
    Each event is a tuple (x, y, t, polarity) with polarity ∈ {+1, −1}.

    Temporal resolution: ~1 µs (vs. ~16 ms for a 60 fps camera).
    """

    def __init__(self, width: int = 64, height: int = 64,
                 threshold: float = 0.1, noise_rate: float = 0.01):
        self.W          = width
        self.H          = height
        self.threshold  = threshold
        self.noise_rate = noise_rate
        self._log_frame = np.zeros((height, width), dtype=np.float32)
        self._t         = 0.0

    def update(self, frame: np.ndarray, dt_us: float = 1000.0) -> list:
        """
        Accept a new greyscale frame [H, W] ∈ [0, 1].
        Returns a time-sorted list of events [(x, y, t, polarity), ...].
        """
        log_new  = np.log(frame + 1e-6)
        diff     = log_new - self._log_frame
        events   = []

        for mask, pol in [(diff >  self.threshold, +1),
                          (diff < -self.threshold, -1)]:
            ys, xs = np.where(mask)
            for y, x in zip(ys, xs):
                events.append((x, y, self._t + np.random.uniform(0, dt_us), pol))

        # Background noise events
        n_noise = np.random.poisson(self.noise_rate * self.W * self.H)
        for _ in range(n_noise):
            events.append((
                np.random.randint(0, self.W),
                np.random.randint(0, self.H),
                self._t + np.random.uniform(0, dt_us),
                np.random.choice([-1, 1])
            ))

        self._log_frame = log_new
        self._t        += dt_us
        return sorted(events, key=lambda e: e[2])


# ---------------------------------------------------------------------------
# Event Frame Accumulator
# ---------------------------------------------------------------------------

class EventAccumulator:
    """
    Accumulates DVS events within a time window [t, t + Δt] into a
    2-D event frame [H, W] suitable as SNN input.
    """

    def __init__(self, width: int, height: int):
        self.W = width
        self.H = height

    def accumulate(self, events: list, window_us: float = 5000.0) -> np.ndarray:
        """Returns a frame where each pixel holds its net accumulated polarity."""
        frame = np.zeros((self.H, self.W), dtype=np.float32)
        if not events:
            return frame
        t_start = events[0][2]
        for x, y, t, pol in events:
            if t - t_start > window_us:
                break
            frame[y, x] += pol
        amax = np.abs(frame).max()
        if amax > 0:
            frame = (frame + amax) / (2 * amax)   # normalise to [0, 1]
        return frame


# ---------------------------------------------------------------------------
# SNN-based Object Detector (lightweight, 3-layer)
# ---------------------------------------------------------------------------

class SNNDetector:
    """
    Lightweight 3-layer SNN detector operating on event frames.
    Output: (centre_x, centre_y, confidence) ∈ [0, 1]³
    """

    def __init__(self, in_size: int = 64 * 64,
                 n_h1: int = 256, n_h2: int = 64, n_out: int = 3,
                 tau_m: float = 0.85, v_th: float = 1.0, seed: int = 42):
        np.random.seed(seed)
        s1 = 1.0 / np.sqrt(in_size)
        s2 = 1.0 / np.sqrt(n_h1)
        self.W1    = (np.random.randn(in_size, n_h1) * s1).astype(np.float32)
        self.W2    = (np.random.randn(n_h1,    n_h2) * s2).astype(np.float32)
        self.W3    = (np.random.randn(n_h2,    n_out) * (1 / np.sqrt(n_h2))).astype(np.float32)
        self.W1   *= (np.random.rand(*self.W1.shape) > 0.85)   # sparse connectivity
        self.tau_m = tau_m
        self.v_th  = v_th

    def _lif_layer(self, x: np.ndarray, W: np.ndarray, T: int = 10) -> np.ndarray:
        """Run a LIF layer for T steps; return the mean firing rate."""
        V    = np.zeros(W.shape[1], dtype=np.float32)
        rate = np.zeros(W.shape[1], dtype=np.float32)
        for _ in range(T):
            V     = self.tau_m * V + x @ W
            fired = V >= self.v_th
            V[fired] = 0.0
            rate += fired.astype(np.float32)
        return rate / T

    def detect(self, event_frame: np.ndarray) -> tuple:
        """event_frame: [H, W] → (cx, cy, confidence)"""
        x   = event_frame.flatten()
        h1  = self._lif_layer(x,  self.W1)
        h2  = self._lif_layer(h1, self.W2)
        out = self._lif_layer(h2, self.W3, T=5)
        return (float(np.clip(out[0], 0, 1)),
                float(np.clip(out[1], 0, 1)),
                float(np.clip(out[2], 0, 1)))


# ---------------------------------------------------------------------------
# Kalman Tracker
# ---------------------------------------------------------------------------

class KalmanTracker:
    """
    4-state Kalman filter for 2-D object tracking.
    State vector: [x, y, vx, vy]
    """

    def __init__(self, init_pos: tuple):
        self.x = np.array([init_pos[0], init_pos[1], 0.0, 0.0], dtype=np.float64)
        dt     = 1.0
        self.F = np.array([[1, 0, dt, 0],
                            [0, 1,  0, dt],
                            [0, 0,  1,  0],
                            [0, 0,  0,  1]], dtype=np.float64)
        self.H = np.array([[1, 0, 0, 0],
                            [0, 1, 0, 0]], dtype=np.float64)
        self.P = np.eye(4) * 0.1
        self.Q = np.eye(4) * 0.01   # Process noise
        self.R = np.eye(2) * 0.05   # Measurement noise

    def predict(self) -> np.ndarray:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:2]

    def update(self, z: np.ndarray) -> np.ndarray:
        z      = np.asarray(z, dtype=np.float64)
        y      = z - self.H @ self.x
        S      = self.H @ self.P @ self.H.T + self.R
        K      = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
        return self.x[:2]


# ---------------------------------------------------------------------------
# Moving target simulator
# ---------------------------------------------------------------------------

class MovingObject:
    """Simulates a ball-like object bouncing inside the camera frame."""

    def __init__(self, W, H):
        self.W, self.H = W, H
        self.cx = W / 2
        self.cy = H / 2
        self.vx = np.random.uniform(0.5, 2.0) * np.random.choice([-1, 1])
        self.vy = np.random.uniform(0.5, 2.0) * np.random.choice([-1, 1])
        self.r  = 5

    def step(self):
        self.cx = np.clip(self.cx + self.vx + np.random.randn() * 0.3, self.r, self.W - self.r)
        self.cy = np.clip(self.cy + self.vy + np.random.randn() * 0.3, self.r, self.H - self.r)
        if self.cx <= self.r or self.cx >= self.W - self.r: self.vx *= -1
        if self.cy <= self.r or self.cy >= self.H - self.r: self.vy *= -1

    def render(self) -> np.ndarray:
        frame     = np.zeros((self.H, self.W), dtype=np.float32)
        y, x      = np.ogrid[:self.H, :self.W]
        mask      = (x - self.cx) ** 2 + (y - self.cy) ** 2 <= self.r ** 2
        frame[mask] = 1.0
        frame    += np.random.rand(self.H, self.W) * 0.05   # background noise
        return np.clip(frame, 0, 1)


if __name__ == '__main__':
    # ---------------------------------------------------------------------------
    # Full system simulation
    # ---------------------------------------------------------------------------
    np.random.seed(3)
    W_CAM, H_CAM = 64, 64
    N_FRAMES     = 150

    dvs_cam  = DVSCamera(W_CAM, H_CAM, threshold=0.08, noise_rate=0.005)
    acc      = EventAccumulator(W_CAM, H_CAM)
    detector = SNNDetector(in_size=W_CAM * H_CAM)
    obj      = MovingObject(W_CAM, H_CAM)

    dvs_cam.update(obj.render())   # initialise log-frame

    tracker                      = None
    latencies, energies          = [], []
    track_cx, track_cy           = [], []
    detect_cx, detect_cy         = [], []
    gt_cx,     gt_cy             = [], []
    event_counts                 = []

    P_SNN_mW = 12.0    # Intel Loihi 2 datasheet, ~12 mW for inference workloads
    P_CNN_mW = 800.0   # NVIDIA Jetson Nano TDP ~5-10 W; typical MobileNetV2 inference ~800 mW

    print("Running neuromorphic edge AI simulation...")
    for _ in range(N_FRAMES):
        obj.step()
        frame = obj.render()

        t0          = time.perf_counter()
        events      = dvs_cam.update(frame)
        event_frame = acc.accumulate(events)

        cx_det, cy_det, conf = detector.detect(event_frame)
        latency_ms           = (time.perf_counter() - t0) * 1000

        cx_px = cx_det * W_CAM
        cy_px = cy_det * H_CAM

        if tracker is None:
            tracker = KalmanTracker((cx_px, cy_px))
        else:
            tracker.predict()
            if conf > 0.1:
                tracker.update(np.array([cx_px, cy_px]))

        cx_track, cy_track = tracker.x[:2]

        n_events   = len(events)
        e_snn      = P_SNN_mW * latency_ms / 1000 * (n_events / (W_CAM * H_CAM))
        e_cnn      = P_CNN_mW * (1.0 / 30)   # CNN processes every frame @ 30 fps

        latencies.append(latency_ms)
        energies.append((e_snn, e_cnn))
        track_cx.append(cx_track);  track_cy.append(cy_track)
        detect_cx.append(cx_px);    detect_cy.append(cy_px)
        gt_cx.append(obj.cx);       gt_cy.append(obj.cy)
        event_counts.append(n_events)

    latencies  = np.array(latencies)
    energies   = np.array(energies)
    track_err  = np.sqrt((np.array(track_cx) - np.array(gt_cx)) ** 2 +
                          (np.array(track_cy) - np.array(gt_cy)) ** 2)

    print(f"\nSimulation results ({N_FRAMES} frames):")
    print(f"  Mean latency       : {latencies.mean():.2f} ms  (max: {latencies.max():.2f} ms)")
    print(f"  Tracking MAE       : {track_err.mean():.2f} px")
    print(f"  Events/frame       : {np.mean(event_counts):.0f}  "
          f"({np.mean(event_counts)/(W_CAM*H_CAM)*100:.1f}% of pixels)")
    print(f"  SNN total energy   : {energies[:,0].sum()*1000:.2f} µJ")
    print(f"  CNN ref. energy    : {energies[:,1].sum()*1000:.2f} µJ")
    print(f"  Energy saving      : {(1 - energies[:,0].sum()/energies[:,1].sum())*100:.1f}%")
    print("\nNote: energy estimates are approximate, based on "
          "[Loihi 2 product brief, Intel 2023] and "
          "[Jetson Nano datasheet, NVIDIA 2019]")
    print("--- Literature comparison (neuromorphic vs. CNN edge inference) ---")
    print("  This work (simulated SNN, 64×64 event frames, 3-layer LIF): "
          f"{energies[:,0].sum()*1000:.2f} µJ total")
    print("  Blouw et al. (2019), Loihi keyword spotter, measured on hardware: "
          "~0.038 mJ/inference (Loihi) vs ~1.6 mJ/inference (Jetson Nano)")
    print("     [Blouw, P. et al. (2019). Benchmarking keyword spotting efficiency")
    print("      on neuromorphic hardware. arXiv:1812.01739]")
    print("  Davies et al. (2018), Loihi 64-core chip: ~0.5 W typical workload power")
    print("     [Davies, M. et al. (2018). Loihi: A neuromorphic manycore processor")
    print("      with on-chip learning. IEEE Micro.]")
    print("  Note: all energy/power numbers in this simulation are estimated,")
    print("  not measured on physical hardware.")

    # ---------------------------------------------------------------------------
    # Plots
    # ---------------------------------------------------------------------------
    fig = plt.figure(figsize=(16, 12))
    gs  = gridspec.GridSpec(2, 3, hspace=0.4, wspace=0.35)

    ax = fig.add_subplot(gs[0, 0])
    ax.plot(gt_cx,     gt_cy,     'k-',  lw=1.0,  alpha=0.5, label='Ground truth')
    ax.plot(detect_cx, detect_cy, 'b.',  ms=2,    alpha=0.5, label='SNN detection')
    ax.plot(track_cx,  track_cy,  'r-',  lw=1.5,  alpha=0.8, label='Kalman track')
    ax.scatter(gt_cx[0], gt_cy[0], c='green', s=60, zorder=5, marker='o', label='Start')
    ax.scatter(gt_cx[-1], gt_cy[-1], c='red', s=60, zorder=5, marker='x')
    ax.set_xlim(0, W_CAM); ax.set_ylim(0, H_CAM)
    ax.set_xlabel('X (px)'); ax.set_ylabel('Y (px)')
    ax.set_title('Object Trajectory – SNN + Kalman Tracker')
    ax.legend(fontsize=8)
    ax.set_facecolor('#fafafa')
    ax.invert_yaxis()

    ax = fig.add_subplot(gs[0, 1])
    ax.plot(track_err, color='#e15759', lw=1.2)
    ax.axhline(track_err.mean(), color='navy', linestyle='--',
               lw=1, label=f'MAE = {track_err.mean():.2f} px')
    ax.fill_between(range(N_FRAMES), track_err, alpha=0.15, color='#e15759')
    ax.set_xlabel('Frame'); ax.set_ylabel('Tracking Error (px)')
    ax.set_title('Tracking Error over Time')
    ax.legend(); ax.grid(True, alpha=0.3); ax.set_facecolor('#fafafa')

    ax = fig.add_subplot(gs[0, 2])
    ax.plot(latencies, color='#4e79a7', lw=1.0)
    ax.axhline(latencies.mean(), color='red', linestyle='--',
               label=f'Mean = {latencies.mean():.2f} ms')
    ax.set_xlabel('Frame'); ax.set_ylabel('Latency (ms)')
    ax.set_title('End-to-End Inference Latency')
    ax.legend(); ax.grid(True, alpha=0.3); ax.set_facecolor('#fafafa')

    ax = fig.add_subplot(gs[1, 0])
    ax.bar(range(N_FRAMES), event_counts, color='#76b7b2', alpha=0.7, width=1.0)
    ax.axhline(W_CAM * H_CAM, color='red', linestyle='--',
               lw=1, label=f'Full frame = {W_CAM*H_CAM} px')
    ax.set_xlabel('Frame'); ax.set_ylabel('Event Count')
    ax.set_title('DVS Events vs. Full Frame\n(event-driven = process only changes)')
    ax.legend(fontsize=8); ax.set_facecolor('#fafafa')

    ax = fig.add_subplot(gs[1, 1])
    ax.plot(np.cumsum(energies[:, 0]) * 1000, color='#59a14f', lw=2, label='SNN (Neuromorphic)')
    ax.plot(np.cumsum(energies[:, 1]) * 1000, color='#e15759', lw=2, label='CNN (Edge GPU)')
    ax.set_xlabel('Frame'); ax.set_ylabel('Cumulative Energy (µJ)')
    ax.set_title('Cumulative Energy Consumption')
    ax.legend(); ax.grid(True, alpha=0.3); ax.set_facecolor('#fafafa')

    ax = fig.add_subplot(gs[1, 2])
    obj_demo = MovingObject(W_CAM, H_CAM)
    dvs_demo = DVSCamera(W_CAM, H_CAM, threshold=0.08)
    dvs_demo.update(obj_demo.render())
    for _ in range(5):
        obj_demo.step()
    ev_demo = dvs_demo.update(obj_demo.render())
    ef_demo = acc.accumulate(ev_demo)
    im = ax.imshow(ef_demo, cmap='RdBu_r', vmin=0, vmax=1, interpolation='nearest')
    cx_d, cy_d, conf_d = detector.detect(ef_demo)
    ax.add_patch(patches.Circle(
        (cx_d * W_CAM, cy_d * H_CAM), radius=6,
        edgecolor='yellow', facecolor='none', linewidth=2
    ))
    ax.set_title(f'Sample Event Frame\nConf={conf_d:.2f}  ({len(ev_demo)} events)')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.04)

    fig.suptitle('Neuromorphic Edge AI – Event-based Vision & Object Tracking',
                 fontsize=14, fontweight='bold')
    plt.savefig(OUTPUT_DIR / 'edge_ai_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\nPlot saved → EdgeAI/edge_ai_results.png")

