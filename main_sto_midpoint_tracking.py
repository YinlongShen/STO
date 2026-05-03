"""
main_sto_midpoint_tracking.py
=============================

Midpoint trajectory tracking for a 2D elastic strip. Supports both training modes:

  --rhc OFF (default): Adjoint-only. 
                       One full-horizon segment of K = n_steps continuation
                       steps; one controller Theta optimized end-to-end.

  --rhc ON           : Adjoint+RHC Fresh controller Theta and fresh per-step
                       STO list per segment. After T_seg = epochs, execute
                       the segment once with best Theta to advance the
                       realized state; next segment continues from there.

The Stateful Tangent Operator (STO) is layered on as **acceleration only**.
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from dataclasses import dataclass
from typing import Callable, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from SimulatorEnv_2D import SimulatorEnv_2D
from dlo_sto_adapter import DLOStatefulTangentOperator
from implicit_grad_tools import implicit_final_control_grad
from utils import plot_sto_lifecycle


# ==========================================================================
# Bounded continuation controller (paper Eq. (3) + App. D)
# ==========================================================================


class BoundaryRateController(nn.Module):
    """
    u_Theta(λ): scalar λ ∈ [0, 1] -> 3D right-end rate, bounded by `u_max`
    via a final tanh. This is the paper's MLP architecture (App. D).
    """

    def __init__(self, hidden: int = 64, n_hidden: int = 2, u_max: float = 1.0):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(1, hidden), nn.Tanh()]
        for _ in range(n_hidden - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers += [nn.Linear(hidden, 3), nn.Tanh()]
        self.net = nn.Sequential(*layers)
        self.u_max = float(u_max)
        final_linear = self.net[-2]
        if isinstance(final_linear, nn.Linear):
            nn.init.zeros_(final_linear.weight)
            nn.init.zeros_(final_linear.bias)

    def forward(self, lam: torch.Tensor) -> torch.Tensor:
        return self.u_max * self.net(lam)


def build_rate_stack(
    net: BoundaryRateController,
    segment_start_step: int,
    segment_horizon: int,
    total_control_steps: int,
    device: torch.device,
) -> torch.Tensor:
    """Differentiable 6D rate stack [0, 0, 0, dx_r, dy_r, dθ_r] for one segment."""
    rates = []
    for k in range(segment_horizon):
        lam = torch.tensor(
            [[(segment_start_step + k) / total_control_steps]],
            dtype=torch.float32,
            device=device,
        )
        right_rate = net(lam).squeeze(0)
        full_rate = torch.cat([
            torch.zeros(3, dtype=torch.float32, device=device),
            right_rate,
        ])
        rates.append(full_rate)
    return torch.stack(rates, dim=0)


# ==========================================================================
# Step record (one realized equilibrium)
# ==========================================================================


@dataclass
class StepRecord:
    J: np.ndarray
    free_index: np.ndarray
    fixed_index: np.ndarray
    control_6d: np.ndarray
    q_full: np.ndarray


def info_to_record(info: dict) -> StepRecord:
    return StepRecord(
        J=info["J"].copy(),
        free_index=info["free_index"].copy(),
        fixed_index=info["fixed_index"].copy(),
        control_6d=info["control"].copy(),
        q_full=info["q"].copy(),
    )


# ==========================================================================
# Env state helpers
# ==========================================================================


def snapshot_env_state(env) -> dict:
    return {
        "q": None if env.q is None else env.q.copy(),
        "u": None if env.u is None else env.u.copy(),
        "ctime": env.ctime,
        "current_step": env.current_step,
        "lamda": env.lamda,
    }


def restore_env_state(env, state: dict) -> None:
    env.q = None if state["q"] is None else state["q"].copy()
    env.u = None if state["u"] is None else state["u"].copy()
    env.ctime = state["ctime"]
    env.current_step = state["current_step"]
    env.lamda = state["lamda"]


def set_env_state(env, q: np.ndarray, state_velocity=None, current_step: int = 0) -> None:
    env.q = np.asarray(q, dtype=float).copy()
    env.u = (
        np.zeros_like(env.q)
        if state_velocity is None
        else np.asarray(state_velocity, dtype=float).copy()
    )
    env.current_step = int(current_step)
    env.ctime = env.current_step * env.dt
    env.lamda = env.current_step * env.dlamda


# ==========================================================================
# Per-step S_k^T v primitive (paper Eq. (29))
# ==========================================================================


def stprod_exact(record: StepRecord, a_free: np.ndarray, deltaL: float) -> np.ndarray:
    """Exact: solve G_x^T p = a_free, return -G_z^T p mapped to 6D control."""
    return implicit_final_control_grad(
        J=record.J,
        free_index=record.free_index,
        fixed_index=record.fixed_index,
        a_free=a_free,
        control_6d=record.control_6d,
        deltaL=deltaL,
    )


def stprod_sto(
    sto: DLOStatefulTangentOperator,
    record: StepRecord,
    a_free: np.ndarray,
    deltaL: float,
) -> tuple[np.ndarray, dict]:
    """STO: validate sentinel probes -> apply cache, or fall back to exact + reinit."""
    g, used_cache, eta, reason = sto.query(
        J_full=record.J,
        free_index=record.free_index,
        fixed_index=record.fixed_index,
        a_free=a_free,
        control_6d=record.control_6d,
        deltaL=deltaL,
        q_full=record.q_full,
    )
    return g, {
        "used_cache": bool(used_cache),
        "rho": sto.last_rho,
        "kappa": sto.last_kappa,
        "eta": eta,
        "reason": reason,
        "kappa_recomputed": bool(sto.last_kappa_recomputed),
        "in_warning": bool(sto.in_warning),
    }


StProdFn = Callable[[int, StepRecord, np.ndarray, float], tuple[np.ndarray, dict]]


def make_stprod(sto_list: Optional[List[DLOStatefulTangentOperator]]) -> StProdFn:
    """
    Return a single S_k^T v primitive that hides the exact-vs-STO choice.

    The Algorithm 1 backward uses the same closure either way; STO is
    therefore a drop-in acceleration of one operation, never a part of
    the algorithm itself.
    """
    exact_meta = {
        "used_cache": False, "rho": np.nan, "kappa": np.nan,
        "eta": np.nan, "reason": "exact",
        "kappa_recomputed": False, "in_warning": False,
    }

    if sto_list is None:
        def call(k, record, a_free, deltaL):
            return stprod_exact(record, a_free, deltaL), dict(exact_meta)
        return call

    def call(k, record, a_free, deltaL):
        return stprod_sto(sto_list[k], record, a_free, deltaL)
    return call


# ==========================================================================
# Algorithm 1 backward (paper App. A)
# ==========================================================================


def midpoint_grad_to_free(
    record: StepRecord,
    grad_xy: np.ndarray,
    mid_node: int,
) -> np.ndarray:
    """Map a 2D gradient on the midpoint coords into the free-DOF basis."""
    ix, iy = 2 * mid_node, 2 * mid_node + 1
    a = np.zeros(record.free_index.shape[0], dtype=float)
    pos_x = np.where(record.free_index == ix)[0]
    pos_y = np.where(record.free_index == iy)[0]
    if len(pos_x) == 1:
        a[pos_x[0]] = grad_xy[0]
    if len(pos_y) == 1:
        a[pos_y[0]] = grad_xy[1]
    return a


def algorithm1_backward(
    records: List[StepRecord],          # length K+1: records[k] = realized x_k for k=0..K
    grad_midpoints: np.ndarray,         # (K, 2): grad_midpoints[i] = ∂L/∂p_mid(x_{i+1})
    mid_node: int,
    dlam: float,
    deltaL: float,
    stprod: StProdFn,
) -> tuple[np.ndarray, List[dict]]:
    """
    Proxy-adjoint backward for direct boundary-control continuation.

    Forward uses z_{i} = z_0 + Δλ * sum_{j < i} rate_j and
    x_i = SolveEq(z_i). Therefore rate_k influences trajectory terms
    i >= k + 1, and the exact path-wise gradient is a cumulative sum of
    each per-step implicit sensitivity h_i = S_i^T w_i in 6D control space.
    """
    K = len(records) - 1
    assert grad_midpoints.shape == (K, 2)
    n_control = records[0].control_6d.shape[0]
    grad_rates = np.zeros((K, n_control), dtype=float)
    metas: list[dict] = []

    cumulative = np.zeros(n_control, dtype=float)
    for i in range(K, 0, -1):
        w_i = midpoint_grad_to_free(records[i], grad_midpoints[i - 1], mid_node)
        h_i, m = stprod(i, records[i], w_i, deltaL)
        m = dict(m); m["k"] = i; m["query"] = "h_i"
        metas.append(m)

        cumulative += h_i
        grad_rates[i - 1] = dlam * cumulative

    return grad_rates, metas


# ==========================================================================
# Autograd Function: forward exact rollout + backward Algorithm 1
# ==========================================================================


class DLOMidpointTrackingLayer(torch.autograd.Function):
    """
    Forward:
      x_0 = SolveEq(z_0)                       (anchor)
      for k = 0..K-1: z_{k+1} = z_k + Δλ·rates[k]; x_{k+1} = SolveEq(z_{k+1}).

    Backward:
      paper Algorithm 1, with the per-step S_k^T v primitive routed through
      `make_stprod(sto_list)`. The autograd Function does not branch on
      exact vs STO; that choice is hidden in the closure.
    """

    @staticmethod
    def forward(
        ctx,
        rate_stack: torch.Tensor,
        control_start: torch.Tensor,
        env,
        sto_list: Optional[List[DLOStatefulTangentOperator]],
        record: dict,
        q_start: Optional[np.ndarray],
        state_velocity_start: Optional[np.ndarray],
        current_step_start: int,
        total_control_steps: int,
    ):
        rates = rate_stack.detach().cpu().numpy()
        control = control_start.detach().cpu().numpy().copy()
        K = rates.shape[0]
        dlam = 1.0 / float(total_control_steps)

        ctx.converged = False
        ctx.K = K
        ctx.n_control = rates.shape[1]
        ctx.record = record

        old_state = snapshot_env_state(env)
        records: list[StepRecord] = []
        midpoint_history: list[list[float]] = []

        try:
            if q_start is None:
                env.reset()
            else:
                set_env_state(env, q_start, state_velocity_start, current_step_start)

            # ----- Anchor: x_0 = SolveEq(z_0) -----
            _, _, info_anchor = env.step(control, use_inertia=False)
            if info_anchor is None or not info_anchor["converged"]:
                record["converged"] = False
                return torch.full(
                    (K, 2), float("nan"),
                    dtype=rate_stack.dtype, device=rate_stack.device,
                )
            records.append(info_to_record(info_anchor))
            anchor_state_velocity = info_anchor["u"].copy()
            set_env_state(
                env, info_anchor["q"], anchor_state_velocity, current_step_start,
            )

            # ----- Rollout: x_{k+1} = SolveEq(z_{k+1}) for k = 0..K-1 -----
            for k in range(K):
                control = control + dlam * rates[k]
                obs, done, info = env.step(control, use_inertia=False)
                if info is None or not info["converged"]:
                    break
                records.append(info_to_record(info))
                midpoint_history.append([obs["midx"], obs["midy"]])
                if done and k < K - 1:
                    break
        finally:
            restore_env_state(env, old_state)

        converged = len(records) == K + 1
        record["converged"] = bool(converged)
        if not converged:
            return torch.full(
                (K, 2), float("nan"),
                dtype=rate_stack.dtype, device=rate_stack.device,
            )

        ctx.records = records
        ctx.sto_list = sto_list
        ctx.deltaL = env.deltaL
        ctx.mid_node = env.mid_node
        ctx.dlam = dlam
        ctx.converged = True

        midpoint_array = np.asarray(midpoint_history, dtype=float)
        record["q_history"] = np.stack(
            [r.q_full for r in records[1:]], axis=0,
        )
        record["control_history"] = np.stack(
            [r.control_6d for r in records[1:]], axis=0,
        )
        record["midpoint_history"] = midpoint_array.copy()
        record["q"] = records[-1].q_full.copy()
        record["u_final"] = records[-1].control_6d.copy()
        record["midpoint"] = (
            float(midpoint_array[-1, 0]),
            float(midpoint_array[-1, 1]),
        )
        record["adjoint_scheme"] = "algorithm1_proxy_adjoint"

        return torch.as_tensor(
            midpoint_array, dtype=rate_stack.dtype, device=rate_stack.device,
        )

    @staticmethod
    def backward(ctx, grad_midpoints):
        if not getattr(ctx, "converged", False):
            return (
                torch.zeros(
                    (getattr(ctx, "K", 1), getattr(ctx, "n_control", 6)),
                    dtype=grad_midpoints.dtype,
                    device=grad_midpoints.device,
                ),
                None, None, None, None, None, None, None, None,
            )

        grad_np = grad_midpoints.detach().cpu().numpy()

        t0 = time.perf_counter()
        stprod = make_stprod(ctx.sto_list)
        grad_rates, metas = algorithm1_backward(
            records=ctx.records,
            grad_midpoints=grad_np,
            mid_node=ctx.mid_node,
            dlam=ctx.dlam,
            deltaL=ctx.deltaL,
            stprod=stprod,
        )
        grad_time = time.perf_counter() - t0

        record_backward_metrics(ctx.record, grad_rates, metas, grad_time, ctx.sto_list)

        return (
            torch.as_tensor(
                grad_rates,
                dtype=grad_midpoints.dtype,
                device=grad_midpoints.device,
            ),
            None, None, None, None, None, None, None, None,
        )


def record_backward_metrics(
    record: dict,
    grad_rates: np.ndarray,
    metas: List[dict],
    grad_time: float,
    sto_list: Optional[List[DLOStatefulTangentOperator]],
) -> None:
    n_q = len(metas)
    n_cache = sum(1 for m in metas if m["used_cache"])
    last = metas[-1] if metas else {}

    rhos = np.array([m["rho"] for m in metas if np.isfinite(m["rho"])], dtype=float)
    etas = np.array([m["eta"] for m in metas if np.isfinite(m["eta"])], dtype=float)
    n_kr = sum(1 for m in metas if m.get("kappa_recomputed", False))
    n_warn = sum(1 for m in metas if m.get("in_warning", False))

    record["implicit_grad_seconds"] = grad_time
    record["grad_rate_norm"] = float(np.linalg.norm(grad_rates))
    record["g_u_final"] = (
        grad_rates[-1].copy() if grad_rates.shape[0] else np.zeros(0)
    )
    record["n_adjoint_queries"] = int(n_q)
    record["sto_cache_hits"] = int(n_cache)
    record["sto_exact_queries"] = int(n_q - n_cache)
    record["sto_cache_hit_rate"] = float(n_cache / max(n_q, 1))
    record["used_cache"] = bool(n_q > 0 and n_cache == n_q)
    record["rho"] = float(rhos[-1]) if rhos.size else np.nan
    record["rho_max_seen"] = float(rhos.max()) if rhos.size else np.nan
    record["kappa"] = last.get("kappa", np.nan)
    record["eta"] = float(etas[-1]) if etas.size else np.nan
    record["eta_max_seen"] = float(etas.max()) if etas.size else np.nan
    record["reason"] = last.get(
        "reason", "exact" if sto_list is None else "",
    )
    record["sto_kappa_recomputes"] = int(n_kr)
    record["sto_warning_queries"] = int(n_warn)


# ==========================================================================
# Per-step STO list (one anchor per continuation step)
# ==========================================================================


def make_sto_list(args, n_records: int) -> List[DLOStatefulTangentOperator]:
    """One STO per index k = 0..K. STO[0] anchors at x_0; STO[k] anchors at x_k."""
    return [
        DLOStatefulTangentOperator(
            rho_max=args.rho_max,
            kappa_max=args.kappa_max,
            rho_warn=args.rho_warn,
            kappa_check_period=args.kappa_check_period,
            cooldown=args.cooldown,
            n_probes=args.n_probes,
            n_power_iter=args.n_power_iter,
            max_reuse=args.max_reuse,
            rng=np.random.default_rng(args.seed + 10_000 + i),
        )
        for i in range(n_records)
    ]


# ==========================================================================
# Target generation
# ==========================================================================


def make_env(args) -> SimulatorEnv_2D:
    return SimulatorEnv_2D(
        nv=args.n_nodes,
        dt=1.0 / args.n_steps,
        rod_length=1.0,
        total_time=1.0,
        R_outer=1.0e-3,
        r_inner=0.0,
        E_al=1.0e5,
        rho_al=1070,
        max_newton_iter=args.max_newton_iter,
        save_history=False,
    )


def smoothstep(t: float) -> float:
    return t * t * (3.0 - 2.0 * t)


def initial_midpoint(args) -> tuple[float, float]:
    """Equilibrium midpoint at the initial control u_0 = [0,0,0,L,0,0]."""
    env = make_env(args)
    env.reset()
    u0 = np.array([0.0, 0.0, 0.0, env.rod_length, 0.0, 0.0], dtype=float)
    obs, _, info = env.step(u0, use_inertia=False)
    if info is None or not info["converged"]:
        raise RuntimeError("Initial equilibrium failed to converge.")
    return float(obs["midx"]), float(obs["midy"])


def teacher_control(
    lam: float, rod_length: float, x_drop: float, y_lift: float, theta_drop: float,
) -> np.ndarray:
    s = smoothstep(lam)
    return np.array([
        0.0, 0.0, 0.0,
        rod_length - x_drop * s,
        y_lift * np.sin(0.5 * np.pi * lam),
        theta_drop * s,
    ], dtype=float)


def _teacher_target(args) -> dict:
    """Legacy: target is the midpoint trajectory of a prescribed teacher boundary
    control rollout (always reachable). Kept for backwards-compatible runs."""
    env = make_env(args)
    env.reset()
    midpoints, controls, q_history = [], [], []
    for k in range(args.n_steps):
        lam = (k + 1) / args.n_steps
        c = teacher_control(
            lam, env.rod_length,
            args.teacher_x_drop, args.teacher_y_lift, args.teacher_theta_drop,
        )
        obs, done, info = env.step(c, use_inertia=False)
        if info is None or not info["converged"]:
            raise RuntimeError(f"Teacher rollout failed to converge at step {k+1}.")
        midpoints.append([obs["midx"], obs["midy"]])
        controls.append(c.copy())
        q_history.append(info["q"].copy())
        if done and k < args.n_steps - 1:
            raise RuntimeError("Teacher rollout ended before n_steps.")
    return {
        "midpoints": np.asarray(midpoints, dtype=float),
        "controls": np.asarray(controls, dtype=float),
        "q_history": np.asarray(q_history, dtype=float),
    }


def generate_target_trajectory(args) -> dict:
    """
    Prescribed midpoint trajectory p*_mid(λ_k) for k = 1..K.

    The trajectory is defined as a math function of λ — NOT generated by a
    teacher controller — so reachability is not guaranteed. The Neural
    Control loss (paper Eq. 22) measures how closely the realized strip
    midpoint follows it.

    Shapes:
      sinusoid : x moves by target_x_span while y follows
                 A sin(2π (phase + f λ)) about the initial y.
                 Fractional f values give a partial sine segment.
      circle   : midpoint traces a circle of radius A around the initial
                 midpoint. Starts and ends at the initial point.
      square   : near-square-wave in y via tanh-clipped sinusoid.
      line     : linear ramp in y from initial to A.
      teacher  : (legacy) midpoint of a prescribed teacher boundary rollout.
    """
    K = args.n_steps
    shape = args.target_shape

    if shape == "teacher":
        return _teacher_target(args)

    x0, y0 = initial_midpoint(args)
    lams = np.array([(k + 1) / K for k in range(K)], dtype=float)
    A = float(args.target_amp)
    f = float(args.target_freq)
    phase0 = float(getattr(args, "target_phase", 0.0))
    x_span = float(args.target_x_span)
    phase = phase0 + f * lams

    if shape == "sinusoid":
        x_t = x0 - x_span * lams
        y_t = y0 + A * np.sin(2.0 * np.pi * phase)
    elif shape == "circle":
        # Phase-shifted so trajectory starts at (x0, y0).
        phase_rel = phase - phase0
        x_t = x0 + A * (np.cos(2.0 * np.pi * phase_rel) - 1.0)
        y_t = y0 + A * np.sin(2.0 * np.pi * phase_rel)
    elif shape == "square":
        x_t = x0 - x_span * lams
        y_t = y0 + A * np.tanh(8.0 * np.sin(2.0 * np.pi * phase))
    elif shape == "line":
        x_t = x0 - x_span * lams
        y_t = y0 + A * lams
    else:
        raise ValueError(
            f"unknown --target_shape '{shape}'; "
            "choose from sinusoid, circle, square, line, teacher."
        )

    midpoints = np.stack([x_t, y_t], axis=1)
    return {
        "midpoints": midpoints,
        "controls": np.empty((0, 6), dtype=float),
        "q_history": np.empty((0, 2 * args.n_nodes), dtype=float),
    }


def choose_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "mps":
        return torch.device("mps")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ==========================================================================
# Segment execution (no grad) — used to advance state across RHC segments
# ==========================================================================


def execute_segment(
    env,
    net: BoundaryRateController,
    q_start: np.ndarray,
    state_velocity_start: np.ndarray,
    control_start_np: np.ndarray,
    current_step_start: int,
    total_K: int,
    H: int,
    device: torch.device,
) -> dict:
    """No-grad rollout of one segment of H steps from a given start state."""
    with torch.no_grad():
        rates_t = build_rate_stack(net, current_step_start, H, total_K, device)
    rates_np = rates_t.detach().cpu().numpy()

    set_env_state(env, q_start, state_velocity_start, current_step_start)
    control = np.asarray(control_start_np, dtype=float).copy()
    dlam = 1.0 / total_K
    midpoints, controls, q_history = [], [], []
    state_velocity_end = np.asarray(state_velocity_start, dtype=float).copy()

    for k in range(H):
        control = control + dlam * rates_np[k]
        obs, done, info = env.step(control, use_inertia=False)
        if info is None or not info["converged"]:
            return {
                "converged": False,
                "midpoints": np.asarray(midpoints, dtype=float),
                "controls": np.asarray(controls, dtype=float),
                "q_history": np.asarray(q_history, dtype=float),
                "state_velocity_end": state_velocity_end,
            }
        midpoints.append([obs["midx"], obs["midy"]])
        controls.append(control.copy())
        q_history.append(info["q"].copy())
        state_velocity_end = info["u"].copy()
        if done and k < H - 1:
            break

    return {
        "converged": len(midpoints) == H,
        "midpoints": np.asarray(midpoints, dtype=float),
        "controls": np.asarray(controls, dtype=float),
        "q_history": np.asarray(q_history, dtype=float),
        "state_velocity_end": state_velocity_end,
    }


# ==========================================================================
# Training: unified segment loop
#   M = 1, H = K  -> Adjoint-only (paper Sec. 3.1–3.2, App. A).
#   M > 1         -> Adjoint+RHC (paper Sec. 3.3) with fresh Theta per segment.
# ==========================================================================


def run_training(args, mode: str, target_midpoints: np.ndarray, run_dir: str) -> dict:
    use_sto = (mode == "sto")
    use_rhc = bool(args.rhc)
    os.makedirs(run_dir, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    K = args.n_steps
    if use_rhc:
        if K % args.segment_horizon != 0:
            raise ValueError(
                f"--n_steps ({K}) must be divisible by --segment_horizon "
                f"({args.segment_horizon}) in RHC mode."
            )
        H = args.segment_horizon
        M = K // H
    else:
        H = K
        M = 1

    env = make_env(args)
    env.reset()
    device = choose_device(args.device)

    # Initial segment-start state.
    q_segment_start = env.q.copy()
    state_velocity_start = env.u.copy()
    control_start_np = np.array(
        [0.0, 0.0, 0.0, env.rod_length, 0.0, 0.0], dtype=float,
    )
    current_step_start = 0

    target_torch_full = torch.as_tensor(
        target_midpoints, dtype=torch.float32, device=device,
    )

    # Aggregated executed trajectory and training metrics.
    executed_midpoints: list[list[float]] = []
    executed_controls: list[np.ndarray] = []
    executed_q: list[np.ndarray] = []
    loss_history: list[tuple[int, float]] = []
    grad_time_history: list[tuple[int, float]] = []
    timing_rows: list[dict] = []
    sto_events: list[dict] = []
    segment_best_losses: list[float] = []
    segment_executed_losses: list[float] = []
    sto_aggregate_stats = {"n_init": 0, "n_cached": 0, "n_exact": 0, "n_kappa_recompute": 0}

    global_ep = 0

    for seg in range(M):
        seg_label = f"seg {seg+1}/{M}" if use_rhc else "training"
        seg_steps = (current_step_start, current_step_start + H - 1)
        print(f"\n[{mode}] {seg_label}: continuation steps {seg_steps[0]}..{seg_steps[1]}")

        # Fresh controller and optimizer per segment (paper Sec. 3.3).
        net = BoundaryRateController(
            hidden=args.hidden, n_hidden=args.n_hidden, u_max=args.u_max,
        ).to(device)
        opt = optim.Adam(net.parameters(), lr=args.lr)
        # Fresh per-segment STO list of length H+1 (one per equilibrium index 0..H).
        sto_list = make_sto_list(args, H + 1) if use_sto else None

        seg_target = target_torch_full[current_step_start:current_step_start + H]
        control_start_torch = torch.as_tensor(
            control_start_np, dtype=torch.float32, device=device,
        )

        best_loss_seg = float("inf")
        best_state_seg: Optional[dict] = None
        consecutive_newton_failures = 0

        pbar = tqdm(
            range(args.epochs),
            desc=f"{mode.upper()} {seg_label}",
            dynamic_ncols=True,
        )
        for local_ep in pbar:
            rate_stack = build_rate_stack(net, current_step_start, H, K, device)
            backward_record: dict = {}
            midpoints_torch = DLOMidpointTrackingLayer.apply(
                rate_stack, control_start_torch, env, sto_list, backward_record,
                q_segment_start, state_velocity_start, current_step_start, K,
            )
            if not backward_record.get("converged", False):
                consecutive_newton_failures += 1
                pbar.set_postfix({"status": "Newton failed"})
                global_ep += 1
                if (
                    args.newton_fail_patience > 0
                    and consecutive_newton_failures >= args.newton_fail_patience
                ):
                    print(
                        f"  {seg_label}: stopped after "
                        f"{consecutive_newton_failures} consecutive Newton failures."
                    )
                    break
                continue
            consecutive_newton_failures = 0

            diff = midpoints_torch - seg_target
            loss_torch = torch.mean(torch.sum(diff * diff, dim=1))
            if not torch.isfinite(loss_torch):
                pbar.set_postfix({"status": "nonfinite loss"})
                global_ep += 1
                continue

            loss_value = float(loss_torch.detach().cpu())
            rmse = float(np.sqrt(loss_value))

            opt.zero_grad()
            loss_torch.backward()
            if args.grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
            grads_finite = all(
                p.grad is None or torch.isfinite(p.grad).all().item()
                for p in net.parameters()
            )
            if not grads_finite:
                pbar.set_postfix({"status": "nonfinite grad"})
                global_ep += 1
                continue

            if loss_value < best_loss_seg:
                best_loss_seg = loss_value
                best_state_seg = {
                    n: v.detach().cpu().clone() for n, v in net.state_dict().items()
                }
            opt.step()

            grad_time = backward_record["implicit_grad_seconds"]
            grad_time_history.append((global_ep, grad_time))
            loss_history.append((global_ep, loss_value))

            row = {
                "epoch": global_ep, "segment": seg, "local_epoch": local_ep,
                "mode": mode, "loss": loss_value, "rmse": rmse,
                "implicit_grad_seconds": grad_time,
                "n_adjoint_queries": backward_record["n_adjoint_queries"],
                "used_cache": int(backward_record["used_cache"]),
                "sto_cache_hits": backward_record["sto_cache_hits"],
                "sto_exact_queries": backward_record["sto_exact_queries"],
                "sto_cache_hit_rate": backward_record["sto_cache_hit_rate"],
                "rho": backward_record["rho"],
                "rho_max_seen": backward_record["rho_max_seen"],
                "kappa": backward_record["kappa"],
                "eta": backward_record["eta"],
                "eta_max_seen": backward_record["eta_max_seen"],
                "reason": backward_record["reason"],
                "grad_rate_norm": backward_record["grad_rate_norm"],
                "sto_kappa_recomputes": backward_record["sto_kappa_recomputes"],
                "sto_warning_queries": backward_record["sto_warning_queries"],
                "midpoint_x": backward_record["midpoint"][0],
                "midpoint_y": backward_record["midpoint"][1],
            }
            timing_rows.append(row)

            if use_sto:
                sto_events.append({
                    "epoch": global_ep, "segment": seg,
                    "n_queries": row["n_adjoint_queries"],
                    "n_cache": row["sto_cache_hits"],
                    "n_exact": row["sto_exact_queries"],
                    "hit_rate": row["sto_cache_hit_rate"],
                    "rho": row["rho"], "rho_max_seen": row["rho_max_seen"],
                    "kappa": row["kappa"], "eta": row["eta"],
                    "eta_max_seen": row["eta_max_seen"],
                    "reason": row["reason"],
                })

            pbar.set_postfix({"loss": f"{loss_value:.3e}", "rmse": f"{rmse:.3e}"})
            if local_ep % args.print_every == 0:
                tag = ""
                if use_sto:
                    tag = (
                        f"  [hit={100*row['sto_cache_hit_rate']:.1f}% "
                        f"exact={row['sto_exact_queries']}/{row['n_adjoint_queries']}"
                        f" rho_max_seen={row['rho_max_seen']:.1e}]"
                    )
                print(
                    f"[{mode} {seg_label} ep {local_ep:04d}] "
                    f"L={loss_value:.6e} rmse={rmse:.3e} "
                    f"implicit_grad={1e3*grad_time:.2f}ms{tag}"
                )

            global_ep += 1

        # ----- Execute segment with best Theta to advance realized state -----
        if best_state_seg is not None:
            net.load_state_dict(best_state_seg)
        elif args.epochs > 0:
            print(f"  {seg_label}: no converged training rollout; stopping training.")
            break

        seg_exec = execute_segment(
            env, net, q_segment_start, state_velocity_start,
            control_start_np, current_step_start, K, H, device,
        )
        if not seg_exec["converged"]:
            print(f"  {seg_label} execution failed; stopping training.")
            break

        # Aggregate executed trajectory and per-segment summary.
        executed_midpoints.extend(seg_exec["midpoints"].tolist())
        executed_controls.extend(seg_exec["controls"].tolist())
        executed_q.extend(seg_exec["q_history"].tolist())
        seg_target_np = target_midpoints[current_step_start:current_step_start + H]
        executed_loss_seg = float(
            np.mean(np.sum((seg_exec["midpoints"] - seg_target_np) ** 2, axis=1))
        )
        segment_best_losses.append(best_loss_seg)
        segment_executed_losses.append(executed_loss_seg)
        print(
            f"  {seg_label} done: best_train={best_loss_seg:.4e} "
            f"executed={executed_loss_seg:.4e}"
        )

        # Aggregate STO stats from this segment's per-step list before discarding it.
        if use_sto and sto_list is not None:
            sto_aggregate_stats["n_init"] += sum(s.n_init for s in sto_list)
            sto_aggregate_stats["n_cached"] += sum(s.n_cached for s in sto_list)
            sto_aggregate_stats["n_exact"] += sum(s.n_fallback for s in sto_list)
            sto_aggregate_stats["n_kappa_recompute"] += sum(
                s.n_kappa_recompute for s in sto_list
            )

        # Advance segment-start for the next iteration.
        q_segment_start = seg_exec["q_history"][-1].copy()
        state_velocity_start = seg_exec["state_velocity_end"].copy()
        control_start_np = seg_exec["controls"][-1].copy()
        current_step_start += H

    executed_midpoints_arr = np.asarray(executed_midpoints, dtype=float)
    executed_controls_arr = np.asarray(executed_controls, dtype=float)
    executed_q_arr = np.asarray(executed_q, dtype=float)

    if executed_midpoints_arr.shape[0] == K:
        final_loss = float(
            np.mean(np.sum((executed_midpoints_arr - target_midpoints) ** 2, axis=1))
        )
        final_rmse = float(np.sqrt(final_loss))
        final_error = float(
            np.linalg.norm(executed_midpoints_arr[-1] - target_midpoints[-1])
        )
    else:
        final_loss = final_rmse = final_error = float("nan")

    save_run_outputs(
        args, run_dir, mode, target_midpoints, executed_midpoints_arr,
        executed_q_arr, executed_controls_arr,
        loss_history, grad_time_history, timing_rows, sto_events,
        segment_best_losses=segment_best_losses,
        segment_executed_losses=segment_executed_losses,
        is_rhc=use_rhc,
    )

    if use_sto:
        print(
            f"  STO aggregate: anchors={H+1}/seg × {M} = {(H+1)*M}  "
            f"inits={sto_aggregate_stats['n_init']}  "
            f"cache_hits={sto_aggregate_stats['n_cached']}  "
            f"fallbacks={sto_aggregate_stats['n_exact']}  "
            f"kappa_recomputes={sto_aggregate_stats['n_kappa_recompute']}"
        )

    if grad_time_history:
        gt = np.array([t for _, t in grad_time_history], dtype=float)
        print(
            f"{mode} backward timing: total={gt.sum():.4f}s "
            f"mean={1e3*gt.mean():.3f}ms median={1e3*np.median(gt):.3f}ms"
        )

    return {
        "mode": mode, "run_dir": run_dir,
        "best_loss": (
            float(min(segment_best_losses))
            if segment_best_losses else float("nan")
        ),
        "final_loss": final_loss,
        "final_rmse": final_rmse,
        "final_error": final_error,
        "loss_history": np.asarray(loss_history, dtype=float),
        "grad_time_history": np.asarray(grad_time_history, dtype=float),
        "final_midpoints": executed_midpoints_arr,
    }


# ==========================================================================
# I/O: artifacts
# ==========================================================================


def save_run_outputs(
    args, run_dir: str, mode: str,
    target_midpoints: np.ndarray, final_midpoints: np.ndarray,
    q_history: np.ndarray, control_history: np.ndarray,
    loss_history: list, grad_time_history: list,
    timing_rows: list, sto_events: list,
    segment_best_losses: Optional[List[float]] = None,
    segment_executed_losses: Optional[List[float]] = None,
    is_rhc: bool = False,
) -> None:
    np.savez(
        os.path.join(run_dir, "tracking_history.npz"),
        target_midpoints=target_midpoints,
        midpoint_history=final_midpoints,
        q_history=q_history,
        control_history=control_history,
    )

    if timing_rows:
        path = os.path.join(run_dir, "tracking_timing.csv")
        fieldnames = list(timing_rows[0].keys())
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(timing_rows)
        print(f"{mode} timing CSV: {path}")

    if loss_history:
        segment_ids = (
            [int(r["segment"]) for r in timing_rows]
            if timing_rows and len(timing_rows) == len(loss_history)
            else None
        )
        save_loss_plot(run_dir, loss_history, segment_ids=segment_ids)

    save_tracking_plot(run_dir, target_midpoints, final_midpoints)
    save_timing_plot(run_dir, mode, grad_time_history, sto_events)

    if sto_events:
        plot_sto_lifecycle(
            sto_events=sto_events,
            timing_rows=sto_events,
            rho_max=args.rho_max,
            rho_warn=args.rho_warn,
            kappa_max=args.kappa_max,
            save_path=os.path.join(run_dir, "sto_diagnostics.png"),
        )

    if is_rhc and segment_best_losses:
        save_segment_losses(
            run_dir, mode,
            segment_best_losses,
            segment_executed_losses or [],
        )


def save_segment_losses(
    run_dir: str, mode: str,
    best_losses: List[float], executed_losses: List[float],
) -> None:
    csv_path = os.path.join(run_dir, "rhc_segment_losses.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["segment", "best_train_loss", "executed_loss"],
        )
        w.writeheader()
        for i in range(len(best_losses)):
            w.writerow({
                "segment": i,
                "best_train_loss": best_losses[i],
                "executed_loss": (
                    executed_losses[i] if i < len(executed_losses) else float("nan")
                ),
            })
    print(f"{mode} RHC segment-loss CSV: {csv_path}")

    seg_idx = np.arange(len(best_losses))
    best = np.asarray(best_losses, dtype=float)
    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    ax.semilogy(seg_idx, np.maximum(best, 1e-30),
                "o-", color="#0072B2", lw=1.4, label="best train loss")
    if executed_losses:
        executed = np.asarray(executed_losses, dtype=float)
        ax.semilogy(seg_idx, np.maximum(executed, 1e-30),
                    "s--", color="#D55E00", lw=1.4, label="executed loss")
    ax.set_xlabel("RHC segment index")
    ax.set_ylabel("loss")
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "rhc_segment_losses.png"), dpi=220)
    plt.close(fig)


def save_loss_plot(
    run_dir: str,
    loss_history: list,
    segment_ids: Optional[List[int]] = None,
) -> None:
    ep_v = np.array([e for e, _ in loss_history], dtype=float)
    ls = np.array([l for _, l in loss_history], dtype=float)
    finite = np.isfinite(ls)
    if not finite.any():
        return
    seg_arr = (
        np.asarray(segment_ids, dtype=int)
        if segment_ids is not None and len(segment_ids) == len(loss_history)
        else None
    )
    if seg_arr is None or np.unique(seg_arr).size == 1:
        best = np.minimum.accumulate(np.where(finite, ls, np.inf))
        best_label = "best"
        title = "Training loss"
    else:
        best = np.full_like(ls, np.nan, dtype=float)
        for seg in np.unique(seg_arr):
            mask = seg_arr == seg
            best[mask] = np.minimum.accumulate(np.where(finite[mask], ls[mask], np.inf))
        best_label = "segment best"
        title = "RHC segment loss"

    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    ax.semilogy(ep_v[finite], ls[finite], color="0.65", lw=0.9, label="epoch")
    ax.semilogy(ep_v[finite], best[finite], color="#0072B2", lw=1.8, label=best_label)
    if seg_arr is not None and np.unique(seg_arr).size > 1:
        for seg in np.unique(seg_arr)[1:]:
            first = int(np.argmax(seg_arr == seg))
            ax.axvline(ep_v[first], color="0.82", lw=0.8, ls=":")
    best_i = int(np.nanargmin(ls))
    ax.scatter([ep_v[best_i]], [ls[best_i]], s=24, color="#0072B2", zorder=3)
    ax.set_title(title)
    ax.set_xlabel("epoch")
    ax.set_ylabel("tracking MSE")
    ax.grid(True, alpha=0.18)
    ax.legend(fontsize=7, frameon=False)
    style_axis(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "loss_curve.png"), dpi=220)
    plt.close(fig)


def style_axis(ax) -> None:
    ax.title.set_fontsize(10)
    ax.xaxis.label.set_size(9)
    ax.yaxis.label.set_size(9)
    ax.tick_params(labelsize=8)


def set_xy_limits(ax, *arrays: np.ndarray, pad_frac: float = 0.08) -> None:
    pts = []
    for arr in arrays:
        arr = np.asarray(arr, dtype=float)
        if arr.ndim == 2 and arr.shape[1] == 2:
            valid = np.all(np.isfinite(arr), axis=1)
            if valid.any():
                pts.append(arr[valid])
    if not pts:
        return
    xy = np.vstack(pts)
    lo = xy.min(axis=0)
    hi = xy.max(axis=0)
    center = 0.5 * (lo + hi)
    span = max(float(np.max(hi - lo)), 5e-2)
    half = 0.5 * span * (1.0 + 2.0 * pad_frac)
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)


def save_tracking_plot(run_dir: str, target: np.ndarray, final: np.ndarray) -> None:
    steps = np.arange(1, len(target) + 1)
    final = np.asarray(final, dtype=float)
    n = min(len(target), len(final))
    err = (
        np.linalg.norm(final[:n] - target[:n], axis=1)
        if n else np.asarray([], dtype=float)
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2))

    ax = axes[0, 0]
    ax.plot(target[:, 0], target[:, 1], color="#D55E00", lw=1.8, label="target path")
    if len(final):
        ax.plot(final[:, 0], final[:, 1], color="#0072B2", lw=1.8, label="learned path")
        ax.plot(final[-1, 0], final[-1, 1], "x", color="#0072B2", ms=7, mew=1.8,
                label="learned final")
    ax.plot(target[-1, 0], target[-1, 1], "o", color="#D55E00", ms=5,
            label="target final")
    ax.plot(target[0, 0], target[0, 1], "o", mfc="white", mec="#D55E00", ms=5,
            label="target start")
    ax.set_title("Midpoint path")
    ax.set_xlabel("midpoint x"); ax.set_ylabel("midpoint y")
    ax.set_aspect("equal", adjustable="box")
    set_xy_limits(ax, target, final)
    ax.grid(True, alpha=0.18); ax.legend(fontsize=6.8, frameon=False)
    style_axis(ax)

    ax = axes[0, 1]
    ax.plot(steps, target[:, 0], color="#D55E00", lw=1.6, label="target")
    if len(final):
        ax.plot(steps[:len(final)], final[:, 0], color="#0072B2", lw=1.6, label="learned")
    ax.set_title("Horizontal tracking")
    ax.set_xlabel("step"); ax.set_ylabel("midpoint x")
    ax.grid(True, alpha=0.18); ax.legend(fontsize=7, frameon=False)
    style_axis(ax)

    ax = axes[1, 0]
    ax.plot(steps, target[:, 1], color="#D55E00", lw=1.6, label="target")
    if len(final):
        ax.plot(steps[:len(final)], final[:, 1], color="#0072B2", lw=1.6, label="learned")
    ax.set_title("Vertical tracking")
    ax.set_xlabel("step"); ax.set_ylabel("midpoint y")
    ax.grid(True, alpha=0.18); ax.legend(fontsize=7, frameon=False)
    style_axis(ax)

    ax = axes[1, 1]
    if err.size:
        ax.plot(steps[:n], err, color="0.2", lw=1.5)
    ax.set_title("Pointwise error")
    ax.set_xlabel("step"); ax.set_ylabel("distance")
    ax.grid(True, alpha=0.18)
    style_axis(ax)

    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "tracking_result.png"), dpi=220)
    plt.close(fig)


def save_timing_plot(
    run_dir: str, mode: str, grad_time_history: list, sto_events: list,
) -> None:
    if not grad_time_history:
        return
    ep_v = np.array([e for e, _ in grad_time_history], dtype=int)
    gt = np.array([t for _, t in grad_time_history], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    ax = axes[0]
    if mode == "sto" and sto_events:
        hits = np.array([e["hit_rate"] for e in sto_events], dtype=float)
        all_hit = hits >= 0.999
        any_miss = (hits > 0) & ~all_hit
        all_miss = hits <= 0
        ax.plot(ep_v, 1e3 * gt, color="0.75", lw=0.8)
        if all_hit.any():
            ax.scatter(ep_v[all_hit], 1e3 * gt[all_hit],
                       s=14, color="#009E73", label="cache")
        if any_miss.any():
            ax.scatter(ep_v[any_miss], 1e3 * gt[any_miss],
                       s=14, color="#E69F00", label="mixed")
        if all_miss.any():
            ax.scatter(ep_v[all_miss], 1e3 * gt[all_miss],
                       s=14, color="#D55E00", label="exact")
        ax.legend(fontsize=7, frameon=False)
    else:
        ax.plot(ep_v, 1e3 * gt, color="#0072B2", lw=1.4)
    ax.set_xlabel("epoch"); ax.set_ylabel("ms"); ax.set_title("backward")
    ax.grid(True, alpha=0.2)

    axes[1].plot(ep_v, np.cumsum(gt), color="#CC79A7", lw=1.5)
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("s"); axes[1].set_title("cumulative")
    axes[1].grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "implicit_gradient_timing.png"), dpi=220)
    plt.close(fig)


def save_target_outputs(log_dir: str, target: dict) -> None:
    os.makedirs(log_dir, exist_ok=True)
    save_kwargs = {"target_midpoints": target["midpoints"]}
    if target.get("controls") is not None:
        save_kwargs["target_controls"] = target["controls"]
    if target.get("q_history") is not None:
        save_kwargs["target_q_history"] = target["q_history"]
    np.savez(os.path.join(log_dir, "tracking_target.npz"), **save_kwargs)

    steps = np.arange(1, len(target["midpoints"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.7))

    axes[0].plot(target["midpoints"][:, 0], target["midpoints"][:, 1],
                 color="#D55E00", lw=1.8, label="target")
    axes[0].plot(target["midpoints"][0, 0], target["midpoints"][0, 1],
                 "o", mfc="white", mec="#D55E00", ms=5)
    axes[0].plot(target["midpoints"][-1, 0], target["midpoints"][-1, 1],
                 "o", color="#D55E00", ms=5)
    axes[0].set_title("Target path")
    axes[0].set_xlabel("midpoint x"); axes[0].set_ylabel("midpoint y")
    axes[0].set_aspect("equal", adjustable="box")
    set_xy_limits(axes[0], target["midpoints"])
    axes[0].grid(True, alpha=0.18); axes[0].legend(fontsize=7, frameon=False)
    style_axis(axes[0])

    axes[1].plot(steps, target["midpoints"][:, 0], color="#D55E00", lw=1.4, label="x")
    axes[1].plot(steps, target["midpoints"][:, 1], color="#0072B2", lw=1.4, label="y")
    axes[1].set_title("Target coordinates")
    axes[1].set_xlabel("step"); axes[1].set_ylabel("midpoint")
    axes[1].grid(True, alpha=0.18); axes[1].legend(fontsize=7, frameon=False)
    style_axis(axes[1])

    fig.tight_layout()
    fig.savefig(os.path.join(log_dir, "tracking_target.png"), dpi=220)
    plt.close(fig)


def save_comparison(
    log_dir: str, summaries: list, target_midpoints: np.ndarray,
) -> None:
    if len(summaries) < 2:
        return

    summary_csv = os.path.join(log_dir, "comparison_summary.csv")
    with open(summary_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "mode", "best_loss", "final_loss",
            "final_rmse", "final_error", "run_dir",
        ])
        w.writeheader()
        for r in summaries:
            w.writerow({
                "mode": r["mode"],
                "best_loss": r["best_loss"],
                "final_loss": r["final_loss"],
                "final_rmse": r["final_rmse"],
                "final_error": r["final_error"],
                "run_dir": r["run_dir"],
            })
    print(f"Comparison summary: {summary_csv}")

    fig, axes = plt.subplots(1, 3, figsize=(10.0, 2.8))
    colors = {"exact": "0.25", "sto": "#0072B2"}

    ax = axes[0]
    for r in summaries:
        h = r["loss_history"]
        if h.size:
            ax.semilogy(h[:, 0], h[:, 1], lw=1.4,
                        color=colors.get(r["mode"]), label=r["mode"])
    ax.set_xlabel("epoch"); ax.set_ylabel("loss")
    ax.grid(True, alpha=0.2); ax.legend(fontsize=7, frameon=False)

    ax = axes[1]
    for r in summaries:
        h = r["grad_time_history"]
        if h.size:
            ax.plot(h[:, 0], np.cumsum(h[:, 1]), lw=1.4,
                    color=colors.get(r["mode"]), label=r["mode"])
    ax.set_xlabel("epoch"); ax.set_ylabel("s")
    ax.grid(True, alpha=0.2); ax.legend(fontsize=7, frameon=False)

    ax = axes[2]
    ax.plot(target_midpoints[:, 0], target_midpoints[:, 1],
            color="#D55E00", lw=1.4, label="target")
    for r in summaries:
        m = r["final_midpoints"]
        if len(m):
            ax.plot(m[:, 0], m[:, 1], lw=1.2,
                    color=colors.get(r["mode"]), label=r["mode"])
            ax.plot(m[-1, 0], m[-1, 1], "x",
                    color=colors.get(r["mode"]), ms=6, mew=1.5)
    ax.plot(target_midpoints[-1, 0], target_midpoints[-1, 1],
            "o", color="#D55E00", ms=5)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    set_xy_limits(
        ax,
        target_midpoints,
        *[r["final_midpoints"] for r in summaries if len(r["final_midpoints"])],
    )
    ax.grid(True, alpha=0.2); ax.legend(fontsize=7, frameon=False)

    fig.tight_layout()
    fig.savefig(os.path.join(log_dir, "comparison.png"), dpi=220)
    plt.close(fig)
    print(f"Comparison plot: {os.path.join(log_dir, 'comparison.png')}")


# ==========================================================================
# CLI
# ==========================================================================


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Neural Control midpoint trajectory tracking with optional STO "
            "acceleration. STO is a drop-in replacement for the per-step "
            "S_k^T v primitive (paper Eq. 29); the algorithm is identical "
            "with or without."
        )
    )
    p.add_argument("--mode", choices=["both", "exact", "sto"], default="both")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--seed", type=int, default=0)

    # Controller
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--n_hidden", type=int, default=2)
    p.add_argument("--u_max", type=float, default=1.0,
                   help="bound on ||u(λ)|| via final tanh (paper App. D)")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--grad_clip", type=float, default=10.0)

    # Strip / continuation
    p.add_argument("--n_nodes", type=int, default=41)
    p.add_argument("--n_steps", type=int, default=80,
                   help="K = total continuation steps (must be M*H in RHC mode)")
    p.add_argument("--max_newton_iter", type=int, default=1000)
    p.add_argument("--newton_fail_patience", type=int, default=20,
                   help="stop the current segment after this many consecutive "
                        "failed Newton rollouts; set <=0 to disable")
    p.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    p.add_argument("--print_every", type=int, default=10)
    p.add_argument("--log_dir", type=str, default="runs/dlo_midpoint_tracking_compare")

    # RHC (paper Sec. 3.3, Algorithm 1 outer loop). Default off (Adjoint-only).
    p.add_argument("--rhc", action="store_true",
                   help="enable receding-horizon continuation (paper Sec. 3.3)")
    p.add_argument("--segment_horizon", type=int, default=16,
                   help="H = continuation steps per RHC segment (only used with --rhc); "
                        "M = n_steps / segment_horizon segments")

    # Target trajectory (paper Task 2): prescribed function of λ, not a teacher rollout.
    p.add_argument("--target_shape",
                   choices=["sinusoid", "circle", "square", "line", "teacher"],
                   default="sinusoid",
                   help="prescribed midpoint trajectory shape p*_mid(λ)")
    p.add_argument("--target_amp", type=float, default=0.15,
                   help="amplitude of the prescribed target (units of strip length)")
    p.add_argument("--target_freq", type=float, default=1.0,
                   help="number of sine cycles traversed over λ ∈ [0, 1]; "
                        "use fractions like 0.5 or 0.25 for a partial sine segment")
    p.add_argument("--target_phase", type=float, default=0.0,
                   help="starting sine phase in cycles for sinusoid/square targets")
    p.add_argument("--target_x_span", type=float, default=0.20,
                   help="horizontal span for sinusoid/square/line midpoint targets")

    # Legacy teacher params (used only when --target_shape teacher).
    p.add_argument("--teacher_x_drop", type=float, default=0.28)
    p.add_argument("--teacher_y_lift", type=float, default=0.34)
    p.add_argument("--teacher_theta_drop", type=float, default=-0.55)

    # STO (acceleration knobs only — algorithm unchanged)
    p.add_argument("--rho_max", type=float, default=1e-1)
    p.add_argument("--rho_warn", type=float, default=1e-2)
    p.add_argument("--kappa_max", type=float, default=1e10)
    p.add_argument("--kappa_check_period", type=int, default=10)
    p.add_argument("--cooldown", type=int, default=3)
    p.add_argument("--n_probes", type=int, default=4)
    p.add_argument("--n_power_iter", type=int, default=3)
    p.add_argument("--max_reuse", type=int, default=10000)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.log_dir, exist_ok=True)

    method = "Adjoint+RHC" if args.rhc else "Adjoint-only"
    print(f"Neural Control midpoint trajectory tracking ({method})")
    print(f"  n_nodes={args.n_nodes} K={args.n_steps} epochs/seg={args.epochs}")
    if args.rhc:
        if args.n_steps % args.segment_horizon != 0:
            raise ValueError(
                f"--n_steps ({args.n_steps}) must be divisible by "
                f"--segment_horizon ({args.segment_horizon}) in RHC mode."
            )
        M = args.n_steps // args.segment_horizon
        print(f"  RHC: M={M} segments × H={args.segment_horizon} steps")
    print(f"  controller: hidden={args.hidden} n_hidden={args.n_hidden} u_max={args.u_max}")
    if args.target_shape == "teacher":
        print(
            f"  target: teacher x_drop={args.teacher_x_drop:.3f} "
            f"y_lift={args.teacher_y_lift:.3f} "
            f"theta_drop={args.teacher_theta_drop:.3f}"
        )
    else:
        print(
            f"  target: {args.target_shape} "
            f"amp={args.target_amp:.3f} cycles={args.target_freq:.3f} "
            f"phase={args.target_phase:.3f} x_span={args.target_x_span:.3f}"
        )
    print(
        f"  STO: rho_max={args.rho_max:.0e} rho_warn={args.rho_warn:.0e} "
        f"kappa_max={args.kappa_max:.0e} period={args.kappa_check_period} "
        f"cooldown={args.cooldown}"
    )

    target = generate_target_trajectory(args)
    save_target_outputs(args.log_dir, target)

    modes = ["exact", "sto"] if args.mode == "both" else [args.mode]
    summaries = []
    for mode in modes:
        run_dir = os.path.join(args.log_dir, mode)
        print(f"\n--- {mode.upper()} training ---")
        s = run_training(args, mode, target["midpoints"], run_dir)
        summaries.append(s)
        print(
            f"{mode} final: loss={s['final_loss']:.6e} "
            f"rmse={s['final_rmse']:.3e} final_error={s['final_error']:.3e}"
        )

    save_comparison(args.log_dir, summaries, target["midpoints"])


if __name__ == "__main__":
    main()
