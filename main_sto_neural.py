import numpy as np
import os
import csv
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import argparse

from SimulatorEnv_2D import SimulatorEnv_2D
from implicit_grad_tools import implicit_final_control_grad
from dlo_sto_adapter import DLOStatefulTangentOperator
from utils import plot_sto_lifecycle, visualize_results


class BoundaryVelocityNet(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            # for right end only
            nn.Linear(hidden, 3),
        )

    def forward(self, lam):
        return self.mlp(lam)


def final_midpoint_loss(q, nv, mid_node, target_xy):
    x_arr = q[::2]
    y_arr = q[1::2]
    mx = x_arr[mid_node]
    my = y_arr[mid_node]
    dx = mx - target_xy[0]
    dy = my - target_xy[1]
    L = dx * dx + dy * dy
    return L, (mx, my)


def dloss_dq_free_midpoint(q, free_index, nv, mid_node, target_xy):
    ix = 2 * mid_node
    iy = 2 * mid_node + 1

    mx = q[ix]
    my = q[iy]

    dL_dmx = 2.0 * (mx - target_xy[0])
    dL_dmy = 2.0 * (my - target_xy[1])

    a_free = np.zeros(len(free_index), dtype=float)

    pos_x = np.where(free_index == ix)[0]
    pos_y = np.where(free_index == iy)[0]

    if len(pos_x) == 1:
        a_free[pos_x[0]] = dL_dmx
    if len(pos_y) == 1:
        a_free[pos_y[0]] = dL_dmy

    return a_free


def midpoint_adjoint_to_free(free_index, mid_node, grad_mid):
    """Map dL/d(midpoint) into dL/dq_free."""
    ix = 2 * mid_node
    iy = 2 * mid_node + 1

    a_free = np.zeros(len(free_index), dtype=float)
    pos_x = np.where(free_index == ix)[0]
    pos_y = np.where(free_index == iy)[0]

    if len(pos_x) == 1:
        a_free[pos_x[0]] = grad_mid[0]
    if len(pos_y) == 1:
        a_free[pos_y[0]] = grad_mid[1]

    return a_free


class DLOFinalMidpointImplicitLayer(torch.autograd.Function):
    """
    Detached DLO equilibrium forward with an exact implicit-adjoint backward.

    This mirrors the professor's DEQ example: do not backpropagate through the
    Newton iterations; instead, use the converged equilibrium Jacobian in
    backward. For the current quasi-static final-midpoint loss, the differentiable
    dependence is through the final boundary control u_N.
    """

    @staticmethod
    def forward(
        ctx,
        u_stack,
        env,
        sto,
        record,
        q_start=None,
        state_velocity_start=None,
        current_step_start=0,
    ):
        controls = u_stack.detach().cpu().numpy()

        old_state = snapshot_env_state(env)
        if q_start is None:
            env.reset()
        else:
            set_env_state(
                env,
                q_start,
                state_velocity_start,
                current_step_start,
            )

        info_last = None
        obs_last = None
        try:
            for k in range(controls.shape[0]):
                obs, done, info = env.step(controls[k], use_inertia=False)
                obs_last = obs
                info_last = info
                if done and (k < controls.shape[0] - 1):
                    break
        finally:
            restore_env_state(env, old_state)

        converged = info_last is not None and info_last["converged"]
        record["converged"] = bool(converged)

        if info_last is None:
            midpoint = np.array([np.nan, np.nan], dtype=float)
            ctx.converged = False
        else:
            q = info_last["q"]
            mx = q[2 * env.mid_node]
            my = q[2 * env.mid_node + 1]
            midpoint = np.array([mx, my], dtype=float)

            ctx.converged = bool(converged)
            ctx.J = info_last["J"]
            ctx.free_index = info_last["free_index"]
            ctx.fixed_index = info_last["fixed_index"]
            ctx.control_6d = controls[-1].copy()
            ctx.q_full = q.copy()
            ctx.deltaL = env.deltaL
            ctx.mid_node = env.mid_node
            ctx.sto = sto
            ctx.record = record
            ctx.n_steps = controls.shape[0]
            ctx.n_control = controls.shape[1]

            record["q"] = q.copy()
            record["u_final"] = controls[-1].copy()
            record["midpoint"] = (float(mx), float(my))

        return torch.as_tensor(midpoint, dtype=u_stack.dtype, device=u_stack.device)

    @staticmethod
    def backward(ctx, grad_mid):
        if not getattr(ctx, "converged", False):
            grad_u_stack = torch.zeros(
                (getattr(ctx, "n_steps", 1), getattr(ctx, "n_control", 6)),
                dtype=grad_mid.dtype,
                device=grad_mid.device,
            )
            return grad_u_stack, None, None, None, None, None, None

        grad_mid_np = grad_mid.detach().cpu().numpy()
        a_free = midpoint_adjoint_to_free(
            ctx.free_index,
            ctx.mid_node,
            grad_mid_np,
        )

        t0 = time.perf_counter()
        if ctx.sto is not None:
            g_u_final, used_cache, eta, reason = ctx.sto.query(
                J_full=ctx.J,
                free_index=ctx.free_index,
                fixed_index=ctx.fixed_index,
                a_free=a_free,
                control_6d=ctx.control_6d,
                deltaL=ctx.deltaL,
                q_full=ctx.q_full,
            )
            rho = ctx.sto.last_rho
            kappa = ctx.sto.last_kappa
            matrix_metrics = {
                "sigma_G": ctx.sto.last_metric("sigma_G"),
                "sigma_M": ctx.sto.last_metric("sigma_M"),
                "Gx_norm_fro": ctx.sto.last_metric("Gx_norm_fro"),
                "Gx_norm_1": ctx.sto.last_metric("Gx_norm_1"),
                "rel_Gx_drift_fro": ctx.sto.last_metric("rel_Gx_drift_fro"),
                "anchor_M_norm_fro": ctx.sto.last_metric("anchor_M_norm_fro"),
                "anchor_cond_1_proxy": ctx.sto.last_metric("anchor_cond_1_proxy"),
                "reuse_count": ctx.sto.last_metric("reuse_count"),
            }
        else:
            g_u_final = implicit_final_control_grad(
                J=ctx.J,
                free_index=ctx.free_index,
                fixed_index=ctx.fixed_index,
                a_free=a_free,
                control_6d=ctx.control_6d,
                deltaL=ctx.deltaL,
            )
            used_cache = False
            eta = np.nan
            rho = np.nan
            kappa = np.nan
            reason = "exact_no_sto"
            matrix_metrics = {
                "sigma_G": np.nan,
                "sigma_M": np.nan,
                "Gx_norm_fro": np.nan,
                "Gx_norm_1": np.nan,
                "rel_Gx_drift_fro": np.nan,
                "anchor_M_norm_fro": np.nan,
                "anchor_cond_1_proxy": np.nan,
                "reuse_count": np.nan,
            }
        grad_time = time.perf_counter() - t0

        ctx.record["implicit_grad_seconds"] = grad_time
        ctx.record["g_u_final"] = g_u_final.copy()
        ctx.record["used_cache"] = bool(used_cache)
        ctx.record["rho"] = rho
        ctx.record["kappa"] = kappa
        ctx.record["eta"] = eta
        ctx.record["reason"] = reason
        ctx.record.update(matrix_metrics)

        grad_u_stack = torch.zeros(
            (ctx.n_steps, ctx.n_control),
            dtype=grad_mid.dtype,
            device=grad_mid.device,
        )
        grad_u_stack[-1] = torch.as_tensor(
            g_u_final,
            dtype=grad_mid.dtype,
            device=grad_mid.device,
        )
        return grad_u_stack, None, None, None, None, None, None


class DLOTask1ProxyAdjointLayer(torch.autograd.Function):
    """
    Task-1 terminal midpoint layer with an exact terminal implicit adjoint.

    Forward:
      z_{k+1} = z_k + dlam * rate_k
      x_{k+1} = SolveEq(z_{k+1}; init=x_k)

    Backward:
      for the current terminal-only midpoint objective, only the terminal
      equilibrium contributes a smooth same-branch sensitivity. Since

          z_N = z_0 + dlam * sum_k rate_k,

      every rate receives the same terminal-control gradient

          dL / d rate_k = dlam * S_N^T dL/dx_N.

      This does not differentiate through Newton iterations.
    """

    @staticmethod
    def forward(
        ctx,
        rate_stack,
        control_start,
        env,
        sto,
        record,
        q_start=None,
        state_velocity_start=None,
        current_step_start=0,
        total_control_steps=None,
    ):
        rates = rate_stack.detach().cpu().numpy()
        control = control_start.detach().cpu().numpy().copy()
        n_steps = rates.shape[0]
        if total_control_steps is None:
            total_control_steps = n_steps
        dlam = 1.0 / float(total_control_steps)

        old_state = snapshot_env_state(env)
        step_records = []
        info_last = None
        obs_last = None

        try:
            if q_start is None:
                env.reset()
            else:
                set_env_state(
                    env,
                    q_start,
                    state_velocity_start,
                    current_step_start,
                )

            # Cache the tangent at the segment anchor x_0,z_0. Algorithm 1
            # needs StProd(k, .) before each control update; the first one is
            # the realized equilibrium at the current segment start.
            _, _, info_anchor = env.step(control, use_inertia=False)
            if info_anchor is None or not info_anchor["converged"]:
                record["converged"] = False
                ctx.converged = False
                midpoint = np.array([np.nan, np.nan], dtype=float)
                return torch.as_tensor(
                    midpoint,
                    dtype=rate_stack.dtype,
                    device=rate_stack.device,
                )
            anchor_state_velocity = info_anchor["u"].copy()
            step_records.append(_copy_step_record(info_anchor))
            set_env_state(
                env,
                info_anchor["q"],
                anchor_state_velocity,
                current_step_start,
            )

            for k in range(n_steps):
                control = control + dlam * rates[k]
                obs, done, info = env.step(control, use_inertia=False)
                obs_last = obs
                info_last = info
                if info is None or not info["converged"]:
                    break
                if k < n_steps - 1:
                    step_records.append(_copy_step_record(info))
                if done and (k < n_steps - 1):
                    break
        finally:
            restore_env_state(env, old_state)

        converged = (
            info_last is not None
            and info_last["converged"]
            and len(step_records) == n_steps
        )
        record["converged"] = bool(converged)

        if not converged:
            ctx.converged = False
            midpoint = np.array([np.nan, np.nan], dtype=float)
            return torch.as_tensor(
                midpoint,
                dtype=rate_stack.dtype,
                device=rate_stack.device,
            )

        q = info_last["q"]
        mx = q[2 * env.mid_node]
        my = q[2 * env.mid_node + 1]
        midpoint = np.array([mx, my], dtype=float)
        terminal_record = _copy_step_record(info_last)

        ctx.converged = True
        ctx.step_records = step_records
        ctx.terminal_record = terminal_record
        ctx.deltaL = env.deltaL
        ctx.mid_node = env.mid_node
        ctx.sto = sto
        ctx.record = record
        ctx.n_steps = n_steps
        ctx.n_control = rates.shape[1]
        ctx.dlam = dlam

        record["q"] = q.copy()
        record["u_final"] = control.copy()
        record["midpoint"] = (float(mx), float(my))
        record["adjoint_scheme"] = "terminal_midpoint_adjoint"
        record["n_adjoint_queries"] = 1
        record["q_history"] = np.stack(
            [row["q_full"] for row in step_records] + [terminal_record["q_full"]],
            axis=0,
        )
        record["control_history"] = np.stack(
            [row["control_6d"] for row in step_records] + [terminal_record["control_6d"]],
            axis=0,
        )

        return torch.as_tensor(
            midpoint,
            dtype=rate_stack.dtype,
            device=rate_stack.device,
        )

    @staticmethod
    def backward(ctx, grad_mid):
        if not getattr(ctx, "converged", False):
            grad_rate_stack = torch.zeros(
                (getattr(ctx, "n_steps", 1), getattr(ctx, "n_control", 6)),
                dtype=grad_mid.dtype,
                device=grad_mid.device,
            )
            return grad_rate_stack, None, None, None, None, None, None, None, None

        grad_mid_np = grad_mid.detach().cpu().numpy()
        a_free = midpoint_adjoint_to_free(
            ctx.terminal_record["free_index"],
            ctx.mid_node,
            grad_mid_np,
        )

        t0 = time.perf_counter()
        query_metrics = []

        terminal_s = _stprod_control(
            ctx.terminal_record,
            a_free,
            ctx.deltaL,
            ctx.sto,
            query_metrics,
        )
        grad_rates = np.repeat(
            (ctx.dlam * terminal_s)[None, :],
            ctx.n_steps,
            axis=0,
        )

        grad_time = time.perf_counter() - t0
        _write_proxy_adjoint_record(
            ctx.record,
            grad_time,
            terminal_s,
            grad_rates,
            query_metrics,
            ctx.sto,
        )

        grad_rate_stack = torch.as_tensor(
            grad_rates,
            dtype=grad_mid.dtype,
            device=grad_mid.device,
        )
        return grad_rate_stack, None, None, None, None, None, None, None, None


def _copy_step_record(info):
    return {
        "J": info["J"].copy(),
        "free_index": info["free_index"].copy(),
        "fixed_index": info["fixed_index"].copy(),
        "control_6d": info["control"].copy(),
        "q_full": info["q"].copy(),
    }


def _exact_stprod_control(step_record, a_free, deltaL):
    return implicit_final_control_grad(
        J=step_record["J"],
        free_index=step_record["free_index"],
        fixed_index=step_record["fixed_index"],
        a_free=a_free,
        control_6d=step_record["control_6d"],
        deltaL=deltaL,
    )


def _stprod_control(step_record, a_free, deltaL, sto, query_metrics):
    if sto is None:
        g_control = _exact_stprod_control(step_record, a_free, deltaL)
        query_metrics.append({
            "used_cache": False,
            "rho": np.nan,
            "kappa": np.nan,
            "eta": np.nan,
            "reason": "exact_terminal_adjoint",
            "sigma_G": np.nan,
            "sigma_M": np.nan,
            "Gx_norm_fro": np.nan,
            "Gx_norm_1": np.nan,
            "rel_Gx_drift_fro": np.nan,
            "anchor_M_norm_fro": np.nan,
            "anchor_cond_1_proxy": np.nan,
            "reuse_count": np.nan,
        })
        return g_control

    g_control, used_cache, eta, reason = sto.query(
        J_full=step_record["J"],
        free_index=step_record["free_index"],
        fixed_index=step_record["fixed_index"],
        a_free=a_free,
        control_6d=step_record["control_6d"],
        deltaL=deltaL,
        q_full=step_record["q_full"],
    )
    query_metrics.append({
        "used_cache": bool(used_cache),
        "rho": sto.last_rho,
        "kappa": sto.last_kappa,
        "eta": eta,
        "reason": reason,
        "sigma_G": sto.last_metric("sigma_G"),
        "sigma_M": sto.last_metric("sigma_M"),
        "Gx_norm_fro": sto.last_metric("Gx_norm_fro"),
        "Gx_norm_1": sto.last_metric("Gx_norm_1"),
        "rel_Gx_drift_fro": sto.last_metric("rel_Gx_drift_fro"),
        "anchor_M_norm_fro": sto.last_metric("anchor_M_norm_fro"),
        "anchor_cond_1_proxy": sto.last_metric("anchor_cond_1_proxy"),
        "reuse_count": sto.last_metric("reuse_count"),
    })
    return g_control


def _write_proxy_adjoint_record(
    record,
    grad_time,
    terminal_s,
    grad_rates,
    query_metrics,
    sto,
):
    n_queries = len(query_metrics)
    n_cache = sum(1 for row in query_metrics if row["used_cache"])
    n_exact = n_queries - n_cache if sto is not None else n_queries
    last = query_metrics[-1] if query_metrics else {}
    etas = np.array(
        [row["eta"] for row in query_metrics if np.isfinite(row["eta"])],
        dtype=float,
    )
    rhos = np.array(
        [row["rho"] for row in query_metrics if np.isfinite(row["rho"])],
        dtype=float,
    )

    record["implicit_grad_seconds"] = grad_time
    record["g_u_final"] = terminal_s.copy()
    record["grad_rate_norm"] = float(np.linalg.norm(grad_rates))
    record["used_cache"] = bool(n_queries > 0 and n_cache == n_queries)
    record["sto_cache_hits"] = int(n_cache)
    record["sto_exact_queries"] = int(n_exact)
    record["sto_cache_hit_rate"] = float(n_cache / max(n_queries, 1))
    record["n_adjoint_queries"] = int(n_queries)
    record["rho"] = float(rhos[-1]) if rhos.size else np.nan
    record["rho_max_seen"] = float(rhos.max()) if rhos.size else np.nan
    record["kappa"] = last.get("kappa", np.nan)
    record["eta"] = float(etas[-1]) if etas.size else np.nan
    record["eta_max_seen"] = float(etas.max()) if etas.size else np.nan
    record["reason"] = last.get("reason", "exact_terminal_adjoint")
    for name in [
        "sigma_G",
        "sigma_M",
        "Gx_norm_fro",
        "Gx_norm_1",
        "rel_Gx_drift_fro",
        "anchor_M_norm_fro",
        "anchor_cond_1_proxy",
        "reuse_count",
    ]:
        record[name] = last.get(name, np.nan)


def snapshot_env_state(env):
    """Copy enough simulator state to restore after a differentiable rollout."""
    return {
        "q": None if env.q is None else env.q.copy(),
        "u": None if env.u is None else env.u.copy(),
        "ctime": env.ctime,
        "current_step": env.current_step,
        "lamda": env.lamda,
    }


def restore_env_state(env, state):
    env.q = None if state["q"] is None else state["q"].copy()
    env.u = None if state["u"] is None else state["u"].copy()
    env.ctime = state["ctime"]
    env.current_step = state["current_step"]
    env.lamda = state["lamda"]


def set_env_state(env, q, state_velocity=None, current_step=0):
    env.q = np.asarray(q, dtype=float).copy()
    if state_velocity is None:
        env.u = np.zeros_like(env.q)
    else:
        env.u = np.asarray(state_velocity, dtype=float).copy()
    env.current_step = int(current_step)
    env.ctime = env.current_step * env.dt
    env.lamda = env.current_step * env.dlamda


def build_right_end_control_stack(
    net,
    control_start,
    segment_start_step,
    segment_horizon,
    total_control_steps,
    device,
):
    """Build differentiable 6D boundary controls for one RHC segment."""
    controls = []
    control = control_start
    dlam = 1.0 / total_control_steps

    for local_k in range(segment_horizon):
        global_k = segment_start_step + local_k
        lam = torch.tensor(
            [[global_k / total_control_steps]],
            dtype=torch.float32,
            device=device,
        )
        u_dot_full = torch.cat(
            [torch.zeros(3, dtype=torch.float32, device=device), net(lam).squeeze(0)]
        )
        control = control + dlam * u_dot_full
        controls.append(control)

    return torch.stack(controls, dim=0)


def build_right_end_rate_stack(
    net,
    segment_start_step,
    segment_horizon,
    total_control_steps,
    device,
):
    """Build differentiable 6D boundary-control rates for one segment."""
    rates = []
    for local_k in range(segment_horizon):
        global_k = segment_start_step + local_k
        lam = torch.tensor(
            [[global_k / total_control_steps]],
            dtype=torch.float32,
            device=device,
        )
        rate_full = torch.cat(
            [torch.zeros(3, dtype=torch.float32, device=device), net(lam).squeeze(0)]
        )
        rates.append(rate_full)
    return torch.stack(rates, dim=0)


def rollout_controls_from_state(env, controls, q_start, state_velocity_start, current_step_start):
    """Execute a fixed control segment from a saved simulator state."""
    set_env_state(env, q_start, state_velocity_start, current_step_start)
    info_last = None
    midpoint_history = []
    control_history = []
    q_history = []

    for k, control in enumerate(controls):
        obs, done, info = env.step(control, use_inertia=False)
        info_last = info
        midpoint_history.append((obs["midx"], obs["midy"]))
        control_history.append(np.asarray(control, dtype=float).copy())
        if info is not None and "q" in info:
            q_history.append(info["q"].copy())
        if done and (k < len(controls) - 1):
            break

    converged = info_last is not None and info_last["converged"]
    return {
        "converged": converged,
        "info": info_last,
        "midpoint_history": midpoint_history,
        "control_history": control_history,
        "q_history": q_history,
    }


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200, help="epochs")
    parser.add_argument("--seed", type=int, default=0,
                        help="random seed for reproducible controller initialization")
    parser.add_argument("--rhc", action="store_true",
                        help="train with receding-horizon control segments")
    parser.add_argument("--rhc_segments", type=int, default=5,
                        help="number of RHC segments")
    parser.add_argument("--segment_horizon", type=int, default=10,
                        help="number of control steps optimized per RHC segment")
    parser.add_argument("--hidden", type=int, default=64,
                        help="hidden width for the boundary velocity network")
    parser.add_argument("--n_nodes", type=int, default=51,
                        help="number of strip nodes")
    parser.add_argument("--n_steps", type=int, default=100,
                        help="number of continuation steps in a full rollout")
    parser.add_argument("--use_sto", action="store_true", default=True,
                        help="use STO(default: True)")
    parser.add_argument("--no_sto", action="store_true",
                        help="no STO，exact solve every epoch")
    parser.add_argument("--rho_max", type=float, default=1e-1,
                        help="STO probe-residual gate: reject if rho > rho_max")
    parser.add_argument("--kappa_max", type=float, default=1e10,
                        help="STO conditioning gate: reject if kappa > kappa_max")
    parser.add_argument("--rho_warn", type=float, default=1e-2,
                        help="STO warning band: rho above this latches kappa-every-epoch mode")
    parser.add_argument("--kappa_check_period", type=int, default=10,
                        help="STO cadence: recompute kappa every N validates when calm")
    parser.add_argument("--cooldown", type=int, default=3,
                        help="STO cadence: consecutive low-rho validates needed to exit warning")
    parser.add_argument("--n_probes", type=int, default=4,
                        help="number of fixed sentinel probes for STO validation")
    parser.add_argument("--n_power_iter", type=int, default=3,
                        help="power iterations for STO conditioning estimate")
    parser.add_argument("--max_reuse", type=int, default=50,
                        help="STO invalidation: max cache reuse count")
    parser.add_argument("--log_dir", type=str, default=None,
                        help="directory for TensorBoard logs, plots, and timing CSV")
    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    epochs = args.epochs
    use_sto = args.use_sto and (not args.no_sto)
    use_rhc = bool(args.rhc)

    # -----------------------------
    # 1) env
    # -----------------------------
    env = SimulatorEnv_2D(
        nv=args.n_nodes,
        dt=1.0 / args.n_steps,
        rod_length=1.0,
        total_time=1.0,
        R_outer=1.0e-3,
        r_inner=0.0,
        E_al=1.0e5,
        rho_al=1070,
        max_newton_iter=1000,
        save_history=False
    )

    _ = env.reset()

    N = env.Nsteps
    dlam = 1.0 / N

    # -----------------------------
    # 2) target
    # -----------------------------
    target_xy = np.array([0.2, 0.4], dtype=float)

    # -----------------------------
    # 3) nn
    # -----------------------------
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    u0 = torch.tensor([0.0, 0.0, 0.0, env.rod_length, 0.0, 0.0],
                       dtype=torch.float32, device=device)

    net = BoundaryVelocityNet(hidden=args.hidden).to(device)
    opt = optim.Adam(net.parameters(), lr=1e-3)

    log_dir = args.log_dir or (
        "runs/dlo_task1_proxy_sto" if use_sto else "runs/dlo_task1_proxy_exact"
    )
    writer = SummaryWriter(log_dir=log_dir)
    os.makedirs(log_dir, exist_ok=True)

    # -----------------------------
    # 4) STO
    # -----------------------------
    sto = None
    if use_sto:
        sto = DLOStatefulTangentOperator(
            rho_max=args.rho_max,
            kappa_max=args.kappa_max,
            rho_warn=args.rho_warn,
            kappa_check_period=args.kappa_check_period,
            cooldown=args.cooldown,
            n_probes=args.n_probes,
            n_power_iter=args.n_power_iter,
            max_reuse=args.max_reuse,
        )

    print("Network structure:")
    print(net)
    print(f"Training on device: {device}")
    print(f"STO enabled: {use_sto}")
    print(f"RHC enabled: {use_rhc}")
    print(f"Strip nodes: {args.n_nodes}  continuation steps: {N}")
    if use_rhc:
        print(f"  rhc_segments={args.rhc_segments}  "
              f"segment_horizon={args.segment_horizon}  "
              f"epochs_per_segment={epochs}")
    if use_sto:
        print(f"  rho_max={args.rho_max:.2e}  kappa_max={args.kappa_max:.0e}  "
              f"rho_warn={args.rho_warn:.2e}  kappa_check_period={args.kappa_check_period}  "
              f"cooldown={args.cooldown}  n_probes={args.n_probes}  "
              f"n_power_iter={args.n_power_iter}  max_reuse={args.max_reuse}")
    print("Starting training...")

    final_loss = None
    loss_history = []
    # (epoch, all_cache, eta, reason, n_queries, n_cache, n_exact, hit_rate)
    sto_events = []
    grad_time_history = []  # (epoch, implicit-gradient seconds)
    timing_rows = []
    q_last = None
    q_history_last = None
    control_history_last = None
    L_np = None
    segment_loss_history = []
    execution_midpoints = []
    execution_controls = []
    execution_q_history = []

    def log_backward_record(
        *,
        global_ep,
        segment,
        local_epoch,
        loss_value,
        midpoint,
        q_value,
        g_u_final,
        u_final_norm,
        backward_record,
        pbar=None,
    ):
        nonlocal final_loss, L_np, q_last, q_history_last, control_history_last

        mx, my = midpoint
        q_last = q_value
        q_history_last = backward_record.get("q_history")
        control_history_last = backward_record.get("control_history")
        grad_time = backward_record["implicit_grad_seconds"]
        used_cache = backward_record["used_cache"]
        rho = backward_record["rho"]
        kappa = backward_record["kappa"]
        eta = backward_record["eta"]
        reason = backward_record["reason"]
        n_adjoint_queries = backward_record.get("n_adjoint_queries", 1)
        sto_cache_hits = backward_record.get("sto_cache_hits", int(used_cache))
        sto_exact_queries = backward_record.get(
            "sto_exact_queries",
            0 if used_cache else n_adjoint_queries,
        )
        sto_cache_hit_rate = backward_record.get(
            "sto_cache_hit_rate",
            float(used_cache),
        )
        grad_rate_norm = backward_record.get(
            "grad_rate_norm",
            float(np.linalg.norm(g_u_final)),
        )
        matrix_metrics = {
            "sigma_G": backward_record["sigma_G"],
            "sigma_M": backward_record["sigma_M"],
            "Gx_norm_fro": backward_record["Gx_norm_fro"],
            "Gx_norm_1": backward_record["Gx_norm_1"],
            "rel_Gx_drift_fro": backward_record["rel_Gx_drift_fro"],
            "anchor_M_norm_fro": backward_record["anchor_M_norm_fro"],
            "anchor_cond_1_proxy": backward_record["anchor_cond_1_proxy"],
            "reuse_count": backward_record["reuse_count"],
        }

        if use_sto:
            sto_events.append((
                global_ep,
                used_cache,
                eta,
                reason,
                n_adjoint_queries,
                sto_cache_hits,
                sto_exact_queries,
                sto_cache_hit_rate,
            ))
            writer.add_scalar("sto/used_cache", float(used_cache), global_ep)
            writer.add_scalar("sto/cache_hit_rate", sto_cache_hit_rate, global_ep)
            writer.add_scalar("sto/cache_hits", sto_cache_hits, global_ep)
            writer.add_scalar("sto/exact_queries", sto_exact_queries, global_ep)
            writer.add_scalar("sto/eta", eta, global_ep)
            writer.add_scalar("sto/rho", rho, global_ep)
            writer.add_scalar("sto/kappa", kappa, global_ep)
            writer.add_scalar("sto/rel_Gx_drift_fro", matrix_metrics["rel_Gx_drift_fro"], global_ep)
            writer.add_scalar("sto/sigma_M", matrix_metrics["sigma_M"], global_ep)
            writer.add_scalar("sto/sigma_G", matrix_metrics["sigma_G"], global_ep)

        grad_time_history.append((global_ep, grad_time))
        timing_rows.append({
            "epoch": global_ep,
            "segment": segment,
            "local_epoch": local_epoch,
            "mode": "sto" if use_sto else "exact",
            "implicit_grad_seconds": grad_time,
            "used_cache": int(used_cache),
            "rho": rho,
            "kappa": kappa,
            "eta": eta,
            "reason": reason,
            "n_adjoint_queries": n_adjoint_queries,
            "sto_cache_hits": sto_cache_hits,
            "sto_exact_queries": sto_exact_queries,
            "sto_cache_hit_rate": sto_cache_hit_rate,
            "grad_rate_norm": grad_rate_norm,
            **matrix_metrics,
            "loss": loss_value,
            "midpoint_x": mx,
            "midpoint_y": my,
        })
        writer.add_scalar("timing/implicit_grad_seconds", grad_time, global_ep)
        writer.add_scalar(
            "timing/cumulative_implicit_grad_seconds",
            sum(t for _, t in grad_time_history),
            global_ep,
        )

        writer.add_scalar("loss/final", loss_value, global_ep)
        writer.add_scalar("midpoint/x", mx, global_ep)
        writer.add_scalar("midpoint/y", my, global_ep)
        writer.add_scalar("physics/newton_converged", 1.0, global_ep)
        writer.add_scalar("grad/gu_final_norm", np.linalg.norm(g_u_final), global_ep)
        writer.add_scalar("grad/rate_stack_norm", grad_rate_norm, global_ep)
        writer.add_scalar("control/u_final_norm", u_final_norm, global_ep)

        if pbar is not None:
            pbar.set_postfix({"loss": f"{loss_value:.6f}"})

        L_np = loss_value
        final_loss = loss_value
        loss_history.append((global_ep, loss_value))

    target_torch = torch.as_tensor(target_xy, dtype=torch.float32, device=device)

    if use_rhc:
        env.reset()
        q_segment_start = env.q.copy()
        state_velocity_start = env.u.copy()
        control_start_np = np.array([0.0, 0.0, 0.0, env.rod_length, 0.0, 0.0], dtype=float)
        current_step_start = 0
        total_control_steps = args.rhc_segments * args.segment_horizon
        global_ep = 0

        for seg in range(args.rhc_segments):
            print(f"\nRHC segment {seg + 1}/{args.rhc_segments}")
            net = BoundaryVelocityNet(hidden=args.hidden).to(device)
            opt = optim.Adam(net.parameters(), lr=1e-3)

            best_loss = float("inf")
            best_state = None

            pbar = tqdm(range(epochs), desc=f"Segment {seg + 1}", dynamic_ncols=True)
            for local_ep in pbar:
                control_start_torch = torch.as_tensor(
                    control_start_np,
                    dtype=torch.float32,
                    device=device,
                )
                rate_stack = build_right_end_rate_stack(
                    net,
                    current_step_start,
                    args.segment_horizon,
                    total_control_steps,
                    device,
                )
                backward_record = {}
                midpoint_torch = DLOTask1ProxyAdjointLayer.apply(
                    rate_stack,
                    control_start_torch,
                    env,
                    sto if use_sto else None,
                    backward_record,
                    q_segment_start,
                    state_velocity_start,
                    current_step_start,
                    total_control_steps,
                )

                if not backward_record.get("converged", False):
                    pbar.set_postfix({"status": "Newton failed"})
                    writer.add_scalar("physics/newton_converged", 0.0, global_ep)
                    global_ep += 1
                    continue

                loss_torch = torch.sum((midpoint_torch - target_torch) ** 2)

                opt.zero_grad()
                loss_torch.backward()
                opt.step()

                loss_value = float(loss_torch.detach().cpu())
                if loss_value < best_loss:
                    best_loss = loss_value
                    best_state = {
                        name: value.detach().cpu().clone()
                        for name, value in net.state_dict().items()
                    }

                log_backward_record(
                    global_ep=global_ep,
                    segment=seg,
                    local_epoch=local_ep,
                    loss_value=loss_value,
                    midpoint=backward_record["midpoint"],
                    q_value=backward_record["q"],
                    g_u_final=backward_record["g_u_final"],
                    u_final_norm=float(np.linalg.norm(backward_record["u_final"])),
                    backward_record=backward_record,
                    pbar=pbar,
                )

                if global_ep % 10 == 0:
                    sto_tag = ""
                    if use_sto and sto_events:
                        last_event = sto_events[-1]
                        sto_tag = (
                            f"  [STO hit={100.0 * last_event[7]:.1f}%  "
                            f"eta={last_event[2]:.1e}]"
                        )
                    mx, my = backward_record["midpoint"]
                    print(f"[seg {seg:02d} ep {local_ep:04d}] L={loss_value:.6f}  "
                          f"mid=({mx:.3f},{my:.3f})  target=({target_xy[0]:.3f},{target_xy[1]:.3f})  "
                          f"implicit_grad={1e3 * backward_record['implicit_grad_seconds']:.2f} ms{sto_tag}")

                global_ep += 1

            if best_state is not None:
                net.load_state_dict(best_state)

            control_start_torch = torch.as_tensor(
                control_start_np,
                dtype=torch.float32,
                device=device,
            )
            with torch.no_grad():
                exec_controls = build_right_end_control_stack(
                    net,
                    control_start_torch,
                    current_step_start,
                    args.segment_horizon,
                    total_control_steps,
                    device,
                ).detach().cpu().numpy()

            exec_result = rollout_controls_from_state(
                env,
                exec_controls,
                q_segment_start,
                state_velocity_start,
                current_step_start,
            )
            if not exec_result["converged"]:
                print(f"RHC segment {seg + 1} execution failed; stopping.")
                break

            info_exec = exec_result["info"]
            q_segment_start = info_exec["q"].copy()
            state_velocity_start = info_exec["u"].copy()
            control_start_np = exec_controls[-1].copy()
            current_step_start += len(exec_controls)
            q_last = q_segment_start.copy()
            L_np, midpoint_np = final_midpoint_loss(q_last, env.nv, env.mid_node, target_xy)
            final_loss = float(L_np)
            segment_loss_history.append((seg, best_loss, final_loss))
            execution_midpoints.extend(exec_result["midpoint_history"])
            execution_controls.extend(exec_result["control_history"])
            execution_q_history.extend(exec_result["q_history"])

            print(f"Executed segment {seg + 1}: best_train_loss={best_loss:.6f}  "
                  f"exec_loss={final_loss:.6f}  "
                  f"mid=({midpoint_np[0]:.3f},{midpoint_np[1]:.3f})")
    else:
        pbar = tqdm(range(epochs), desc="Training", dynamic_ncols=True)
        for ep in pbar:
            rate_stack = build_right_end_rate_stack(
                net,
                0,
                N,
                N,
                device,
            )
            backward_record = {}
            midpoint_torch = DLOTask1ProxyAdjointLayer.apply(
                rate_stack,
                u0,
                env,
                sto if use_sto else None,
                backward_record,
                None,
                None,
                0,
                N,
            )

            if not backward_record.get("converged", False):
                pbar.set_postfix({"status": "Newton failed"})
                writer.add_scalar("physics/newton_converged", 0.0, ep)
                continue

            loss_torch = torch.sum((midpoint_torch - target_torch) ** 2)

            opt.zero_grad()
            loss_torch.backward()
            opt.step()

            loss_value = float(loss_torch.detach().cpu())
            log_backward_record(
                global_ep=ep,
                segment=0,
                local_epoch=ep,
                loss_value=loss_value,
                midpoint=backward_record["midpoint"],
                q_value=backward_record["q"],
                g_u_final=backward_record["g_u_final"],
                u_final_norm=float(np.linalg.norm(backward_record["u_final"])),
                backward_record=backward_record,
                pbar=pbar,
            )

            if ep % 10 == 0:
                sto_tag = ""
                if use_sto and sto_events:
                    last_event = sto_events[-1]
                    sto_tag = (
                        f"  [STO hit={100.0 * last_event[7]:.1f}%  "
                        f"eta={last_event[2]:.1e}]"
                    )
                mx, my = backward_record["midpoint"]
                print(f"[ep {ep:04d}] L={loss_value:.6f}  mid=({mx:.3f},{my:.3f})  "
                      f"target=({target_xy[0]:.3f},{target_xy[1]:.3f})  "
                      f"implicit_grad={1e3 * backward_record['implicit_grad_seconds']:.2f} ms{sto_tag}")

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------
    if final_loss is not None:
        print(f"\nTraining finished. Final loss: {final_loss:.6f}")
    else:
        print("\nTraining finished. No converged epoch recorded.")

    if use_sto and sto is not None:
        print(sto.stats_str())

    if grad_time_history:
        grad_times = np.array([t for _, t in grad_time_history], dtype=float)
        print(
            "Implicit gradient timing: "
            f"total={grad_times.sum():.4f}s  "
            f"mean={grad_times.mean() * 1e3:.3f}ms  "
            f"median={np.median(grad_times) * 1e3:.3f}ms"
        )
        if use_sto and sto_events:
            cache_times = np.array(
                [t for (_, t), event in zip(grad_time_history, sto_events)
                 if event[7] >= 0.999],
                dtype=float,
            )
            exact_times = np.array(
                [t for (_, t), event in zip(grad_time_history, sto_events)
                 if event[6] > 0],
                dtype=float,
            )
            if cache_times.size > 0:
                print(f"  STO all-cache backward mean: {cache_times.mean() * 1e3:.3f}ms")
            if exact_times.size > 0:
                print(f"  STO with exact/fallback mean: {exact_times.mean() * 1e3:.3f}ms")

    # ---------------------------------------------------------
    # Visualize results
    # ---------------------------------------------------------
    results = {
        'final_q': q_last,
        'final_error': float(L_np ** 0.5) if L_np is not None else float('inf'),
        'all_losses': [l for _, l in loss_history],
        'q_history': (
            np.asarray(execution_q_history, dtype=float)
            if use_rhc and execution_q_history
            else q_history_last
        ),
    }
    visualize_results(results, env, target_xy, save_path=log_dir)

    if results["q_history"] is not None:
        q_hist_arr = np.asarray(results["q_history"], dtype=float)
        shape_path = f"{log_dir}/shape_history.npz"
        np.savez(
            shape_path,
            q_history=q_hist_arr,
            target_xy=target_xy,
            final_q=q_last,
            control_history=(
                np.asarray(execution_controls, dtype=float)
                if use_rhc and execution_controls
                else (
                    np.asarray(control_history_last, dtype=float)
                    if control_history_last is not None
                    else np.empty((0, 6), dtype=float)
                )
            ),
        )
        print(f"Shape history saved to: {shape_path}")

    # Loss curve
    if loss_history:
        eps_list, losses = zip(*loss_history)
        fig, ax = plt.subplots(figsize=(4.2, 2.8))
        ax.plot(eps_list, losses, linewidth=1.4, color="#1f77b4")
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.set_title("loss")
        ax.grid(True, alpha=0.2, linewidth=0.6)
        fig.tight_layout()
        save_path = f"{log_dir}/loss_curve.png"
        fig.savefig(save_path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        print(f"Loss curve saved to: {save_path}")

    if use_rhc and segment_loss_history:
        segment_csv = f"{log_dir}/rhc_segment_losses.csv"
        with open(segment_csv, "w", newline="") as f:
            writer_csv = csv.DictWriter(
                f,
                fieldnames=["segment", "best_train_loss", "executed_loss"],
            )
            writer_csv.writeheader()
            for seg, best_train_loss, executed_loss in segment_loss_history:
                writer_csv.writerow({
                    "segment": seg,
                    "best_train_loss": best_train_loss,
                    "executed_loss": executed_loss,
                })
        print(f"RHC segment-loss CSV saved to: {segment_csv}")

        seg_vals = np.array([seg for seg, _, _ in segment_loss_history], dtype=int)
        best_losses = np.array([loss for _, loss, _ in segment_loss_history], dtype=float)
        executed_losses = np.array([loss for _, _, loss in segment_loss_history], dtype=float)

        fig, ax = plt.subplots(figsize=(4.2, 2.8))
        ax.semilogy(seg_vals, best_losses, "o-", lw=1.4, ms=3.5, label="train")
        ax.semilogy(seg_vals, executed_losses, "s--", lw=1.4, ms=3.5, label="exec")
        ax.set_xlabel("segment")
        ax.set_ylabel("loss")
        ax.set_title("RHC loss")
        ax.grid(True, alpha=0.2, linewidth=0.6)
        ax.legend(fontsize=7, frameon=False)
        fig.tight_layout()
        seg_plot = f"{log_dir}/rhc_segment_losses.png"
        fig.savefig(seg_plot, dpi=220, bbox_inches="tight")
        plt.close(fig)
        print(f"RHC segment-loss plot saved to: {seg_plot}")

    if use_rhc and execution_midpoints:
        mids = np.array(execution_midpoints, dtype=float)
        controls = np.array(execution_controls, dtype=float)
        step_vals = np.arange(1, len(mids) + 1)

        fig, axes = plt.subplots(2, 1, figsize=(5.2, 4.0), sharex=True)
        axes[0].plot(step_vals, mids[:, 0], lw=1.5, label="x")
        axes[0].plot(step_vals, mids[:, 1], lw=1.5, label="y")
        axes[0].axhline(target_xy[0], color="0.35", ls="--", lw=0.9)
        axes[0].axhline(target_xy[1], color="0.35", ls=":", lw=1.0)
        axes[0].set_ylabel("midpoint")
        axes[0].set_title("execution")
        axes[0].grid(True, alpha=0.2, linewidth=0.6)
        axes[0].legend(fontsize=7, frameon=False, ncol=2)

        axes[1].plot(step_vals, controls[:, 3], lw=1.3, label="x")
        axes[1].plot(step_vals, controls[:, 4], lw=1.3, label="y")
        axes[1].plot(step_vals, controls[:, 5], lw=1.3, label="theta")
        axes[1].set_xlabel("step")
        axes[1].set_ylabel("right end")
        axes[1].grid(True, alpha=0.2, linewidth=0.6)
        axes[1].legend(fontsize=7, frameon=False, ncol=3)

        fig.tight_layout()
        exec_plot = f"{log_dir}/rhc_execution.png"
        fig.savefig(exec_plot, dpi=220, bbox_inches="tight")
        plt.close(fig)
        print(f"RHC execution plot saved to: {exec_plot}")

    # Backward-gradient timing artifacts
    if timing_rows:
        timing_csv = f"{log_dir}/implicit_gradient_timing.csv"
        with open(timing_csv, "w", newline="") as f:
            fieldnames = [
                "epoch", "segment", "local_epoch", "mode",
                "implicit_grad_seconds", "used_cache",
                "rho", "kappa", "eta", "reason",
                "n_adjoint_queries", "sto_cache_hits", "sto_exact_queries",
                "sto_cache_hit_rate", "grad_rate_norm",
                "sigma_G", "sigma_M", "Gx_norm_fro", "Gx_norm_1",
                "rel_Gx_drift_fro", "anchor_M_norm_fro", "anchor_cond_1_proxy",
                "reuse_count",
                "loss", "midpoint_x", "midpoint_y",
            ]
            writer_csv = csv.DictWriter(f, fieldnames=fieldnames)
            writer_csv.writeheader()
            writer_csv.writerows(timing_rows)
        print(f"Implicit-gradient timing CSV saved to: {timing_csv}")

        ep_vals = np.array([ep_i for ep_i, _ in grad_time_history], dtype=int)
        grad_times = np.array([t for _, t in grad_time_history], dtype=float)
        cumulative = np.cumsum(grad_times)

        fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))

        ax = axes[0]
        if use_sto and sto_events:
            cache_mask = np.array([event[7] >= 0.999 for event in sto_events], dtype=bool)
            ax.plot(ep_vals, 1e3 * grad_times, color="0.75", lw=0.9, zorder=1)
            ax.scatter(ep_vals[cache_mask], 1e3 * grad_times[cache_mask],
                       s=16, color="#2ca02c", label="cache", zorder=2)
            ax.scatter(ep_vals[~cache_mask], 1e3 * grad_times[~cache_mask],
                       s=16, color="#d62728", label="exact", zorder=2)
            ax.legend(fontsize=7, frameon=False)
        else:
            ax.plot(ep_vals, 1e3 * grad_times, "o-", color="#1f77b4", ms=3,
                    label="exact")
            ax.legend(fontsize=7, frameon=False)
        ax.set_xlabel("epoch")
        ax.set_ylabel("ms")
        ax.set_title("backward")
        ax.grid(True, alpha=0.2, linewidth=0.6)

        ax = axes[1]
        ax.plot(ep_vals, cumulative, lw=1.5, color="#9467bd")
        ax.set_xlabel("epoch")
        ax.set_ylabel("s")
        ax.set_title("cumulative")
        ax.grid(True, alpha=0.2, linewidth=0.6)

        fig.tight_layout()
        timing_plot = f"{log_dir}/implicit_gradient_timing.png"
        fig.savefig(timing_plot, dpi=220, bbox_inches="tight")
        plt.close(fig)
        print(f"Implicit-gradient timing plot saved to: {timing_plot}")

    # STO lifecycle plot (if STO was used)
    if use_sto and sto_events:
        plot_sto_lifecycle(
            sto_events=sto_events,
            timing_rows=timing_rows,
            rho_max=args.rho_max,
            rho_warn=args.rho_warn,
            kappa_max=args.kappa_max,
            save_path=f"{log_dir}/sto_diagnostics.png",
        )

    writer.close()


"""
Usage:
  python main_sto_neural --no_sto         # exact terminal adjoint
  python main_sto_neural                  # STO diagnostic run
  python main_sto_neural --epochs 500     # more epochs
  python main_sto_neural --rho_max 1e-2   # stricter STO residual gate
  python main_sto_neural --n_probes 8     # more sentinel probes
  python main_sto_neural --n_power_iter 5 # stricter conditioning estimate
"""

if __name__ == "__main__":
    main()
