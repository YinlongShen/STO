import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

def visualize_results(results, env, target_pos, save_path='img'):
    """Visualize results"""
    final_q = results['final_q']
    x_arr = final_q[::2]
    y_arr = final_q[1::2]
    mid_idx = env.mid_node
    midx, midy = x_arr[mid_idx], y_arr[mid_idx]
    
    # Plot configuration
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.8))
    
    # Configuration plot
    ax1.plot(x_arr, y_arr, 'o-', color='0.15', linewidth=1.4,
             markersize=3.5, label='strip')
    ax1.plot([x_arr[0], x_arr[-1]], [y_arr[0], y_arr[-1]], 's',
             color='0.45', markersize=5, label='ends')
    ax1.plot(target_pos[0], target_pos[1], 'o', color='#d62728',
             markersize=5, label='target', zorder=10)
    ax1.plot(midx, midy, 'x', color='#1f77b4',
             markersize=7, markeredgewidth=1.8, label='final mid', zorder=11)
    ax1.plot([midx, target_pos[0]], [midy, target_pos[1]], '--',
             color='0.55', linewidth=0.9)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('shape')
    ax1.axis('equal')
    ax1.legend(fontsize=7, frameon=False, loc='best')
    ax1.grid(True, alpha=0.2, linewidth=0.6)
    
    # Loss curve
    ax2.semilogy(results['all_losses'], linewidth=1.5, color='#1f77b4')
    ax2.set_xlabel('epoch')
    ax2.set_ylabel('loss')
    ax2.set_title('training')
    ax2.grid(True, alpha=0.2, linewidth=0.6)
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/neural_control_result.png', dpi=220, bbox_inches='tight')
    print(f"\nSaved to {save_path}/neural_control_result.png")
    plt.close()

    q_history = results.get('q_history')
    if q_history is None:
        return

    q_history = np.asarray(q_history, dtype=float)
    if q_history.ndim != 2 or q_history.shape[0] == 0:
        return

    n_draw = min(8, q_history.shape[0])
    draw_idx = np.unique(np.linspace(0, q_history.shape[0] - 1, n_draw, dtype=int))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(draw_idx)))

    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    for color, idx in zip(colors, draw_idx):
        q = q_history[idx]
        ax.plot(q[::2], q[1::2], 'o-', color=color, lw=1.0,
                ms=2.4, alpha=0.85)

    q0 = q_history[draw_idx[0]]
    qN = q_history[draw_idx[-1]]
    final_mid = q_history[-1, 2 * mid_idx:2 * mid_idx + 2]
    ax.plot(q0[::2], q0[1::2], color='0.6', lw=1.0, alpha=0.7)
    ax.plot(qN[::2], qN[1::2], color='0.05', lw=1.4, label='strip')
    ax.plot(target_pos[0], target_pos[1], 'o', color='#d62728',
            markersize=5, label='target')
    ax.plot(final_mid[0], final_mid[1], 'x', color='#1f77b4',
            markersize=7, markeredgewidth=1.8, label='final mid')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('continuation')
    ax.axis('equal')
    ax.grid(True, alpha=0.2, linewidth=0.6)
    ax.legend(fontsize=7, frameon=False, loc='best')
    fig.tight_layout()
    fig.savefig(f'{save_path}/intermediate_shapes.png', dpi=220, bbox_inches='tight')
    print(f"Intermediate shapes saved to {save_path}/intermediate_shapes.png")
    plt.close(fig)


def _sto_event_value(event, tuple_idx, key, default=np.nan):
    if isinstance(event, dict):
        return event.get(key, default)
    if tuple_idx is None or tuple_idx >= len(event):
        return default
    return event[tuple_idx]


def _sto_timing_value(row, *keys):
    for key in keys:
        if isinstance(row, dict) and key in row:
            value = row[key]
            if np.isfinite(value):
                return float(value)
    return np.nan


def plot_sto_lifecycle(
    sto_events,
    timing_rows,
    rho_max,
    rho_warn,
    kappa_max,
    save_path,
    title="STO lifecycle",
):
    """
    One-panel STO lifecycle plot.

    Background bands show the lifecycle decision:
      green = all cached, orange = mixed, red = exact/fallback.
    Lines show raw gate values on separate axes:
      left = rho, right = kappa.
    """
    if not sto_events:
        return

    ep_vals = np.asarray(
        [_sto_event_value(event, 0, "epoch", i) for i, event in enumerate(sto_events)],
        dtype=float,
    )
    n_queries = np.asarray(
        [_sto_event_value(event, 4, "n_queries", 1.0) for event in sto_events],
        dtype=float,
    )
    n_cache = np.asarray(
        [_sto_event_value(event, 5, "n_cache", 0.0) for event in sto_events],
        dtype=float,
    )
    n_exact = np.asarray(
        [_sto_event_value(event, 6, "n_exact", 0.0) for event in sto_events],
        dtype=float,
    )

    status = []
    for cache_i, exact_i, total_i in zip(n_cache, n_exact, n_queries):
        if total_i > 0 and cache_i >= total_i:
            status.append("cache")
        elif cache_i > 0:
            status.append("mixed")
        else:
            status.append("exact")

    status_colors = {
        "cache": "#009E73",
        "mixed": "#E69F00",
        "exact": "#D55E00",
    }
    status_labels = {
        "cache": "reuse",
        "mixed": "mixed",
        "exact": "solve",
    }

    rho_vals = []
    kappa_vals = []
    for row in timing_rows:
        rho_vals.append(_sto_timing_value(row, "rho_max_seen", "rho"))
        kappa_vals.append(_sto_timing_value(row, "kappa"))
    rho_vals = np.asarray(rho_vals, dtype=float)
    kappa_vals = np.asarray(kappa_vals, dtype=float)

    n = min(len(ep_vals), max(len(rho_vals), len(kappa_vals), len(ep_vals)))
    ep_vals = ep_vals[:n]
    status = status[:n]
    if rho_vals.size < n:
        rho_vals = np.pad(rho_vals, (0, n - rho_vals.size), constant_values=np.nan)
    if kappa_vals.size < n:
        kappa_vals = np.pad(kappa_vals, (0, n - kappa_vals.size), constant_values=np.nan)
    rho_vals = rho_vals[:n]
    kappa_vals = kappa_vals[:n]

    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    ax_kappa = ax.twinx()
    ax.set_yscale("log")
    ax_kappa.set_yscale("log")
    for ep, state in zip(ep_vals, status):
        ax.axvspan(
            ep - 0.5,
            ep + 0.5,
            color=status_colors[state],
            alpha=0.14,
            linewidth=0,
        )

    line_handles = []
    finite_rho = np.isfinite(rho_vals) & (rho_vals > 0)
    if np.any(finite_rho):
        (rho_line,) = ax.semilogy(
            ep_vals[finite_rho],
            np.maximum(rho_vals[finite_rho], 1e-30),
            color="#0072B2",
            lw=1.5,
            label="rho",
        )
        line_handles.append(rho_line)

    finite_kappa = np.isfinite(kappa_vals) & (kappa_vals > 0)
    if np.any(finite_kappa):
        (kappa_line,) = ax_kappa.semilogy(
            ep_vals[finite_kappa],
            np.maximum(kappa_vals[finite_kappa], 1e-30),
            color="#CC79A7",
            lw=1.2,
            drawstyle="steps-post",
            label="kappa",
        )
        line_handles.append(kappa_line)

    if rho_max > 0:
        ax.axhline(rho_max, color="#0072B2", lw=0.9, ls="--")
    if rho_warn > 0:
        ax.axhline(rho_warn, color="#E69F00", lw=0.9, ls=":")
    if kappa_max > 0:
        ax_kappa.axhline(kappa_max, color="#CC79A7", lw=0.9, ls="--")

    rho_axis_values = rho_vals[np.isfinite(rho_vals) & (rho_vals > 0)]
    if rho_max > 0:
        rho_axis_values = np.concatenate([rho_axis_values, [rho_max]])
    if rho_warn > 0:
        rho_axis_values = np.concatenate([rho_axis_values, [rho_warn]])
    if rho_axis_values.size:
        y_min = max(np.nanmin(rho_axis_values) * 0.4, 1e-12)
        y_max = max(np.nanmax(rho_axis_values) * 2.5, 1.0)
        ax.set_ylim(y_min, y_max)

    kappa_axis_values = kappa_vals[np.isfinite(kappa_vals) & (kappa_vals > 0)]
    if kappa_max > 0:
        kappa_axis_values = np.concatenate([kappa_axis_values, [kappa_max]])
    if kappa_axis_values.size:
        y_min = max(np.nanmin(kappa_axis_values) * 0.4, 1e-12)
        y_max = max(np.nanmax(kappa_axis_values) * 2.5, 1.0)
        ax_kappa.set_ylim(y_min, y_max)

    ax.set_xlabel("epoch")
    ax.set_ylabel("rho", color="#0072B2")
    ax_kappa.set_ylabel("kappa", color="#CC79A7")
    ax.tick_params(axis="y", labelcolor="#0072B2")
    ax_kappa.tick_params(axis="y", labelcolor="#CC79A7")
    ax.set_title(title)
    ax.grid(True, alpha=0.18, linewidth=0.6)

    patch_handles = []
    for key in ["cache", "mixed", "exact"]:
        if key in status:
            patch_handles.append(
                Patch(
                    facecolor=status_colors[key],
                    edgecolor="none",
                    alpha=0.22,
                    label=status_labels[key],
                )
            )
    ax.legend(
        handles=line_handles + patch_handles,
        fontsize=7,
        frameon=False,
        ncol=3,
        loc="best",
    )

    fig.tight_layout()
    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"STO lifecycle saved to: {save_path}")
