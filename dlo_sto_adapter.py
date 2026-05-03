"""
DLO adapter for the theory-oriented Stateful Tangent Operator.

The core STO in stateful_tangent_operator.py operates on the generic
implicit system

    G(x, z) = 0

and answers adjoint queries

    v -> -G_z.T @ inv(G_x.T) @ v.

This adapter maps the 2D strip simulator's block Jacobian notation onto that
generic interface and returns the 6D boundary-control gradient used by
implicit_grad_tools. The adapter is intentionally thin so the STO theory
stays in one place.
"""

from __future__ import annotations

import numpy as np

from implicit_grad_tools import boundary_node_jacobian
from stateful_tangent_operator import StatefulTangentOperator


class DLOStatefulTangentOperator:
    """STO wrapper for the neural-control DLO boundary-control example."""

    def __init__(
        self,
        rho_max: float = 1e-1,
        kappa_max: float = 1e10,
        rho_warn: float = 1e-2,
        kappa_check_period: int = 10,
        cooldown: int = 3,
        drift_radius_x: float = np.inf,
        drift_radius_z: float = np.inf,
        n_probes: int = 5,
        n_power_iter: int = 5,
        max_reuse: int = 10000,
        singular_rcond: float = 1e-12,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.core = StatefulTangentOperator(
            rho_max=rho_max,
            kappa_max=kappa_max,
            rho_warn=rho_warn,
            kappa_check_period=kappa_check_period,
            cooldown=cooldown,
            drift_radius_x=drift_radius_x,
            drift_radius_z=drift_radius_z,
            n_probes=n_probes,
            n_power_iter=n_power_iter,
            max_reuse=max_reuse,
            singular_rcond=singular_rcond,
            rng=rng,
        )
        self._last_n_free = 0

    def query(
        self,
        J_full: np.ndarray,
        free_index: np.ndarray,
        fixed_index: np.ndarray,
        a_free: np.ndarray,
        control_6d: np.ndarray,
        deltaL: float,
        q_full: np.ndarray | None = None,
    ) -> tuple[np.ndarray, bool, float, str]:
        """
        Return dL/du for the 6D boundary control.

        Simulator notation:
            J_ff = dG_free / dq_free
            J_fb = dG_free / dq_boundary
            J_map = dq_boundary / du

        Generic STO notation:
            G_x = J_ff
            G_z = J_fb @ J_map

        Therefore the generic adjoint action returns
            -G_z.T @ inv(G_x.T) @ a_free == dL/du.
        """
        free_index = np.asarray(free_index)
        fixed_index = np.asarray(fixed_index)
        control_6d = np.asarray(control_6d, dtype=float)
        a_free = np.asarray(a_free, dtype=float)

        G_x = J_full[np.ix_(free_index, free_index)]
        J_fb = J_full[np.ix_(free_index, fixed_index)]
        J_map = boundary_node_jacobian(control_6d, deltaL)
        G_z = J_fb @ J_map

        x_new = None if q_full is None else np.asarray(q_full, dtype=float)[free_index]
        z_new = control_6d
        self._last_n_free = G_x.shape[0]

        result = self.core.query(
            v=a_free,
            G_x_new=G_x,
            G_z_new=G_z,
            x_new=x_new,
            z_new=z_new,
        )
        return result.g_z, result.used_cache, result.eta, result.reason

    def amortized_speedup(self, cost_ratio: float | None = None) -> float:
        """Init-aware speedup against always-exact adjoint solves."""
        if cost_ratio is None:
            if self._last_n_free <= 0:
                return self.core.amortized_speedup(cost_ratio=1.0)
            cost_ratio = 1.0 / self._last_n_free
        return self.core.amortized_speedup(cost_ratio=cost_ratio)

    def stats_str(self) -> str:
        return (
            self.core.stats.summary()
            + f"\n  Amortized speedup     : {self.amortized_speedup():.2f}x"
        )

    @property
    def n_init(self) -> int:
        return self.core.stats.n_init

    @property
    def n_cached(self) -> int:
        return self.core.stats.n_cached

    @property
    def n_fallback(self) -> int:
        return self.core.stats.n_exact

    @property
    def n_kappa_recompute(self) -> int:
        return self.core.stats.n_kappa_recompute

    @property
    def in_warning(self) -> bool:
        return self.core.in_warning

    @property
    def last_rho(self) -> float:
        gate = self.core.last_validation
        return np.nan if gate is None else gate.rho

    @property
    def last_kappa(self) -> float:
        gate = self.core.last_validation
        return np.nan if gate is None else gate.kappa

    @property
    def last_eta(self) -> float:
        gate = self.core.last_validation
        return np.nan if gate is None else gate.eta

    @property
    def last_kappa_recomputed(self) -> bool:
        gate = self.core.last_validation
        return False if gate is None else gate.kappa_recomputed

    def last_metric(self, name: str, default: float = np.nan) -> float:
        return self.core.last_query_metrics.get(name, default)
