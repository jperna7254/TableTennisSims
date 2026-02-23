"""Integration loop for the bounce simulation."""

import warnings
import numpy as np
from scipy.integrate import solve_ivp

from .parameters import SimConfig
from .physics import derivatives, total_energy


def make_initial_state(
    y0: float = 0.305,
    x_dot0: float = 0.0,
    y_dot0: float = 0.0,
    omega0: float = 0.0,
    config: SimConfig | None = None,
) -> np.ndarray:
    """Build the 7-element initial state vector.

    Parameters
    ----------
    y0 : float
        Initial ball centre height [m]. Default 0.305 m (ITTF drop test).
    x_dot0 : float
        Initial horizontal velocity [m/s].
    y_dot0 : float
        Initial vertical velocity [m/s] (negative = downward).
    omega0 : float
        Initial spin [rad/s] (positive = topspin).
    config : SimConfig, optional
        Used to get ball radius. Defaults to standard ball.

    Returns
    -------
    state : array, shape (7,)
        [x, x_dot, y, y_dot, omega, y_t, y_t_dot]
    """
    if config is None:
        config = SimConfig()
    return np.array([0.0, x_dot0, y0, y_dot0, omega0, 0.0, 0.0])


def run_simulation(
    config: SimConfig | None = None,
    y0: float = 0.305,
    x_dot0: float = 0.0,
    y_dot0: float = 0.0,
    omega0: float = 0.0,
    t_span: tuple[float, float] = (0.0, 0.5),
    rtol: float = 1e-9,
    atol: float = 1e-9,
    max_step: float = 1e-3,
) -> dict:
    """Run a single bounce simulation.

    Returns
    -------
    result : dict
        Keys: 't', 'state' (7 x N), 'energy', 'config', 'sol' (dense output)
    """
    if config is None:
        config = SimConfig()

    s0 = make_initial_state(y0, x_dot0, y_dot0, omega0, config)

    def rhs(t, s):
        return derivatives(t, s, config)

    # Suppress benign overflow warnings from Radau's numerical Jacobian
    # estimation — intermediate scaling factors overflow for long
    # integrations of soft floors but the solution remains correct.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="overflow", category=RuntimeWarning)
        sol = solve_ivp(
            rhs,
            t_span,
            s0,
            method="Radau",
            rtol=rtol,
            atol=atol,
            max_step=max_step,
            dense_output=True,
        )

    if not sol.success:
        raise RuntimeError(f"Integration failed: {sol.message}")

    # Compute energy at each output time
    energy = np.array([total_energy(sol.y[:, i], config) for i in range(len(sol.t))])

    return {
        "t": sol.t,
        "state": sol.y,       # shape (7, N)
        "energy": energy,
        "config": config,
        "sol": sol.sol,        # dense output callable
    }


class _StitchedSolution:
    """Callable that stitches two OdeSolution objects at a seam time.

    Mimics the ``sol.sol`` dense-output interface so that callers can
    evaluate ``stitched(t_array)`` transparently.
    """

    def __init__(self, sol_a, sol_b, t_seam: float):
        self._a = sol_a
        self._b = sol_b
        self._t_seam = t_seam

    def __call__(self, t):
        t = np.asarray(t, dtype=float)
        scalar = t.ndim == 0
        t = np.atleast_1d(t)

        out = np.empty((7, len(t)))
        mask_a = t <= self._t_seam
        mask_b = ~mask_a

        if mask_a.any():
            out[:, mask_a] = self._a(t[mask_a])
        if mask_b.any():
            out[:, mask_b] = self._b(t[mask_b])

        if scalar:
            return out[:, 0]
        return out


def run_single_bounce_simulation(
    config: SimConfig | None = None,
    y0: float = 0.305,
    x_dot0: float = 0.0,
    y_dot0: float = 0.0,
    omega0: float = 0.0,
    t_span: tuple[float, float] = (0.0, 0.5),
    rtol: float = 1e-9,
    atol: float = 1e-9,
    max_step: float = 1e-3,
) -> dict:
    """Run a simulation that allows exactly one ball-table contact, then
    continues with only the table-floor subsystem ringing down.

    After the ball separates from the table for the first time, the ball is
    placed far above the surface so it never re-contacts.  The table and
    floor continue to evolve normally.  The two phases are stitched into a
    single dense-output callable.

    Returns the same dict shape as :func:`run_simulation`.
    """
    if config is None:
        config = SimConfig()

    R = config.ball.R
    s0 = make_initial_state(y0, x_dot0, y_dot0, omega0, config)

    # -- Phase 1: integrate until the ball *leaves* contact ----------------
    #
    # Event: delta crosses zero from positive to negative (ball lifts off).
    # We want this to fire only once the ball has actually been in contact,
    # so we track a "was_in_contact" flag via a closure.
    _was_in_contact = [False]

    def _ball_separates(t, s):
        delta = R - (s[2] - s[5])
        if delta > 0:
            _was_in_contact[0] = True
        if not _was_in_contact[0]:
            return 1.0          # positive -> no zero-crossing before contact
        return delta            # crosses zero when ball lifts off

    _ball_separates.terminal = True
    _ball_separates.direction = -1   # trigger when delta goes + -> -

    def rhs(t, s):
        return derivatives(t, s, config)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="overflow", category=RuntimeWarning)
        sol1 = solve_ivp(
            rhs,
            t_span,
            s0,
            method="Radau",
            rtol=rtol,
            atol=atol,
            max_step=max_step,
            dense_output=True,
            events=_ball_separates,
        )

    if not sol1.success:
        raise RuntimeError(f"Phase-1 integration failed: {sol1.message}")

    # If no separation event fired (ball never contacted, or t_span too
    # short), just return the phase-1 result as-is.
    if len(sol1.t_events[0]) == 0:
        energy = np.array([total_energy(sol1.y[:, i], config)
                           for i in range(len(sol1.t))])
        return {
            "t": sol1.t,
            "state": sol1.y,
            "energy": energy,
            "config": config,
            "sol": sol1.sol,
        }

    t_seam = sol1.t_events[0][0]

    # -- Phase 2: table-floor ring-down (ball removed) ---------------------
    #
    # Take the state at the seam, move the ball far away so delta is
    # always negative, and continue integrating.
    s_seam = sol1.sol(t_seam).copy()
    s_seam[2] = 100.0       # y  = 100 m (far above table)
    s_seam[3] = 1000.0      # y_dot = 1 km/s upward (ensures no return within 60 s)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="overflow", category=RuntimeWarning)
        sol2 = solve_ivp(
            rhs,
            (t_seam, t_span[1]),
            s_seam,
            method="Radau",
            rtol=rtol,
            atol=atol,
            max_step=max_step,
            dense_output=True,
        )

    if not sol2.success:
        raise RuntimeError(f"Phase-2 integration failed: {sol2.message}")

    # -- Stitch results ----------------------------------------------------
    t_all = np.concatenate([sol1.t, sol2.t[1:]])
    state_all = np.concatenate([sol1.y, sol2.y[:, 1:]], axis=1)
    energy_all = np.array([total_energy(state_all[:, i], config)
                           for i in range(state_all.shape[1])])

    stitched_sol = _StitchedSolution(sol1.sol, sol2.sol, t_seam)

    return {
        "t": t_all,
        "state": state_all,
        "energy": energy_all,
        "config": config,
        "sol": stitched_sol,
    }
