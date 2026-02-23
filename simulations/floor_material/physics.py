"""ODE right-hand side and energy computation for the bounce model.

State vector: s = [x, x_dot, y, y_dot, omega, y_t, y_t_dot]
See docs/model_description.tex Eq. 14 for the full ODE system.
"""

import numpy as np

from .parameters import SimConfig


def derivatives(t: float, state: np.ndarray, config: SimConfig) -> np.ndarray:
    """Compute ds/dt for the 7-element state vector.

    Parameters
    ----------
    t : float
        Current time (unused — system is autonomous).
    state : array, shape (7,)
        [x, x_dot, y, y_dot, omega, y_t, y_t_dot]
    config : SimConfig
        All physical parameters.

    Returns
    -------
    d_state : array, shape (7,)
    """
    x, x_dot, y, y_dot, omega, y_t, y_t_dot = state

    m = config.ball.m
    R = config.ball.R
    I = config.ball.I
    g = config.g

    k_c = config.contact.k_c
    c_c = config.contact.c_c
    mu = config.contact.mu
    v_0 = config.contact.v_0

    m_t = config.table.m_t
    k_f = config.floor.k_f
    c_f = config.floor.c_f

    # --- Contact geometry ---
    delta = R - (y - y_t)           # overlap (Eq. 2)
    delta_dot = -(y_dot - y_t_dot)  # overlap rate (Eq. 3)

    # --- Normal force (Eq. 4) ---
    if delta > 0:
        F_n = k_c * delta + c_c * delta_dot
        if F_n < 0:
            F_n = 0.0
    else:
        F_n = 0.0

    # --- Tangential (friction) force (Eq. 6) ---
    if F_n > 0:
        v_slip = x_dot - omega * R   # Eq. 5
        F_t = -mu * F_n * np.tanh(v_slip / v_0)
    else:
        F_t = 0.0

    # --- Derivatives (Eq. 14) ---
    d_state = np.empty(7)
    d_state[0] = x_dot                                         # dx/dt
    d_state[1] = F_t / m                                       # d(x_dot)/dt
    d_state[2] = y_dot                                         # dy/dt
    d_state[3] = -g + F_n / m                                  # d(y_dot)/dt
    d_state[4] = -R * F_t / I                                  # d(omega)/dt
    d_state[5] = y_t_dot                                       # d(y_t)/dt
    d_state[6] = (F_n - k_f * y_t - c_f * y_t_dot) / m_t      # d(y_t_dot)/dt

    return d_state


def total_energy(state: np.ndarray, config: SimConfig) -> float:
    """Compute total mechanical energy of the system (Eq. 11).

    Parameters
    ----------
    state : array, shape (7,)
    config : SimConfig

    Returns
    -------
    E : float
        Total energy [J].
    """
    x, x_dot, y, y_dot, omega, y_t, y_t_dot = state

    m = config.ball.m
    R = config.ball.R
    I = config.ball.I
    g = config.g
    k_f = config.floor.k_f
    k_c = config.contact.k_c

    # Ball kinetic + gravitational potential
    E_ball = 0.5 * m * (x_dot**2 + y_dot**2) + 0.5 * I * omega**2 + m * g * y

    # Table kinetic + floor spring potential
    E_table = 0.5 * config.table.m_t * y_t_dot**2 + 0.5 * k_f * y_t**2

    # Contact spring potential (only when in contact)
    delta = R - (y - y_t)
    E_contact = 0.5 * k_c * delta**2 if delta > 0 else 0.0

    return E_ball + E_table + E_contact
