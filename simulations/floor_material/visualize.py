"""Floor comparison visualization for the topspin bounce simulation.

Produces two outputs:
  1. bounce_animation.gif — ball trajectory + per-floor table displacement
  2. ball_mechanics.png   — static 3-panel proving floor has no effect on ball

Usage:
    python -m simulations.floor_material.visualize
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation

from .parameters import SimConfig, REALISTIC_FLOORS
from .runner import run_simulation, run_single_bounce_simulation


FLOOR_COLORS = {
    "Concrete": "tab:blue",
    "Hardwood": "tab:green",
    "Rubber Mat": "tab:orange",
    "Carpet": "tab:red",
}


# -- Helper functions ----------------------------------------------------------

def _compute_contact_force(states, t_array, config):
    """Recompute normal contact force F_n from state arrays."""
    R = config.ball.R
    k_c = config.contact.k_c
    c_c = config.contact.c_c

    y = states[2, :]
    y_dot = states[3, :]
    y_t = states[5, :]
    y_t_dot = states[6, :]

    delta = R - (y - y_t)
    delta_dot = -(y_dot - y_t_dot)

    F_n = k_c * delta + c_c * delta_dot
    F_n = np.where(delta > 0, np.maximum(F_n, 0.0), 0.0)
    return F_n


def _find_contact_windows(states, t_array, config):
    """Return list of (t_start, t_end) intervals where ball is in contact."""
    R = config.ball.R
    y = states[2, :]
    y_t = states[5, :]
    delta = R - (y - y_t)
    in_contact = delta > 0

    windows = []
    inside = False
    for i in range(len(t_array)):
        if in_contact[i] and not inside:
            t_start = t_array[i]
            inside = True
        elif not in_contact[i] and inside:
            windows.append((t_start, t_array[i]))
            inside = False
    if inside:
        windows.append((t_start, t_array[-1]))
    return windows


# -- Output 1: bounce_animation.gif -------------------------------------------

def animate_bounce_comparison(
    y0: float = 0.305,
    x_dot0: float = 2.0,
    omega0: float = 150.0,
    save_path: str = "bounce_animation.gif",
    fps: int = 30,
):
    """Animated GIF: shared ball trajectory (left) + 4 stacked table displacement panels (right).

    Ball trajectory panel spans 0-500 ms. Each table displacement panel is
    simulated for 3 natural periods (capped 60 s) with its own time/y-axis
    scale. All panels animate 0%->100% simultaneously.

    Extended table sims use :func:`run_single_bounce_simulation` so that
    only a single ball-table contact occurs; subsequent table motion is
    purely the table-floor ring-down with no further energy input from
    additional bounces.
    """
    ball_t_span = (0.0, 0.5)
    m_t = SimConfig().table.m_t
    duration_anim = 10.0
    n_frames = int(duration_anim * fps)

    # -- Run simulations --
    sim_ball = {}   # ball-timescale sims (0-500 ms)
    sim_table = {}  # per-floor extended sims (single bounce only)
    t_ends = {}

    for floor in REALISTIC_FLOORS:
        cfg = SimConfig(floor=floor)
        # Ball sim (shared trajectory)
        r_ball = run_simulation(config=cfg, y0=y0, x_dot0=x_dot0, omega0=omega0,
                                t_span=ball_t_span, max_step=1e-3)
        sim_ball[floor.name] = r_ball

        # Extended table sim — single bounce then ring-down only
        period = 2 * np.pi * np.sqrt(m_t / floor.k_f) if floor.k_f > 0 else 10.0
        t_end = min(period * 3, 60.0)
        t_end = max(t_end, 0.5)
        t_ends[floor.name] = t_end

        r_table = run_single_bounce_simulation(
            config=cfg, y0=y0, x_dot0=x_dot0, omega0=omega0,
            t_span=(0.0, t_end), max_step=0.1,
        )
        sim_table[floor.name] = r_table

    # -- Compute contact statistics (identical across all floors) --
    ref_name = REALISTIC_FLOORS[0].name
    ref_r = sim_ball[ref_name]
    ref_cfg = ref_r["config"]
    t_fine = np.linspace(0.0, 0.5, 50000)
    ref_states_fine = ref_r["sol"](t_fine)
    F_n = _compute_contact_force(ref_states_fine, t_fine, ref_cfg)
    contact_windows = _find_contact_windows(ref_states_fine, t_fine, ref_cfg)
    F_peak = np.max(F_n)
    a_table_peak = F_peak / m_t
    if contact_windows:
        cw_start, cw_end = contact_windows[0]
        contact_dur_ms = (cw_end - cw_start) * 1000
    else:
        contact_dur_ms = 0.0

    # -- Two-phase time mapping --
    # Phase 1 (first half): all panels sync at 0-500 ms with ball trajectory
    # Phase 2 (second half): ball freezes, table panels continue ring-down
    n_phase1 = n_frames // 2
    n_phase2 = n_frames - n_phase1

    # Ball: 0->500ms in phase 1, frozen at 500ms in phase 2
    t_ball_anim = np.concatenate([
        np.linspace(0.0, 0.5, n_phase1),
        np.full(n_phase2, 0.5),
    ])
    ref_ball_interp = sim_ball[ref_name]["sol"](t_ball_anim)

    # Table panels: 0->500ms in phase 1 (synced), 500ms->t_end in phase 2
    t_table_anim = {}
    table_interp = {}
    for floor in REALISTIC_FLOORS:
        name = floor.name
        t_end = t_ends[name]
        t_p1 = np.linspace(0.0, 0.5, n_phase1)
        if t_end > 0.5:
            t_p2 = np.linspace(0.5, t_end, n_phase2 + 1)[1:]  # skip duplicate 0.5
        else:
            t_p2 = np.full(n_phase2, t_end)  # already done, freeze
        t_arr = np.concatenate([t_p1, t_p2])
        t_table_anim[name] = t_arr
        table_interp[name] = sim_table[name]["sol"](t_arr)

    # -- Figure layout with gridspec --
    fig = plt.figure(figsize=(14, 9))
    fig.suptitle(
        "Floor Material Has No Effect on Ball Bounce Mechanics\n"
        f"Table: {m_t:.0f} kg  |  Ball: {SimConfig().ball.m*1000:.1f} g  |  "
        f"Topspin: {omega0:.0f} rad/s",
        fontsize=12, fontweight="bold",
    )

    gs = gridspec.GridSpec(4, 2, figure=fig, width_ratios=[1.2, 1],
                           hspace=0.4, wspace=0.3)

    # Left: ball trajectory spanning all 4 rows
    ax_traj = fig.add_subplot(gs[:, 0])

    # Right: 4 stacked table displacement panels
    ax_tables = [fig.add_subplot(gs[i, 1]) for i in range(4)]

    # -- Ball trajectory panel --
    R_cm = SimConfig().ball.R * 100
    x_all = ref_ball_interp[0, :] * 100
    y_all = ref_ball_interp[2, :] * 100
    x_margin, y_margin = 2, 2
    ax_traj.set_xlim(np.min(x_all) - x_margin, np.max(x_all) + x_margin)
    ax_traj.set_ylim(-0.5, np.max(y_all) + y_margin)
    ax_traj.set_aspect("equal")
    ax_traj.set_xlabel(r"$x$ [cm]")
    ax_traj.set_ylabel(r"$y$ [cm]")
    ax_traj.set_title("Ball Trajectory")
    ax_traj.axhline(0, color="k", linewidth=1.5)

    # Ball patch + trail (single -- all floors give same trajectory)
    ball_patch = patches.Circle((0, 0), R_cm, fill=False, edgecolor="tab:blue", linewidth=2)
    ax_traj.add_patch(ball_patch)
    trail_line, = ax_traj.plot([], [], color="tab:blue", alpha=0.3, linewidth=0.8)

    time_text = ax_traj.text(0.02, 0.95, "", transform=ax_traj.transAxes,
                             fontsize=10, verticalalignment="top",
                             fontfamily="monospace")

    # Contact statistics annotation
    stats_str = (
        f"Contact time: {contact_dur_ms:.2f} ms\n"
        f"Peak force on table: {F_peak:.1f} N\n"
        f"Peak table accel: {a_table_peak*1000:.2f} mm/s$^2$\n"
        f"Table mass: {m_t:.0f} kg"
    )
    ax_traj.text(0.02, 0.02, stats_str, transform=ax_traj.transAxes,
                 fontsize=8, verticalalignment="bottom",
                 fontfamily="monospace",
                 bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow",
                           edgecolor="gray", alpha=0.9))

    # -- Table displacement panels --
    table_lines = {}
    table_cursors = {}
    for idx, floor in enumerate(REALISTIC_FLOORS):
        name = floor.name
        ax = ax_tables[idx]
        color = FLOOR_COLORS[name]

        st = table_interp[name]
        yt_um = st[5, :] * 1e6
        yt_max = max(np.max(np.abs(yt_um)) * 1.3, 1e-6)

        t_end_ms = t_ends[name] * 1000
        ax.set_xlim(0, t_end_ms)
        ax.set_ylim(-yt_max, yt_max)
        ax.set_ylabel(r"$y_t$ [$\mu$m]", fontsize=8)
        ax.set_title(f"{name} ($k_f$ = {floor.k_f:.0e} N/m)", fontsize=9, loc="left")
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        ax.tick_params(labelsize=7)

        if idx == 3:
            ax.set_xlabel("Time [ms]", fontsize=8)

        line, = ax.plot([], [], color=color, linewidth=1.2)
        table_lines[name] = line
        cl = ax.axvline(0, color="gray", linewidth=0.8, linestyle=":")
        table_cursors[name] = cl

    fig.subplots_adjust(top=0.88)

    def init():
        ball_patch.center = (0, 0)
        trail_line.set_data([], [])
        time_text.set_text("")
        for name in table_lines:
            table_lines[name].set_data([], [])
        return []

    def update(frame):
        # Ball trajectory
        cx = ref_ball_interp[0, frame] * 100
        cy = ref_ball_interp[2, frame] * 100
        ball_patch.center = (cx, cy)
        trail_line.set_data(ref_ball_interp[0, :frame+1] * 100,
                            ref_ball_interp[2, :frame+1] * 100)
        t_ms = t_ball_anim[frame] * 1000
        time_text.set_text(f"t = {t_ms:.1f} ms")

        # Table displacement panels (each at its own time scale)
        sl = slice(None, frame + 1)
        for floor in REALISTIC_FLOORS:
            name = floor.name
            t_arr = t_table_anim[name]
            st = table_interp[name]
            table_lines[name].set_data(t_arr[sl] * 1000, st[5, sl] * 1e6)
            table_cursors[name].set_xdata([t_arr[frame] * 1000])

        return []

    anim = FuncAnimation(fig, update, frames=n_frames, init_func=init,
                         blit=False, interval=1000 / fps)
    anim.save(save_path, writer="pillow", fps=fps, dpi=80)
    plt.close(fig)
    print(f"Saved: {save_path}")
    return anim


# -- Output 2: ball_mechanics.png ---------------------------------------------

def plot_ball_mechanics(
    y0: float = 0.305,
    x_dot0: float = 2.0,
    omega0: float = 150.0,
    t_span: tuple = (0.0, 0.5),
    save_path: str | None = None,
):
    """3-panel static figure: horiz velocity, spin, ball height -- all 4 floors overlaid.

    Traces overlap perfectly, proving floor has no effect on ball mechanics.
    """
    results = {}
    for floor in REALISTIC_FLOORS:
        cfg = SimConfig(floor=floor)
        r = run_simulation(config=cfg, y0=y0, x_dot0=x_dot0, omega0=omega0,
                           t_span=t_span, max_step=0.01)
        results[floor.name] = r

    fig, (ax_vx, ax_spin, ax_h) = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        "Ball Mechanics Are Identical Across All Floor Types\n"
        f"Table: {SimConfig().table.m_t:.0f} kg  |  "
        f"Topspin: {omega0:.0f} rad/s  |  "
        f"Drop height: {y0*100:.1f} cm",
        fontsize=12, fontweight="bold",
    )

    for floor in REALISTIC_FLOORS:
        name = floor.name
        r = results[name]
        t_ms = r["t"] * 1000
        color = FLOOR_COLORS[name]
        label = f"{name} ($k_f$ = {floor.k_f:.0e})"

        ax_vx.plot(t_ms, r["state"][1, :], color=color, linewidth=1.5, label=label)
        ax_spin.plot(t_ms, r["state"][4, :], color=color, linewidth=1.5, label=label)
        ax_h.plot(t_ms, r["state"][2, :] * 100, color=color, linewidth=1.5, label=label)

    ax_vx.set_xlabel(r"Time [ms]")
    ax_vx.set_ylabel(r"$\dot{x}$ [m/s]")
    ax_vx.set_title("Horizontal Velocity")
    ax_vx.legend(fontsize=7)

    ax_spin.set_xlabel(r"Time [ms]")
    ax_spin.set_ylabel(r"$\omega$ [rad/s]")
    ax_spin.set_title("Spin")
    ax_spin.legend(fontsize=7)

    ax_h.set_xlabel(r"Time [ms]")
    ax_h.set_ylabel(r"Ball Height [cm]")
    ax_h.set_title("Ball Height")
    ax_h.legend(fontsize=7)

    fig.tight_layout(rect=[0, 0, 1, 0.88])
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    return fig


# -- Console output ------------------------------------------------------------

def print_results_table(
    y0: float = 0.305,
    x_dot0: float = 2.0,
    omega0: float = 150.0,
    t_span: tuple = (0.0, 0.5),
):
    """Print a summary table of post-bounce observables for all floor types."""
    print(f"{'Floor':>14s}  {'k_f':>10s}  {'c_f':>8s}  "
          f"{'x_dot post':>10s}  {'y peak':>10s}  {'omega post':>10s}  {'y_t peak':>12s}")
    print(f"{'':>14s}  {'[N/m]':>10s}  {'[Ns/m]':>8s}  "
          f"{'[m/s]':>10s}  {'[cm]':>10s}  {'[rad/s]':>10s}  {'[um]':>12s}")
    print("-" * 90)

    for floor in REALISTIC_FLOORS:
        cfg = SimConfig(floor=floor)
        r = run_simulation(config=cfg, y0=y0, x_dot0=x_dot0, omega0=omega0, t_span=t_span)

        y = r["state"][2, :]
        y_min_idx = np.argmin(y)
        y_peak = np.max(y[y_min_idx:]) * 100
        x_dot_post = r["state"][1, -1]
        omega_post = r["state"][4, -1]
        y_t_peak = np.max(np.abs(r["state"][5, :])) * 1e6

        print(f"{floor.name:>14s}  {floor.k_f:10.0e}  {floor.c_f:8.0e}  "
              f"{x_dot_post:10.6f}  {y_peak:10.4f}  {omega_post:10.4f}  {y_t_peak:12.3f}")

    print()
    print(f"Table mass: {cfg.table.m_t} kg (Butterfly Centrefold 25)")
    print(f"Ball mass:  {cfg.ball.m * 1000} g")
    print(f"Mass ratio: 1:{cfg.table.m_t / cfg.ball.m:.0f}")
    print(f"Initial conditions: y0={y0}m, x_dot0={x_dot0}m/s, omega0={omega0}rad/s")
    print()
    print("Conclusion: Floor stiffness spans 6 orders of magnitude but ball")
    print("observables (x_dot, y peak, omega) are identical. Only table displacement")
    print("changes. The 127 kg table mass blocks floor effects from reaching")
    print("the ball during the ~1 ms contact.")


# -- Main ----------------------------------------------------------------------

if __name__ == "__main__":
    import os
    import glob as _glob
    out_dir = os.path.join(os.path.dirname(__file__), "..", "..", "output")
    os.makedirs(out_dir, exist_ok=True)

    # Clear previous outputs
    for f in _glob.glob(os.path.join(out_dir, "*")):
        os.remove(f)

    print("=" * 90)
    print("FLOOR MATERIAL IMPACT SIMULATION -- TOPSPIN BOUNCE")
    print("=" * 90)
    print()

    # 1. Print numerical results table (4 realistic floors)
    print_results_table()

    # 2. Static PNG -- ball mechanics (4 floors overlaid)
    print()
    plot_ball_mechanics(save_path=os.path.join(out_dir, "ball_mechanics.png"))

    # 3. Animated GIF -- bounce + table displacement
    print("\nGenerating bounce comparison animation...")
    animate_bounce_comparison(save_path=os.path.join(out_dir, "bounce_animation.gif"))

    print("\nDone! Check the output/ directory.")
