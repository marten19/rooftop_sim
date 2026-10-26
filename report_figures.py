"""
Figures for the report: the reflected sunlight on a roof at several times of
day, one figure per roof. Every panel uses the same colour scale and view, so
they can be compared.

Each roof (shape + materials) is defined in ROOFS below. The time of day ->
sun angle conversion comes from sun_day_animation.py. Edit the parameters
below and run:

    python3 report_figures.py
"""

import os
from concurrent.futures import ProcessPoolExecutor

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

import rooftop_sim as sim
import sun_day_animation as day


# =============================================================================
# PARAMETERS
# =============================================================================

# ---- times of day ----------------------------------------------------------
TIMES = [8.0, 10.0, 12.0, 14.0, 16.0, 18.0]   # [h], one panel per time

# ---- roofs -----------------------------------------------------------------
# One figure per roof. Every roof is a straight roof line (slope + length).
#   steps = 0 : flat roof, one smooth surface made of material A
#   steps > 0 : stair roof, each step = one segment A + one segment B
#               (angles absolute: 0 = horizontal, 90 = vertical; the slope
#               must lie between angle_a and angle_b)
# diffusion : 0 = perfect mirror ... 1 = complete scatter (matte)
# absorption: 0 = absorbs nothing ... 1 = absorbs everything
ROOFS = [
    dict(
        name="Stair roof",
        slope=40.0,              # [deg]
        length=25.0,
        steps=5,
        angle_a=30.0,            # [deg]
        angle_b=90.0,            # [deg]
        material_a=dict(diffusion=0.6, absorption=0.6),
        material_b=dict(diffusion=0.6, absorption=0.5),
    ),
    dict(
        name="Stair roof super reflective",
        slope=40.0,              # [deg]
        length=25.0,
        steps=5,
        angle_a=30.0,            # [deg]
        angle_b=90.0,            # [deg]
        material_a=dict(diffusion=0.1, absorption=0.1),
        material_b=dict(diffusion=0.6, absorption=0.5),
    ),
    dict(
        name="Flat angled roof",
        slope=40.0,              # [deg] 0 = horizontal
        length=25.0,
        steps=0,
        material_a=dict(diffusion=0.6, absorption=0.6),
    ),
    
    dict(
        name="Flat roof",
        slope=0.0,              # [deg] 0 = horizontal
        length=25.0,
        steps=0,
        material_a=dict(diffusion=0.6, absorption=0.6),
    ),
]

# ---- simulation ------------------------------------------------------------
N_SUN_RAYS = 300         # fewer rays than rooftop_sim.py keeps the pictures readable
N_DIFFUSE = 60

# ---- output ----------------------------------------------------------------
OUTPUT_DIR = "sim_figures"
N_COLUMNS = 3            # panels per row in the combined figure
SAVE_SINGLE = True       # also save every panel as its own picture
DPI = 300
FILE_TYPE = "png"        # "png", or "pdf" / "svg" for vector graphics


# =============================================================================
# SIMULATION
# =============================================================================

def use_report_settings():
    sim.N_SUN_RAYS = N_SUN_RAYS
    sim.N_DIFFUSE = N_DIFFUSE


def build_roof(roof):
    """Segments of one roof from ROOFS."""
    start = (0.0, 0.0)
    if roof["steps"] == 0:
        s = np.radians(roof["slope"])
        end = (roof["length"] * np.cos(s), roof["length"] * np.sin(s))
        return [sim.Segment(start, end, **roof["material_a"], name="flat", kind="A")]
    segments, _, _ = sim.stair_roof(start, roof["slope"], roof["length"], roof["steps"],
                                    roof["angle_a"], roof["angle_b"],
                                    roof["material_a"], roof["material_b"])
    return segments


def sun_angle(hour):
    """Same linear time -> sun angle conversion as the day animation."""
    f = (hour - day.SUNRISE_HOUR) / (day.SUNSET_HOUR - day.SUNRISE_HOUR)
    return day.SUN_START_ANGLE + f * (day.SUN_END_ANGLE - day.SUN_START_ANGLE)


def simulate(task):
    """One roof at one time of day. Returns only what is needed for drawing."""
    roof_def, hour = task
    roof = build_roof(roof_def)
    angle = sun_angle(hour)

    arrows = []
    for ray in sim.sun_rays(roof, angle, sim.N_ARROWS):
        t, seg = sim.closest_hit(ray, roof)
        if seg is not None and np.dot(seg.normal(), ray.direction) <= 0:
            arrows.append((ray.origin + ray.direction * (t - sim.ARROW_GAP), ray.direction))

    try:
        pieces, stats = sim.trace(roof, angle)
    except ValueError:                       # sun behind the roof: nothing to show
        return dict(lines=np.zeros((0, 2, 2)), strength=np.zeros(0),
                    max_sky_strength=0.0, absorbed=0.0, arrows=arrows)

    refl = sorted((p for p in pieces if p.depth > 0), key=lambda p: p.strength)

    # strongest reflected ray that leaves the scene upwards (to the sky).
    # Rays leaving the scene are drawn exactly SCENE_SIZE long.
    to_sky = [p.strength for p in refl
              if p.end[1] > p.start[1]
              and np.isclose(np.linalg.norm(p.end - p.start), sim.SCENE_SIZE)]

    return dict(lines=np.array([[p.start, p.end] for p in refl]).reshape(-1, 2, 2),
                strength=np.array([p.strength for p in refl]),
                max_sky_strength=max(to_sky, default=0.0),
                absorbed=sum(seg.absorbed for seg in roof), arrows=arrows)


# =============================================================================
# DRAWING
# =============================================================================

def view_limits():
    """One view that fits every roof, so all panels show the same area."""
    pts = np.array([p for roof in ROOFS for seg in build_roof(roof)
                    for p in (seg.p1, seg.p2)])
    pad = 5
    return ((pts[:, 0].min() - pad, pts[:, 0].max() + pad),
            (pts[:, 1].min() - pad, pts[:, 1].max() + 2 * pad))


def draw_panel(ax, roof, hour, result, xlim, ylim):
    norm = sim.color_norm()
    rel = np.clip(norm(result["strength"]), 0, 1)
    rgba = sim.RAY_COLORMAP(rel)
    rgba[:, 3] = sim.RAY_OPACITY * rel
    visible = rel > 1e-3
    ax.add_collection(LineCollection(result["lines"][visible], colors=rgba[visible],
                                     linewidths=sim.RAY_WIDTH))

    for seg in build_roof(roof):
        ax.plot([seg.p1[0], seg.p2[0]], [seg.p1[1], seg.p2[1]],
                color=sim.SEGMENT_COLORS.get(seg.kind, "black"), lw=3,
                solid_capstyle="round", zorder=4)

    length = 0.12 * (ylim[1] - ylim[0])
    for head, d in result["arrows"]:
        ax.annotate("", xy=head, xytext=head - d * length, zorder=6,
                    arrowprops=dict(arrowstyle="-|>", color="darkorange", lw=2,
                                    mutation_scale=15))

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.set_facecolor("white")
    ax.set_title(f"sun {sun_angle(hour):.0f}°   |   max sky intensity "
                 f"{result['max_sky_strength']:.2f}   |   absorption {result['absorbed']:.2f}",
                 fontsize=10)


def add_colorbar(fig, axes):
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=sim.color_norm(), cmap=sim.RAY_COLORMAP),
                        ax=axes, fraction=0.03, pad=0.02)
    cbar.set_label("reflected ray strength (1 = incoming sunlight)")


def roof_title(roof):
    # if roof["steps"] == 0:
    #     return f"{roof['name']}  (slope {roof['slope']:g}°)"
    # return (f"{roof['name']}  ({roof['steps']} steps, slope {roof['slope']:g}°, "
    #         f"A {roof['angle_a']:g}°, B {roof['angle_b']:g}°)")
        return f"{roof['name']}"


def file_name(roof):
    return "".join(c if c.isalnum() else "_" for c in roof["name"].lower()).strip("_")


# =============================================================================
# MAIN
# =============================================================================

def main():
    use_report_settings()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tasks = [(roof, hour) for roof in ROOFS for hour in TIMES]
    print(f"simulating {len(tasks)} pictures ...")
    with ProcessPoolExecutor(initializer=use_report_settings) as pool:
        results = list(pool.map(simulate, tasks))

    xlim, ylim = view_limits()
    n_rows = int(np.ceil(len(TIMES) / N_COLUMNS))
    for k, roof in enumerate(ROOFS):
        roof_results = results[k * len(TIMES):(k + 1) * len(TIMES)]
        name = file_name(roof)

        # combined figure: all times of day for this roof
        fig, axes = plt.subplots(n_rows, N_COLUMNS, figsize=(5 * N_COLUMNS, 4.2 * n_rows),
                                 squeeze=False, layout="constrained")
        for ax, hour, result in zip(axes.flat, TIMES, roof_results):
            draw_panel(ax, roof, hour, result, xlim, ylim)
        for ax in axes.flat[len(TIMES):]:
            ax.axis("off")
        add_colorbar(fig, axes)
        fig.suptitle(roof_title(roof), fontsize=12)
        path = os.path.join(OUTPUT_DIR, f"{name}_day.{FILE_TYPE}")
        fig.savefig(path, dpi=DPI)
        plt.close(fig)
        print(f"saved {path}")

        # one picture per time of day
        if SAVE_SINGLE:
            for hour, result in zip(TIMES, roof_results):
                fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
                draw_panel(ax, roof, hour, result, xlim, ylim)
                add_colorbar(fig, ax)
                fig.suptitle(roof_title(roof), fontsize=10)
                path = os.path.join(OUTPUT_DIR, f"{name}_{int(hour):02d}"
                                                f"{int(round(hour % 1 * 60)):02d}.{FILE_TYPE}")
                fig.savefig(path, dpi=DPI)
                plt.close(fig)
            print(f"saved {len(TIMES)} single pictures of the {roof['name']}")


if __name__ == "__main__":
    main()
