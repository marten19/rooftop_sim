"""
Animation of the rooftop over a full day: the sun moves from sunrise (right,
east) to sunset (left, west) and the reflected light is traced for every
frame.

The roof, materials and display settings are taken from rooftop_sim.py.
Only the parameters below are specific to the animation.

    python3 sun_day_animation.py
"""

import os
from concurrent.futures import ProcessPoolExecutor

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection

import rooftop_sim as sim


# =============================================================================
# PARAMETERS
# =============================================================================

# ---- day -------------------------------------------------------------------
SUNRISE_HOUR = 6.0       # time at which the sun is at SUN_START_ANGLE
SUNSET_HOUR = 18.0       # time at which the sun is at SUN_END_ANGLE
SUN_START_ANGLE = 5.0    # [deg] sun elevation at sunrise (right side, east)
SUN_END_ANGLE = 175.0    # [deg] sun elevation at sunset (left side, west)
N_FRAMES = 20            # frames over the day (the sun angle changes linearly)

# ---- simulation (lower than rooftop_sim.py to keep it fast) ----------------
N_SUN_RAYS = 500
N_DIFFUSE = 100

# ---- output ----------------------------------------------------------------
FPS = 5
SAVE_MP4 = "sun_day.mp4"  # file name, or None to not save (needs ffmpeg)
SAVE_GIF = None           # file name, or None to not save
SHOW = False               # also open a window with the animation


# =============================================================================
# FRAME COMPUTATION
# =============================================================================

def use_animation_settings():
    sim.N_SUN_RAYS = N_SUN_RAYS
    sim.N_DIFFUSE = N_DIFFUSE


def build_roof(material_a=None, material_b=None):
    """The roof from rooftop_sim.py, optionally with other materials."""
    roof, _, _ = sim.stair_roof(sim.ROOF_START, sim.ROOF_SLOPE, sim.ROOF_LENGTH,
                                sim.N_STEPS, sim.ANGLE_A, sim.ANGLE_B,
                                material_a or sim.MATERIAL_A,
                                material_b or sim.MATERIAL_B)
    return roof


def compute_frame(sun_angle, material_a=None, material_b=None):
    """Trace one sun position. Returns everything needed to draw the frame."""
    roof = build_roof(material_a, material_b)

    # arrows: real sun rays, stopping just before they hit the roof
    ray_len = 0.12 * (sim.ROOF_LENGTH + 15)
    arrows = []
    for ray in sim.sun_rays(roof, sun_angle, sim.N_ARROWS):
        t, seg = sim.closest_hit(ray, roof)
        if seg is not None and np.dot(seg.normal(), ray.direction) <= 0:
            head = ray.origin + ray.direction * (t - sim.ARROW_GAP)
            arrows.append((head - ray.direction * ray_len, head))

    try:
        pieces, stats = sim.trace(roof, sun_angle)
    except ValueError:            # sun behind the roof: no light on it
        return dict(lines=np.zeros((0, 2, 2)), strength=np.zeros(0),
                    reflectivity=np.nan, absorbed=np.nan, incident=0.0,
                    reflected_below=np.nan, max_sky_strength=np.nan,
                    arrows=arrows)

    refl = sorted((p for p in pieces if p.depth > 0), key=lambda p: p.strength)
    lines = np.array([[p.start, p.end] for p in refl]).reshape(-1, 2, 2)
    strength = np.array([p.strength for p in refl])

    # sunlight landing on the roof (constant sun strength per unit beam width),
    # used to weight the daily average
    spacing = sim.sun_rays(roof, sun_angle, sim.N_SUN_RAYS)[0].intensity
    incident = spacing * sum(1 for p in pieces if p.depth == 0)

    # strongest reflected ray that leaves the scene upwards (to the sky).
    # Rays leaving the scene are drawn exactly SCENE_SIZE long.
    to_sky = [p.strength for p in refl
              if p.end[1] > p.start[1]
              and np.isclose(np.linalg.norm(p.end - p.start), sim.SCENE_SIZE)]

    return dict(lines=lines, strength=strength,
                reflectivity=stats["reflectivity"],
                absorbed=sum(s.absorbed for s in roof),
                reflected_below=stats["escaped"] - stats["escaped_up"],
                max_sky_strength=max(to_sky, default=0.0),
                incident=incident, arrows=arrows)


def clock(hour):
    return f"{int(hour):02d}:{int(round(hour % 1 * 60)) % 60:02d}"


def day_results(frames, hours):
    """
    The four results of the day simulation, as a dict. Fractions are of the
    sunlight hitting the roof; the sky intensity is the strongest single ray
    (1 = direct sunlight). Returns None if the sun never reaches the roof.
    """
    def col(key):
        return np.array([f[key] for f in frames], dtype=float)

    incident = col("incident")
    lit = incident > 0
    if not lit.any():
        return None

    def peak(key):
        values = col(key)
        i = np.nanargmax(values)
        return values[i], clock(hours[i])

    def whole_day(key):   # weighted by the sunlight hitting the roof
        values = col(key)
        return np.sum(values[lit] * incident[lit]) / np.sum(incident[lit])

    r1, t1 = peak("reflected_below")
    r2, t2 = peak("max_sky_strength")
    r3, t3 = peak("absorbed")
    return dict(max_reflected_below=r1, max_reflected_below_time=t1,
                max_sky_intensity=r2, max_sky_intensity_time=t2,
                max_absorption=r3, max_absorption_time=t3,
                overall_absorption=whole_day("absorbed"),
                overall_reflectivity=whole_day("reflectivity"))


def print_results(frames, hours):
    r = day_results(frames, hours)
    if r is None:
        print("the sun never reaches the outside of the roof")
        return
    print()
    print("RESULTS (day simulation)")
    print(f"  #1 max power reflected below the horizon : {r['max_reflected_below']:.3f}"
          f"   at {r['max_reflected_below_time']}   (fraction of sunlight on the roof)")
    print(f"  #2 max reflection intensity to the sky   : {r['max_sky_intensity']:.3f}"
          f"   at {r['max_sky_intensity_time']}   (strongest ray, 1 = direct sunlight)")
    print(f"  #3 max absorption                        : {r['max_absorption']:.3f}"
          f"   at {r['max_absorption_time']}   (fraction of sunlight on the roof)")
    print(f"  #4 overall absorption (whole day)        : {r['overall_absorption']:.3f}"
          "            (weighted by sunlight on the roof)")
    print(f"     overall reflectivity (whole day)      : {r['overall_reflectivity']:.3f}")
    print()


# =============================================================================
# ANIMATION
# =============================================================================

def main():
    use_animation_settings()
    hours = np.linspace(SUNRISE_HOUR, SUNSET_HOUR, N_FRAMES)
    angles = np.linspace(SUN_START_ANGLE, SUN_END_ANGLE, N_FRAMES)

    print(f"computing {N_FRAMES} frames on {os.cpu_count()} cores ...")
    with ProcessPoolExecutor(initializer=use_animation_settings) as pool:
        frames = list(pool.map(compute_frame, angles))

    print_results(frames, hours)

    # ---- figure layout
    fig, ax = plt.subplots(figsize=(10, 7))

    roof = build_roof()
    pts = np.array([s.p1 for s in roof] + [s.p2 for s in roof])
    center = pts.mean(axis=0)
    sun_r = 0.5 * np.ptp(pts, axis=0).max() + 8      # sun drawn on this arc
    pad = 5
    ax.set_xlim(min(pts[:, 0].min(), center[0] - sun_r) - pad,
                max(pts[:, 0].max(), center[0] + sun_r) + pad)
    ax.set_ylim(pts[:, 1].min() - pad, center[1] + sun_r + pad)
    ax.set_aspect("equal")
    ax.set_facecolor("white")

    # sky arc and roof (static)
    arc = np.radians(np.linspace(SUN_START_ANGLE, SUN_END_ANGLE, 100))
    ax.plot(center[0] + sun_r * np.cos(arc), center[1] + sun_r * np.sin(arc),
            ls=":", color="0.75", lw=1)
    for seg in roof:
        ax.plot([seg.p1[0], seg.p2[0]], [seg.p1[1], seg.p2[1]],
                color=sim.SEGMENT_COLORS.get(seg.kind, "black"), lw=3,
                solid_capstyle="round", zorder=4)
    for kind in sorted({s.kind for s in roof if s.kind in sim.SEGMENT_COLORS}):
        ax.plot([], [], color=sim.SEGMENT_COLORS[kind], lw=3, label=f"segment {kind}")
    ax.legend(loc="upper left", fontsize=8)

    norm = sim.color_norm()
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=sim.RAY_COLORMAP),
                        ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("reflected ray strength (1 = incoming sunlight)")

    # dynamic parts
    rays = LineCollection([], linewidths=sim.RAY_WIDTH)
    ax.add_collection(rays)
    sun_dot, = ax.plot([], [], "o", ms=18, color="gold", mec="darkorange", zorder=7)
    clock_text = ax.text(0.98, 0.97, "", transform=ax.transAxes, ha="right", va="top",
                    fontsize=14, family="monospace")
    arrow_artists = []

    def update(i):
        f = frames[i]

        rel = np.clip(norm(f["strength"]), 0, 1)
        rgba = sim.RAY_COLORMAP(rel)
        rgba[:, 3] = sim.RAY_OPACITY * rel
        rays.set_segments(f["lines"])
        rays.set_color(rgba)

        e = np.radians(angles[i])
        sun_dot.set_data([center[0] + sun_r * np.cos(e)], [center[1] + sun_r * np.sin(e)])
        clock_text.set_text(f"{clock(hours[i])}\nsun {angles[i]:5.1f}°")
        title = "Rooftop over the day"
        if np.isnan(f["reflectivity"]):
            title += "   (sun behind the roof)"
        ax.set_title(title)

        for a in arrow_artists:
            a.remove()
        arrow_artists.clear()
        for tail, head in f["arrows"]:
            arrow_artists.append(ax.annotate(
                "", xy=head, xytext=tail, zorder=6,
                arrowprops=dict(arrowstyle="-|>", color="darkorange", lw=2,
                                mutation_scale=15)))

        return [rays, sun_dot, clock_text, *arrow_artists]

    plt.tight_layout()
    anim = FuncAnimation(fig, update, frames=N_FRAMES, interval=1000 / FPS)

    if SAVE_MP4:
        if FFMpegWriter.isAvailable():
            print(f"saving {SAVE_MP4} ...")
            anim.save(SAVE_MP4, writer=FFMpegWriter(fps=FPS, bitrate=4000), dpi=150)
            print("done")
        else:
            print("cannot save mp4: ffmpeg not found (install with: sudo apt install ffmpeg)")
    if SAVE_GIF:
        print(f"saving {SAVE_GIF} ...")
        anim.save(SAVE_GIF, writer=PillowWriter(fps=FPS))
        print("done")
    if SHOW:
        plt.show()


if __name__ == "__main__":
    main()
