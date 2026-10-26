"""
2D ray tracer for testing how reflective a rooftop shape is.

The roof is a stair made of N steps, each step one segment A followed by
one segment B. Sunlight comes in as parallel rays; every hit absorbs part of
the light and reflects the rest (mirror-like, glossy or fully scattered).
The result is the reflectivity: the share of the sunlight hitting the roof
that leaves it again.

Change the values in the PARAMETERS section below and run:

    python3 rooftop_sim.py
"""

from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, LogNorm


# =============================================================================
# PARAMETERS
# =============================================================================

# ---- roof ------------------------------------------------------------------
ROOF_START = (0.0, 0.0)  # (x, y) of the bottom of the roof
ROOF_SLOPE = 40.0        # overall slope of the roof [deg], positive = up to the right
ROOF_LENGTH = 25.0       # overall length of the roof, measured along the slope
N_STEPS = 5              # number of A+B steps

# ---- segments --------------------------------------------------------------
# Angles are absolute: 0 = horizontal (facing up), 90 = vertical (facing left).
# The lengths of A and B are calculated so the stair follows the roof line,
# which requires ROOF_SLOPE to lie between ANGLE_A and ANGLE_B.
ANGLE_A = 30.0           # [deg]
ANGLE_B = 90.0           # [deg]

# diffusion : 0 = perfect mirror ... 1 = complete scatter (matte)
# absorption: 0 = absorbs nothing ... 1 = absorbs everything
MATERIAL_A = dict(diffusion=0.6, absorption=0.6)
MATERIAL_B = dict(diffusion=0.6, absorption=0.5)

# ---- sun -------------------------------------------------------------------
SUN_ELEVATION = 140.0    # [deg] 90 = overhead, < 90 sun on the right, > 90 on the left

# ---- simulation ------------------------------------------------------------
N_SUN_RAYS = 10        # parallel rays of sunlight
N_DIFFUSE = 10          # rays a (non-mirror) reflection is split into
MAX_DEPTH = 4            # max number of bounces per ray
MIN_INTENSITY = 1e-4     # rays carrying less energy than this are dropped
GLOSS_SHARPNESS = 20.0   # how fast the reflection lobe narrows as diffusion -> 0

# ---- display ---------------------------------------------------------------
SHOW_INCOMING = False    # also draw the incoming sun rays (thin grey)
RAY_COLORMAP = LinearSegmentedColormap.from_list(
    "blue_yellow_red", ["blue", "gold", "red"])   # blue = weak, red = strong
INTENSITY_SCALE = "linear"  # "linear" or "log" colour scale
COLOR_MAX = 1.0          # top of the colour scale (1.0 = incoming sunlight);
                         # fixed, so colours can be compared between runs
LOG_DECADES = 3          # log scale only: show rays down to COLOR_MAX * 10^-3
BEAM_WIDTH_DEG = 10.0    # a mirror reflection counts as a beam this wide; a lobe
                         # as concentrated as that is shown at full strength
RAY_OPACITY = 0.5        # opacity of the strongest ray (lower = less pile-up)
RAY_WIDTH = 0.8          # line width of the reflected rays
N_ARROWS = 6             # sunlight direction arrows drawn along the roof
ARROW_GAP = 1.5          # distance between arrow tip and roof
SEGMENT_COLORS = {"A": "tab:green", "B": "black"}

# ---- internals -------------------------------------------------------------
SCENE_SIZE = 500.0       # how far away sun rays start / escaped rays are drawn
EPS = 1e-9


# =============================================================================
# DATA TYPES
# =============================================================================

@dataclass
class Segment:
    start: tuple
    end: tuple
    diffusion: float = 0.5   # 0 = perfect mirror, 1 = complete scatter
    absorption: float = 0.3  # 0 = absorbs nothing, 1 = absorbs everything
    name: str = ""
    kind: str = ""           # segment type, e.g. "A" or "B" (used for colouring)
    absorbed: float = field(default=0.0, init=False)

    @property
    def p1(self):
        return np.asarray(self.start, dtype=float)

    @property
    def p2(self):
        return np.asarray(self.end, dtype=float)

    def normal(self):
        """Outward normal: the outside of the roof is on the LEFT when walking
        from start to end (light hitting the other side is blocked by the
        building)."""
        d = self.p2 - self.p1
        n = np.array([-d[1], d[0]])
        return n / np.linalg.norm(n)


@dataclass
class Ray:
    origin: np.ndarray
    direction: np.ndarray
    intensity: float         # energy the ray carries (used for the energy balance)
    depth: int = 0           # number of bounces so far
    strength: float = 1.0    # brightness for the colours, independent of ray
                             # counts: 1.0 = as strong as the incoming sunlight


@dataclass
class PathPiece:
    """A straight piece of a ray path, kept for plotting."""
    start: np.ndarray
    end: np.ndarray
    intensity: float
    depth: int
    strength: float


# =============================================================================
# ROOF
# =============================================================================

def stair_roof(start, slope, length, n_steps, angle_a, angle_b,
               material_a, material_b):
    """
    Build a connected stair-shaped roof made of n_steps x (segment A + segment B).
    Returns (segments, length of A, length of B).

    The lengths of A and B are calculated so the steps exactly follow the
    roof line. This only works when the slope lies between angle_a and angle_b.
    """
    ua = np.array([np.cos(np.radians(angle_a)), np.sin(np.radians(angle_a))])
    ub = np.array([np.cos(np.radians(angle_b)), np.sin(np.radians(angle_b))])
    step = length / n_steps * np.array([np.cos(np.radians(slope)),
                                        np.sin(np.radians(slope))])

    # solve len_a * ua + len_b * ub = step
    m = np.column_stack([ua, ub])
    if abs(np.linalg.det(m)) < EPS:
        raise ValueError("angle_a and angle_b must not be parallel")
    len_a, len_b = np.linalg.solve(m, step)
    if len_a <= 0 or len_b <= 0:
        raise ValueError(
            f"slope {slope}° must lie between angle_a ({angle_a}°) and "
            f"angle_b ({angle_b}°): got len_a={len_a:.3f}, len_b={len_b:.3f}")

    segs = []
    p = np.asarray(start, dtype=float)
    for i in range(n_steps):
        q = p + len_a * ua
        segs.append(Segment(tuple(p), tuple(q), **material_a, name=f"A {i}", kind="A"))
        p = q + len_b * ub
        segs.append(Segment(tuple(q), tuple(p), **material_b, name=f"B {i}", kind="B"))
    return segs, len_a, len_b


# =============================================================================
# GEOMETRY
# =============================================================================

def intersect(ray: Ray, seg: Segment):
    """Return distance t along the ray to the segment, or None."""
    p, r = ray.origin, ray.direction
    q, s = seg.p1, seg.p2 - seg.p1
    denom = r[0] * s[1] - r[1] * s[0]
    if abs(denom) < EPS:
        return None  # parallel
    qp = q - p
    t = (qp[0] * s[1] - qp[1] * s[0]) / denom   # along ray
    u = (qp[0] * r[1] - qp[1] * r[0]) / denom   # along segment (0..1)
    if t > 1e-7 and -EPS <= u <= 1 + EPS:
        return t
    return None


def closest_hit(ray: Ray, segments):
    best_t, best_seg = None, None
    for seg in segments:
        t = intersect(ray, seg)
        if t is not None and (best_t is None or t < best_t):
            best_t, best_seg = t, seg
    return best_t, best_seg


def reflect(d, n):
    return d - 2 * np.dot(d, n) * n


def scatter_directions(n, mirror, diffusion, count):
    """
    Split a reflection into `count` rays around the mirror direction.
    Returns (directions, energy weights summing to 1, angle between rays).

    Energy per direction (angle t from the normal, t_m = mirror angle):
        cos(t) * cos(t - t_m) ^ p,   p = GLOSS_SHARPNESS * (1 - d) / d
    d = 1 gives p = 0: pure Lambertian scatter around the normal.
    Smaller d gives a narrower lobe around the mirror direction.
    The rays are placed evenly over the part of the hemisphere where the
    lobe carries energy, so narrow lobes still get all `count` rays.
    """
    base = np.arctan2(n[1], n[0])
    t_m = np.arctan2(n[0] * mirror[1] - n[1] * mirror[0], np.dot(n, mirror))
    p = GLOSS_SHARPNESS * (1 - diffusion) / diffusion

    def weight(t):
        return np.cos(t) * np.clip(np.cos(t - t_m), 0, None) ** p

    # find the angle range where the lobe has energy (strictly inside ±90°)
    fine = np.linspace(-np.pi / 2, np.pi / 2, 2001)[1:-1]
    w = weight(fine)
    inside = fine[w >= 1e-3 * w.max()]
    lo, hi = inside.min(), inside.max()

    offsets = lo + (np.arange(count) + 0.5) * (hi - lo) / count
    weights = weight(offsets)
    angles = base + offsets
    dirs = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    spacing = (hi - lo) / count
    return dirs, weights / weights.sum(), spacing


# =============================================================================
# TRACER
# =============================================================================

def sun_rays(segments, sun_elevation, n_rays):
    """
    `n_rays` parallel sun rays, spread evenly over the width of the roof as
    seen from the sun. sun_elevation is measured from the +x side
    (90 = straight overhead, < 90 sun on the right, > 90 sun on the left).
    """
    e = np.radians(sun_elevation)
    d = -np.array([np.cos(e), np.sin(e)])           # direction the light travels
    perp = np.array([-d[1], d[0]])
    pts = np.array([p for seg in segments for p in (seg.p1, seg.p2)])
    across = pts @ perp
    lo, hi = across.min(), across.max()
    spacing = (hi - lo) / n_rays
    start_along = (pts @ d).min() - SCENE_SIZE      # start well before the roof
    return [Ray(perp * (lo + (i + 0.5) * spacing) + d * start_along, d, spacing)
            for i in range(n_rays)]


def trace(segments, sun_elevation):
    """
    Trace all sunlight over the roof. Returns (path pieces, stats); all energies
    are fractions of the sunlight that hits the roof.
    """
    for seg in segments:
        seg.absorbed = 0.0

    stack = sun_rays(segments, sun_elevation, N_SUN_RAYS)

    pieces = []
    escaped = 0.0        # energy that leaves the scene
    escaped_up = 0.0     # part of it going upwards (reflected back to the sky)
    dropped = 0.0        # energy lost due to depth / intensity cut-off,
                         # or going into the inside of the roof
    incident = 0.0       # energy of sun rays that directly hit the roof

    while stack:
        ray = stack.pop()
        t, seg = closest_hit(ray, segments)

        if seg is None:
            if ray.depth == 0:
                continue  # never touched the roof: not drawn, not measured
            end = ray.origin + ray.direction * SCENE_SIZE
            pieces.append(PathPiece(ray.origin, end, ray.intensity, ray.depth, ray.strength))
            escaped += ray.intensity
            if ray.direction[1] > 0:
                escaped_up += ray.intensity
            continue

        n = seg.normal()
        hit = ray.origin + ray.direction * t

        # hitting the inside of the roof: that light is blocked by the building
        if np.dot(n, ray.direction) > 0:
            if ray.depth > 0:
                pieces.append(PathPiece(ray.origin, hit, ray.intensity, ray.depth, ray.strength))
                dropped += ray.intensity
            continue

        if ray.depth == 0:
            incident += ray.intensity
        pieces.append(PathPiece(ray.origin, hit, ray.intensity, ray.depth, ray.strength))

        absorbed = ray.intensity * seg.absorption
        seg.absorbed += absorbed
        remaining = ray.intensity - absorbed

        if ray.depth + 1 > MAX_DEPTH:
            dropped += remaining
            continue

        new_origin = hit + n * 1e-6
        mirror = reflect(ray.direction, n)
        reflected = ray.strength * (1 - seg.absorption)

        if seg.diffusion <= 0:
            dirs, weights = [mirror], [1.0]       # perfect mirror
            strengths = [reflected]
        else:
            dirs, weights, spacing = scatter_directions(n, mirror, seg.diffusion, N_DIFFUSE)
            # strength = how concentrated the light is: energy per angle,
            # compared to a mirror beam of BEAM_WIDTH_DEG. Wide lobes spread
            # the light, so each direction is weaker. Independent of N_DIFFUSE.
            per_angle = np.asarray(weights) / spacing
            strengths = reflected * np.minimum(per_angle * np.radians(BEAM_WIDTH_DEG), 1.0)

        for d, w, st in zip(dirs, weights, strengths):
            if remaining * w > MIN_INTENSITY:
                stack.append(Ray(new_origin, d, remaining * w, ray.depth + 1, st))
            else:
                dropped += remaining * w

    if incident == 0:
        raise ValueError("no light hits the outside of the roof "
                         "(is the sun behind the roof?)")

    # normalise everything to the energy that actually hit the roof
    for seg in segments:
        seg.absorbed /= incident
    for p in pieces:
        p.intensity /= incident
    stats = dict(escaped=escaped / incident,
                 escaped_up=escaped_up / incident,
                 dropped=dropped / incident,
                 reflectivity=escaped / incident)
    return pieces, stats


# =============================================================================
# OUTPUT
# =============================================================================

def print_report(segments, stats, len_a, len_b):
    print(f"segment A length: {len_a:.3f}   segment B length: {len_b:.3f}\n")
    print(f"{'segment':<12}{'diff':>6}{'abs':>6}{'absorbed':>12}")
    for s in segments:
        print(f"{s.name:<12}{s.diffusion:>6.2f}{s.absorption:>6.2f}{s.absorbed:>12.4f}")
    total_abs = sum(s.absorbed for s in segments)
    print("\n(values below are fractions of the energy hitting the roof)")
    print(f"total absorbed by roof : {total_abs:.4f}")
    print(f"escaped upwards (sky)  : {stats['escaped_up']:.4f}")
    print(f"escaped other dirs     : {stats['escaped'] - stats['escaped_up']:.4f}")
    print(f"dropped (cut-off)      : {stats['dropped']:.4f}")
    print(f"REFLECTIVITY           : {stats['reflectivity']:.4f}")


def color_norm():
    if INTENSITY_SCALE == "log":
        return LogNorm(COLOR_MAX / 10 ** LOG_DECADES, COLOR_MAX)
    return plt.Normalize(0, COLOR_MAX)


def plot(segments, pieces, stats, sun_elevation, title="Rooftop reflectivity"):
    fig, ax = plt.subplots(figsize=(10, 7))

    # incoming rays (optional), drawn thin and grey
    if SHOW_INCOMING:
        incoming = [[p.start, p.end] for p in pieces if p.depth == 0]
        ax.add_collection(LineCollection(incoming, colors="0.6", linewidths=0.5))

    # reflected rays: colour and opacity follow the ray strength on a fixed
    # scale (0..COLOR_MAX). Strongest rays are drawn on top.
    refl = sorted((p for p in pieces if p.depth > 0), key=lambda p: p.strength)
    if refl:
        rel = np.clip(color_norm()(np.array([p.strength for p in refl])), 0, 1)
        rgba = RAY_COLORMAP(rel)
        rgba[:, 3] = RAY_OPACITY * rel
        visible = rel > 1e-3          # skip lines too faint to see
        lines = [[p.start, p.end] for p, v in zip(refl, visible) if v]
        ax.add_collection(LineCollection(lines, colors=rgba[visible], linewidths=RAY_WIDTH))
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=color_norm(), cmap=RAY_COLORMAP),
                        ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("reflected ray strength (1 = incoming sunlight)")

    # roof
    for seg in segments:
        ax.plot([seg.p1[0], seg.p2[0]], [seg.p1[1], seg.p2[1]],
                color=SEGMENT_COLORS.get(seg.kind, "black"), lw=3,
                solid_capstyle="round", zorder=4)

    # legend
    if SHOW_INCOMING:
        ax.plot([], [], color="0.6", lw=0.5, label="incoming light")
    for kind in sorted({seg.kind for seg in segments if seg.kind in SEGMENT_COLORS}):
        ax.plot([], [], color=SEGMENT_COLORS[kind], lw=3, label=f"segment {kind}")
    ax.legend(loc="upper left", fontsize=8)

    # view: the roof plus some room around it (extra room above for the light)
    pts = np.array([s.p1 for s in segments] + [s.p2 for s in segments])
    pad = 5
    ax.set_xlim(pts[:, 0].min() - pad, pts[:, 0].max() + pad)
    ax.set_ylim(pts[:, 1].min() - pad, pts[:, 1].max() + 2 * pad)
    ax.set_aspect("equal")
    ax.set_facecolor("white")
    ax.set_title(f"{title}   (reflectivity = {stats['reflectivity']:.3f})")

    # arrows along the roof showing the direction of the incoming sunlight:
    # each arrow follows a real sun ray and stops just before it hits the roof
    ylo, yhi = ax.get_ylim()
    length = 0.12 * (yhi - ylo)
    for ray in sun_rays(segments, sun_elevation, N_ARROWS):
        t, seg = closest_hit(ray, segments)
        if seg is None or np.dot(seg.normal(), ray.direction) > 0:
            continue                       # misses the roof or hits its inside
        head = ray.origin + ray.direction * (t - ARROW_GAP)
        ax.annotate("", xy=head, xytext=head - ray.direction * length, zorder=6,
                    arrowprops=dict(arrowstyle="-|>", color="darkorange", lw=2,
                                    mutation_scale=15))

    plt.tight_layout()
    plt.show()


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    roof, len_a, len_b = stair_roof(ROOF_START, ROOF_SLOPE, ROOF_LENGTH, N_STEPS,
                                    ANGLE_A, ANGLE_B, MATERIAL_A, MATERIAL_B)
    pieces, stats = trace(roof, SUN_ELEVATION)
    print_report(roof, stats, len_a, len_b)
    plot(roof, pieces, stats, SUN_ELEVATION)
