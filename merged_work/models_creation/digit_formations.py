"""
Digit formation offsets for drone swarms (digits 0-9).

Deterministic placement: explicit junction anchors (one slot each) plus interior
slots distributed evenly along line/arc edges by arc length. No relaxation or projection.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

Point = Tuple[float, float]
Stroke = Union["LineStroke", "ArcStroke", "EllipseStroke"]

# --- Refinement tuning (canonical XY units) ---
DIGIT_THREE_LOWER_CY_SHIFT = 0.10
DIGIT_THREE_LOWER_TRIM_FRAC = 0.10

DIGIT_EIGHT_LOWER_TRIM_FRAC = 0.20


def _point_on_circle(cx: float, cy: float, r: float, theta: float) -> Point:
    return (cx + r * math.cos(theta), cy + r * math.sin(theta))


@dataclass(frozen=True)
class LineStroke:
    p0: Point
    p1: Point

    def length(self) -> float:
        dx = self.p1[0] - self.p0[0]
        dy = self.p1[1] - self.p0[1]
        return float(math.hypot(dx, dy))

    def sample(self, t: float) -> Point:
        return (
            self.p0[0] + t * (self.p1[0] - self.p0[0]),
            self.p0[1] + t * (self.p1[1] - self.p0[1]),
        )


@dataclass(frozen=True)
class ArcStroke:
    cx: float
    cy: float
    radius: float
    theta0: float
    theta1: float

    def length(self) -> float:
        return abs(self.theta1 - self.theta0) * float(self.radius)

    def sample(self, t: float) -> Point:
        theta = self.theta0 + t * (self.theta1 - self.theta0)
        return (
            self.cx + self.radius * math.cos(theta),
            self.cy + self.radius * math.sin(theta),
        )


@dataclass(frozen=True)
class EllipseStroke:
    cx: float
    cy: float
    rx: float
    ry: float
    theta0: float
    theta1: float

    def length(self) -> float:
        a, b = self.rx, self.ry
        if a + b <= 0.0:
            return 0.0
        h = ((a - b) / (a + b)) ** 2
        full = math.pi * (a + b) * (1.0 + 3.0 * h / (10.0 + math.sqrt(4.0 - 3.0 * h)))
        return float(full * abs(self.theta1 - self.theta0) / (2.0 * math.pi))

    def sample(self, t: float) -> Point:
        th = self.theta0 + t * (self.theta1 - self.theta0)
        return (
            self.cx + self.rx * math.cos(th),
            self.cy + self.ry * math.sin(th),
        )


@dataclass(frozen=True)
class EdgeDef:
    """Edge primitive with optional anchor indices at t=0 and t=1."""

    stroke: Stroke
    anchor_t0: Optional[int] = None  # index into anchors at parameter t=0
    anchor_t1: Optional[int] = None  # index into anchors at parameter t=1
    closed_loop: bool = False  # full circle anchored at t=0 only
    # Remap interior samples from nominal t in (0,1) to stroke param in [param_lo, param_hi]
    param_lo: float = 0.0
    param_hi: float = 1.0


@dataclass(frozen=True)
class DigitDef:
    anchors: Tuple[Point, ...]
    edges: Tuple[EdgeDef, ...]


def _build_nine_real_digit() -> DigitDef:
    """Digit 9: closed head loop with stem from lower-right junction."""
    cx, cy, r = 0.0, 0.8, 0.8
    junction_theta = -math.pi / 4.0
    junction = _point_on_circle(cx, cy, r, junction_theta)
    tail = (junction[0], -1.6)
    return DigitDef(
        anchors=(junction,),
        edges=(
            EdgeDef(
                ArcStroke(cx, cy, r, junction_theta, junction_theta + 2.0 * math.pi),
                anchor_t0=0,
                anchor_t1=0,
                closed_loop=True,
            ),
            EdgeDef(
                LineStroke(junction, tail),
                anchor_t0=0,
                anchor_t1=None,
            ),
        ),
    )


def _mirror_origin_digit_def(defn: DigitDef) -> DigitDef:
    """Flip every point through the origin — digit 6 = digit 9 with same formulas."""
    anchors = tuple((-x, -y) for x, y in defn.anchors)
    edges: List[EdgeDef] = []
    for edge in defn.edges:
        stroke = edge.stroke
        if isinstance(stroke, LineStroke):
            mirrored: Stroke = LineStroke(
                (-stroke.p0[0], -stroke.p0[1]),
                (-stroke.p1[0], -stroke.p1[1]),
            )
        elif isinstance(stroke, ArcStroke):
            mirrored = ArcStroke(
                -stroke.cx,
                -stroke.cy,
                stroke.radius,
                stroke.theta0 + math.pi,
                stroke.theta1 + math.pi,
            )
        elif isinstance(stroke, EllipseStroke):
            mirrored = EllipseStroke(
                -stroke.cx,
                -stroke.cy,
                stroke.rx,
                stroke.ry,
                stroke.theta0 + math.pi,
                stroke.theta1 + math.pi,
            )
        else:
            raise TypeError(f"Unsupported stroke type: {type(stroke)}")
        edges.append(
            EdgeDef(
                mirrored,
                edge.anchor_t0,
                edge.anchor_t1,
                edge.closed_loop,
                edge.param_lo,
                edge.param_hi,
            )
        )
    return DigitDef(anchors=anchors, edges=tuple(edges))


def _distribute_counts(total: int, weights: Sequence[float]) -> List[int]:
    """Largest-remainder allocation of `total` across positive weights."""
    n = len(weights)
    if total <= 0 or n == 0:
        return [0] * n
    w = np.asarray(weights, dtype=np.float64)
    if w.sum() <= 0:
        return [total // n + (1 if i < total % n else 0) for i in range(n)]
    exact = total * w / w.sum()
    base = np.floor(exact).astype(int)
    rem = int(total - base.sum())
    order = np.argsort(-(exact - base))
    for i in order[:rem]:
        base[i] += 1
    return base.tolist()


def _interior_t_values(
    r: int,
    anchor_t0: Optional[int],
    anchor_t1: Optional[int],
    closed_loop: bool,
) -> List[float]:
    """
    Interior parameters strictly in (0, 1), never on anchored endpoints.
    Same as rectangle/triangle: step / (num_edge + 1) measured from the
    free side when exactly one endpoint is an anchor.
    """
    if r <= 0:
        return []
    if closed_loop:
        if anchor_t0 is not None:
            return [k / (r + 1) for k in range(1, r + 1)]
        # Closed loop with no anchor (e.g. digit 0): uniform around circle.
        return [(k - 0.5) / r for k in range(1, r + 1)]
    has0 = anchor_t0 is not None
    has1 = anchor_t1 is not None
    if has0 and has1:
        return [k / (r + 1) for k in range(1, r + 1)]
    if has0 and not has1:
        # Anchor at p0; interior spans toward free endpoint p1.
        return [1.0 - k / (r + 1) for k in range(1, r + 1)]
    if has1 and not has0:
        # Anchor at p1; interior spans from free endpoint p0.
        return [k / (r + 1) for k in range(1, r + 1)]
    return [(k - 0.5) / r for k in range(1, r + 1)]


def _edge_effective_length(edge: EdgeDef) -> float:
    span = float(edge.param_hi - edge.param_lo)
    if span <= 0.0:
        return 0.0
    return float(edge.stroke.length()) * span


def _canonical_perimeter(defn: DigitDef) -> float:
    return float(sum(_edge_effective_length(e) for e in defn.edges))


def _build_digit_def(digit: int) -> DigitDef:
    """Per-digit anchors and edges in canonical coordinates."""
    if digit == 0:
        return DigitDef(
            anchors=(),
            edges=(
                EdgeDef(
                    EllipseStroke(0.0, 0.0, 1.0, 1.6, 0.0, 2.0 * math.pi),
                    anchor_t0=None,
                    anchor_t1=None,
                    closed_loop=True,
                ),
            ),
        )

    if digit == 1:
        return DigitDef(
            anchors=((0.0, 1.6), (0.0, -1.6)),
            edges=(
                EdgeDef(LineStroke((-0.4, 1.0), (0.0, 1.6)), anchor_t0=None, anchor_t1=0),
                EdgeDef(LineStroke((0.0, 1.6), (0.0, -1.6)), anchor_t0=0, anchor_t1=1),
                EdgeDef(LineStroke((-0.6, -1.6), (0.0, -1.6)), anchor_t0=None, anchor_t1=1),
                EdgeDef(LineStroke((0.0, -1.6), (0.6, -1.6)), anchor_t0=1, anchor_t1=None),
            ),
        )

    if digit == 2:
        return DigitDef(
            anchors=((0.8, 0.6), (-0.8, -1.6)),
            edges=(
                EdgeDef(ArcStroke(0.0, 0.6, 0.8, math.pi, 0.0), anchor_t0=None, anchor_t1=0),
                EdgeDef(LineStroke((0.8, 0.6), (-0.8, -1.6)), anchor_t0=0, anchor_t1=1),
                EdgeDef(LineStroke((-0.8, -1.6), (0.8, -1.6)), anchor_t0=1, anchor_t1=None),
            ),
        )

    if digit == 3:
        cy_lo = -0.8 + DIGIT_THREE_LOWER_CY_SHIFT
        r_lo = -cy_lo  # keep waist (0, 0) on lower arc at theta = pi/2
        return DigitDef(
            anchors=((0.0, 0.0),),
            edges=(
                EdgeDef(
                    ArcStroke(0.0, 0.8, 0.8, math.pi, -math.pi / 2),
                    anchor_t0=None,
                    anchor_t1=0,
                ),
                EdgeDef(
                    ArcStroke(0.0, cy_lo, r_lo, math.pi / 2, -math.pi),
                    anchor_t0=0,
                    anchor_t1=None,
                    param_lo=DIGIT_THREE_LOWER_TRIM_FRAC,
                    param_hi=1.0,
                ),
            ),
        )

    if digit == 4:
        return DigitDef(
            anchors=((-0.8, 0.0), (0.8, 0.0)),
            edges=(
                EdgeDef(LineStroke((-0.8, 1.6), (-0.8, 0.0)), anchor_t0=None, anchor_t1=0),
                EdgeDef(LineStroke((-0.8, 0.0), (0.8, 0.0)), anchor_t0=0, anchor_t1=1),
                EdgeDef(LineStroke((0.8, 1.6), (0.8, 0.0)), anchor_t0=None, anchor_t1=1),
                EdgeDef(LineStroke((0.8, 0.0), (0.8, -1.6)), anchor_t0=1, anchor_t1=None),
            ),
        )

    if digit == 5:
        # All corners are anchors; bottom is a U made of lines (no arc through (0, 0)).
        return DigitDef(
            anchors=(
                (-0.8, 1.6),
                (-0.8, 0.0),
                (0.8, 0.0),
                (0.8, -1.6),
                (-0.8, -1.6),
            ),
            edges=(
                EdgeDef(LineStroke((-0.8, 1.6), (0.8, 1.6)), anchor_t0=0, anchor_t1=None),
                EdgeDef(LineStroke((-0.8, 1.6), (-0.8, 0.0)), anchor_t0=0, anchor_t1=1),
                EdgeDef(LineStroke((-0.8, 0.0), (0.8, 0.0)), anchor_t0=1, anchor_t1=2),
                EdgeDef(LineStroke((0.8, 0.0), (0.8, -1.6)), anchor_t0=2, anchor_t1=3),
                EdgeDef(LineStroke((0.8, -1.6), (-0.8, -1.6)), anchor_t0=3, anchor_t1=4),
            ),
        )

    if digit == 6:
        return _mirror_origin_digit_def(_build_nine_real_digit())

    if digit == 7:
        return DigitDef(
            anchors=((0.8, 1.6),),
            edges=(
                EdgeDef(LineStroke((-0.8, 1.6), (0.8, 1.6)), anchor_t0=None, anchor_t1=0),
                EdgeDef(LineStroke((0.8, 1.6), (-0.4, -1.6)), anchor_t0=0, anchor_t1=None),
            ),
        )

    if digit == 8:
        waist = (0.0, 0.0)
        wedge = DIGIT_EIGHT_LOWER_TRIM_FRAC * 2.0 * math.pi
        half_w = 0.5 * wedge
        th_lo0 = math.pi / 2 + half_w
        th_lo1 = math.pi / 2 + 2.0 * math.pi - half_w
        return DigitDef(
            anchors=(waist,),
            edges=(
                EdgeDef(
                    ArcStroke(0.0, 0.8, 0.8, -math.pi / 2, -math.pi / 2 + 2.0 * math.pi),
                    anchor_t0=0,
                    anchor_t1=0,
                    closed_loop=True,
                ),
                EdgeDef(
                    ArcStroke(0.0, -0.8, 0.8, th_lo0, th_lo1),
                    anchor_t0=None,
                    anchor_t1=None,
                    closed_loop=False,
                ),
            ),
        )

    if digit == 9:
        return _build_nine_real_digit()

    raise ValueError(f"Unsupported digit: {digit}")


def _allocate_slots(defn: DigitDef, num_drones: int) -> np.ndarray:
    """Place anchors and interior points; return (N, 2) before scaling/centering."""
    anchors = list(defn.anchors)
    k = len(anchors)
    if num_drones <= 0:
        return np.zeros((0, 2), dtype=np.float32)

    if num_drones <= k:
        xy = np.asarray(anchors[:num_drones], dtype=np.float32)
        return xy

    slots: List[Point] = list(anchors)
    remaining = num_drones - k
    lengths = [_edge_effective_length(e) for e in defn.edges]
    per_edge = _distribute_counts(remaining, lengths)

    for edge, r_i in zip(defn.edges, per_edge):
        ts = _interior_t_values(
            r_i,
            edge.anchor_t0,
            edge.anchor_t1,
            edge.closed_loop,
        )
        for t in ts:
            tau = edge.param_lo + t * (edge.param_hi - edge.param_lo)
            slots.append(edge.stroke.sample(tau))

    xy = np.asarray(slots, dtype=np.float32)
    if xy.shape[0] != num_drones:
        raise RuntimeError(
            f"slot count mismatch: expected {num_drones}, got {xy.shape[0]}"
        )
    return xy


_DIGIT_DEFS: Dict[int, DigitDef] = {d: _build_digit_def(d) for d in range(10)}


def _enforce_min_sep(xy: np.ndarray, min_sep: float) -> np.ndarray:
    """
    Uniformly scale XY so every pairwise distance is >= min_sep (negotiator-style).
    """
    if xy.shape[0] < 2 or min_sep <= 0:
        return xy
    pts = np.asarray(xy, dtype=np.float64).copy()
    for i, p in enumerate(pts):
        key = (round(float(p[0]), 6), round(float(p[1]), 6))
        for j in range(i):
            k2 = (round(float(pts[j, 0]), 6), round(float(pts[j, 1]), 6))
            if k2 == key:
                pts[i] += np.random.uniform(-1e-3, 1e-3, size=2)
                break
    diff = pts[:, None, :] - pts[None, :, :]
    dists = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(dists, np.inf)
    min_d = float(dists.min())
    if min_d > 0 and min_d < min_sep:
        pts *= min_sep / min_d
    elif min_d == 0:
        pts += np.random.uniform(-0.01, 0.01, size=pts.shape)
        return _enforce_min_sep(pts.astype(np.float32), min_sep)
    return pts.astype(np.float32)


def sample_digit_offsets(
    digit: int,
    num_drones: int,
    spacing: float = 2.0,
    min_separation: float = 0.3,
) -> np.ndarray:
    """
    Build (num_drones, 3) formation offsets centered at the origin.

    Every digit is scaled to the same bounding-box height (N * spacing / 2) and
    centered on its bbox midpoint, then scaled so pairwise distance >= min_separation.
    """
    if num_drones <= 0:
        return np.zeros((0, 3), dtype=np.float32)
    if digit not in _DIGIT_DEFS:
        raise ValueError(f"Unsupported digit: {digit}")

    defn = _DIGIT_DEFS[digit]
    xy = _allocate_slots(defn, num_drones)

    bbox_h = float(xy[:, 1].max() - xy[:, 1].min())
    target_h = float(num_drones) * float(spacing) / 2.0
    if bbox_h > 1e-9 and target_h > 0:
        xy = xy * (target_h / bbox_h)

    bbox_min = xy.min(axis=0)
    bbox_max = xy.max(axis=0)
    xy -= 0.5 * (bbox_min + bbox_max)

    min_sep = float(min_separation) * float(spacing)
    xy = _enforce_min_sep(xy, min_sep)

    offsets = np.zeros((num_drones, 3), dtype=np.float32)
    offsets[:, :2] = xy
    return offsets


def build_digit_formation(
    digit: int,
    num_drones: int,
    spacing: float = 2.0,
    altitude: float = 2.0,
) -> np.ndarray:
    offsets = sample_digit_offsets(digit=digit, num_drones=num_drones, spacing=spacing)
    slots = np.zeros((num_drones, 3), dtype=np.float32)
    slots[:, :2] = offsets[:, :2]
    slots[:, 2] = float(altitude)
    return slots


def build_digit_one_hot(digit: int) -> np.ndarray:
    if digit < 0 or digit > 9:
        raise ValueError(f"Digit out of range 0-9: {digit}")
    one_hot = np.zeros(10, dtype=np.float32)
    one_hot[digit] = 1.0
    return one_hot


def min_pairwise_distance_xy(xy: np.ndarray) -> float:
    """Minimum pairwise distance in 2D (inf if fewer than 2 points)."""
    if xy.shape[0] < 2:
        return float("inf")
    diff = xy[None, :, :] - xy[:, None, :]
    dist_sq = np.einsum("ijk,ijk->ij", diff, diff)
    np.fill_diagonal(dist_sq, np.inf)
    return float(np.sqrt(dist_sq.min()))
