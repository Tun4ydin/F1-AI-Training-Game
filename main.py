from __future__ import annotations

import json
import math
import random
import re
from bisect import bisect_right
from dataclasses import dataclass, field
from pathlib import Path

import pygame
from pygame import Vector2
from safe_algorithm import AlgorithmError, SafeAlgorithm

WIDTH, HEIGHT = 1280, 760
DISPLAY_WIDTH, DISPLAY_HEIGHT = 1920, 1080
PANEL = 280
CANVAS_W = WIDTH - PANEL
FPS = 60
# All world geometry is measured in metres.
ROAD_W = 12.0
KERB_W = 14.0
KERB_COLOR_INTERVAL_M = 100.0
BORDER_W = 30.0
CAR_WIDTH_M = 2.0
CAR_WHEELBASE_M = 3.4
CAR_LENGTH_M = 5.6
PITLANE_WIDTH_M = 6.0
PIT_SPEED_LIMIT_KPH = 80.0
PIT_SPEED_LIMIT_MPF = PIT_SPEED_LIMIT_KPH / (FPS * 3.6)
GEAR_COUNT = 8
MAX_ENGINE_RPM = 13000.0
IDLE_ENGINE_RPM = 4000.0
REFERENCE_TOP_SPEED = 1.67
REFERENCE_SPEED_KPH = REFERENCE_TOP_SPEED * FPS * 3.6
# A modern eight-speed sequential gearbox. Each gear's speed ceiling follows
# from its ratio relative to eighth gear, so low gears are shorter and stronger.
# Seventh is deliberately close to sixth so the car reaches eighth at roughly
# 292 km/h instead of stalling below the former 314 km/h shift point.
GEAR_RATIOS = (3.20, 2.55, 2.08, 1.72, 1.45, 1.25, 1.16, 0.94)
GEAR_SPEED_LIMITS = tuple(
    min(1.0, GEAR_RATIOS[-1] / ratio)
    for ratio in GEAR_RATIOS
)
# Straight-line calibration targets:
# combustion only: 0–100 ≈ 2.6 s and 0–200 ≈ 5.0 s;
# deployment from third gear: 0–200 ≈ 4.6 s.
BASE_ENGINE_ACCELERATION = 0.0066
LOW_SPEED_TRACTION = 0.32
FULL_TRACTION_SPEED = 0.60
HYBRID_COMBUSTION_SHARE = 0.80
HYBRID_ELECTRIC_SHARE = 0.20
HYBRID_TOTAL_POWER_SCALE = 1.1875
ROLLING_RESISTANCE = 0.00035
AERO_DRAG_COEFFICIENT = 0.00076
OPPONENT_SENSOR_RANGE_M = 60.0
OVERTAKE_REWARD = 150.0
OVERTAKE_COOLDOWN_FRAMES = FPS * 3
# Boundary vision remains responsive at 20 Hz while the physics stays at
# 60 Hz. Staggering the work is important when a full 20-car grid is active.
RAYCAST_UPDATE_FRAMES = 3
DEFAULT_CAMERA_ZOOM = 8.0
MIN_CAMERA_ZOOM = 2.0
MAX_CAMERA_ZOOM = 16.0
ROOT = Path(__file__).resolve().parent
TRACK_DIR = ROOT / "saved_data" / "tracks"
BRAIN_DIR = ROOT / "saved_data" / "brains"
ALGORITHM_DIR = ROOT / "saved_data" / "algorithms"
REPLAY_DIR = ROOT / "saved_data" / "replays"

BG = (7, 15, 18)
GRASS = (31, 92, 49)
ROAD = (57, 61, 66)
ROAD_EDGE = (225, 228, 220)
RED = (213, 44, 54)
WHITE = (239, 247, 244)
YELLOW = (255, 196, 77)
CYAN = (67, 225, 190)
MUTED = (132, 153, 148)
INK = (4, 9, 12)
UI_SURFACE = (13, 28, 27)
UI_SURFACE_RAISED = (18, 39, 36)
UI_SURFACE_HOVER = (25, 55, 49)
UI_BORDER = (48, 78, 72)
UI_BORDER_BRIGHT = (78, 122, 111)
UI_SHADOW = (3, 8, 10)
UI_BLUE = (83, 160, 255)
UI_VIOLET = (166, 115, 255)
UI_ORANGE = (255, 137, 79)
UI_ACCENTS = (CYAN, UI_BLUE, UI_ORANGE, UI_VIOLET)
COLORS = [
    (239, 65, 54), (56, 189, 248), (250, 204, 21), (34, 197, 94),
    (168, 85, 247), (249, 115, 22), (236, 72, 153), (20, 184, 166),
    (248, 250, 252), (148, 163, 184), (220, 38, 38), (14, 165, 233),
    (234, 179, 8), (22, 163, 74), (147, 51, 234), (234, 88, 12),
    (219, 39, 119), (13, 148, 136), (203, 213, 225), (100, 116, 139),
]
_CAR_SPRITE_CACHE = {}


def clamp(value, low, high):
    return max(low, min(high, value))


def safe_component_name(name, fallback="Untitled"):
    """Return a readable, filesystem-safe component name."""
    cleaned = "".join(
        character
        for character in str(name).strip()
        if character.isalnum() or character in "-_ "
    )
    cleaned = " ".join(cleaned.split())[:40].strip(" .-_")
    return cleaned or fallback


def component_filename(name, suffix):
    safe = safe_component_name(name)
    slug = safe.replace(" ", "_").lower()
    return f"{slug}{suffix}"


def default_track():
    return [
        (245, 180), (420, 105), (650, 115), (825, 180), (900, 315),
        (850, 500), (710, 610), (520, 625), (360, 575), (215, 550),
        (115, 430), (145, 285),
    ]


def seg_distance(point, a, b):
    p, a, b = Vector2(point), Vector2(a), Vector2(b)
    ab = b - a
    t = clamp((p - a).dot(ab) / max(ab.length_squared(), 0.001), 0, 1)
    nearest = a + ab * t
    return (p - nearest).length(), nearest, t


class Track:
    SAMPLES_PER_SECTION = 14

    def __init__(
        self, points=None, name="Starter Ring", kerb_points=None, features=None,
        geometry="spline", declared_length_m=None, road_width_m=ROAD_W,
        road_widths_m=None, pitlane_points=None,
    ):
        self.points = [Vector2(p) for p in (points or default_track())]
        self.name = name
        self.geometry = geometry
        self.declared_length_m = declared_length_m
        base_width = float(road_width_m or ROAD_W)
        supplied_widths = list(road_widths_m or [])
        self.road_widths_m = [
            clamp(
                float(supplied_widths[i]) if i < len(supplied_widths) else base_width,
                6.0, 24.0,
            )
            for i in range(len(self.points))
        ]
        self.road_width_m = (
            sum(self.road_widths_m) / len(self.road_widths_m)
            if self.road_widths_m else base_width
        )
        self.variable_width = (
            max(self.road_widths_m) - min(self.road_widths_m) > 0.01
        )
        self.kerb_width_m = self.road_width_m + 2.0
        self.samples_per_section = 1 if geometry == "sampled" else self.SAMPLES_PER_SECTION
        self.kerb_points = set(kerb_points) if kerb_points is not None else self._automatic_kerbs()
        default_sectors = [
            len(self.points) // 4, len(self.points) // 2, len(self.points) * 3 // 4,
        ] if len(self.points) >= 8 else []
        base_features = {
            "start_finish": 0,
            "sectors": default_sectors,
            "pit_entry": None,
            "pit_exit": None,
            "pit_boxes": [],
            "border_margin": BORDER_W,
        }
        self.features = {**base_features, **(features or {})}
        self.pitlane_points = [
            Vector2(point) for point in (pitlane_points or [])
        ]
        (
            self.centerline,
            self.kerb_segments,
            self.centerline_widths_m,
        ) = self._build_spline()
        self.pitlane_centerline = self._build_pitlane()
        self._pit_box_positions_cache = None
        self._ribbon_cache = {}
        self.segment_lengths_m = [
            self.centerline[i].distance_to(self.centerline[(i + 1) % len(self.centerline)])
            for i in range(len(self.centerline))
        ]
        self.cumulative_lengths_m = [0.0]
        for segment_length in self.segment_lengths_m:
            self.cumulative_lengths_m.append(
                self.cumulative_lengths_m[-1] + segment_length
            )
        self.measured_length_m = self.cumulative_lengths_m[-1]
        self.lap_length_m = float(declared_length_m or self.measured_length_m)
        self._build_spatial_index()

    def _build_spatial_index(self):
        self.spatial_cell_m = 50.0
        self.spatial_segments = {}
        for index, start in enumerate(self.centerline):
            end = self.centerline[(index + 1) % len(self.centerline)]
            minimum_x = math.floor((min(start.x, end.x) - BORDER_W) / self.spatial_cell_m)
            maximum_x = math.floor((max(start.x, end.x) + BORDER_W) / self.spatial_cell_m)
            minimum_y = math.floor((min(start.y, end.y) - BORDER_W) / self.spatial_cell_m)
            maximum_y = math.floor((max(start.y, end.y) + BORDER_W) / self.spatial_cell_m)
            for cell_x in range(minimum_x, maximum_x + 1):
                for cell_y in range(minimum_y, maximum_y + 1):
                    self.spatial_segments.setdefault((cell_x, cell_y), set()).add(index)

    def _turn_angle(self, index):
        if len(self.points) < 3:
            return 0
        point = self.points[index]
        incoming = self.points[index - 1] - point
        outgoing = self.points[(index + 1) % len(self.points)] - point
        if not incoming.length() or not outgoing.length():
            return 0
        dot = clamp(incoming.normalize().dot(outgoing.normalize()), -1, 1)
        return 180 - math.degrees(math.acos(dot))

    def _automatic_kerbs(self):
        # Kerbs belong at meaningful corners, never along every straight.
        return {i for i in range(len(self.points)) if self._turn_angle(i) >= 36}

    @staticmethod
    def _catmull(p0, p1, p2, p3, t):
        t2, t3 = t * t, t * t * t
        return 0.5 * (
            2 * p1
            + (-p0 + p2) * t
            + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
            + (-p0 + 3 * p1 - 3 * p2 + p3) * t3
        )

    def _build_spline(self):
        count = len(self.points)
        if self.geometry == "sampled":
            kerbs = [i in self.kerb_points for i in range(count)]
            return (
                [point.copy() for point in self.points],
                kerbs,
                list(self.road_widths_m),
            )
        if count < 4:
            return (
                [p.copy() for p in self.points],
                [False] * count,
                list(self.road_widths_m),
            )
        curve, kerbs, widths = [], [], []
        for i in range(count):
            p0 = self.points[(i - 1) % count]
            p1 = self.points[i]
            p2 = self.points[(i + 1) % count]
            p3 = self.points[(i + 2) % count]
            for sample in range(self.SAMPLES_PER_SECTION):
                t = sample / self.SAMPLES_PER_SECTION
                curve.append(self._catmull(p0, p1, p2, p3, t))
                near_corner = (i in self.kerb_points and t < .30) or (
                    (i + 1) % count in self.kerb_points and t > .70
                )
                kerbs.append(near_corner)
                widths.append(
                    self.road_widths_m[i]
                    + (
                        self.road_widths_m[(i + 1) % count]
                        - self.road_widths_m[i]
                    ) * t
                )
        return curve, kerbs, widths

    def set_kerb_points(self, indexes):
        self.kerb_points = set(indexes)
        (
            self.centerline,
            self.kerb_segments,
            self.centerline_widths_m,
        ) = self._build_spline()

    def _pitlane_edge_anchor(self, control_index, toward, is_entry):
        """Return the local asphalt edge facing an authored pit-road node."""
        spline_index = (
            int(control_index) * self.samples_per_section
        ) % len(self.centerline)
        centre = self.centerline[spline_index]
        if is_entry:
            neighbour = self.centerline[
                (spline_index + 1) % len(self.centerline)
            ]
            tangent = neighbour - centre
        else:
            neighbour = self.centerline[spline_index - 1]
            tangent = centre - neighbour
        if not tangent.length_squared():
            tangent = Vector2(1, 0)
        else:
            tangent = tangent.normalize()
        normal = Vector2(-tangent.y, tangent.x)
        toward_edge = Vector2(toward) - centre
        side = 1.0 if toward_edge.dot(normal) >= 0.0 else -1.0
        half_width = self.centerline_widths_m[spline_index] / 2
        return centre + normal * side * half_width

    def _build_pitlane(self):
        """Connect authored pit nodes between the main road's asphalt edges."""
        entry = self.features.get("pit_entry")
        exit_index = self.features.get("pit_exit")
        if (
            not self.pitlane_points
            or entry is None
            or exit_index is None
            or not self.points
        ):
            return []
        entry = int(entry)
        exit_index = int(exit_index)
        if not (
            0 <= entry < len(self.points)
            and 0 <= exit_index < len(self.points)
        ):
            return []
        entry_anchor = self._pitlane_edge_anchor(
            entry, self.pitlane_points[0], True
        )
        exit_anchor = self._pitlane_edge_anchor(
            exit_index, self.pitlane_points[-1], False
        )
        return [
            entry_anchor,
            *(point.copy() for point in self.pitlane_points),
            exit_anchor,
        ]

    def pitlane_nearest(self, point):
        """Return distance and position on the open pitlane polyline."""
        point = Vector2(point)
        best = (float("inf"), Vector2(), None, 0.0)
        for index in range(max(0, len(self.pitlane_centerline) - 1)):
            distance, nearest, ratio = seg_distance(
                point,
                self.pitlane_centerline[index],
                self.pitlane_centerline[index + 1],
            )
            if distance < best[0]:
                best = (distance, nearest, index, ratio)
        return best

    def is_in_pitlane(self, point):
        return (
            len(self.pitlane_centerline) >= 2
            and self.pitlane_nearest(point)[0] <= PITLANE_WIDTH_M / 2
        )

    def pit_box_positions(self):
        """Resolve pit boxes on new pitlanes or legacy main-track saves."""
        if self._pit_box_positions_cache is not None:
            return self._pit_box_positions_cache
        boxes = self.features.get("pit_boxes", [])
        if self.pitlane_points:
            positions = [
                self.pitlane_points[int(index)].copy()
                for index in boxes
                if 0 <= int(index) < len(self.pitlane_points)
            ]
        else:
            # Compatibility for tracks saved before dedicated pit roads existed.
            positions = [
                self.points[int(index) % len(self.points)].copy()
                for index in boxes
                if self.points
            ]
        self._pit_box_positions_cache = tuple(positions)
        return self._pit_box_positions_cache

    def nearest(self, point):
        best = (1e9, Vector2(), 0, 0)
        point_vector = Vector2(point)
        cell = (
            math.floor(point_vector.x / self.spatial_cell_m),
            math.floor(point_vector.y / self.spatial_cell_m),
        )
        candidates = set()
        for offset_x in (-1, 0, 1):
            for offset_y in (-1, 0, 1):
                candidates.update(self.spatial_segments.get((cell[0] + offset_x, cell[1] + offset_y), ()))
        if not candidates:
            candidates = range(len(self.centerline))
        for i in candidates:
            a = self.centerline[i]
            b = self.centerline[(i + 1) % len(self.centerline)]
            d, near, t = seg_distance(point_vector, a, b)
            if d < best[0]:
                best = (d, near, i, t)
        return best

    def progress(self, point):
        _, _, index, t = self.nearest(point)
        return index + t

    def progress_metres(self, point):
        """Return continuous distance around the racing line in world metres."""
        _, _, index, t = self.nearest(point)
        return self.cumulative_lengths_m[index] + self.segment_lengths_m[index] * t

    def tangent(self, point):
        _, _, index, _ = self.nearest(point)
        delta = self.centerline[(index + 1) % len(self.centerline)] - self.centerline[index]
        return delta.normalize() if delta.length() else Vector2(1, 0)

    def width_at_segment(self, index, t=0.0):
        current = self.centerline_widths_m[index]
        following = self.centerline_widths_m[
            (index + 1) % len(self.centerline_widths_m)
        ]
        return current + (following - current) * t

    def point_at_distance(self, distance_m):
        """Interpolate a centreline position at a wrapped lap distance."""
        if not self.centerline:
            return Vector2()
        if self.measured_length_m <= 1e-9:
            return self.centerline[0].copy()
        wrapped = distance_m % self.measured_length_m
        index = min(
            bisect_right(self.cumulative_lengths_m, wrapped) - 1,
            len(self.centerline) - 1,
        )
        segment_length = self.segment_lengths_m[index]
        ratio = (
            (wrapped - self.cumulative_lengths_m[index]) / segment_length
            if segment_length > 1e-9 else 0.0
        )
        return self.centerline[index].lerp(
            self.centerline[(index + 1) % len(self.centerline)],
            ratio,
        )

    def ribbon_edges(self, widths):
        """Build gap-free mitered left/right edges for a closed-width ribbon."""
        cache_key = tuple(round(float(width), 6) for width in widths)
        cached = self._ribbon_cache.get(cache_key)
        if cached is not None:
            return cached
        count = len(self.centerline)
        if count < 2:
            return [], []
        left, right = [], []
        for index, point in enumerate(self.centerline):
            previous = self.centerline[index - 1]
            following = self.centerline[(index + 1) % count]
            incoming = point - previous
            outgoing = following - point
            if incoming.length_squared() <= 1e-12:
                incoming = outgoing.copy()
            if outgoing.length_squared() <= 1e-12:
                outgoing = incoming.copy()
            incoming = (
                incoming.normalize()
                if incoming.length_squared() else Vector2(1, 0)
            )
            outgoing = (
                outgoing.normalize()
                if outgoing.length_squared() else incoming
            )
            incoming_normal = Vector2(-incoming.y, incoming.x)
            outgoing_normal = Vector2(-outgoing.y, outgoing.x)
            joined = incoming_normal + outgoing_normal
            if joined.length_squared() <= 1e-8:
                joined = outgoing_normal
            else:
                joined = joined.normalize()
            half_width = max(0.5, float(widths[index]) / 2)
            projection = abs(joined.dot(outgoing_normal))
            miter_length = min(
                half_width / max(projection, 0.25),
                half_width * 2.5,
            )
            offset = joined * miter_length
            left.append(point + offset)
            right.append(point - offset)
        self._ribbon_cache[cache_key] = (left, right)
        return left, right

    @staticmethod
    def _screen_points(points, offset, scale):
        return [point * scale + offset for point in points]

    def visible_segments(self, screen, offset, scale):
        """Return track segments that can intersect the current clip area."""
        if scale <= 0:
            return tuple(range(len(self.centerline)))
        clip = screen.get_clip()
        world_left = (clip.left - offset.x) / scale
        world_right = (clip.right - offset.x) / scale
        world_top = (clip.top - offset.y) / scale
        world_bottom = (clip.bottom - offset.y) / scale
        minimum_x = math.floor(
            min(world_left, world_right) / self.spatial_cell_m
        ) - 1
        maximum_x = math.floor(
            max(world_left, world_right) / self.spatial_cell_m
        ) + 1
        minimum_y = math.floor(
            min(world_top, world_bottom) / self.spatial_cell_m
        ) - 1
        maximum_y = math.floor(
            max(world_top, world_bottom) / self.spatial_cell_m
        ) + 1
        visible = set()
        for cell_x in range(minimum_x, maximum_x + 1):
            for cell_y in range(minimum_y, maximum_y + 1):
                visible.update(
                    self.spatial_segments.get((cell_x, cell_y), ())
                )
        return tuple(sorted(visible))

    def surface(self, point):
        if self.is_in_pitlane(point):
            return "pitlane"
        distance, _, index, t = self.nearest(point)
        local_width = self.width_at_segment(index, t)
        if distance <= local_width / 2:
            return "asphalt"
        if distance <= (local_width + 2.0) / 2 and self.kerb_segments[index]:
            return "kerb"
        if distance <= float(self.features["border_margin"]) / 2:
            return "grass"
        return "wall"

    def kerb_color_at_segment(self, index):
        """Alternate red and white kerb bands every 100 world metres."""
        distance_m = self.cumulative_lengths_m[
            int(index) % len(self.centerline)
        ]
        band = int(distance_m // KERB_COLOR_INTERVAL_M)
        return RED if band % 2 == 0 else WHITE

    def draw(self, screen, offset=Vector2(), scale=1.0):
        pts = self._screen_points(self.centerline, offset, scale)
        if len(pts) < 2:
            return
        visible = self.visible_segments(screen, offset, scale)

        # Every visual layer is derived from the same mitered ribbon vertices.
        # This removes the triangular gaps produced by independent thick-line
        # caps at sharp or variable-width nodes.
        border_widths = [
            max(width + 4.0, float(self.features["border_margin"]))
            for width in self.centerline_widths_m
        ]
        border_left, border_right = self.ribbon_edges(border_widths)
        border_left_px = self._screen_points(border_left, offset, scale)
        border_right_px = self._screen_points(border_right, offset, scale)
        for index in visible:
            following = (index + 1) % len(pts)
            pygame.draw.polygon(
                screen, (17, 43, 28),
                (
                    border_left_px[index],
                    border_left_px[following],
                    border_right_px[following],
                    border_right_px[index],
                ),
            )

        road_left, road_right = self.ribbon_edges(
            self.centerline_widths_m
        )
        kerb_left, kerb_right = self.ribbon_edges(
            [width + 2.0 for width in self.centerline_widths_m]
        )
        road_left_px = self._screen_points(road_left, offset, scale)
        road_right_px = self._screen_points(road_right, offset, scale)
        kerb_left_px = self._screen_points(kerb_left, offset, scale)
        kerb_right_px = self._screen_points(kerb_right, offset, scale)
        for i in visible:
            if self.kerb_segments[i]:
                following = (i + 1) % len(pts)
                color = self.kerb_color_at_segment(i)
                pygame.draw.polygon(
                    screen, color,
                    (
                        road_left_px[i],
                        road_left_px[following],
                        kerb_left_px[following],
                        kerb_left_px[i],
                    ),
                )
                pygame.draw.polygon(
                    screen, color,
                    (
                        road_right_px[i],
                        road_right_px[following],
                        kerb_right_px[following],
                        kerb_right_px[i],
                    ),
                )

        for index in visible:
            following = (index + 1) % len(pts)
            pygame.draw.polygon(
                screen, ROAD,
                (
                    road_left_px[index],
                    road_left_px[following],
                    road_right_px[following],
                    road_right_px[index],
                ),
            )
        if len(self.pitlane_centerline) >= 2:
            pit_points = self._screen_points(
                self.pitlane_centerline, offset, scale
            )
            border_width = max(
                2, int(round((PITLANE_WIDTH_M + 2.0) * scale))
            )
            road_width = max(
                1, int(round(PITLANE_WIDTH_M * scale))
            )
            pygame.draw.lines(
                screen, (17, 43, 28), False, pit_points, border_width
            )
            pygame.draw.lines(
                screen, (67, 70, 74), False, pit_points, road_width
            )
            joint_radius = max(1, road_width // 2)
            for pit_point in pit_points[1:-1]:
                pygame.draw.circle(
                    screen, (67, 70, 74), pit_point, joint_radius
                )
            pygame.draw.aalines(
                screen, (160, 164, 164), False, pit_points
            )
        edge_color = (112, 116, 117)
        if len(visible) < len(pts) * 0.65:
            for index in visible:
                following = (index + 1) % len(pts)
                pygame.draw.aaline(
                    screen, edge_color,
                    road_left_px[index], road_left_px[following],
                )
                pygame.draw.aaline(
                    screen, edge_color,
                    road_right_px[index], road_right_px[following],
                )
        else:
            pygame.draw.aalines(screen, edge_color, True, road_left_px)
            pygame.draw.aalines(screen, edge_color, True, road_right_px)
        # One start/finish line; editor control points are not physical seams.
        if len(self.centerline) > 1:
            start_index = int(self.features.get("start_finish", 0)) % len(self.points)
            spline_index = start_index * self.samples_per_section
            p = self.centerline[spline_index] * scale + offset
            q = self.centerline[(spline_index + 1) % len(self.centerline)] * scale + offset
            tangent = (q - p).normalize()
            normal = Vector2(-tangent.y, tangent.x)
            local_width = self.centerline_widths_m[spline_index]
            pygame.draw.line(screen, YELLOW, p - normal * local_width * scale / 2, p + normal * local_width * scale / 2, max(1, int(scale)))
        for control_index in self.features.get("sectors", []):
            if not self.points:
                continue
            spline_index = (int(control_index) % len(self.points)) * self.samples_per_section
            p = self.centerline[spline_index] * scale + offset
            q = self.centerline[(spline_index + 1) % len(self.centerline)] * scale + offset
            tangent = (q - p).normalize()
            normal = Vector2(-tangent.y, tangent.x)
            local_width = self.centerline_widths_m[spline_index]
            pygame.draw.line(screen, (77, 171, 247), p - normal * local_width * scale / 2, p + normal * local_width * scale / 2, max(1, int(scale)))
        feature_colors = {
            "pit_entry": (249, 115, 22),
            "pit_exit": (34, 197, 94),
        }
        for feature, color in feature_colors.items():
            index = self.features.get(feature)
            if index is not None and self.points:
                position = self.points[int(index) % len(self.points)] * scale + offset
                pygame.draw.circle(screen, color, position, max(3, int(2.2 * scale)), max(1, int(.6 * scale)))
        for position in self.pit_box_positions():
            pygame.draw.rect(
                screen, (168, 85, 247),
                pygame.Rect(
                    0, 0, max(4, int(3 * scale)), max(4, int(3 * scale))
                ).move(
                    position * scale + offset
                    - Vector2(max(2, int(1.5 * scale))),
                ),
                max(1, int(.5 * scale)),
            )

    def draw_preview(self, screen, rect):
        """Draw a fitted, decorative circuit map without affecting world physics."""
        rect = pygame.Rect(rect)
        if len(self.points) < 2:
            return
        bounds_points = self.centerline + self.pitlane_centerline
        xs = [p.x for p in bounds_points]
        ys = [p.y for p in bounds_points]
        source_w, source_h = max(xs) - min(xs), max(ys) - min(ys)
        scale = min((rect.width - 70) / max(source_w, 1), (rect.height - 70) / max(source_h, 1))
        centre = Vector2((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2)
        target = Vector2(rect.center)
        pts = [target + (p - centre) * scale for p in self.centerline]
        pygame.draw.lines(screen, (18, 38, 32), True, pts, 32)
        for i, enabled in enumerate(self.kerb_segments):
            if enabled:
                pygame.draw.line(
                    screen, self.kerb_color_at_segment(i),
                    pts[i], pts[(i + 1) % len(pts)], 22,
                )
        if not self.variable_width:
            pygame.draw.lines(screen, (69, 74, 78), True, pts, 12)
        else:
            for i, point in enumerate(pts):
                following = (i + 1) % len(pts)
                relative_width = (
                    self.centerline_widths_m[i]
                    + self.centerline_widths_m[following]
                ) / max(self.road_width_m * 2, 0.1)
                pygame.draw.line(
                    screen, (69, 74, 78), point, pts[following],
                    max(5, int(12 * relative_width)),
                )
        pygame.draw.circle(screen, YELLOW, pts[0], 6)
        if len(self.pitlane_centerline) >= 2:
            pit_points = [
                target + (point - centre) * scale
                for point in self.pitlane_centerline
            ]
            pygame.draw.lines(
                screen, (18, 38, 32), False, pit_points, 13
            )
            pygame.draw.lines(
                screen, (104, 108, 111), False, pit_points, 7
            )

    def save(self, name):
        TRACK_DIR.mkdir(parents=True, exist_ok=True)
        safe = safe_component_name(name, "Custom Circuit")
        path = TRACK_DIR / component_filename(safe, ".json")
        path.write_text(json.dumps({
            "name": safe,
            "points": [list(p) for p in self.points],
            "kerb_points": sorted(self.kerb_points),
            "features": self.features,
            "geometry": self.geometry,
            "declared_length_m": self.lap_length_m,
            "road_width_m": self.road_width_m,
            "road_widths_m": self.road_widths_m,
            "pitlane_points": [list(point) for point in self.pitlane_points],
        }, indent=2))
        self.name = safe
        return path

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text())
        return cls(
            data["points"],
            data.get("name", Path(path).stem),
            data.get("kerb_points"),
            data.get("features"),
            data.get("geometry", "spline"),
            data.get("declared_length_m"),
            data.get("road_width_m", ROAD_W),
            data.get("road_widths_m"),
            data.get("pitlane_points"),
        )


class Brain:
    """Dependency-free driving and pit-strategy policy."""
    INPUT_NAMES = (
        "far_left", "left", "forward", "right", "far_right",
        "heading_error", "speed", "dirty_tyres",
        "tyre_wear", "tyre_age", "fuel", "fuel_kg", "health",
        "puncture", "rain", "slipstream", "lap", "lap_progress", "pitstops",
        "pit_available", "tyre_soft", "tyre_medium", "tyre_hard", "tyre_wet",
        "battery", "battery_percent", "regen", "is_hybrid",
        "overtake_active", "off_track", "car_collision",
        "understeer", "oversteer", "racing_line_offset",
        "car_ahead", "car_ahead_distance", "car_ahead_side",
        "closing_speed", "passing", "passing_side",
        "local_velocity_forward", "local_velocity_lateral",
        "angular_velocity", "traction", "tire_slip",
        "rpm", "rpm_value", "gear", "gear_number", "speed_kph",
        "ray_left_90", "ray_left_18", "ray_right_18", "ray_right_90",
        "waypoint_5_forward", "waypoint_5_right",
        "waypoint_10_forward", "waypoint_10_right",
        "waypoint_20_forward", "waypoint_20_right",
        "waypoint_40_forward", "waypoint_40_right",
        "opponent_1_forward", "opponent_1_right",
        "opponent_1_velocity_forward", "opponent_1_velocity_right",
        "opponent_2_forward", "opponent_2_right",
        "opponent_2_velocity_forward", "opponent_2_velocity_right",
        "opponent_3_forward", "opponent_3_right",
        "opponent_3_velocity_forward", "opponent_3_velocity_right",
        "opponent_1_present", "opponent_2_present", "opponent_3_present",
        "previous_steering", "previous_throttle", "previous_brake",
        "corner_curvature_10", "corner_curvature_20",
        "corner_curvature_40",
        "race_position", "field_size", "position_deficit",
        "gap_to_leader_m", "gap_to_next_m",
        "race_aggression", "aggression_error",
    )
    DRIVING_INPUTS = 8

    def __init__(self, weights=None, config=None, program=None, parameters=None, source=None):
        self.config = {
            "steering": 1.0, "aggression": .82, "braking": .55,
            "recovery": .8, "mutation": .22, **(config or {}),
        }
        self.program = program
        self.source = source or (program.source if program else None)
        self.parameters = dict(parameters or (program.defaults() if program else {}))
        self.weights = None if program else (weights or self._user_weights())

    def _user_weights(self):
        """Translate the user's controller design into its initial policy."""
        turn = self.config["steering"]
        recovery = self.config["recovery"]
        return [
            [-.40 * turn, -1.0 * turn, 0, 1.0 * turn, .40 * turn,
             .75 * recovery, 0, 0, 0],
            [0, 0, 1.0, 0, 0, 0, -1.0, -.4, 0],
        ]

    def think(self, inputs):
        state = dict(zip(self.INPUT_NAMES, inputs))
        if self.program:
            try:
                return self.program.run(state, self.parameters)
            except (AlgorithmError, ArithmeticError, KeyError, TypeError, ValueError):
                # A bad runtime calculation stops this agent safely, never the game.
                return 0.0, 0.0, 0.0, 0.0, 0.0, 1
        outputs = []
        for row in self.weights:
            outputs.append(math.tanh(sum(a * b for a, b in zip(row[:-1], inputs)) + row[-1]))
        steer = clamp(outputs[0], -1, 1)
        target_speed = (
            inputs[2] * (0.9 + self.config["aggression"] * 0.25)
            - abs(steer) * 0.25
        )
        throttle = clamp(
            (target_speed - inputs[6]) * (0.4 + self.config["braking"] * 0.5),
            0, 1,
        )
        brake = 0.0
        throttle *= 1.0 - inputs[7] * 0.5
        # The fallback neural policy needs a deterministic last-resort response
        # before evolution has tuned its weights. Coded user policies remain
        # completely responsible for their own emergency behavior.
        if inputs[2] < 0.25:
            open_side = (
                inputs[3] - inputs[1] + (inputs[4] - inputs[0]) * 0.4
            )
            if abs(open_side) > 0.01:
                steer = ((open_side > 0) - (open_side < 0)) * 0.9
            throttle = 0.05
            brake = 0.35 if state.get("is_hybrid", 0) >= 0.5 else 0.0
        pit_request = (
            state.get("pit_available", 0) >= 0.5
            and (
                state.get("tyre_wear", 0) >= 0.65
                or state.get("puncture", 0) >= 0.5
            )
        )
        pit_tyre = 3 if state.get("rain", 0) >= 0.45 else 1
        overtake = (
            state.get("is_hybrid", 0) >= 0.5
            and state.get("battery", 0) >= 0.18
            and state.get("forward", 0) >= 0.72
            and abs(steer) <= 0.20
            and brake <= 0.05
        )
        return (
            steer, throttle, brake, float(overtake),
            float(pit_request), pit_tyre,
        )

    def mutate(self, amount=None):
        amount = self.config["mutation"] if amount is None else amount
        if self.program:
            return Brain(
                config=self.config.copy(),
                program=self.program,
                parameters=self.program.mutate(self.parameters, amount),
                source=self.source,
            )
        values = [[w + random.gauss(0, amount) if random.random() < 0.35 else w for w in row] for row in self.weights]
        return Brain(values, self.config.copy())

    def save(self, name="champion"):
        BRAIN_DIR.mkdir(parents=True, exist_ok=True)
        safe = safe_component_name(name, "Champion")
        path = BRAIN_DIR / component_filename(safe, ".json")
        path.write_text(json.dumps({
            "version": 3,
            "name": safe,
            "weights": self.weights,
            "algorithm": self.config,
            "source": self.source,
            "parameters": self.parameters,
        }, indent=2))
        return path

    @staticmethod
    def migrate_legacy_source(source):
        """Repair the exact reversed-sign bug from the bundled legacy template."""
        legacy_marker = "# Formula AI Controller"
        legacy_balance = "open_side = left_space - right_space"
        legacy_steering = (
            "raw_steering = sign(open_side) * steer_strength "
            "- (heading_error * recovery_gain)"
        )
        legacy_tuning = "steer_strength = parameter(0.85, 0.20, 1.80)"
        if (
            legacy_marker in source
            and (
                legacy_balance in source
                or legacy_steering in source
                or legacy_tuning in source
            )
        ):
            migrated = (
                source.replace(
                    legacy_balance,
                    "open_side = right_space - left_space",
                )
                .replace(
                    legacy_steering,
                    "raw_steering = (open_side * steer_strength) "
                    "+ (heading_error * recovery_gain)",
                )
                .replace(
                    "steer_strength = parameter(0.85, 0.20, 1.80)",
                    "steer_strength = parameter(1.00, 0.20, 1.80)",
                )
                .replace(
                    "recovery_gain  = parameter(0.75, 0.10, 1.50)",
                    "recovery_gain  = parameter(0.60, 0.10, 1.50)",
                )
                .replace(
                    "power          = parameter(1.20, 0.30, 2.00)",
                    "power          = parameter(1.10, 0.30, 2.00)",
                )
                .replace(
                    "braking        = parameter(0.60, 0.05, 1.50)",
                    "braking        = parameter(0.80, 0.05, 1.50)",
                )
            )
            if "pit_request =" not in migrated:
                migrated += (
                    "\n# Controller-owned pit strategy (legacy template migration)\n"
                    "pit_request = 0.0\n"
                    "pit_tyre = 1.0\n"
                    "if pit_available > 0.5 and "
                    "(tyre_wear >= 0.68 or puncture > 0.5):\n"
                    "    pit_request = 1.0\n"
                    "if rain >= 0.45:\n"
                    "    pit_tyre = 3.0\n"
                )
            return migrated
        return source

    @classmethod
    def load_file(cls, path):
        try:
            data = json.loads(Path(path).read_text())
            if data.get("source"):
                stored_source = data["source"]
                source = cls.migrate_legacy_source(stored_source)
                program = SafeAlgorithm(source)
                return cls(
                    config=data.get("algorithm"),
                    program=program,
                    # Values selected by the legacy, broken fitness function
                    # are not meaningful after correcting its steering signs.
                    parameters=(
                        program.defaults()
                        if source != stored_source
                        else data.get("parameters")
                    ),
                    source=source,
                )
            return cls(data["weights"], data.get("algorithm"))
        except (AlgorithmError, KeyError, ValueError, OSError):
            return cls()

    @classmethod
    def load_best(cls):
        files = sorted(
            BRAIN_DIR.glob("*.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        return cls.load_file(files[0]) if files else cls()


@dataclass
class Car:
    position: Vector2
    angle: float
    color: tuple
    brain: Brain
    name: str = "AI"
    generation: str = "ICE"
    tyre: str = "Medium"
    fuel: float = 50.0
    velocity: Vector2 = field(default_factory=Vector2)
    health: float = 100.0
    tyre_wear: float = 0.0
    tyre_laps: int = 0
    dirty: float = 0.0
    lap: int = 0
    checkpoint: int = 0
    score: float = 0.0
    fitness: float = 0.0
    forward_distance_m: float = 0.0
    reverse_distance_m: float = 0.0
    control_penalty: float = 0.0
    collision_penalty: float = 0.0
    collision_count: int = 0
    overtake_reward: float = 0.0
    overtakes: int = 0
    off_track_frames: float = 0.0
    stagnant_frames: float = 0.0
    alive: bool = True
    track_limits: int = 0
    puncture: bool = False
    pitstops: int = 0
    finish_time: float | None = None
    previous_progress: float = 0.0
    previous_progress_m: float | None = None
    outside_limits: bool = False
    pit_timer: float = 0.0
    pit_requested: bool = False
    requested_tyre: int = 1
    slipstream: float = 0.0
    drafting_car: str = ""
    brain_name: str = "CURRENT SESSION"
    team: str = ""
    pit_box_index: int | None = None
    in_pitlane: bool = False
    battery: float = 100.0
    battery_regen: float = 0.0
    overtake_active: bool = False
    brake_input: float = 0.0
    car_collision: bool = False
    understeer: float = 0.0
    oversteer: float = 0.0
    car_ahead: float = 0.0
    car_ahead_distance: float = 1.0
    car_ahead_side: float = 0.0
    closing_speed: float = 0.0
    passing: bool = False
    passing_side: float = 0.0
    gear: int = 1
    rpm: float = IDLE_ENGINE_RPM
    angular_velocity: float = 0.0
    traction: float = 1.0
    tire_slip: float = 0.0
    steering_input: float = 0.0
    throttle_input: float = 0.0
    previous_steering: float = 0.0
    previous_throttle: float = 0.0
    previous_brake: float = 0.0
    race_position: int = 1
    field_size: int = 1
    position_deficit: float = 0.0
    gap_to_leader_m: float = 0.0
    gap_to_next_m: float = 0.0
    race_aggression: float = 0.0
    aggression_error: float = 0.0
    aggression_mistake_frames: float = field(
        default=0.0, repr=False, compare=False
    )
    aggression_mistake_cooldown: float = field(
        default=0.0, repr=False, compare=False
    )
    opponent_data: tuple = field(
        default_factory=lambda: (0.0,) * 12,
        repr=False,
        compare=False,
    )
    opponent_presence: tuple = field(
        default_factory=lambda: (0.0,) * 3,
        repr=False,
        compare=False,
    )
    raycast_cache: tuple | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    sensor_frame: int = field(default=0, repr=False, compare=False)
    sensor_phase: int = field(default=-1, repr=False, compare=False)
    track_segment: int = field(default=0, repr=False, compare=False)
    overtake_candidates: set = field(
        default_factory=set,
        repr=False,
        compare=False,
    )
    overtake_cooldowns: dict = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )
    passing_target: object = field(default=None, repr=False, compare=False)

    @property
    def speed_kph(self):
        return self.velocity.length() * FPS * 3.6

    def update_drivetrain(self, throttle=0.0):
        """Update the automatic eight-speed gearbox and 13,000 RPM engine."""
        heading = Vector2(1, 0).rotate(self.angle)
        forward_speed = max(0.0, self.velocity.dot(heading))
        speed_ratio = clamp(
            forward_speed / REFERENCE_TOP_SPEED, 0.0, 1.0
        )
        self.gear = next(
            (
                index
                for index, limit in enumerate(GEAR_SPEED_LIMITS, start=1)
                if speed_ratio <= limit
            ),
            GEAR_COUNT,
        )
        gear_low = (
            0.0 if self.gear == 1
            else GEAR_SPEED_LIMITS[self.gear - 2]
        )
        gear_high = GEAR_SPEED_LIMITS[self.gear - 1]
        within_gear = clamp(
            (speed_ratio - gear_low) / max(gear_high - gear_low, 1e-9),
            0.0, 1.0,
        )
        if forward_speed < 0.025:
            self.rpm = IDLE_ENGINE_RPM + clamp(
                throttle, 0.0, 1.0
            ) * 2500.0
        else:
            self.rpm = (
                IDLE_ENGINE_RPM
                + within_gear * (MAX_ENGINE_RPM - IDLE_ENGINE_RPM)
            )
        self.rpm = clamp(self.rpm, IDLE_ENGINE_RPM, MAX_ENGINE_RPM)

    def drivetrain_power_factor(self):
        """Return available wheel force for the current gear and engine RPM."""
        ratio_strength = GEAR_RATIOS[self.gear - 1] / GEAR_RATIOS[0]
        gear_force = 0.10 + ratio_strength * 0.90
        rpm_fraction = clamp(
            (self.rpm - IDLE_ENGINE_RPM)
            / (MAX_ENGINE_RPM - IDLE_ENGINE_RPM),
            0.0, 1.0,
        )
        # Power stays strong through the useful rev range, then falls
        # progressively over the final third of the gear.
        redline_zone = clamp((rpm_fraction - 0.67) / 0.33, 0.0, 1.0)
        redline_falloff = 1.0 - 0.20 * redline_zone ** 2
        return gear_force * redline_falloff

    def raycast_distances(self, track):
        """Return nine normalized boundary rays from left side to right side."""
        values = []
        for relative in (-90, -70, -35, -18, 0, 18, 35, 70, 90):
            direction = Vector2(1, 0).rotate(self.angle + relative)
            previous_distance = 0.0
            distance = 4.0
            while distance <= 120.0:
                if track.surface(self.position + direction * distance) not in (
                    "asphalt", "kerb", "pitlane",
                ):
                    # Refine the four-metre crossing so sensor precision stays
                    # close to the old two-metre marching implementation.
                    low, high = previous_distance, distance
                    for _ in range(2):
                        midpoint = (low + high) / 2
                        if track.surface(
                            self.position + direction * midpoint
                        ) in ("asphalt", "kerb", "pitlane"):
                            low = midpoint
                        else:
                            high = midpoint
                    distance = low
                    break
                previous_distance = distance
                distance += 4.0
            else:
                distance = 120.0
            values.append(distance / 120)
        return values

    def sensors(self, track, raycasts=None, tangent=None):
        raycasts = raycasts or self.raycast_distances(track)
        # Preserve the original five inputs and their ordering for existing
        # user algorithms. Four additional rays are appended later.
        values = [
            raycasts[1], raycasts[2], raycasts[4],
            raycasts[6], raycasts[7],
        ]
        tangent = tangent or track.tangent(self.position)
        heading = Vector2(1, 0).rotate(self.angle)
        signed = math.atan2(heading.cross(tangent), heading.dot(tangent)) / math.pi
        # Keep the original 1.7 normalization divisor so legacy brains retain
        # exactly the same throttle behavior.
        speed = clamp(self.velocity.length() / 1.7, 0.0, 1.0)
        return values + [signed, speed, self.dirty / 180]

    def controller_inputs(self, track, rain=0.0):
        """Build normalized driving and race-strategy data for user code."""
        tyre_flags = {
            "Soft": (1.0, 0.0, 0.0, 0.0),
            "Medium": (0.0, 1.0, 0.0, 0.0),
            "Hard": (0.0, 0.0, 1.0, 0.0),
            "Wet": (0.0, 0.0, 0.0, 1.0),
        }
        _, nearest_line, line_segment, line_ratio = track.nearest(
            self.position
        )
        self.track_segment = line_segment
        progress_m = (
            track.cumulative_lengths_m[line_segment]
            + track.segment_lengths_m[line_segment] * line_ratio
        )
        lap_progress = progress_m / max(track.measured_length_m, 1e-9)
        line_delta = (
            track.centerline[(line_segment + 1) % len(track.centerline)]
            - track.centerline[line_segment]
        )
        line_tangent = (
            line_delta.normalize()
            if line_delta.length_squared() else Vector2(1, 0)
        )
        line_normal = Vector2(-line_tangent.y, line_tangent.x)
        heading = Vector2(1, 0).rotate(self.angle)
        local_normal = Vector2(-heading.y, heading.x)
        racing_line_offset = clamp(
            (self.position - nearest_line).dot(line_normal)
            / max(track.width_at_segment(line_segment) / 2, 1.0),
            -1.0, 1.0,
        )
        local_velocity_forward = clamp(
            self.velocity.dot(heading) / REFERENCE_TOP_SPEED, -1.0, 1.0
        )
        local_velocity_lateral = clamp(
            self.velocity.dot(local_normal) / REFERENCE_TOP_SPEED,
            -1.0, 1.0,
        )
        if self.sensor_phase < 0:
            self.sensor_phase = sum(ord(character) for character in self.name) % (
                RAYCAST_UPDATE_FRAMES
            )
        if (
            self.raycast_cache is None
            or self.sensor_frame % RAYCAST_UPDATE_FRAMES == self.sensor_phase
        ):
            self.raycast_cache = tuple(self.raycast_distances(track))
        self.sensor_frame += 1
        raycasts = self.raycast_cache
        waypoint_values = []
        waypoint_points = []
        for lookahead in (5.0, 10.0, 20.0, 40.0):
            waypoint = track.point_at_distance(progress_m + lookahead)
            waypoint_points.append(waypoint)
            relative = waypoint - self.position
            waypoint_values.extend((
                clamp(relative.dot(heading) / lookahead, -1.0, 1.0),
                clamp(relative.dot(local_normal) / lookahead, -1.0, 1.0),
            ))
        # Signed centreline curvature, normalized so 1.0 represents roughly a
        # 20-metre radius. Unlike lateral waypoint displacement, these values
        # are independent of the car's chosen racing-line offset.
        curvature_values = []
        for waypoint, lookahead in zip(
            waypoint_points[1:],
            (10.0, 20.0, 40.0),
        ):
            chord = waypoint - nearest_line
            if chord.length_squared() > 1e-9:
                chord_direction = chord.normalize()
                angle_change = math.atan2(
                    line_tangent.cross(chord_direction),
                    line_tangent.dot(chord_direction),
                )
                curvature_values.append(
                    clamp(angle_change * 40.0 / lookahead, -1.0, 1.0)
                )
            else:
                curvature_values.append(0.0)
        return self.sensors(track, raycasts, line_tangent) + [
            clamp(self.tyre_wear / 100.0, 0.0, 1.0),
            float(self.tyre_laps),
            clamp(self.fuel / 110.0, 0.0, 1.0),
            self.fuel,
            clamp(self.health / 100.0, 0.0, 1.0),
            float(self.puncture),
            clamp(rain, 0.0, 1.0),
            clamp(self.slipstream, 0.0, 1.0),
            float(self.lap),
            lap_progress,
            float(self.pitstops),
            float(bool(track.pit_box_positions())),
            *tyre_flags.get(self.tyre, (0.0, 1.0, 0.0, 0.0)),
            (
                clamp(self.battery / 100.0, 0.0, 1.0)
                if self.generation == "Hybrid" else 0.0
            ),
            self.battery if self.generation == "Hybrid" else 0.0,
            (
                clamp(self.battery_regen / 0.05, 0.0, 1.0)
                if self.generation == "Hybrid" else 0.0
            ),
            float(self.generation == "Hybrid"),
            float(self.overtake_active),
            float(self.outside_limits),
            float(self.car_collision),
            clamp(self.understeer, 0.0, 1.0),
            clamp(self.oversteer, 0.0, 1.0),
            racing_line_offset,
            clamp(self.car_ahead, 0.0, 1.0),
            clamp(self.car_ahead_distance, 0.0, 1.0),
            clamp(self.car_ahead_side, -1.0, 1.0),
            clamp(self.closing_speed, -1.0, 1.0),
            float(self.passing),
            clamp(self.passing_side, -1.0, 1.0),
            local_velocity_forward,
            local_velocity_lateral,
            clamp(self.angular_velocity / 6.0, -1.0, 1.0),
            clamp(self.traction, 0.0, 1.0),
            clamp(self.tire_slip, 0.0, 1.0),
            clamp(self.rpm / MAX_ENGINE_RPM, 0.0, 1.0),
            self.rpm,
            self.gear / GEAR_COUNT,
            float(self.gear),
            self.speed_kph,
            raycasts[0],
            raycasts[3],
            raycasts[5],
            raycasts[8],
            *waypoint_values,
            *self.opponent_data,
            *self.opponent_presence,
            clamp(self.previous_steering, -1.0, 1.0),
            clamp(self.previous_throttle, 0.0, 1.0),
            clamp(self.previous_brake, 0.0, 1.0),
            *curvature_values,
            float(self.race_position),
            float(self.field_size),
            clamp(self.position_deficit, 0.0, 1.0),
            max(0.0, self.gap_to_leader_m),
            max(0.0, self.gap_to_next_m),
            clamp(self.race_aggression, 0.0, 1.0),
            clamp(self.aggression_error, -1.0, 1.0),
        ]

    def update(self, track, dt=1.0, rain=0.0, damage_enabled=True):
        if not self.alive:
            return
        if self.pit_timer > 0:
            self.pit_timer -= dt
            self.overtake_active = False
            self.brake_input = 0.0
            self.steering_input = 0.0
            self.throttle_input = 0.0
            self.battery_regen = 0.0
            self.velocity *= .72
            self.update_drivetrain(0.0)
            self.previous_steering = 0.0
            self.previous_throttle = 0.0
            self.previous_brake = 0.0
            if self.pit_timer <= 0:
                self.tyre_wear = 0
                self.tyre_laps = 0
                self.puncture = False
                self.dirty = 0
                self.tyre = ("Soft", "Medium", "Hard", "Wet")[
                    int(clamp(self.requested_tyre, 0, 3))
                ]
                self.pitstops += 1
                self.pit_requested = False
            return
        # Aggressive trailing drivers occasionally misjudge a line or braking
        # point. Mistakes are short pulses rather than frame-by-frame noise,
        # making them visible, consequential, and recoverable.
        if self.race_aggression <= 0.0:
            self.aggression_error = 0.0
            self.aggression_mistake_frames = 0.0
            self.aggression_mistake_cooldown = 0.0
        elif self.aggression_mistake_frames > 0.0:
            self.aggression_mistake_frames = max(
                0.0, self.aggression_mistake_frames - dt
            )
            if self.aggression_mistake_frames <= 0.0:
                self.aggression_mistake_cooldown = random.uniform(90.0, 240.0)
        else:
            self.aggression_error *= 0.84 ** dt
            self.aggression_mistake_cooldown = max(
                0.0, self.aggression_mistake_cooldown - dt
            )
            mistake_probability = (
                self.race_aggression ** 2 * 0.0022 * dt
            )
            if (
                self.aggression_mistake_cooldown <= 0.0
                and random.random() < mistake_probability
            ):
                mistake_sign = 1.0 if random.random() < 0.62 else -1.0
                self.aggression_error = (
                    mistake_sign * random.uniform(0.45, 1.0)
                )
                self.aggression_mistake_frames = random.uniform(20.0, 55.0)
        controller_inputs = self.controller_inputs(track, rain)
        # Contact is latched by collision resolution after physics, then
        # consumed exactly once by the following controller evaluation.
        self.car_collision = False
        (
            steer, throttle, brake, overtake_request,
            pit_request, pit_tyre,
        ) = self.brain.think(controller_inputs)
        steer = clamp(steer, -1.0, 1.0)
        throttle = clamp(throttle, 0.0, 1.0)
        previous_angle = self.angle
        self.pit_requested = pit_request >= 0.5
        self.requested_tyre = int(clamp(pit_tyre, 0, 3))
        self.brake_input = clamp(brake, 0.0, 1.0)
        self.steering_input = steer
        self.throttle_input = throttle
        sensor_values = controller_inputs[:Brain.DRIVING_INPUTS]
        heading_error = sensor_values[5]
        forward = Vector2(1, 0).rotate(self.angle)
        normal = Vector2(-forward.y, forward.x)
        wheel_surfaces = [
            track.surface(self.position + forward * longitudinal + normal * lateral)
            for longitudinal in (-CAR_WHEELBASE_M / 2, CAR_WHEELBASE_M / 2)
            for lateral in (-CAR_WIDTH_M / 2, CAR_WIDTH_M / 2)
        ]
        priority = {
            "asphalt": 0, "pitlane": 0, "kerb": 1,
            "grass": 2, "wall": 3,
        }
        surface = max(wheel_surfaces, key=lambda value: priority[value])
        grip = {
            "asphalt": 1.0, "pitlane": 1.0, "kerb": 0.78,
            "grass": 0.34, "wall": 0.15,
        }[surface]
        tyre_grip = {"Soft": 1.08, "Medium": 1.0, "Hard": 0.93, "Wet": 0.85}.get(self.tyre, 1)
        if self.tyre == "Wet":
            tyre_grip *= .72 + rain * .45
        else:
            tyre_grip *= 1.0 - rain * .68
        tyre_grip *= 1 - min(self.tyre_wear, 100) * 0.004
        if self.dirty > 0:
            tyre_grip *= 0.72
            self.dirty -= dt
        if self.puncture:
            tyre_grip *= 0.28
        # Roughly 100 m/s at 60 FPS, expressed as metres per simulation frame.
        draft = clamp(self.slipstream, 0.0, 1.0)
        speed = self.velocity.length()
        if self.fuel <= 0:
            throttle = 0.0
        is_hybrid = self.generation == "Hybrid"
        self.overtake_active = bool(
            is_hybrid
            and overtake_request >= 0.5
            and self.battery > 0.0
            and throttle > 0.0
            and self.brake_input < 0.05
        )
        deployment = 1.0 if self.overtake_active else 0.0
        # Hybrid output is an 80% combustion / 20% electric split. Scaling the
        # complete power unit by 1.1875 makes the 80% combustion portion equal
        # to the calibrated 95% wheel-power level, enough for 0–200 in 5.0 s.
        power_ratio = (
            HYBRID_TOTAL_POWER_SCALE
            * (
                HYBRID_COMBUSTION_SHARE
                + HYBRID_ELECTRIC_SHARE * deployment
            )
            if is_hybrid else 1.0
        )
        if is_hybrid:
            speed_factor = clamp(speed / 1.35, 0.0, 1.0)
            self.battery_regen = self.brake_input * speed_factor * 0.045 * dt
            battery_drain = deployment * throttle * 0.08 * dt
            self.battery = clamp(
                self.battery + self.battery_regen - battery_drain,
                0.0, 100.0,
            )
        else:
            self.battery = 0.0
            self.battery_regen = 0.0
            self.overtake_active = False
        hybrid_speed = 1.55 + 0.12 * deployment
        max_speed = (
            (hybrid_speed if is_hybrid else 1.67)
            * (1 - self.fuel * 0.0015)
            * (1.0 + draft * 0.075)
        )
        aero_grip = 1.0 - draft * 0.22
        front_grip = clamp(grip * tyre_grip * aero_grip, 0.05, 1.15)
        cornering_load = abs(steer) * speed
        understeer_target = clamp(
            (cornering_load - front_grip * 1.28) * 0.30
            + draft * abs(steer) * 0.04,
            0.0, 1.0,
        )
        oversteer_target = clamp(
            (
                cornering_load * throttle
                * max(0.0, 1.05 - grip * tyre_grip)
                - 0.16
            ) * 1.35
            + float(self.puncture) * 0.35,
            0.0, 1.0,
        )
        # Understeer now builds more gradually, clears faster when the front
        # tyres recover, and removes at most 18% of steering authority. It
        # remains useful telemetry without overwhelming the driving command.
        understeer_response = clamp(
            (0.14 if understeer_target > self.understeer else 0.26) * dt,
            0.0, 1.0,
        )
        handling_response = clamp(0.18 * dt, 0.0, 1.0)
        self.understeer += (
            understeer_target - self.understeer
        ) * understeer_response
        self.oversteer += (
            oversteer_target - self.oversteer
        ) * handling_response
        # Fast steering-rack response and stronger lateral tyre recovery make
        # the car rotate into an apex sooner, then settle cleanly on exit.
        self.angle += (
            steer * (3.25 + speed * 0.68) * grip * tyre_grip * aero_grip
            * (1.0 - self.understeer * 0.18 + self.oversteer * 0.20)
            * dt
        )
        self.update_drivetrain(throttle)
        drivetrain_power = self.drivetrain_power_factor()
        launch_traction = (
            LOW_SPEED_TRACTION
            + (1.0 - LOW_SPEED_TRACTION)
            * clamp(speed / FULL_TRACTION_SPEED, 0.0, 1.0)
        )
        # Compounds and conditions affect the traction-limited launch, but do
        # not multiply engine horsepower once the car is at speed.
        tyre_traction_factor = clamp(
            1.0 + (tyre_grip - 1.0) * 0.40, 0.30, 1.05
        )
        available_traction = clamp(
            launch_traction * tyre_traction_factor, 0.10, 1.0
        )
        self.velocity += (
            forward * throttle * (1.0 - self.brake_input)
            * BASE_ENGINE_ACCELERATION
            * drivetrain_power
            * power_ratio * (1.0 + draft * 0.08)
            * grip * available_traction * dt
        )
        forward_speed = max(0.0, self.velocity.dot(forward))
        braking_delta = min(
            forward_speed,
            self.brake_input * 0.028 * grip * tyre_grip * dt,
        )
        self.velocity -= forward * braking_delta

        # A closed or nearly closed throttle makes the driven wheels resist
        # rotation. Lower gears provide stronger engine braking.
        engine_brake_request = clamp(
            (0.25 - throttle) / 0.25, 0.0, 1.0
        )
        ratio_strength = GEAR_RATIOS[self.gear - 1] / GEAR_RATIOS[0]
        engine_braking_delta = min(
            max(0.0, self.velocity.dot(forward)),
            engine_brake_request
            * (0.0012 + 0.0028 * ratio_strength)
            * grip * dt,
        )
        self.velocity -= forward * engine_braking_delta

        lateral = self.velocity - forward * self.velocity.dot(forward)
        self.velocity -= lateral * clamp(
            grip * tyre_grip * 0.22
            * (1.0 - draft * 0.28)
            * (1.0 - self.oversteer * 0.62),
            0.025, 0.27,
        ) * dt

        # Hard steering scrubs speed through the tyres. The loss rises quickly
        # with both steering angle and velocity, preventing flat-out cornering.
        current_speed = self.velocity.length()
        cornering_scrub = min(
            current_speed,
            abs(steer) ** 1.7
            * current_speed ** 2
            * 0.004
            * (1.0 + self.understeer * 0.45 + self.oversteer * 0.65)
            * dt,
        )
        if current_speed > 1e-9 and cornering_scrub > 0:
            self.velocity.scale_to_length(current_speed - cornering_scrub)

        # Rolling resistance plus quadratic aerodynamic drag means a small
        # maintenance throttle is no longer enough to hold maximum speed.
        current_speed = self.velocity.length()
        coast_drag = min(
            current_speed,
            (
                ROLLING_RESISTANCE
                + current_speed ** 2 * AERO_DRAG_COEFFICIENT
            ) * dt,
        )
        if current_speed > 1e-9 and coast_drag > 0:
            self.velocity.scale_to_length(current_speed - coast_drag)
        if self.velocity.length() > max_speed:
            self.velocity.scale_to_length(max_speed)
        if "grass" in wheel_surfaces:
            self.dirty = 180
            self.velocity *= 0.975 ** dt
        all_out = all(
            value not in ("asphalt", "kerb", "pitlane")
            for value in wheel_surfaces
        )
        if all_out:
            self.off_track_frames += dt
        if all_out and not self.outside_limits:
            self.track_limits += 1
        self.outside_limits = all_out
        if surface == "wall":
            impact = speed
            if damage_enabled:
                self.health -= impact * 7.0
            self.velocity *= -0.2
            _, near, near_segment, _ = track.nearest(self.position)
            direction = self.position - near
            if direction.length():
                margin = float(track.features.get("border_margin", BORDER_W))
                # Move the complete collision box back inside the barrier.
                # Repositioning only the centre at the wall edge can leave a
                # wheel beyond the border forever, preventing AI recovery.
                car_clearance = math.hypot(
                    CAR_LENGTH_M / 2, CAR_WIDTH_M / 2
                )
                safe_distance = max(
                    track.width_at_segment(near_segment) / 2 + .5,
                    margin / 2 - car_clearance - .5,
                )
                self.position = (
                    near + direction.normalize() * safe_distance
                )
        self.position += self.velocity * dt
        self.in_pitlane = track.is_in_pitlane(self.position)
        if (
            self.in_pitlane
            and self.velocity.length() > PIT_SPEED_LIMIT_MPF
        ):
            self.velocity.scale_to_length(PIT_SPEED_LIMIT_MPF)
        angle_delta = (
            (self.angle - previous_angle + 180.0) % 360.0 - 180.0
        )
        self.angular_velocity = angle_delta / max(dt, 1e-9)
        current_heading = Vector2(1, 0).rotate(self.angle)
        current_normal = Vector2(-current_heading.y, current_heading.x)
        lateral_speed = abs(self.velocity.dot(current_normal))
        velocity_slip = lateral_speed / max(self.velocity.length(), 0.08)
        self.tire_slip = clamp(
            velocity_slip * 1.15
            + self.understeer * 0.12
            + self.oversteer * 0.48,
            0.0, 1.0,
        )
        self.traction = 1.0 - self.tire_slip
        self.update_drivetrain(throttle)
        self.previous_steering = steer
        self.previous_throttle = throttle
        self.previous_brake = self.brake_input
        wear_rate = {"Soft": .0018, "Medium": .0012, "Hard": .0008, "Wet": .0025}.get(self.tyre, .0012)
        if self.tyre == "Wet" and rain < .2:
            wear_rate *= 3.6
        wear_rate *= 1.0 + self.race_aggression * 0.12
        self.tyre_wear += speed * wear_rate * dt
        self.fuel = max(
            0,
            self.fuel
            - throttle * (1.0 - self.brake_input) * 0.0009 * dt,
        )
        if self.tyre_wear > 70 and random.random() < ((self.tyre_wear - 70) / 30) ** 4 * .0005:
            self.puncture = True
        pit_boxes = track.pit_box_positions()
        if self.pit_requested and pit_boxes:
            assigned = self.pit_box_index if self.pit_box_index is not None else 0
            box_position = pit_boxes[assigned % len(pit_boxes)]
            if self.position.distance_to(box_position) < 6:
                self.pit_timer = 120
                self.velocity *= .25
        _, _, progress_segment, progress_ratio = track.nearest(self.position)
        progress = progress_segment + progress_ratio
        progress_m = (
            track.cumulative_lengths_m[progress_segment]
            + track.segment_lengths_m[progress_segment] * progress_ratio
        )
        if self.previous_progress_m is None:
            self.previous_progress_m = progress_m
        progress_delta_m = (
            progress_m - self.previous_progress_m + track.measured_length_m / 2
        ) % track.measured_length_m - track.measured_length_m / 2
        # Nearest-line queries can jump at crossings or close parallel sections.
        # A real car cannot advance more than this in one simulation frame.
        maximum_progress = 3.0 * max(dt, 0.25)
        progress_delta_m = clamp(
            progress_delta_m, -maximum_progress, maximum_progress
        )
        self.forward_distance_m += max(0.0, progress_delta_m)
        self.reverse_distance_m += max(0.0, -progress_delta_m)
        self.previous_progress_m = progress_m
        if speed < 0.08:
            self.stagnant_frames += dt
        else:
            self.stagnant_frames = 0.0
        self.control_penalty += (
            abs(heading_error) * 0.025 + max(0.0, abs(steer) - 0.65) * 0.01
        ) * max(speed / 0.5, 0.15) * dt
        def crossed(marker):
            if progress >= self.previous_progress:
                return self.previous_progress < marker <= progress
            return marker > self.previous_progress or marker <= progress

        sectors = [
            (int(index) % len(track.points)) * track.samples_per_section
            for index in track.features.get("sectors", [])
        ]
        if self.checkpoint < len(sectors) and crossed(sectors[self.checkpoint]):
            self.checkpoint += 1
        start_marker = (
            int(track.features.get("start_finish", 0)) % len(track.points)
        ) * track.samples_per_section
        if crossed(start_marker) and speed > 1 and (not sectors or self.checkpoint >= len(sectors)):
            self.lap += 1
            self.tyre_laps += 1
            self.checkpoint = 0
        self.previous_progress = progress
        lap_progress_m = progress_m
        if self.lap == 0 and progress_m > track.measured_length_m * 0.8:
            # Cars in the rows behind the line begin a few metres before zero.
            lap_progress_m -= track.measured_length_m
        self.score = (
            self.lap * track.measured_length_m
            + lap_progress_m
            - self.track_limits * 10.0
        )
        stalled_penalty = min(
            max(0.0, self.stagnant_frames - 180.0) * 0.025, 100.0
        )
        self.fitness = (
            self.forward_distance_m
            + self.overtake_reward
            - self.reverse_distance_m * 2.0
            - self.track_limits * 20.0
            - self.off_track_frames * 0.04
            - (100.0 - self.health) * 1.5
            - self.control_penalty
            - self.collision_penalty
            - stalled_penalty
        )
        self.alive = self.health > 0

    def draw(self, screen, focused=False, offset=Vector2(), scale=1.0):
        # Cars keep their true 5.6 m x 2.0 m footprint. Four-times
        # oversampling and a per-era cache preserve small mechanical details
        # after the world is scaled and the sprite is rotated.
        cache_key = (
            self.generation,
            tuple(self.color),
            round(float(scale), 3),
        )
        sprite = _CAR_SPRITE_CACHE.get(cache_key)
        if sprite is None:
            aa = 4
            logical_length = max(14, round(CAR_LENGTH_M * scale))
            logical_width = max(7, round(CAR_WIDTH_M * scale))
            length = logical_length * aa
            width = logical_width * aa
            padding = max(4 * aa, round(scale * .65 * aa))
            sprite = pygame.Surface(
                (length + padding * 2, width + padding * 2),
                pygame.SRCALPHA,
            )
            x0, x1 = padding, padding + length
            centre_x = (x0 + x1) // 2
            centre_y = sprite.get_height() // 2
            half_width = width / 2
            highlight = tuple(
                min(255, channel + 72) for channel in self.color
            )
            shade = tuple(
                max(0, round(channel * .42)) for channel in self.color
            )
            dark_livery = tuple(
                max(0, round(channel * .70)) for channel in self.color
            )
            hybrid = self.generation == "Hybrid"

            tyre_length = max(
                4 * aa, round(scale * (1.10 if hybrid else .96) * aa)
            )
            tyre_width = max(
                2 * aa, round(scale * (.52 if hybrid else .42) * aa)
            )
            axle_offset = CAR_WHEELBASE_M / 2 * scale * aa
            wheel_centres = []
            for longitudinal in (-axle_offset, axle_offset):
                wheel_x = round(
                    centre_x + longitudinal - tyre_length / 2
                )
                for side in (-1, 1):
                    wheel_y = round(
                        centre_y
                        + side * (half_width - tyre_width / 2)
                        - tyre_width / 2
                    )
                    tyre = pygame.Rect(
                        wheel_x, wheel_y, tyre_length, tyre_width
                    )
                    pygame.draw.rect(
                        sprite, (5, 7, 8), tyre,
                        border_radius=max(1, tyre_width // 3),
                    )
                    pygame.draw.rect(
                        sprite, (38, 43, 43), tyre, max(1, aa // 2),
                        border_radius=max(1, tyre_width // 3),
                    )
                    wheel_centres.append((tyre.center, side, longitudinal))
                    if hybrid:
                        # 18-inch wheel cover and central locking nut.
                        rim = tyre.inflate(
                            -max(2, tyre.width // 3),
                            -max(2, tyre.height // 3),
                        )
                        pygame.draw.ellipse(sprite, (95, 101, 100), rim)
                        pygame.draw.ellipse(
                            sprite, highlight,
                            rim.inflate(-max(1, aa), -max(1, aa)),
                            max(1, aa // 2),
                        )
                    else:
                        # Early-2000s grooved tyre.
                        for groove in (.24, .43, .62, .81):
                            groove_y = tyre.y + round(tyre.height * groove)
                            pygame.draw.line(
                                sprite, (83, 87, 84),
                                (tyre.x + aa, groove_y),
                                (tyre.right - aa, groove_y),
                                max(1, aa // 2),
                            )

            # Carbon suspension arms remain visible between chassis and tyres.
            for (wheel_x, wheel_y), side, longitudinal in wheel_centres:
                chassis_x = centre_x + longitudinal * .72
                chassis_y = centre_y + side * half_width * .16
                pygame.draw.line(
                    sprite, (67, 73, 72),
                    (round(chassis_x), round(chassis_y)),
                    (wheel_x, wheel_y), max(1, aa),
                )
                pygame.draw.line(
                    sprite, (18, 22, 22),
                    (
                        round(chassis_x - length * .06),
                        round(chassis_y),
                    ),
                    (wheel_x, wheel_y), max(1, aa // 2),
                )

            if hybrid:
                # 2022 ground-effect car: full-width wings, wide sculpted
                # floor/sidepods, large tyres and a prominent halo.
                front_wing = pygame.Rect(
                    round(x1 - length * .13),
                    round(centre_y - half_width * .50),
                    round(length * .14),
                    round(half_width),
                )
                rear_wing = pygame.Rect(
                    x0, round(centre_y - half_width * .46),
                    round(length * .115), round(half_width * .92),
                )
                pygame.draw.rect(
                    sprite, shade, front_wing, border_radius=aa
                )
                pygame.draw.rect(
                    sprite, shade, rear_wing, border_radius=aa
                )
                for wing in (front_wing, rear_wing):
                    pygame.draw.line(
                        sprite, highlight,
                        (wing.left + aa, wing.top + aa),
                        (wing.right - aa, wing.top + aa),
                        max(1, aa),
                    )
                    pygame.draw.line(
                        sprite, (8, 12, 12),
                        (wing.left + aa, wing.bottom - aa),
                        (wing.right - aa, wing.bottom - aa),
                        max(1, aa),
                    )
                floor = [
                    (x0 + length * .13, centre_y - half_width * .29),
                    (x0 + length * .40, centre_y - half_width * .43),
                    (x0 + length * .67, centre_y - half_width * .38),
                    (x1 - length * .22, centre_y - half_width * .18),
                    (x1 - length * .22, centre_y + half_width * .18),
                    (x0 + length * .67, centre_y + half_width * .38),
                    (x0 + length * .40, centre_y + half_width * .43),
                    (x0 + length * .13, centre_y + half_width * .29),
                ]
                pygame.draw.polygon(sprite, (14, 20, 20), floor)
                pygame.draw.lines(
                    sprite, (66, 72, 70), True, floor, max(1, aa // 2)
                )
                body = [
                    (x1 - length * .025, centre_y),
                    (x1 - length * .31, centre_y - half_width * .105),
                    (x0 + length * .67, centre_y - half_width * .26),
                    (x0 + length * .48, centre_y - half_width * .34),
                    (x0 + length * .23, centre_y - half_width * .28),
                    (x0 + length * .105, centre_y - half_width * .14),
                    (x0 + length * .105, centre_y + half_width * .14),
                    (x0 + length * .23, centre_y + half_width * .28),
                    (x0 + length * .48, centre_y + half_width * .34),
                    (x0 + length * .67, centre_y + half_width * .26),
                    (x1 - length * .31, centre_y + half_width * .105),
                ]
                pygame.draw.polygon(sprite, self.color, body)
                # Sidepod inlets, floor edge and modern livery flashes.
                for side in (-1, 1):
                    sidepod = [
                        (
                            x0 + length * .40,
                            centre_y + side * half_width * .14,
                        ),
                        (
                            x0 + length * .62,
                            centre_y + side * half_width * .22,
                        ),
                        (
                            x0 + length * .55,
                            centre_y + side * half_width * .33,
                        ),
                        (
                            x0 + length * .35,
                            centre_y + side * half_width * .25,
                        ),
                    ]
                    pygame.draw.polygon(sprite, dark_livery, sidepod)
                    pygame.draw.line(
                        sprite, highlight,
                        (
                            round(x0 + length * .18),
                            round(centre_y + side * half_width * .22),
                        ),
                        (
                            round(x0 + length * .67),
                            round(centre_y + side * half_width * .28),
                        ),
                        max(1, aa),
                    )
                pygame.draw.line(
                    sprite, highlight,
                    (round(x0 + length * .64), centre_y),
                    (round(x1 - length * .055), centre_y),
                    max(1, aa),
                )
                cockpit = pygame.Rect(
                    round(x0 + length * .43),
                    round(centre_y - half_width * .22),
                    round(length * .20), round(half_width * .44),
                )
                pygame.draw.ellipse(sprite, (5, 10, 12), cockpit)
                helmet_radius = max(aa, round(half_width * .12))
                helmet_center = (
                    round(cockpit.x + cockpit.width * .43), centre_y
                )
                pygame.draw.circle(
                    sprite, YELLOW, helmet_center, helmet_radius
                )
                pygame.draw.circle(
                    sprite, (26, 34, 35), helmet_center,
                    max(1, helmet_radius // 2),
                )
                halo = cockpit.inflate(
                    round(length * .04), round(half_width * .17)
                )
                pygame.draw.ellipse(
                    sprite, highlight, halo, max(1, aa)
                )
                pygame.draw.line(
                    sprite, highlight,
                    (halo.centerx, halo.top),
                    (halo.centerx, centre_y), max(1, aa)
                )
            else:
                # Early-2000s car: narrow multi-element wings, a long raised
                # nose, compact sidepods and an exposed open cockpit.
                front_wing = pygame.Rect(
                    round(x1 - length * .105),
                    round(centre_y - half_width * .42),
                    round(length * .115), round(half_width * .84),
                )
                rear_wing = pygame.Rect(
                    x0, round(centre_y - half_width * .39),
                    round(length * .105), round(half_width * .78),
                )
                pygame.draw.rect(
                    sprite, shade, front_wing, border_radius=aa
                )
                pygame.draw.rect(
                    sprite, shade, rear_wing, border_radius=aa
                )
                for wing in (front_wing, rear_wing):
                    pygame.draw.line(
                        sprite, highlight,
                        (wing.left + aa, wing.centery),
                        (wing.right - aa, wing.centery),
                        max(1, aa),
                    )
                body = [
                    (x1 - length * .02, centre_y),
                    (x1 - length * .43, centre_y - half_width * .065),
                    (x0 + length * .64, centre_y - half_width * .19),
                    (x0 + length * .45, centre_y - half_width * .29),
                    (x0 + length * .23, centre_y - half_width * .25),
                    (x0 + length * .095, centre_y - half_width * .14),
                    (x0 + length * .095, centre_y + half_width * .14),
                    (x0 + length * .23, centre_y + half_width * .25),
                    (x0 + length * .45, centre_y + half_width * .29),
                    (x0 + length * .64, centre_y + half_width * .19),
                    (x1 - length * .43, centre_y + half_width * .065),
                ]
                pygame.draw.polygon(sprite, self.color, body)
                for side in (-1, 1):
                    pygame.draw.polygon(
                        sprite, dark_livery,
                        [
                            (
                                x0 + length * .28,
                                centre_y + side * half_width * .13,
                            ),
                            (
                                x0 + length * .53,
                                centre_y + side * half_width * .18,
                            ),
                            (
                                x0 + length * .48,
                                centre_y + side * half_width * .27,
                            ),
                            (
                                x0 + length * .24,
                                centre_y + side * half_width * .21,
                            ),
                        ],
                    )
                pygame.draw.line(
                    sprite, highlight,
                    (round(x0 + length * .13), centre_y),
                    (round(x1 - length * .045), centre_y),
                    max(1, aa),
                )
                cockpit = pygame.Rect(
                    round(x0 + length * .42),
                    round(centre_y - half_width * .20),
                    round(length * .19), round(half_width * .40),
                )
                pygame.draw.ellipse(sprite, (5, 10, 12), cockpit)
                helmet_radius = max(aa, round(half_width * .11))
                helmet_center = (
                    round(cockpit.x + cockpit.width * .43), centre_y
                )
                pygame.draw.circle(
                    sprite, WHITE, helmet_center, helmet_radius
                )
                pygame.draw.circle(
                    sprite, (25, 34, 35), helmet_center,
                    max(1, helmet_radius // 2),
                )
                pygame.draw.arc(
                    sprite, highlight,
                    cockpit.inflate(aa * 2, aa * 2),
                    math.pi * .60, math.pi * 1.40, max(1, aa),
                )
                # Tall airbox/roll hoop behind the open cockpit.
                airbox = pygame.Rect(
                    round(x0 + length * .31),
                    round(centre_y - half_width * .105),
                    round(length * .085), round(half_width * .21),
                )
                pygame.draw.ellipse(sprite, shade, airbox)

            sprite = pygame.transform.smoothscale(
                sprite,
                (
                    max(1, sprite.get_width() // aa),
                    max(1, sprite.get_height() // aa),
                ),
            )
            _CAR_SPRITE_CACHE[cache_key] = sprite

        rotated = pygame.transform.rotate(sprite, -self.angle)
        screen_position = self.position * scale + offset
        # A soft offset shadow separates the car from asphalt without changing
        # its collision geometry.
        shadow_sprite = rotated.copy()
        shadow_sprite.fill((18, 22, 22, 105), special_flags=pygame.BLEND_RGBA_MULT)
        screen.blit(
            shadow_sprite,
            shadow_sprite.get_rect(
                center=(screen_position.x + 2, screen_position.y + 3)
            ),
        )
        screen.blit(rotated, rotated.get_rect(center=screen_position))
        if focused:
            pygame.draw.circle(
                screen, CYAN, screen_position,
                max(12, round(2.2 * scale)), 2,
            )


def spawn_car(
    track, brain, color, name="AI", offset=0, wide_start=False
):
    # Racecraft training uses one staggered car per longitudinal slot so a
    # large population is already separated before its first decision.
    row = offset if wide_start else offset // 2
    longitudinal_gap = 4.0 if wide_start else 2.0
    target_distance = row * (CAR_LENGTH_M + longitudinal_gap)
    current_index = 0
    current = track.centerline[0]
    remaining = target_distance
    tangent = (track.centerline[1] - current).normalize()
    start = current
    for _ in range(len(track.centerline)):
        previous_index = (current_index - 1) % len(track.centerline)
        previous = track.centerline[previous_index]
        segment_length = current.distance_to(previous)
        if remaining <= segment_length:
            ratio = remaining / max(segment_length, 1e-9)
            start = current.lerp(previous, ratio)
            tangent = (current - previous).normalize()
            break
        remaining -= segment_length
        current_index = previous_index
        current = previous
    normal = Vector2(-tangent.y, tangent.x)
    grid_half_width = CAR_WIDTH_M / 2 + (.75 if wide_start else .45)
    position = start + normal * (-grid_half_width if offset % 2 else grid_half_width)
    car = Car(
        position.copy(), tangent.angle_to(Vector2(1, 0)) * -1,
        color, brain, name=name,
    )
    car.previous_progress = track.progress(car.position)
    car.previous_progress_m = track.progress_metres(car.position)
    return car


class Button:
    def __init__(self, rect, title, subtitle=""):
        self.rect = pygame.Rect(rect)
        self.title = title
        self.subtitle = subtitle

    def draw(self, screen, fonts, mouse):
        hover = self.rect.collidepoint(mouse)
        try:
            index = max(0, int(self.title.split()[0]) - 1)
        except (ValueError, IndexError):
            index = 0
        accent = UI_ACCENTS[index % len(UI_ACCENTS)]
        shadow = self.rect.move(0, 6)
        pygame.draw.rect(screen, UI_SHADOW, shadow, border_radius=16)
        pygame.draw.rect(
            screen,
            UI_SURFACE_HOVER if hover else UI_SURFACE_RAISED,
            self.rect,
            border_radius=16,
        )
        pygame.draw.rect(
            screen,
            accent if hover else UI_BORDER,
            self.rect,
            2 if hover else 1,
            border_radius=16,
        )
        pygame.draw.rect(
            screen, accent,
            (self.rect.x, self.rect.y + 17, 4, self.rect.height - 34),
            border_radius=2,
        )
        badge = pygame.Rect(self.rect.x + 20, self.rect.y + 20, 44, 44)
        pygame.draw.rect(
            screen,
            tuple(min(255, channel + 8) for channel in UI_SURFACE_HOVER),
            badge,
            border_radius=12,
        )
        pygame.draw.rect(screen, accent, badge, 1, border_radius=12)
        number = self.title.split()[0]
        number_surface = fonts["mono"].render(number, True, accent)
        screen.blit(
            number_surface,
            number_surface.get_rect(center=badge.center),
        )
        clean_title = self.title[len(number):].strip()
        screen.blit(
            fonts["h2"].render(clean_title, True, WHITE),
            (self.rect.x + 82, self.rect.y + 17),
        )
        screen.blit(
            fonts["small"].render(self.subtitle, True, MUTED),
            (self.rect.x + 82, self.rect.y + 52),
        )
        arrow = pygame.Rect(self.rect.right - 48, self.rect.centery - 16, 32, 32)
        pygame.draw.rect(
            screen,
            accent if hover else UI_SURFACE,
            arrow,
            border_radius=10,
        )
        arrow_text = fonts["body"].render(
            "→", True, INK if hover else MUTED
        )
        screen.blit(arrow_text, arrow_text.get_rect(center=arrow.center))
        return hover


class Game:
    def __init__(self):
        pygame.init()
        try:
            pygame.scrap.init()
        except pygame.error:
            pass
        pygame.display.set_caption("Formula AI Lab")
        self.display_flags = pygame.RESIZABLE | pygame.DOUBLEBUF
        self.window = pygame.display.set_mode(
            (DISPLAY_WIDTH, DISPLAY_HEIGHT), self.display_flags
        )
        self.screen = pygame.Surface((WIDTH, HEIGHT)).convert()
        self._ui_background = None
        self._present_surface = None
        self._present_size = None
        self._minimap_cache = {}
        self.viewport_scale = 1.0
        self.viewport_rect = pygame.Rect(0, 0, WIDTH, HEIGHT)
        self.update_viewport()
        self.clock = pygame.time.Clock()
        self.show_fps = False
        self.fonts = {
            "title": pygame.font.SysFont(
                "Inter, SF Pro Display, Avenir Next, Arial", 44, bold=True
            ),
            "h2": pygame.font.SysFont(
                "Inter, SF Pro Display, Avenir Next, Arial", 25, bold=True
            ),
            "body": pygame.font.SysFont(
                "Inter, SF Pro Text, Avenir Next, Arial", 18
            ),
            "small": pygame.font.SysFont(
                "Inter, SF Pro Text, Avenir Next, Arial", 14
            ),
            "tiny": pygame.font.SysFont(
                "Inter, SF Pro Text, Avenir Next, Arial", 12
            ),
            "mono": pygame.font.SysFont(
                "SF Mono, Menlo, Consolas, monospace", 16, bold=True
            ),
        }
        spa_path = TRACK_DIR / "spa_francorchamps.json"
        self.track = Track.load(spa_path) if spa_path.exists() else Track()
        self.selected_track = spa_path.name if spa_path.exists() else None
        self.mode = "menu"
        self.population = 20
        self.training_generation = "Hybrid"
        self.training_racecraft = False
        self.training_base_brain = "__empty__"
        self.training_duration = 60.0
        self.cars = []
        self.generation = 0
        self.best_brain = Brain.load_best()
        self.paused = False
        self.session_time = 0.0
        self.follow = 0
        self.message = ""
        self.message_until = 0
        self.metric = 0
        self.target_laps = 3
        self.editor_points = []
        self.editor_widths = []
        self.editor_pitlane_points = []
        self.editor_width_node = None
        self.editor_width_dragging = False
        self.editor_kerbs = set()
        self.editor_manual_kerbs = False
        self.editor_features = {}
        self.editor_tool = "route"
        self.editor_camera = Vector2()
        self.editor_zoom = 1.0
        self.editor_geometry = "spline"
        self.editor_declared_length = None
        self.editor_road_width = ROAD_W
        self.algorithm_paths = {
            "ICE": ALGORITHM_DIR / "ice_controller.fai",
            "Hybrid": ALGORITHM_DIR / "hybrid_controller.fai",
        }
        self.algorithm_sources = {
            generation: path.read_text() if path.exists() else ""
            for generation, path in self.algorithm_paths.items()
        }
        self.algorithm_path = self.algorithm_paths[self.training_generation]
        self.algorithm_source = self.algorithm_sources[self.training_generation]
        self.algorithm_cursor = len(self.algorithm_source)
        self.algorithm_anchor = None
        self.algorithm_undo = []
        self.algorithm_redo = []
        self.algorithm_scroll_line = 0
        self.algorithm_docs_scroll_line = 0
        self.algorithm_dragging = False
        self.algorithm_clipboard = ""
        self.algorithm_preferred_col = None
        self.algorithm_find_query = ""
        self.algorithm_error = ""
        driver_names = [
            "NOVA", "APEX", "VOLT", "ZENITH", "ORBIT", "PULSE", "COMET", "ECHO",
            "BLAZE", "KITE", "ONYX", "RAPTOR", "SOLAR", "DRIFT", "TITAN", "FLUX",
            "VEGA", "STORM", "RUNE", "FROST",
        ]
        self.race_entries = [
            {
                "name": name, "color": i, "tyre": ("Soft", "Medium", "Hard")[i % 3],
                "fuel": 50, "brain": "__session__",
            }
            for i, name in enumerate(driver_names)
        ]
        self.race_settings = {
            "cars": 8, "laps": 5, "weather": "Dry",
            "generation": "Hybrid", "teams": False,
        }
        self.team_names = [f"Team {i + 1}" for i in range(10)]
        self.selected_entry = 0
        self.editing_name = None
        self.race_name_value = ""
        self.race_name_cursor = 0
        self.race_name_anchor = None
        self.race_name_target = None
        self.rain_level = 0.0
        self.weather_forecast = 0.0
        self.last_weather_lap = -1
        self.weather_popup_until = 0
        self.event_log = []
        self.replay_frames = []
        self.replay_tick = 0
        self.replay_saved = False
        self.event_camera = False
        self.camera_until = 0
        self.flag_state = "GREEN"
        self.flag_until = 0
        self.camera_zoom = DEFAULT_CAMERA_ZOOM
        self.hotlap_brain = "__session__"
        self.hotlap_generation = "Hybrid"
        self.hotlap_car = None
        self.hotlap_time = 0.0
        self.hotlap_splits = []
        self.hotlap_finished = False
        self.name_dialog = None
        self.name_dialog_background = None
        self.race_countdown = 0.0
        self.race_lights_out_flash = 0.0

    def update_viewport(self):
        """Fit the logical GUI into the high-resolution display."""
        window_width, window_height = self.window.get_size()
        self.viewport_scale = min(
            window_width / WIDTH, window_height / HEIGHT
        )
        render_width = max(1, round(WIDTH * self.viewport_scale))
        render_height = max(1, round(HEIGHT * self.viewport_scale))
        self.viewport_rect = pygame.Rect(
            (window_width - render_width) // 2,
            (window_height - render_height) // 2,
            render_width,
            render_height,
        )

    def logical_position(self, position):
        """Convert a physical window coordinate into the GUI design grid."""
        return (
            (position[0] - self.viewport_rect.x) / self.viewport_scale,
            (position[1] - self.viewport_rect.y) / self.viewport_scale,
        )

    def logical_mouse_position(self):
        return self.logical_position(pygame.mouse.get_pos())

    def translate_events(self, events):
        translated = []
        for event in events:
            if event.type == pygame.VIDEORESIZE:
                self.window = pygame.display.set_mode(
                    event.size, self.display_flags
                )
                self.update_viewport()
                continue
            attributes = event.dict.copy()
            if "pos" in attributes:
                attributes["pos"] = self.logical_position(attributes["pos"])
            if "rel" in attributes:
                attributes["rel"] = (
                    attributes["rel"][0] / self.viewport_scale,
                    attributes["rel"][1] / self.viewport_scale,
                )
            translated.append(pygame.event.Event(event.type, attributes))
        return translated

    def present(self):
        """Upscale the logical frame without allocating a new surface."""
        self.window.fill(INK)
        viewport_size = self.viewport_rect.size
        if self._present_size != viewport_size:
            self._present_surface = pygame.Surface(viewport_size).convert()
            self._present_size = viewport_size
        pygame.transform.smoothscale(
            self.screen,
            viewport_size,
            self._present_surface,
        )
        self.window.blit(self._present_surface, self.viewport_rect)
        pygame.display.flip()

    def text(self, value, pos, font="body", color=WHITE):
        self.screen.blit(self.fonts[font].render(str(value), True, color), pos)

    def draw_app_background(self):
        """Draw a cached gradient-and-grid background for menu-style screens."""
        if self._ui_background is None:
            background = pygame.Surface((WIDTH, HEIGHT)).convert()
            top = (6, 14, 18)
            bottom = (10, 24, 25)
            for y in range(HEIGHT):
                ratio = y / max(HEIGHT - 1, 1)
                color = tuple(
                    round(start + (end - start) * ratio)
                    for start, end in zip(top, bottom)
                )
                pygame.draw.line(background, color, (0, y), (WIDTH, y))
            for x in range(0, WIDTH, 64):
                pygame.draw.line(
                    background, (14, 34, 34), (x, 0), (x, HEIGHT)
                )
            for y in range(0, HEIGHT, 64):
                pygame.draw.line(
                    background, (14, 34, 34), (0, y), (WIDTH, y)
                )
            glow = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            pygame.draw.circle(glow, (*CYAN, 18), (1120, 40), 330)
            pygame.draw.circle(glow, (*UI_BLUE, 12), (1050, 690), 280)
            pygame.draw.circle(glow, (*UI_VIOLET, 9), (100, 660), 220)
            background.blit(glow, (0, 0))
            self._ui_background = background
        self.screen.blit(self._ui_background, (0, 0))

    def glass_card(
        self, rect, accent=None, elevated=True, radius=14
    ):
        """Draw one modern reusable card and return its normalized rectangle."""
        rect = pygame.Rect(rect)
        if elevated:
            pygame.draw.rect(
                self.screen, UI_SHADOW, rect.move(0, 5),
                border_radius=radius,
            )
        pygame.draw.rect(
            self.screen, UI_SURFACE_RAISED, rect, border_radius=radius
        )
        pygame.draw.rect(
            self.screen,
            accent if accent is not None else UI_BORDER,
            rect,
            1,
            border_radius=radius,
        )
        if accent is not None:
            pygame.draw.rect(
                self.screen,
                accent,
                (rect.x + 18, rect.y, max(34, rect.width // 7), 3),
                border_radius=2,
            )
        return rect

    def draw_fps_counter(self):
        """Draw a compact optional performance counter."""
        value = self.fonts["mono"].render(
            f"FPS {self.clock.get_fps():5.1f}", True, CYAN
        )
        card = value.get_rect(topright=(CANVAS_W - 14, 14)).inflate(30, 12)
        pygame.draw.rect(
            self.screen, UI_SHADOW, card.move(0, 3), border_radius=9
        )
        pygame.draw.rect(
            self.screen, UI_SURFACE_RAISED, card, border_radius=9
        )
        pygame.draw.rect(
            self.screen, UI_BORDER_BRIGHT, card, 1, border_radius=9
        )
        pygame.draw.circle(
            self.screen, CYAN, (card.x + 11, card.centery), 3
        )
        self.screen.blit(
            value,
            value.get_rect(midleft=(card.x + 20, card.centery)),
        )

    def camera_transform(self, focus):
        centre = Vector2(CANVAS_W / 2, HEIGHT / 2)
        return centre - Vector2(focus) * self.camera_zoom, self.camera_zoom

    @staticmethod
    def camera_zoom_control_rects():
        card = pygame.Rect(CANVAS_W - 164, 54, 148, 38)
        return (
            card,
            pygame.Rect(card.x + 5, card.y + 5, 28, 28),
            pygame.Rect(card.right - 33, card.y + 5, 28, 28),
        )

    def adjust_camera_zoom(self, steps):
        """Apply bounded multiplicative camera zoom steps."""
        if not steps:
            return
        self.camera_zoom = clamp(
            self.camera_zoom * (1.15 ** steps),
            MIN_CAMERA_ZOOM,
            MAX_CAMERA_ZOOM,
        )

    def handle_camera_zoom_event(self, event):
        """Handle wheel and clickable camera controls in driving modes."""
        _, minus_rect, plus_rect = self.camera_zoom_control_rects()
        if event.type == pygame.MOUSEWHEEL:
            mouse_x, _ = self.logical_mouse_position()
            if mouse_x < CANVAS_W:
                self.adjust_camera_zoom(event.y)
                return True
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button in (4, 5) and event.pos[0] < CANVAS_W:
                self.adjust_camera_zoom(1 if event.button == 4 else -1)
                return True
            if event.button == 1:
                if minus_rect.collidepoint(event.pos):
                    self.adjust_camera_zoom(-1)
                    return True
                if plus_rect.collidepoint(event.pos):
                    self.adjust_camera_zoom(1)
                    return True
        return False

    def draw_camera_zoom_control(self):
        """Draw shared zoom controls over the driving canvas."""
        card, minus_rect, plus_rect = self.camera_zoom_control_rects()
        mouse = self.logical_mouse_position()
        pygame.draw.rect(
            self.screen, UI_SHADOW, card.move(0, 3), border_radius=10
        )
        pygame.draw.rect(
            self.screen, UI_SURFACE_RAISED, card, border_radius=10
        )
        pygame.draw.rect(
            self.screen, UI_BORDER_BRIGHT, card, 1, border_radius=10
        )
        for rect, symbol in ((minus_rect, "−"), (plus_rect, "+")):
            hovered = rect.collidepoint(mouse)
            pygame.draw.rect(
                self.screen,
                CYAN if hovered else UI_SURFACE_HOVER,
                rect,
                border_radius=6,
            )
            label = self.fonts["body"].render(
                symbol, True, INK if hovered else WHITE
            )
            self.screen.blit(label, label.get_rect(center=rect.center))
        zoom_label = self.fonts["tiny"].render(
            f"ZOOM {self.camera_zoom:04.1f}x", True, CYAN
        )
        self.screen.blit(
            zoom_label, zoom_label.get_rect(center=card.center)
        )

    @staticmethod
    def minimap_projection(points, rect):
        """Fit world-space points into a rectangle without distorting them."""
        rect = pygame.Rect(rect)
        vectors = [Vector2(point) for point in points]
        if not vectors:
            return 1.0, Vector2(rect.center), []
        minimum_x = min(point.x for point in vectors)
        maximum_x = max(point.x for point in vectors)
        minimum_y = min(point.y for point in vectors)
        maximum_y = max(point.y for point in vectors)
        source_width = max(maximum_x - minimum_x, 1.0)
        source_height = max(maximum_y - minimum_y, 1.0)
        scale = min(
            rect.width / source_width,
            rect.height / source_height,
        )
        source_centre = Vector2(
            (minimum_x + maximum_x) / 2,
            (minimum_y + maximum_y) / 2,
        )
        offset = Vector2(rect.center) - source_centre * scale
        return (
            scale,
            offset,
            [point * scale + offset for point in vectors],
        )

    def draw_minimap(self, cars, focused=None, rect=(16, 16, 220, 165)):
        """Draw the complete circuit and color-coded live car positions."""
        rect = pygame.Rect(rect)
        cache_key = (id(self.track), rect.size, self.track.name)
        cached = self._minimap_cache.get(cache_key)
        if cached is None:
            cached = self._build_minimap_background(rect)
            # A track replacement invalidates every old projection.
            self._minimap_cache = {cache_key: cached}
        background, scale, offset, map_rect = cached
        overlay = background.copy()

        # Minimap markers have no collision or separation. Draw cars from back
        # to front so the car farther ahead naturally covers cars behind it
        # when their projected positions overlap.
        visible_cars = [
            car for car in cars
            if car.alive or car.finish_time is not None
        ]
        for car in sorted(visible_cars, key=lambda item: item.score):
            marker = Vector2(car.position) * scale + offset
            marker.x = clamp(marker.x, map_rect.left, map_rect.right)
            marker.y = clamp(marker.y, map_rect.top, map_rect.bottom)
            pygame.draw.circle(overlay, INK, marker, 5)
            pygame.draw.circle(overlay, car.color, marker, 3)
        if focused in visible_cars:
            focused_marker = Vector2(focused.position) * scale + offset
            focused_marker.x = clamp(
                focused_marker.x, map_rect.left, map_rect.right
            )
            focused_marker.y = clamp(
                focused_marker.y, map_rect.top, map_rect.bottom
            )
            pygame.draw.circle(overlay, WHITE, focused_marker, 6, 1)
        self.screen.blit(overlay, rect)

    def _build_minimap_background(self, rect):
        """Render static minimap geometry once for the active track."""
        overlay = pygame.Surface(rect.size, pygame.SRCALPHA)
        pygame.draw.rect(
            overlay, (*UI_SURFACE, 238), overlay.get_rect(),
            border_radius=13,
        )
        pygame.draw.rect(
            overlay, (*UI_BORDER_BRIGHT, 235), overlay.get_rect(), 1,
            border_radius=13,
        )
        title = self.fonts["tiny"].render(
            f"TRACK MAP  •  {self.track.name[:14].upper()}",
            True, CYAN,
        )
        overlay.blit(title, (10, 7))
        # Keep thick circuit strokes and car markers away from the card edge.
        map_rect = pygame.Rect(
            16, 32, rect.width - 32, rect.height - 47
        )
        track_count = len(self.track.centerline)
        scale, offset, projected_points = self.minimap_projection(
            self.track.centerline + self.track.pitlane_centerline,
            map_rect,
        )
        track_points = projected_points[:track_count]
        pit_points = projected_points[track_count:]
        if len(track_points) >= 2:
            pygame.draw.lines(
                overlay, (5, 13, 15), True, track_points, 9
            )
            pygame.draw.lines(
                overlay, (91, 105, 108), True, track_points, 4
            )
            pygame.draw.aalines(
                overlay, (196, 213, 208), True, track_points
            )
            start_index = (
                int(self.track.features.get("start_finish", 0))
                % max(len(self.track.points), 1)
            ) * self.track.samples_per_section
            start = track_points[start_index % len(track_points)]
            pygame.draw.circle(overlay, YELLOW, start, 3)
        if len(pit_points) >= 2:
            pygame.draw.lines(
                overlay, (10, 31, 23), False, pit_points, 6
            )
            pygame.draw.lines(
                overlay, (249, 115, 22), False, pit_points, 2
            )
        return overlay, scale, offset, map_rect

    def fit_editor_camera(self):
        if not self.editor_points:
            self.editor_camera = Vector2()
            self.editor_zoom = 1.0
            return
        camera_points = self.editor_points + self.editor_pitlane_points
        xs = [point.x for point in camera_points]
        ys = [point.y for point in camera_points]
        width = max(xs) - min(xs)
        height = max(ys) - min(ys)
        self.editor_zoom = clamp(
            min((CANVAS_W - 100) / max(width, 1), (HEIGHT - 100) / max(height, 1)),
            .08, 2.5,
        )
        self.editor_camera = Vector2(
            min(xs) - 50 / self.editor_zoom,
            min(ys) - 50 / self.editor_zoom,
        )

    def panel(self, title, eyebrow):
        x = CANVAS_W
        pygame.draw.rect(
            self.screen, (8, 18, 20), (x, 0, PANEL, HEIGHT)
        )
        pygame.draw.rect(self.screen, CYAN, (x, 0, PANEL, 3))
        pygame.draw.line(
            self.screen, UI_BORDER, (x, 0), (x, HEIGHT), 1
        )
        eyebrow_card = pygame.Rect(x + 20, 18, PANEL - 40, 23)
        pygame.draw.rect(
            self.screen, UI_SURFACE_RAISED, eyebrow_card, border_radius=7
        )
        pygame.draw.circle(
            self.screen, CYAN,
            (eyebrow_card.x + 11, eyebrow_card.centery), 3,
        )
        self.text(eyebrow.upper(), (x + 39, 22), "tiny", CYAN)
        self.text(title, (x + 20, 52), "h2")
        pygame.draw.line(
            self.screen, UI_BORDER,
            (x + 20, 91), (WIDTH - 20, 91), 1,
        )

    def pill(self, label, value, x, y, width=112, accent=CYAN):
        rect = pygame.Rect(x, y, width, 54)
        pygame.draw.rect(
            self.screen, UI_SURFACE_RAISED, rect, border_radius=10
        )
        pygame.draw.rect(
            self.screen, UI_BORDER, rect, 1, border_radius=10
        )
        pygame.draw.rect(
            self.screen, accent, (x, y + 10, 3, 34), border_radius=2
        )
        self.text(label.upper(), (x + 11, y + 8), "small", MUTED)
        self.text(value, (x + 11, y + 27), "mono", accent)

    def footer_hint(self, items):
        x = CANVAS_W + 18
        pygame.draw.rect(
            self.screen, (8, 19, 21),
            (CANVAS_W + 1, HEIGHT - 70, PANEL - 1, 70),
        )
        pygame.draw.line(
            self.screen, UI_BORDER,
            (CANVAS_W + 18, HEIGHT - 70),
            (WIDTH - 18, HEIGHT - 70),
        )
        for i, item in enumerate(items[:2]):
            self.text(item, (x, HEIGHT - 57 + i * 25), "small", MUTED)

    def notice(self, message):
        self.message = message
        self.message_until = pygame.time.get_ticks() + 2600

    def track_choices(self):
        choices = []
        for path in sorted(TRACK_DIR.glob("*.json")):
            try:
                data = json.loads(path.read_text())
                if not isinstance(data.get("points"), list):
                    continue
            except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
                continue
            choices.append((path.name, str(data.get("name", path.stem))))
        return choices

    def track_label(self):
        choices = dict(self.track_choices())
        return choices.get(self.selected_track, self.track.name)

    def select_track(self, track_id):
        available = {path.name: path for path in TRACK_DIR.glob("*.json")}
        path = available.get(track_id)
        if path is None:
            return False
        try:
            self.track = Track.load(path)
        except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
            return False
        self.selected_track = track_id
        self.camera_zoom = DEFAULT_CAMERA_ZOOM
        return True

    def cycle_track(self, amount=1):
        choices = self.track_choices()
        if not choices:
            return None
        identifiers = [identifier for identifier, _ in choices]
        try:
            index = identifiers.index(self.selected_track)
        except ValueError:
            index = -1 if amount > 0 else 0
        selected = identifiers[(index + amount) % len(identifiers)]
        self.select_track(selected)
        return selected

    def brain_choices(self):
        choices = [("__session__", "CURRENT SESSION")]
        for path in sorted(BRAIN_DIR.glob("*.json")):
            label = path.stem.replace("_", " ")
            try:
                label = json.loads(path.read_text()).get("name", label)
            except (OSError, ValueError, json.JSONDecodeError):
                pass
            choices.append((path.name, str(label).upper()))
        return choices

    def training_brain_choices(self):
        """Brains that can seed generation zero in the training workspace."""
        return [("__empty__", "EMPTY BRAIN")] + self.brain_choices()[1:]

    def training_brain_label(self, brain_id=None):
        brain_id = (
            getattr(self, "training_base_brain", "__empty__")
            if brain_id is None else brain_id
        )
        return dict(self.training_brain_choices()).get(
            brain_id, "EMPTY BRAIN"
        )

    def cycle_training_brain(self, amount=1):
        choices = self.training_brain_choices()
        identifiers = [identifier for identifier, _ in choices]
        current = getattr(self, "training_base_brain", "__empty__")
        try:
            index = identifiers.index(current)
        except ValueError:
            index = 0
        self.training_base_brain = identifiers[
            (index + amount) % len(identifiers)
        ]
        return self.training_base_brain

    def brain_label(self, brain_id):
        choices = dict(self.brain_choices())
        return choices.get(brain_id, "CURRENT SESSION")

    def cycle_brain(self, brain_id, amount=1):
        choices = self.brain_choices()
        identifiers = [identifier for identifier, _ in choices]
        try:
            index = identifiers.index(brain_id)
        except ValueError:
            index = 0
        return identifiers[(index + amount) % len(identifiers)]

    def load_brain_choice(self, brain_id):
        if brain_id == "__session__":
            return self.best_brain
        available = {path.name: path for path in BRAIN_DIR.glob("*.json")}
        path = available.get(brain_id)
        return Brain.load_file(path) if path else self.best_brain

    def create_training_seed(self, program, source):
        """Build generation zero from controller defaults or a saved brain.

        Controller code remains authoritative. A saved brain contributes its
        compatible learned parameters and mutation configuration, allowing old
        brains to seed revised controllers without importing stale code.
        """
        defaults = program.defaults()
        brain_id = getattr(self, "training_base_brain", "__empty__")
        if brain_id == "__empty__":
            return Brain(program=program, source=source)

        available = {path.name: path for path in BRAIN_DIR.glob("*.json")}
        path = available.get(brain_id)
        if path is None:
            self.training_base_brain = "__empty__"
            return Brain(program=program, source=source)

        saved = Brain.load_file(path)
        parameters = defaults.copy()
        if saved.program:
            for name, specification in program.parameters.items():
                if name in saved.parameters:
                    try:
                        parameters[name] = clamp(
                            float(saved.parameters[name]),
                            specification.low,
                            specification.high,
                        )
                    except (TypeError, ValueError):
                        pass
        return Brain(
            config=saved.config.copy(),
            program=program,
            parameters=parameters,
            source=source,
        )

    def open_name_dialog(self, kind, title, default_name, payload=None):
        value = safe_component_name(default_name)
        self.name_dialog = {
            "kind": kind,
            "title": title,
            "value": value,
            "cursor": len(value),
            "anchor": 0,
            "return_mode": self.mode,
            "payload": payload,
            "error": "",
            "replace": False,
        }
        self.name_dialog_background = self.screen.copy()
        self.mode = "name_dialog"
        pygame.key.start_text_input()

    def name_dialog_selection(self):
        dialog = self.name_dialog
        if not dialog or dialog["anchor"] is None:
            return None
        if dialog["anchor"] == dialog["cursor"]:
            return None
        return tuple(sorted((dialog["anchor"], dialog["cursor"])))

    def replace_name_dialog_selection(self, text):
        dialog = self.name_dialog
        if not dialog:
            return
        text = "".join(
            character
            for character in text
            if character.isalnum() or character in "-_ "
        )
        selection = self.name_dialog_selection()
        start, end = selection or (dialog["cursor"], dialog["cursor"])
        value = (dialog["value"][:start] + text + dialog["value"][end:])[:40]
        dialog["value"] = value
        dialog["cursor"] = min(start + len(text), len(value))
        dialog["anchor"] = None
        dialog["error"] = ""
        dialog["replace"] = False

    def name_dialog_path(self, kind, name):
        if kind == "track":
            return TRACK_DIR / component_filename(name, ".json")
        if kind == "brain":
            return BRAIN_DIR / component_filename(name, ".json")
        return ALGORITHM_DIR / component_filename(name, ".fai")

    def close_name_dialog(self):
        return_mode = self.name_dialog["return_mode"]
        self.name_dialog = None
        self.name_dialog_background = None
        self.mode = return_mode
        if return_mode == "algorithm":
            pygame.key.start_text_input()
        else:
            pygame.key.stop_text_input()

    def confirm_name_dialog(self):
        dialog = self.name_dialog
        if not dialog:
            return None
        name = safe_component_name(dialog["value"], "")
        if not name:
            dialog["error"] = "Enter a name before saving."
            return None
        destination = self.name_dialog_path(dialog["kind"], name)
        if destination.exists() and not dialog["replace"]:
            dialog["error"] = "That name already exists. Press Save again to replace it."
            dialog["replace"] = True
            return None
        kind = dialog["kind"]
        if kind == "track":
            self.track = self.fit_editor_track()
            path = self.track.save(name)
            self.selected_track = path.name
            message = f"Track saved: {path.name}"
        elif kind == "brain":
            brain = dialog["payload"]
            path = brain.save(name)
            message = f"Brain exported: {path.name}"
        else:
            path = self.save_algorithm(name)
            message = f"Controller saved: {path.name}"
        self.close_name_dialog()
        self.notice(message)
        return path

    def draw_name_dialog(self, events):
        dialog = self.name_dialog
        if not dialog:
            self.mode = "menu"
            return
        field = pygame.Rect(390, 326, 500, 58)
        cancel = pygame.Rect(535, 423, 150, 48)
        save = pygame.Rect(705, 423, 185, 48)
        mono = self.fonts["mono"]
        char_width = max(1, mono.size("M")[0])

        for event in events:
            if event.type == pygame.TEXTINPUT:
                self.replace_name_dialog_selection(event.text)
            elif event.type == pygame.KEYDOWN:
                command = bool(event.mod & (pygame.KMOD_CTRL | pygame.KMOD_META))
                shift = bool(event.mod & pygame.KMOD_SHIFT)
                selection = self.name_dialog_selection()
                if command and event.key == pygame.K_a:
                    dialog["anchor"] = 0
                    dialog["cursor"] = len(dialog["value"])
                elif command and event.key == pygame.K_c and selection:
                    self.editor_set_clipboard(dialog["value"][selection[0]:selection[1]])
                elif command and event.key == pygame.K_x and selection:
                    self.editor_set_clipboard(dialog["value"][selection[0]:selection[1]])
                    self.replace_name_dialog_selection("")
                elif command and event.key == pygame.K_v:
                    self.replace_name_dialog_selection(self.editor_get_clipboard())
                elif event.key == pygame.K_ESCAPE:
                    self.close_name_dialog()
                    return
                elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                    self.confirm_name_dialog()
                    return
                elif event.key == pygame.K_BACKSPACE:
                    if selection:
                        self.replace_name_dialog_selection("")
                    elif dialog["cursor"] > 0:
                        dialog["anchor"] = dialog["cursor"] - 1
                        self.replace_name_dialog_selection("")
                elif event.key == pygame.K_DELETE:
                    if selection:
                        self.replace_name_dialog_selection("")
                    elif dialog["cursor"] < len(dialog["value"]):
                        dialog["anchor"] = dialog["cursor"] + 1
                        self.replace_name_dialog_selection("")
                elif event.key in (pygame.K_LEFT, pygame.K_RIGHT):
                    direction = -1 if event.key == pygame.K_LEFT else 1
                    if selection and not shift:
                        destination = selection[0] if direction < 0 else selection[1]
                    else:
                        destination = clamp(
                            dialog["cursor"] + direction, 0, len(dialog["value"])
                        )
                    if shift and dialog["anchor"] is None:
                        dialog["anchor"] = dialog["cursor"]
                    elif not shift:
                        dialog["anchor"] = None
                    dialog["cursor"] = destination
                elif event.key in (pygame.K_HOME, pygame.K_END):
                    destination = 0 if event.key == pygame.K_HOME else len(dialog["value"])
                    if shift and dialog["anchor"] is None:
                        dialog["anchor"] = dialog["cursor"]
                    elif not shift:
                        dialog["anchor"] = None
                    dialog["cursor"] = destination
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if cancel.collidepoint(event.pos):
                    self.close_name_dialog()
                    return
                if save.collidepoint(event.pos):
                    self.confirm_name_dialog()
                    return
                if field.collidepoint(event.pos):
                    index = clamp(
                        round((event.pos[0] - field.x - 18) / char_width),
                        0, len(dialog["value"]),
                    )
                    dialog["cursor"] = index
                    dialog["anchor"] = None

        if self.name_dialog_background:
            self.screen.blit(self.name_dialog_background, (0, 0))
        overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        overlay.fill((2, 7, 10, 218))
        self.screen.blit(overlay, (0, 0))
        card = pygame.Rect(345, 220, 590, 300)
        self.glass_card(card, accent=CYAN, radius=18)
        self.text(dialog["title"].upper(), (390, 255), "small", CYAN)
        self.text("Name this component", (390, 280), "h2", WHITE)
        pygame.draw.rect(
            self.screen, (7, 18, 21), field, border_radius=10
        )
        pygame.draw.rect(
            self.screen, UI_BORDER_BRIGHT, field, 1, border_radius=10
        )
        selection = self.name_dialog_selection()
        if selection:
            highlight = pygame.Rect(
                field.x + 18 + selection[0] * char_width,
                field.y + 14,
                max(char_width, (selection[1] - selection[0]) * char_width),
                28,
            )
            pygame.draw.rect(self.screen, (32, 91, 103), highlight, border_radius=3)
        self.text(dialog["value"], (field.x + 18, field.y + 18), "mono", WHITE)
        if pygame.time.get_ticks() % 1000 < 550:
            caret_x = field.x + 18 + dialog["cursor"] * char_width
            pygame.draw.line(
                self.screen, YELLOW,
                (caret_x, field.y + 14), (caret_x, field.y + 43), 2,
            )
        pygame.draw.rect(
            self.screen, UI_SURFACE_HOVER, cancel, border_radius=10
        )
        pygame.draw.rect(self.screen, CYAN, save, border_radius=10)
        cancel_text = self.fonts["mono"].render("CANCEL", True, MUTED)
        save_label = "REPLACE" if dialog["replace"] else "SAVE"
        save_text = self.fonts["mono"].render(save_label, True, INK)
        self.screen.blit(cancel_text, cancel_text.get_rect(center=cancel.center))
        self.screen.blit(save_text, save_text.get_rect(center=save.center))
        helper = dialog["error"] or "Letters, numbers, spaces, hyphens and underscores"
        color = (255, 170, 170) if dialog["error"] else MUTED
        self.text(helper, (390, 397), "small", color)
        self.text("Enter save  •  Esc cancel  •  Ctrl/Cmd+A/C/X/V", (390, 487), "small", MUTED)

    def start_race_name_edit(self, entry):
        self.editing_name = "driver"
        self.race_name_target = entry
        self.race_name_value = str(entry.get("name", ""))[:12]
        self.race_name_cursor = len(self.race_name_value)
        self.race_name_anchor = 0
        pygame.key.start_text_input()

    def race_name_selection(self):
        if (
            self.editing_name != "driver"
            or self.race_name_anchor is None
            or self.race_name_anchor == self.race_name_cursor
        ):
            return None
        return tuple(sorted((self.race_name_anchor, self.race_name_cursor)))

    def replace_race_name_selection(self, text):
        allowed = "".join(
            character
            for character in str(text).upper()
            if character.isalnum() or character in "-_ "
        )
        selection = self.race_name_selection()
        start, end = selection or (
            self.race_name_cursor, self.race_name_cursor
        )
        value = (
            self.race_name_value[:start]
            + allowed
            + self.race_name_value[end:]
        )[:12]
        self.race_name_value = value
        self.race_name_cursor = min(start + len(allowed), len(value))
        self.race_name_anchor = None

    def finish_race_name_edit(self, commit=True):
        if self.editing_name != "driver":
            return
        if commit and self.race_name_target is not None:
            cleaned = " ".join(self.race_name_value.split()).strip()
            if cleaned:
                self.race_name_target["name"] = cleaned[:12]
            else:
                self.notice("A race car must have a name")
        self.editing_name = None
        self.race_name_target = None
        self.race_name_anchor = None
        pygame.key.stop_text_input()

    def reset_training(self, evolve=False):
        if evolve and self.cars:
            champion = max(self.cars, key=lambda car: car.fitness)
            self.best_brain = champion.brain
            self.generation += 1
        self.cars = []
        powertrain = getattr(self, "training_generation", "Hybrid")
        racecraft = getattr(self, "training_racecraft", False)
        self.training_duration = clamp(self.track.lap_length_m / 45.0, 30.0, 180.0)
        for i in range(self.population):
            brain = self.best_brain if i == 0 else self.best_brain.mutate()
            car = spawn_car(
                self.track, brain, COLORS[i % len(COLORS)],
                f"Agent {i + 1}", i if racecraft else 0,
                wide_start=racecraft,
            )
            car.generation = powertrain
            car.battery = 100.0 if car.generation == "Hybrid" else 0.0
            self.cars.append(car)
        self.session_time = 0
        self.follow = 0

    def change_population(self, delta):
        """Change the live field without throwing away the running generation."""
        target = clamp(self.population + delta, 1, 50)
        if target == self.population:
            return
        if target > self.population:
            for i in range(self.population, target):
                brain = self.best_brain.mutate()
                car = spawn_car(
                    self.track, brain, COLORS[i % len(COLORS)],
                    f"Agent {i + 1}",
                    i if getattr(self, "training_racecraft", False) else 0,
                    wide_start=getattr(
                        self, "training_racecraft", False
                    ),
                )
                car.generation = getattr(
                    self, "training_generation", "Hybrid"
                )
                car.battery = 100.0 if car.generation == "Hybrid" else 0.0
                self.cars.append(car)
        else:
            self.cars = self.cars[:target]
        self.population = target
        self.notice(f"Training field: {self.population} agents")

    def switch_training_generation(self, generation):
        if generation not in ("ICE", "Hybrid"):
            return
        current = getattr(self, "training_generation", "Hybrid")
        if generation == current:
            return
        self.algorithm_sources[current] = self.algorithm_source
        self.algorithm_paths[current] = self.algorithm_path
        self.training_generation = generation
        self.algorithm_path = self.algorithm_paths[generation]
        self.algorithm_source = self.algorithm_sources[generation]
        self.algorithm_cursor = len(self.algorithm_source)
        self.algorithm_anchor = None
        self.algorithm_undo.clear()
        self.algorithm_redo.clear()
        self.algorithm_scroll_line = 0
        self.algorithm_docs_scroll_line = 0
        self.algorithm_error = ""

    def save_algorithm(self, name=None):
        ALGORITHM_DIR.mkdir(parents=True, exist_ok=True)
        if name is not None:
            self.algorithm_path = (
                ALGORITHM_DIR / component_filename(name, ".fai")
            )
        self.algorithm_path.write_text(self.algorithm_source)
        if hasattr(self, "algorithm_paths"):
            self.algorithm_paths[self.training_generation] = self.algorithm_path
            self.algorithm_sources[self.training_generation] = (
                self.algorithm_source
            )
        return self.algorithm_path

    def start_user_training(self):
        try:
            program = SafeAlgorithm(self.algorithm_source)
            self.algorithm_sources[self.training_generation] = (
                self.algorithm_source
            )
            self.best_brain = self.create_training_seed(
                program, self.algorithm_source
            )
            self.generation = 0
            self.rain_level = 0.0
            self.event_log = []
            self.reset_training()
            self.algorithm_error = ""
            self.mode = "training"
            pygame.key.stop_text_input()
        except AlgorithmError as exc:
            self.algorithm_error = str(exc)
            match = re.search(r"Line (\d+)", self.algorithm_error)
            if match:
                line = max(0, int(match.group(1)) - 1)
                self.algorithm_cursor = self.editor_line_col_to_index(line, 0)
                self.algorithm_anchor = None
                self.algorithm_scroll_line = max(0, line - 4)

    def editor_selection(self):
        if self.algorithm_anchor is None or self.algorithm_anchor == self.algorithm_cursor:
            return None
        return tuple(sorted((self.algorithm_anchor, self.algorithm_cursor)))

    def editor_snapshot(self):
        self.algorithm_undo.append((
            self.algorithm_source, self.algorithm_cursor, self.algorithm_anchor,
        ))
        self.algorithm_undo = self.algorithm_undo[-120:]
        self.algorithm_redo.clear()

    def editor_replace_selection(self, text):
        self.editor_snapshot()
        selection = self.editor_selection()
        start, end = selection if selection else (self.algorithm_cursor, self.algorithm_cursor)
        self.algorithm_source = self.algorithm_source[:start] + text + self.algorithm_source[end:]
        self.algorithm_cursor = start + len(text)
        self.algorithm_anchor = None
        self.algorithm_preferred_col = None
        self.algorithm_error = ""

    def editor_set_clipboard(self, text):
        self.algorithm_clipboard = text
        try:
            pygame.scrap.put(pygame.SCRAP_TEXT, text.encode("utf-8") + b"\0")
        except pygame.error:
            pass

    def editor_get_clipboard(self):
        try:
            raw = pygame.scrap.get(pygame.SCRAP_TEXT)
            if raw:
                return raw.decode("utf-8", errors="ignore").rstrip("\0")
        except pygame.error:
            pass
        return self.algorithm_clipboard

    def editor_restore(self, undo=True):
        source_stack = self.algorithm_undo if undo else self.algorithm_redo
        target_stack = self.algorithm_redo if undo else self.algorithm_undo
        if not source_stack:
            return
        target_stack.append((self.algorithm_source, self.algorithm_cursor, self.algorithm_anchor))
        self.algorithm_source, self.algorithm_cursor, self.algorithm_anchor = source_stack.pop()
        self.algorithm_error = ""

    def editor_move(self, position, selecting=False):
        position = clamp(int(position), 0, len(self.algorithm_source))
        if selecting:
            if self.algorithm_anchor is None:
                self.algorithm_anchor = self.algorithm_cursor
        else:
            self.algorithm_anchor = None
        self.algorithm_cursor = position

    def editor_line_data(self):
        lines = self.algorithm_source.splitlines(keepends=True)
        if not lines or self.algorithm_source.endswith("\n"):
            lines.append("")
        starts, total = [], 0
        for line in lines:
            starts.append(total)
            total += len(line)
        return lines, starts

    def editor_cursor_line_col(self):
        lines, starts = self.editor_line_data()
        line = max(i for i, start in enumerate(starts) if start <= self.algorithm_cursor)
        return line, self.algorithm_cursor - starts[line]

    def editor_line_col_to_index(self, line, column):
        lines, starts = self.editor_line_data()
        line = clamp(int(line), 0, len(lines) - 1)
        content_length = len(lines[line].rstrip("\r\n"))
        return starts[line] + clamp(int(column), 0, content_length)

    def editor_move_vertical(self, amount, selecting=False):
        line, column = self.editor_cursor_line_col()
        if self.algorithm_preferred_col is None:
            self.algorithm_preferred_col = column
        self.editor_move(
            self.editor_line_col_to_index(line + amount, self.algorithm_preferred_col),
            selecting,
        )

    def editor_transform_lines(self, action):
        selection = self.editor_selection()
        start, end = selection if selection else (self.algorithm_cursor, self.algorithm_cursor)
        first_line = self.algorithm_source.count("\n", 0, start)
        last_line = self.algorithm_source.count("\n", 0, end)
        lines, starts = self.editor_line_data()
        block_start = starts[first_line]
        block_end = starts[last_line] + len(lines[last_line])
        original = self.algorithm_source[block_start:block_end]
        transformed = action(original)
        self.algorithm_anchor = block_start
        self.algorithm_cursor = block_end
        self.editor_replace_selection(transformed)
        self.algorithm_anchor = block_start
        self.algorithm_cursor = block_start + len(transformed)

    def editor_toggle_comment(self):
        def toggle(block):
            lines = block.splitlines(keepends=True)
            nonempty = [line for line in lines if line.strip()]
            remove = bool(nonempty) and all(line.lstrip().startswith("#") for line in nonempty)
            result = []
            for line in lines:
                if not line.strip():
                    result.append(line)
                elif remove:
                    indent = len(line) - len(line.lstrip())
                    content = line[indent:]
                    content = content[1:]
                    if content.startswith(" "):
                        content = content[1:]
                    result.append(line[:indent] + content)
                else:
                    indent = len(line) - len(line.lstrip())
                    result.append(line[:indent] + "# " + line[indent:])
            return "".join(result)
        self.editor_transform_lines(toggle)

    def editor_indent(self, outdent=False):
        def indent(block):
            lines = block.splitlines(keepends=True)
            if outdent:
                return "".join(line[4:] if line.startswith("    ") else line[1:] if line.startswith("\t") else line for line in lines)
            return "".join("    " + line for line in lines)
        self.editor_transform_lines(indent)

    def start_race(self):
        count = self.race_settings["cars"]
        brain_cache = {}
        self.cars = []
        for i, entry in enumerate(self.race_entries[:count]):
            brain_id = entry.get("brain", "__session__")
            if brain_id not in brain_cache:
                brain_cache[brain_id] = self.load_brain_choice(brain_id)
            car = spawn_car(
                self.track, brain_cache[brain_id],
                COLORS[entry["color"] % len(COLORS)], entry["name"], i,
            )
            car.brain_name = self.brain_label(brain_id)
            car.tyre = entry["tyre"]
            car.fuel = entry["fuel"]
            car.generation = self.race_settings["generation"]
            car.battery = 100.0 if car.generation == "Hybrid" else 0.0
            car.team = self.team_names[i // 2] if self.race_settings["teams"] else ""
            car.pit_box_index = i // 2 if self.race_settings["teams"] else i
            self.cars.append(car)
        for i, car in enumerate(self.cars):
            car.previous_progress = self.track.progress(car.position)
        self.target_laps = self.race_settings["laps"]
        weather = self.race_settings["weather"]
        self.rain_level = 1.0 if weather == "Wet" else 0.0
        self.weather_forecast = random.uniform(.25, .75) if weather == "Changing" else self.rain_level
        self.last_weather_lap = -1
        self.session_time = 0
        self.race_countdown = 5.0
        self.race_lights_out_flash = 0.0
        self.follow = 0
        self.paused = False
        self.event_log = [{
            "time": 0, "type": "start",
            "message": "Five-light start sequence armed",
        }]
        self.replay_frames = []
        self.replay_tick = 0
        self.replay_saved = False
        self.event_camera = False
        self.camera_until = 0
        self.flag_state = "START SEQUENCE"
        self.flag_until = 0

    def advance_race_countdown(self, dt):
        """Advance the five-light sequence; return True while it owns the tick."""
        if self.race_countdown <= 0:
            return False
        self.race_countdown = max(
            0.0, self.race_countdown - max(0.0, dt) / 1000
        )
        if self.race_countdown <= 0:
            self.flag_state = "GREEN"
            self.race_lights_out_flash = 1.0
            self.log_event("start", "LIGHTS OUT — race started", "high")
        return True

    def log_event(self, event_type, message, priority="medium", focus=None):
        self.event_log.append({
            "time": round(self.session_time, 2),
            "type": event_type,
            "priority": priority,
            "message": message,
        })
        self.event_log = self.event_log[-40:]
        if self.event_camera and focus is not None:
            self.follow = int(focus)
            seconds = {"high": 10, "medium": 15, "low": 30}.get(priority, 15)
            self.camera_until = pygame.time.get_ticks() + seconds * 1000

    def capture_replay_frame(self):
        self.replay_frames.append({
            "time": round(self.session_time, 3),
            "rain": round(self.rain_level, 3),
            "focus": self.follow,
            "metric": self.metric,
            "cars": [
                {
                    "name": car.name, "x": round(car.position.x, 2),
                    "y": round(car.position.y, 2), "angle": round(car.angle, 2),
                    "lap": car.lap, "tyre": car.tyre, "wear": round(car.tyre_wear, 2),
                    "fuel": round(car.fuel, 2), "pit_requested": car.pit_requested,
                    "slipstream": round(car.slipstream, 3),
                    "generation": car.generation,
                    "battery": round(car.battery, 2),
                    "overtake": car.overtake_active,
                    "brake": round(car.brake_input, 3),
                    "aggression": round(car.race_aggression, 3),
                    "aggression_error": round(car.aggression_error, 3),
                    "speed_kph": round(car.speed_kph, 1),
                    "gear": car.gear,
                    "rpm": round(car.rpm),
                    "health": round(car.health, 2), "pitstops": car.pitstops,
                    "team": car.team, "brain": car.brain_name,
                }
                for car in self.cars
            ],
        })

    def save_replay(self):
        REPLAY_DIR.mkdir(parents=True, exist_ok=True)
        path = REPLAY_DIR / "latest_race_replay.json"
        path.write_text(json.dumps({
            "version": 1,
            "track": self.track.name,
            "settings": self.race_settings,
            "teams": self.team_names if self.race_settings["teams"] else [],
            "events": self.event_log,
            "frames": self.replay_frames,
        }, indent=2))
        self.replay_saved = True
        self.notice(f"Replay saved: {path.name}")
        return path

    def menu(self, events):
        self.draw_app_background()
        pygame.draw.rect(
            self.screen, UI_SURFACE_RAISED,
            (70, 36, 168, 26), border_radius=8,
        )
        pygame.draw.circle(self.screen, CYAN, (84, 49), 4)
        self.text("SIMULATION ONLINE", (96, 42), "tiny", CYAN)
        self.text("FORMULA AI LAB", (70, 72), "title")
        self.text(
            "DESIGN  /  TRAIN  /  COMPETE",
            (75, 125), "mono", MUTED,
        )
        self.text("SELECT A WORKSPACE", (75, 154), "tiny", CYAN)
        cards = [
            Button((70, 175, 535, 86), "1  Track Studio", "Draw and save a closed circuit"),
            Button((675, 175, 535, 86), "2  AI Training", "Ghost agents share one start point"),
            Button((70, 280, 535, 86), "3  Race Weekend", "Physical collisions, drafting and mixed brains"),
            Button((675, 280, 535, 86), "4  Two-Lap Hotlap", "Choose one AI brain and record its time"),
        ]
        mouse = self.logical_mouse_position()
        for card in cards:
            card.draw(self.screen, self.fonts, mouse)
        preview = pygame.Rect(70, 395, 1140, 235)
        self.glass_card(preview, accent=YELLOW, radius=18)
        pygame.draw.rect(
            self.screen, (24, 39, 37),
            (preview.x + 22, preview.y + 18, 136, 24),
            border_radius=7,
        )
        self.text(
            "ACTIVE CIRCUIT",
            (preview.x + 34, preview.y + 23), "tiny", YELLOW,
        )
        self.text(self.track.name, (preview.x + 24, preview.y + 44), "h2", YELLOW)
        minimum_width = min(self.track.road_widths_m)
        maximum_width = max(self.track.road_widths_m)
        width_summary = (
            f"{self.track.road_width_m:.1f} m"
            if maximum_width - minimum_width < 0.05
            else f"{minimum_width:.1f}–{maximum_width:.1f} m"
        )
        self.text(
            (
                f"{self.track.lap_length_m / 1000:.3f} km  •  "
                f"{width_summary} asphalt"
            ),
            (preview.x + 24, preview.y + 82), "small", CYAN,
        )
        self.track.draw_preview(self.screen, (preview.x + 300, preview.y + 20, preview.width - 325, preview.height - 40))
        pygame.draw.rect(
            self.screen, (7, 17, 20), (0, 655, WIDTH, 105)
        )
        pygame.draw.line(
            self.screen, UI_BORDER, (0, 655), (WIDTH, 655)
        )
        self.text("LOCAL SAVE", (70, 682), "small", CYAN)
        self.text(
            "Tracks and trained brains stay inside this project.",
            (70, 708), "body", MUTED,
        )
        self.text("ESC quits  •  Number keys open workspaces", (832, 702), "small", MUTED)
        for event in events:
            target = None
            if event.type == pygame.KEYDOWN and event.key in (
                pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4,
            ):
                target = event.key - pygame.K_1
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                pygame.event.post(pygame.event.Event(pygame.QUIT))
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                target = next((i for i, c in enumerate(cards) if c.rect.collidepoint(event.pos)), None)
            if target == 0:
                self.editor_points = [p.copy() for p in self.track.points]
                self.editor_widths = list(self.track.road_widths_m)
                self.editor_pitlane_points = [
                    point.copy() for point in self.track.pitlane_points
                ]
                self.editor_width_node = None
                self.editor_width_dragging = False
                self.editor_kerbs = set(self.track.kerb_points)
                self.editor_manual_kerbs = False
                self.editor_features = json.loads(json.dumps(self.track.features))
                self.editor_tool = "route"
                self.editor_geometry = self.track.geometry
                self.editor_declared_length = self.track.declared_length_m
                self.editor_road_width = self.track.road_width_m
                self.fit_editor_camera()
                self.mode = "editor"
            elif target == 1:
                self.mode = "algorithm"
                pygame.key.start_text_input()
            elif target == 2:
                self.mode = "race_setup"
            elif target == 3:
                self.mode = "hotlap_setup"

    def language_reference_docs(self):
        """Return the language reference for the selected powertrain."""
        docs = [
            ("SENSORS", CYAN),
            ("far_left, left, forward, right, far_right", WHITE),
            ("ray_left/right_90, ray_left/right_18", WHITE),
            ("speed, local_velocity_forward/lateral", WHITE),
            ("angular_velocity, traction, tire_slip", WHITE),
            ("rpm, gear 0..1; rpm_value, gear_number", WHITE),
            ("speed_kph; heading_error -1..1", WHITE),
            ("racing_line_offset -1 left / +1 right", WHITE),
            ("waypoint_5_forward/right", WHITE),
            ("waypoint_10_forward/right", WHITE),
            ("waypoint_20_forward/right", WHITE),
            ("waypoint_40_forward/right", WHITE),
            ("corner_curvature_10/20/40 (-1..1)", WHITE),
            ("race_position, field_size, position_deficit", WHITE),
            ("gap_to_leader/next_m, race_aggression", WHITE),
            ("aggression_error: -1 cautious / +1 overcommit", WHITE),
            ("opponent_1/2/3_present + opponent_N_*", WHITE),
            ("  forward/right + velocity_forward/right", MUTED),
            ("car_ahead/distance/side, closing_speed", WHITE),
            ("passing/side race assist; under/oversteer", WHITE),
            ("previous_steering/throttle/brake", WHITE),
            ("off_track, car_collision, dirty_tyres", WHITE),
            ("tyre_wear/age, fuel/fuel_kg, health", WHITE),
            ("puncture, rain, slipstream, lap/progress", WHITE),
        ]
        if self.training_generation == "Hybrid":
            docs.append(("battery/regen/overtake (80% + 20%)", YELLOW))
        else:
            docs.append(("ICE: constant 100% power", YELLOW))
        docs.extend([
            ("OUTPUTS", CYAN),
            ("steering: -1 left / +1 right", WHITE),
            ("throttle / brake: 0.0 .. 1.0", WHITE),
        ])
        if self.training_generation == "Hybrid":
            docs.append(("overtake: 0 off / 1 deploy", WHITE))
        docs.extend([
            ("pit_request: 0 no / 1 yes", WHITE),
            ("pit_tyre: 0 S / 1 M / 2 H / 3 W", WHITE),
            ("PARAMETER / FUNCTIONS / EDITOR", CYAN),
            ("parameter(default, min, max)", WHITE),
            ("clamp min max abs sign sqrt", WHITE),
            ("Cmd/Ctrl A C X V Z Y • F find • L line", WHITE),
        ])
        return docs

    def algorithm(self, events):
        """In-game editor for the restricted, trainable controller language."""
        editor_rect = pygame.Rect(55, 154, 820, 510)
        docs_rect = pygame.Rect(900, 154, 325, 510)
        docs = self.language_reference_docs()
        docs_content_rect = pygame.Rect(
            docs_rect.x + 14, docs_rect.y + 47,
            docs_rect.width - 34, docs_rect.height - 61,
        )
        docs_line_height = 15
        docs_visible_lines = max(
            1, docs_content_rect.height // docs_line_height
        )
        docs_max_scroll = max(0, len(docs) - docs_visible_lines)
        docs_scroll_track = pygame.Rect(
            docs_rect.right - 10, docs_content_rect.y,
            4, docs_content_rect.height,
        )
        self.algorithm_docs_scroll_line = clamp(
            self.algorithm_docs_scroll_line, 0, docs_max_scroll
        )
        start_rect = pygame.Rect(930, 684, 295, 52)
        reload_rect = pygame.Rect(755, 684, 155, 52)
        brain_rect = pygame.Rect(55, 105, 445, 38)
        racecraft_rect = pygame.Rect(515, 105, 160, 38)
        era_rect = pygame.Rect(690, 105, 185, 38)
        track_rect = pygame.Rect(900, 105, 325, 38)
        mono = self.fonts["mono"]
        char_w = max(mono.size("M")[0], 1)

        def caret_from_mouse(position):
            lines, _ = self.editor_line_data()
            clicked_line = clamp(
                (position[1] - editor_rect.y - 15) // 21 + self.algorithm_scroll_line,
                0, len(lines) - 1,
            )
            column = max(0, int((position[0] - editor_rect.x - 52) / char_w + .5))
            return self.editor_line_col_to_index(clicked_line, column)

        for event in events:
            if event.type == pygame.TEXTINPUT:
                self.editor_replace_selection(event.text.replace("\r", ""))
            elif event.type == pygame.KEYDOWN:
                command = bool(event.mod & (pygame.KMOD_CTRL | pygame.KMOD_META))
                shift = bool(event.mod & pygame.KMOD_SHIFT)
                selection = self.editor_selection()
                if command and event.key == pygame.K_a:
                    self.algorithm_anchor = 0
                    self.algorithm_cursor = len(self.algorithm_source)
                elif command and event.key == pygame.K_l:
                    line, _ = self.editor_cursor_line_col()
                    lines, starts = self.editor_line_data()
                    self.algorithm_anchor = starts[line]
                    self.algorithm_cursor = starts[line] + len(lines[line])
                elif command and event.key == pygame.K_f:
                    if selection:
                        self.algorithm_find_query = self.algorithm_source[selection[0]:selection[1]]
                        search_from = selection[1]
                    else:
                        search_from = self.algorithm_cursor
                        if not self.algorithm_find_query:
                            left = self.algorithm_source[:self.algorithm_cursor]
                            right = self.algorithm_source[self.algorithm_cursor:]
                            left_word = re.search(r"\w+$", left)
                            right_word = re.match(r"\w*", right)
                            start = left_word.start() if left_word else self.algorithm_cursor
                            end = self.algorithm_cursor + (right_word.end() if right_word else 0)
                            self.algorithm_find_query = self.algorithm_source[start:end]
                    if self.algorithm_find_query:
                        found = self.algorithm_source.find(self.algorithm_find_query, search_from)
                        if found < 0:
                            found = self.algorithm_source.find(self.algorithm_find_query, 0, search_from)
                        if found >= 0:
                            self.algorithm_anchor = found
                            self.algorithm_cursor = found + len(self.algorithm_find_query)
                    else:
                        self.notice("Place the caret on a word or select text to find")
                elif command and event.key == pygame.K_c:
                    if selection:
                        self.editor_set_clipboard(self.algorithm_source[selection[0]:selection[1]])
                elif command and event.key == pygame.K_x:
                    if selection:
                        self.editor_set_clipboard(self.algorithm_source[selection[0]:selection[1]])
                        self.editor_replace_selection("")
                elif command and event.key == pygame.K_v:
                    self.editor_replace_selection(self.editor_get_clipboard())
                elif command and event.key == pygame.K_z:
                    self.editor_restore(undo=not shift)
                elif command and event.key == pygame.K_y:
                    self.editor_restore(undo=False)
                elif command and event.key == pygame.K_SLASH:
                    self.editor_toggle_comment()
                elif command and event.key == pygame.K_RETURN:
                    self.start_user_training()
                elif command and event.key == pygame.K_s:
                    try:
                        SafeAlgorithm(self.algorithm_source)
                        self.algorithm_error = ""
                        self.open_name_dialog(
                            "algorithm", "Save AI controller",
                            self.algorithm_path.stem.replace("_", " "),
                            self.algorithm_source,
                        )
                    except AlgorithmError as exc:
                        self.algorithm_error = str(exc)
                        match = re.search(r"Line (\d+)", self.algorithm_error)
                        if match:
                            line = max(0, int(match.group(1)) - 1)
                            self.editor_move(self.editor_line_col_to_index(line, 0))
                            self.algorithm_scroll_line = max(0, line - 4)
                elif event.key == pygame.K_ESCAPE:
                    self.mode = "menu"
                    pygame.key.stop_text_input()
                elif event.key == pygame.K_RETURN:
                    line_start = self.algorithm_source.rfind("\n", 0, self.algorithm_cursor) + 1
                    indent = re.match(r"[ \t]*", self.algorithm_source[line_start:self.algorithm_cursor]).group()
                    before = self.algorithm_source[:self.algorithm_cursor].rstrip()
                    extra = "    " if before.endswith(":") else ""
                    self.editor_replace_selection("\n" + indent + extra)
                elif event.key == pygame.K_TAB:
                    if selection:
                        self.editor_indent(outdent=shift)
                    elif shift:
                        line_start = self.algorithm_source.rfind("\n", 0, self.algorithm_cursor) + 1
                        remove = min(4, len(self.algorithm_source[line_start:self.algorithm_cursor]) - len(self.algorithm_source[line_start:self.algorithm_cursor].lstrip(" ")))
                        if remove:
                            self.algorithm_anchor = line_start
                            self.algorithm_cursor = line_start + remove
                            self.editor_replace_selection("")
                    else:
                        self.editor_replace_selection(" " * (4 - self.editor_cursor_line_col()[1] % 4))
                elif event.key == pygame.K_BACKSPACE:
                    if selection:
                        self.editor_replace_selection("")
                    elif self.algorithm_cursor > 0:
                        self.algorithm_anchor = self.algorithm_cursor - 1
                        self.editor_replace_selection("")
                elif event.key == pygame.K_DELETE:
                    if selection:
                        self.editor_replace_selection("")
                    elif self.algorithm_cursor < len(self.algorithm_source):
                        self.algorithm_anchor = self.algorithm_cursor + 1
                        self.editor_replace_selection("")
                elif event.key == pygame.K_LEFT:
                    if selection and not shift:
                        self.editor_move(selection[0])
                    elif command:
                        prefix = self.algorithm_source[:self.algorithm_cursor]
                        match = re.search(r"\w+\W*$", prefix)
                        self.editor_move(match.start() if match else 0, shift)
                    else:
                        self.editor_move(self.algorithm_cursor - 1, shift)
                    self.algorithm_preferred_col = None
                elif event.key == pygame.K_RIGHT:
                    if selection and not shift:
                        self.editor_move(selection[1])
                    elif command:
                        suffix = self.algorithm_source[self.algorithm_cursor:]
                        match = re.match(r"\W*\w+", suffix)
                        self.editor_move(self.algorithm_cursor + (match.end() if match else len(suffix)), shift)
                    else:
                        self.editor_move(self.algorithm_cursor + 1, shift)
                    self.algorithm_preferred_col = None
                elif event.key == pygame.K_UP:
                    self.editor_move_vertical(-1, shift)
                elif event.key == pygame.K_DOWN:
                    self.editor_move_vertical(1, shift)
                elif event.key == pygame.K_HOME:
                    if command:
                        self.editor_move(0, shift)
                    else:
                        line, _ = self.editor_cursor_line_col()
                        self.editor_move(self.editor_line_col_to_index(line, 0), shift)
                elif event.key == pygame.K_END:
                    if command:
                        self.editor_move(len(self.algorithm_source), shift)
                    else:
                        line, _ = self.editor_cursor_line_col()
                        self.editor_move(self.editor_line_col_to_index(line, 10**6), shift)
                elif event.key == pygame.K_PAGEUP:
                    self.editor_move_vertical(-20, shift)
                elif event.key == pygame.K_PAGEDOWN:
                    self.editor_move_vertical(20, shift)
            elif event.type == pygame.MOUSEWHEEL:
                mouse_position = self.logical_mouse_position()
                if docs_rect.collidepoint(mouse_position):
                    self.algorithm_docs_scroll_line = clamp(
                        self.algorithm_docs_scroll_line - event.y * 3,
                        0, docs_max_scroll,
                    )
                elif editor_rect.collidepoint(mouse_position):
                    lines, _ = self.editor_line_data()
                    self.algorithm_scroll_line = clamp(
                        self.algorithm_scroll_line - event.y * 3,
                        0, max(0, len(lines) - 22),
                    )
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if (
                    docs_max_scroll
                    and docs_scroll_track.inflate(14, 0).collidepoint(
                        event.pos
                    )
                ):
                    scroll_fraction = clamp(
                        (
                            event.pos[1] - docs_scroll_track.top
                        ) / max(docs_scroll_track.height, 1),
                        0.0, 1.0,
                    )
                    self.algorithm_docs_scroll_line = round(
                        scroll_fraction * docs_max_scroll
                    )
                elif start_rect.collidepoint(event.pos):
                    self.start_user_training()
                elif brain_rect.collidepoint(event.pos):
                    self.cycle_training_brain(
                        -1 if event.pos[0] < brain_rect.centerx else 1
                    )
                elif racecraft_rect.collidepoint(event.pos):
                    self.training_racecraft = not self.training_racecraft
                elif era_rect.collidepoint(event.pos):
                    self.switch_training_generation(
                        "ICE"
                        if self.training_generation == "Hybrid"
                        else "Hybrid"
                    )
                elif track_rect.collidepoint(event.pos):
                    self.cycle_track(
                        -1 if event.pos[0] < track_rect.centerx else 1
                    )
                elif reload_rect.collidepoint(event.pos) and self.algorithm_path.exists():
                    self.editor_snapshot()
                    self.algorithm_source = self.algorithm_path.read_text()
                    self.algorithm_cursor = len(self.algorithm_source)
                    self.algorithm_anchor = None
                    self.algorithm_error = ""
                elif editor_rect.collidepoint(event.pos):
                    selecting = bool(pygame.key.get_mods() & pygame.KMOD_SHIFT)
                    clicked = caret_from_mouse(event.pos)
                    if getattr(event, "clicks", 1) >= 2:
                        left = self.algorithm_source[:clicked]
                        right = self.algorithm_source[clicked:]
                        left_word = re.search(r"\w+$", left)
                        right_word = re.match(r"\w*", right)
                        start = left_word.start() if left_word else clicked
                        end = clicked + (right_word.end() if right_word else 0)
                        self.algorithm_anchor = start
                        self.algorithm_cursor = end
                    else:
                        self.editor_move(clicked, selecting)
                    self.algorithm_dragging = True
            elif event.type == pygame.MOUSEMOTION and self.algorithm_dragging:
                self.editor_move(caret_from_mouse(event.pos), True)
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                self.algorithm_dragging = False

        current_line, current_col = self.editor_cursor_line_col()
        if current_line < self.algorithm_scroll_line:
            self.algorithm_scroll_line = current_line
        elif current_line >= self.algorithm_scroll_line + 22:
            self.algorithm_scroll_line = current_line - 21

        self.draw_app_background()
        self.text("CODED CONTROLLER", (55, 28), "small", CYAN)
        self.text("Write the driving algorithm", (55, 52), "title")
        pygame.draw.rect(
            self.screen, UI_SURFACE_RAISED, brain_rect, border_radius=9
        )
        pygame.draw.rect(
            self.screen, UI_BORDER, brain_rect, 1, border_radius=9
        )
        self.text("BASE BRAIN", (brain_rect.x + 12, brain_rect.y + 11), "small", MUTED)
        self.text(
            f"‹  {self.training_brain_label()[:27]}  ›",
            (brain_rect.x + 116, brain_rect.y + 10), "small", CYAN,
        )
        pygame.draw.rect(
            self.screen, UI_SURFACE_RAISED, racecraft_rect, border_radius=9
        )
        self.text(
            "RACECRAFT", (racecraft_rect.x + 10, racecraft_rect.y + 11),
            "small", MUTED,
        )
        self.text(
            "ON" if self.training_racecraft else "OFF",
            (racecraft_rect.x + 112, racecraft_rect.y + 10),
            "small", CYAN if self.training_racecraft else MUTED,
        )
        pygame.draw.rect(
            self.screen, UI_SURFACE_RAISED, era_rect, border_radius=9
        )
        self.text("POWER", (era_rect.x + 12, era_rect.y + 11), "small", MUTED)
        self.text(
            self.training_generation.upper(),
            (era_rect.x + 78, era_rect.y + 10), "small", YELLOW,
        )
        pygame.draw.rect(
            self.screen, UI_SURFACE_RAISED, track_rect, border_radius=9
        )
        self.text("TRACK", (track_rect.x + 12, track_rect.y + 11), "small", MUTED)
        self.text(
            f"‹  {self.track_label()[:21]}  ›",
            (track_rect.x + 72, track_rect.y + 10), "small", CYAN,
        )
        pygame.draw.rect(
            self.screen, UI_SHADOW, editor_rect.move(0, 5),
            border_radius=14,
        )
        pygame.draw.rect(
            self.screen, (9, 22, 23), editor_rect, border_radius=14
        )
        pygame.draw.rect(
            self.screen, UI_BORDER, editor_rect, 1, border_radius=14
        )

        lines_keep, line_starts = self.editor_line_data()
        first_line = self.algorithm_scroll_line
        visible = lines_keep[first_line:first_line + 22]
        selection = self.editor_selection()
        self.screen.set_clip(editor_rect.inflate(-8, -8))
        for shown, line in enumerate(visible):
            number = first_line + shown + 1
            y = editor_rect.y + 14 + shown * 21
            if number - 1 == current_line:
                pygame.draw.rect(self.screen, (19, 39, 32), (editor_rect.x + 8, y - 1, editor_rect.width - 16, 21))
            content = line.rstrip("\r\n")
            if selection:
                line_start = line_starts[number - 1]
                line_content_end = line_start + len(content)
                highlight_start = max(selection[0], line_start)
                highlight_end = min(selection[1], line_start + len(line))
                if highlight_end > highlight_start:
                    start_col = highlight_start - line_start
                    end_col = min(highlight_end, line_content_end) - line_start
                    width = max(char_w, (end_col - start_col) * char_w)
                    pygame.draw.rect(
                        self.screen, (32, 91, 103),
                        (editor_rect.x + 52 + start_col * char_w, y - 1, width, 20),
                        border_radius=2,
                    )
            self.text(f"{number:>2}", (editor_rect.x + 14, y), "mono", (76, 103, 92))
            color = MUTED if content.lstrip().startswith("#") else WHITE
            self.text(content, (editor_rect.x + 52, y), "mono", color)
        if pygame.time.get_ticks() % 1000 < 550:
            caret_x = editor_rect.x + 52 + current_col * char_w
            caret_y = editor_rect.y + 14 + (current_line - first_line) * 21
            pygame.draw.line(self.screen, CYAN, (caret_x, caret_y), (caret_x, caret_y + 18), 2)
        self.screen.set_clip(None)
        if len(lines_keep) > 22:
            track_y, track_h = editor_rect.y + 10, editor_rect.height - 20
            thumb_h = max(28, int(track_h * 22 / len(lines_keep)))
            thumb_y = track_y + int((track_h - thumb_h) * first_line / max(1, len(lines_keep) - 22))
            pygame.draw.rect(self.screen, (28, 50, 43), (editor_rect.right - 8, track_y, 3, track_h), border_radius=2)
            pygame.draw.rect(self.screen, CYAN, (editor_rect.right - 9, thumb_y, 5, thumb_h), border_radius=3)

        self.glass_card(docs_rect, accent=UI_BLUE, radius=14)
        self.text(
            f"{self.training_generation.upper()} LANGUAGE REFERENCE",
            (docs_rect.x + 20, docs_rect.y + 18), "small", YELLOW,
        )
        if docs_max_scroll:
            self.text(
                "SCROLL",
                (docs_rect.right - 58, docs_rect.y + 20), "tiny", MUTED,
            )
        first_doc = self.algorithm_docs_scroll_line
        visible_docs = docs[
            first_doc:first_doc + docs_visible_lines
        ]
        self.screen.set_clip(docs_content_rect)
        for i, (line, color) in enumerate(visible_docs):
            self.text(
                line,
                (
                    docs_content_rect.x + 6,
                    docs_content_rect.y + i * docs_line_height,
                ),
                "tiny", color,
            )
        self.screen.set_clip(None)
        if docs_max_scroll:
            pygame.draw.rect(
                self.screen, UI_SURFACE_HOVER, docs_scroll_track,
                border_radius=2,
            )
            docs_thumb_height = max(
                34,
                round(
                    docs_scroll_track.height
                    * docs_visible_lines / len(docs)
                ),
            )
            docs_thumb_y = docs_scroll_track.y + round(
                (
                    docs_scroll_track.height - docs_thumb_height
                ) * first_doc / docs_max_scroll
            )
            pygame.draw.rect(
                self.screen, UI_BLUE,
                (
                    docs_scroll_track.x - 1, docs_thumb_y,
                    6, docs_thumb_height,
                ),
                border_radius=3,
            )
        if self.algorithm_error:
            pygame.draw.rect(self.screen, (72, 25, 28), (55, 681, 680, 55), border_radius=8)
            self.text(self.algorithm_error[:82], (70, 699), "small", (255, 170, 170))
        else:
            self.text("Ctrl/Cmd+S save  •  Ctrl/Cmd+Enter train  •  Tab/Shift+Tab indent", (55, 701), "small", MUTED)
        pygame.draw.rect(self.screen, (18, 38, 31), reload_rect, border_radius=9)
        self.text("RELOAD FILE", (773, 701), "small", MUTED)
        pygame.draw.rect(self.screen, CYAN, start_rect, border_radius=9)
        self.text("VALIDATE & TRAIN", (986, 701), "mono", INK)

    def fit_editor_track(self):
        """Keep authored coordinates in metres; cameras handle screen fitting."""
        return Track(
            [point.copy() for point in self.editor_points],
            "Custom Circuit", self.editor_kerbs, self.editor_features,
            self.editor_geometry, self.editor_declared_length, self.editor_road_width,
            self.editor_widths,
            [point.copy() for point in self.editor_pitlane_points],
        )

    def editor_node_at(self, screen_position, radius=16):
        if not self.editor_points or screen_position[0] >= CANVAS_W:
            return None
        screen_position = Vector2(screen_position)
        screen_points = [
            (point - self.editor_camera) * self.editor_zoom
            for point in self.editor_points
        ]
        nearest = min(
            range(len(screen_points)),
            key=lambda index: screen_points[index].distance_squared_to(screen_position),
        )
        return (
            nearest
            if screen_points[nearest].distance_to(screen_position) <= radius
            else None
        )

    def editor_pitlane_ready(self):
        entry = self.editor_features.get("pit_entry")
        exit_index = self.editor_features.get("pit_exit")
        return (
            entry is not None
            and exit_index is not None
            and entry != exit_index
            and 0 <= int(entry) < len(self.editor_points)
            and 0 <= int(exit_index) < len(self.editor_points)
        )

    def editor_pitlane_node_at(self, screen_position, radius=16):
        if (
            not self.editor_pitlane_points
            or screen_position[0] >= CANVAS_W
        ):
            return None
        screen_position = Vector2(screen_position)
        screen_points = [
            (point - self.editor_camera) * self.editor_zoom
            for point in self.editor_pitlane_points
        ]
        nearest = min(
            range(len(screen_points)),
            key=lambda index: screen_points[index].distance_squared_to(
                screen_position
            ),
        )
        return (
            nearest
            if screen_points[nearest].distance_to(screen_position) <= radius
            else None
        )

    def add_editor_pitlane_node(self, world_position):
        if not self.editor_pitlane_ready():
            return False
        self.editor_pitlane_points.append(Vector2(world_position))
        return True

    def toggle_editor_pit_box(self, pitlane_index):
        if not 0 <= pitlane_index < len(self.editor_pitlane_points):
            return False
        boxes = self.editor_features.setdefault("pit_boxes", [])
        if pitlane_index in boxes:
            boxes.remove(pitlane_index)
        else:
            boxes.append(pitlane_index)
        return True

    def adjust_editor_node_width(self, index, delta):
        if not 0 <= index < len(self.editor_widths):
            return
        self.editor_widths[index] = clamp(
            self.editor_widths[index] + delta, 6.0, 24.0
        )
        self.editor_road_width = (
            sum(self.editor_widths) / len(self.editor_widths)
        )

    def adjust_all_editor_widths(self, delta):
        self.editor_widths = [
            clamp(width + delta, 6.0, 24.0)
            for width in self.editor_widths
        ]
        if self.editor_widths:
            self.editor_road_width = sum(self.editor_widths) / len(self.editor_widths)
        else:
            self.editor_road_width = clamp(
                self.editor_road_width + delta, 6.0, 24.0
            )

    def editor(self, events):
        tools = [
            ("route", "1 ROUTE"), ("kerb", "2 KERB"),
            ("sector", "3 SECTOR"), ("start", "4 START"),
            ("pit_entry", "5 PIT IN"), ("pit_exit", "6 PIT OUT"),
            ("pitlane", "7 PIT ROAD"), ("pit_box", "8 PIT BOX"),
        ]
        x = CANVAS_W + 18
        tool_rects = {}
        for i, (key, _) in enumerate(tools):
            tool_rects[key] = pygame.Rect(x + (i % 2) * 122, 190 + (i // 2) * 46, 114, 37)
        minus_border = pygame.Rect(x + 145, 405, 38, 32)
        plus_border = pygame.Rect(x + 198, 405, 38, 32)
        minus_road = pygame.Rect(x + 145, 459, 38, 32)
        plus_road = pygame.Rect(x + 198, 459, 38, 32)

        for event in events:
            if (
                event.type == pygame.MOUSEWHEEL
                and self.logical_mouse_position()[0] < CANVAS_W
            ):
                if (
                    self.editor_width_dragging
                    and self.editor_width_node is not None
                ):
                    self.adjust_editor_node_width(
                        self.editor_width_node, event.y * 0.5
                    )
                else:
                    cursor = Vector2(self.logical_mouse_position())
                    world_at_cursor = self.editor_camera + cursor / self.editor_zoom
                    self.editor_zoom = clamp(self.editor_zoom * (1.15 ** event.y), .28, 2.5)
                    self.editor_camera = world_at_cursor - cursor / self.editor_zoom
            elif event.type == pygame.MOUSEMOTION and event.buttons[1] and event.pos[0] < CANVAS_W:
                self.editor_camera -= Vector2(event.rel) / self.editor_zoom
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                selected = next((key for key, rect in tool_rects.items() if rect.collidepoint(event.pos)), None)
                if selected:
                    if selected == "pitlane" and not self.editor_pitlane_ready():
                        self.notice("Place PIT IN and PIT OUT before the pit road")
                        continue
                    if selected == "pit_box" and not self.editor_pitlane_points:
                        self.notice("Add at least one PIT ROAD node first")
                        continue
                    self.editor_tool = selected
                    continue
                if minus_border.collidepoint(event.pos):
                    self.editor_features["border_margin"] = max(16, self.editor_features.get("border_margin", BORDER_W) - 2)
                    continue
                if plus_border.collidepoint(event.pos):
                    self.editor_features["border_margin"] = min(80, self.editor_features.get("border_margin", BORDER_W) + 2)
                    continue
                if minus_road.collidepoint(event.pos):
                    self.adjust_all_editor_widths(-0.5)
                    continue
                if plus_road.collidepoint(event.pos):
                    self.adjust_all_editor_widths(0.5)
                    continue
                if event.pos[0] < CANVAS_W:
                    world = self.editor_camera + Vector2(event.pos) / self.editor_zoom
                    if self.editor_tool == "pitlane":
                        self.editor_width_node = None
                        self.editor_width_dragging = False
                        nearest_pit = self.editor_pitlane_node_at(event.pos)
                        if nearest_pit is None:
                            first_pit_node = not self.editor_pitlane_points
                            if not self.add_editor_pitlane_node(world):
                                self.notice(
                                    "Place PIT IN and PIT OUT before the pit road"
                                )
                            else:
                                self.editor_features["pit_boxes"] = (
                                    []
                                    if first_pit_node
                                    else [
                                        index for index in
                                        self.editor_features.get(
                                            "pit_boxes", []
                                        )
                                        if index
                                        < len(self.editor_pitlane_points)
                                    ]
                                )
                                self.editor_declared_length = None
                        continue
                    if self.editor_tool == "pit_box":
                        self.editor_width_node = None
                        self.editor_width_dragging = False
                        nearest_pit = self.editor_pitlane_node_at(event.pos)
                        if nearest_pit is None:
                            self.notice("Pit boxes must be placed on a PIT ROAD node")
                        else:
                            self.toggle_editor_pit_box(nearest_pit)
                        continue

                    nearest = self.editor_node_at(event.pos)
                    self.editor_width_node = nearest
                    self.editor_width_dragging = nearest is not None
                    if self.editor_tool == "route":
                        if nearest is None:
                            self.editor_points.append(world)
                            self.editor_widths.append(self.editor_road_width)
                            self.editor_width_node = len(self.editor_points) - 1
                            self.editor_declared_length = None
                            if not self.editor_manual_kerbs and len(self.editor_points) >= 3:
                                self.editor_kerbs = Track(
                                    self.editor_points, geometry=self.editor_geometry,
                                    road_widths_m=self.editor_widths,
                                ).kerb_points
                    elif nearest is not None:
                        if self.editor_points[nearest].distance_to(world) <= 24 / self.editor_zoom:
                            if self.editor_tool == "kerb":
                                self.editor_manual_kerbs = True
                                if nearest in self.editor_kerbs:
                                    self.editor_kerbs.remove(nearest)
                                else:
                                    self.editor_kerbs.add(nearest)
                            elif self.editor_tool == "sector":
                                sectors = self.editor_features.setdefault("sectors", [])
                                sectors.remove(nearest) if nearest in sectors else sectors.append(nearest)
                            elif self.editor_tool == "start":
                                self.editor_features["start_finish"] = nearest
                            elif self.editor_tool in ("pit_entry", "pit_exit"):
                                self.editor_features[self.editor_tool] = nearest
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                self.editor_width_dragging = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.mode = "menu"
                elif pygame.K_1 <= event.key <= pygame.K_8:
                    selected = tools[event.key - pygame.K_1][0]
                    if selected == "pitlane" and not self.editor_pitlane_ready():
                        self.notice("Place PIT IN and PIT OUT before the pit road")
                    elif selected == "pit_box" and not self.editor_pitlane_points:
                        self.notice("Add at least one PIT ROAD node first")
                    else:
                        self.editor_tool = selected
                elif event.key == pygame.K_BACKSPACE:
                    if self.editor_tool in ("pitlane", "pit_box"):
                        if self.editor_pitlane_points:
                            removed = len(self.editor_pitlane_points) - 1
                            self.editor_pitlane_points.pop()
                            self.editor_features["pit_boxes"] = [
                                index for index in
                                self.editor_features.get("pit_boxes", [])
                                if index != removed
                                and index < len(self.editor_pitlane_points)
                            ]
                    elif self.editor_points:
                        self.editor_points.pop()
                        if self.editor_widths:
                            self.editor_widths.pop()
                        self.editor_width_node = None
                        count = len(self.editor_points)
                        self.editor_kerbs = {
                            i for i in self.editor_kerbs if i < count
                        }
                        self.editor_features["sectors"] = [
                            i for i in
                            self.editor_features.get("sectors", [])
                            if i < count
                        ]
                        for endpoint in ("pit_entry", "pit_exit"):
                            index = self.editor_features.get(endpoint)
                            if index is not None and index >= count:
                                self.editor_features[endpoint] = None
                        if not self.editor_pitlane_ready():
                            self.editor_pitlane_points.clear()
                            self.editor_features["pit_boxes"] = []
                elif event.key == pygame.K_c:
                    self.editor_points.clear()
                    self.editor_widths.clear()
                    self.editor_pitlane_points.clear()
                    self.editor_width_node = None
                    self.editor_width_dragging = False
                    self.editor_kerbs.clear()
                    self.editor_features = {
                        "start_finish": 0, "sectors": [], "pit_entry": None,
                        "pit_exit": None, "pit_boxes": [], "border_margin": BORDER_W,
                    }
                    self.editor_manual_kerbs = False
                elif event.key in (pygame.K_RETURN, pygame.K_s):
                    if len(self.editor_points) < 8:
                        self.notice("Add at least 8 route points")
                    elif (
                        self.editor_pitlane_points
                        and not self.editor_pitlane_ready()
                    ):
                        self.notice(
                            "Pit road requires different PIT IN and PIT OUT nodes"
                        )
                    elif (
                        self.editor_features.get("pit_boxes")
                        and not self.editor_pitlane_points
                    ):
                        self.notice("Pit boxes require a PIT ROAD")
                    elif event.key == pygame.K_s:
                        self.open_name_dialog(
                            "track", "Save track",
                            self.track.name or "Custom Circuit",
                        )
                    else:
                        self.track = self.fit_editor_track()
                        self.selected_track = None
                        self.mode = "menu"

        self.screen.fill((25, 76, 43))
        self.screen.set_clip(pygame.Rect(0, 0, CANVAS_W, HEIGHT))
        # World-space grid makes zoom and pan visible.
        spacing = 100
        world_left = int(self.editor_camera.x // spacing) * spacing
        world_top = int(self.editor_camera.y // spacing) * spacing
        world_right = int((self.editor_camera.x + CANVAS_W / self.editor_zoom) // spacing + 1) * spacing
        world_bottom = int((self.editor_camera.y + HEIGHT / self.editor_zoom) // spacing + 1) * spacing
        for gx in range(world_left, world_right + 1, spacing):
            sx = int((gx - self.editor_camera.x) * self.editor_zoom)
            pygame.draw.line(self.screen, (32, 89, 52), (sx, 0), (sx, HEIGHT), 1)
        for gy in range(world_top, world_bottom + 1, spacing):
            sy = int((gy - self.editor_camera.y) * self.editor_zoom)
            pygame.draw.line(self.screen, (32, 89, 52), (0, sy), (CANVAS_W, sy), 1)
        screen_points = [(point - self.editor_camera) * self.editor_zoom for point in self.editor_points]
        temp = Track(
            self.editor_points, "Custom", self.editor_kerbs, self.editor_features,
            self.editor_geometry, self.editor_declared_length, self.editor_road_width,
            self.editor_widths, self.editor_pitlane_points,
        ) if screen_points else None
        if temp and len(screen_points) >= 2:
            temp.draw(self.screen, -self.editor_camera * self.editor_zoom, self.editor_zoom)
        pitlane_screen_points = [
            (point - self.editor_camera) * self.editor_zoom
            for point in self.editor_pitlane_points
        ]
        for index, point in enumerate(pitlane_screen_points):
            is_box = index in self.editor_features.get("pit_boxes", [])
            pygame.draw.circle(
                self.screen,
                (168, 85, 247) if is_box else (249, 115, 22),
                point, 6 if is_box else 5,
            )
            pygame.draw.circle(self.screen, WHITE, point, 7, 1)
            self.text(
                f"P{index + 1}", point + Vector2(7, -16),
                "tiny", WHITE,
            )
        if (
            self.editor_width_node is not None
            and 0 <= self.editor_width_node < len(screen_points)
            and len(screen_points) >= 2
        ):
            index = self.editor_width_node
            previous = screen_points[index - 1]
            following = screen_points[(index + 1) % len(screen_points)]
            tangent = following - previous
            if tangent.length():
                normal = Vector2(-tangent.y, tangent.x).normalize()
                half_width = self.editor_widths[index] * self.editor_zoom / 2
                pygame.draw.line(
                    self.screen, YELLOW,
                    screen_points[index] - normal * half_width,
                    screen_points[index] + normal * half_width,
                    2,
                )
        feature_point_colors = {
            "pit_entry": (249, 115, 22), "pit_exit": (34, 197, 94),
        }
        for i, point in enumerate(screen_points):
            point_color = YELLOW if i == self.editor_features.get("start_finish", 0) else (RED if i in self.editor_kerbs else CYAN)
            if i in self.editor_features.get("sectors", []):
                point_color = (77, 171, 247)
            for feature, color in feature_point_colors.items():
                if i == self.editor_features.get(feature):
                    point_color = color
            is_feature = (
                i == self.editor_features.get("start_finish", 0)
                or i in self.editor_features.get("sectors", [])
                or i == self.editor_features.get("pit_entry")
                or i == self.editor_features.get("pit_exit")
            )
            if (
                self.editor_geometry != "sampled"
                or i % 6 == 0
                or is_feature
                or i == self.editor_width_node
            ):
                radius = 6 if self.editor_geometry != "sampled" else (5 if is_feature else 3)
                pygame.draw.circle(self.screen, point_color, point, radius)
                if i == self.editor_width_node:
                    pygame.draw.circle(self.screen, YELLOW, point, radius + 4, 2)
            if (
                self.editor_geometry != "sampled"
                or i % 30 == 0
                or is_feature
                or i == self.editor_width_node
            ):
                self.text(str(i + 1), point + Vector2(6, -15), "small", WHITE)
        self.screen.set_clip(None)

        self.panel("Track Studio", f"World editor • {temp.lap_length_m / 1000:.3f} km" if temp else "World editor")
        self.pill("Points", f"{len(self.editor_points)} / 8", x, 108, 112, YELLOW)
        self.pill("Zoom", f"{self.editor_zoom:.2f}x", x + 122, 108, 114)
        self.text("AUTHORING TOOL", (x, 169), "small", MUTED)
        for key, label in tools:
            rect = tool_rects[key]
            active = key == self.editor_tool
            available = (
                self.editor_pitlane_ready()
                if key == "pitlane"
                else bool(self.editor_pitlane_points)
                if key == "pit_box"
                else True
            )
            pygame.draw.rect(
                self.screen,
                (41, 78, 63) if active else
                (15, 30, 25) if available else (17, 23, 21),
                rect, border_radius=6,
            )
            pygame.draw.rect(
                self.screen,
                CYAN if active else
                (37, 58, 50) if available else (35, 40, 38),
                rect, 1, border_radius=6,
            )
            self.text(
                label, (rect.x + 9, rect.y + 10), "small",
                WHITE if active else MUTED if available else (73, 82, 78),
            )
        self.text("BORDER DISTANCE", (x, 381), "small", MUTED)
        self.text(f"{int(self.editor_features.get('border_margin', BORDER_W))} m", (x, 412), "mono", YELLOW)
        for rect, symbol in ((minus_border, "−"), (plus_border, "+")):
            pygame.draw.rect(self.screen, (28, 51, 43), rect, border_radius=5)
            self.text(symbol, (rect.x + 12, rect.y + 4), "body", WHITE)
        self.text("ALL-NODE WIDTH", (x, 444), "small", MUTED)
        self.text(f"{self.editor_road_width:.1f} m", (x, 466), "mono", YELLOW)
        for rect, symbol in ((minus_road, "−"), (plus_road, "+")):
            pygame.draw.rect(self.screen, (28, 51, 43), rect, border_radius=5)
            self.text(symbol, (rect.x + 12, rect.y + 4), "body", WHITE)
        pygame.draw.rect(self.screen, (14, 28, 23), (x, 510, 236, 126), border_radius=8)
        self.text("PHYSICAL SCALE", (x + 12, 522), "small", CYAN)
        self.text("Hold left on node + wheel", (x + 12, 548), "small", WHITE)
        selected_width = (
            f"Node {self.editor_width_node + 1}: "
            f"{self.editor_widths[self.editor_width_node]:.1f} m"
            if self.editor_width_node is not None
            and self.editor_width_node < len(self.editor_widths)
            else "Select a node to inspect width"
        )
        self.text(selected_width, (x + 12, 573), "small", YELLOW)
        self.text(
            (
                f"PIT ROAD {len(self.editor_pitlane_points)} nodes"
                f" • {len(self.editor_features.get('pit_boxes', []))} boxes"
            ),
            (x + 12, 598), "small",
            (249, 115, 22) if self.editor_pitlane_points else MUTED,
        )
        self.text(
            "80 km/h • PIT IN → PIT OUT",
            (x + 12, 623), "small",
            CYAN if self.editor_pitlane_ready() else MUTED,
        )
        self.footer_hint(("[Esc] Paddock  •  [C] Clear", "[S] Save  •  [Enter] Apply"))

    def advance_training_cars(self, dt=1.0):
        """Advance one training step with the selected interaction rules."""
        collisions = 0
        if self.training_racecraft:
            # Racecraft training exposes traffic but does not choose passes.
            # The user's controller must interpret the sensors and steer.
            self.update_race_awareness(assist_passing=False)
            self.update_slipstreams()
        else:
            for car in self.cars:
                car.slipstream = 0.0
                car.drafting_car = ""
                car.car_ahead = 0.0
                car.car_ahead_distance = 1.0
                car.car_ahead_side = 0.0
                car.closing_speed = 0.0
                car.passing = False
                car.passing_side = 0.0
                car.opponent_data = (0.0,) * 12
                car.opponent_presence = (0.0,) * 3
                car.passing_target = None
                car.race_position = 1
                car.field_size = 1
                car.position_deficit = 0.0
                car.gap_to_leader_m = 0.0
                car.gap_to_next_m = 0.0
                car.race_aggression = 0.0
        for car in self.cars:
            car.update(
                self.track, dt=dt, rain=self.rain_level,
                damage_enabled=False,
            )
        if self.training_racecraft:
            collisions = self.resolve_collisions(apply_damage=False)
            self.update_training_overtakes()
        return collisions

    def update_training_overtakes(self):
        """Reward a clean move from fully behind to fully ahead of a rival."""
        car_by_id = {id(car): car for car in self.cars}
        approach_distance = CAR_LENGTH_M * 5.0
        completion_distance = CAR_LENGTH_M * 0.75
        for follower in self.cars:
            follower.overtake_cooldowns = {
                opponent_id: frames - 1
                for opponent_id, frames
                in follower.overtake_cooldowns.items()
                if frames > 1 and opponent_id in car_by_id
            }
            if (
                not follower.alive
                or follower.finish_time is not None
                or follower.car_collision
            ):
                follower.overtake_candidates.clear()
                continue

            heading = Vector2(1, 0).rotate(follower.angle)
            normal = Vector2(-heading.y, heading.x)
            maximum_lateral = (
                self.track.width_at_segment(follower.track_segment) * 0.72
            )
            for opponent in self.cars:
                if (
                    opponent is follower
                    or not opponent.alive
                    or opponent.finish_time is not None
                ):
                    continue
                opponent_id = id(opponent)
                relative = opponent.position - follower.position
                longitudinal = relative.dot(heading)
                lateral = relative.dot(normal)
                opponent_heading = Vector2(1, 0).rotate(opponent.angle)
                aligned = heading.dot(opponent_heading) >= 0.78

                if opponent_id in follower.overtake_candidates:
                    invalid = (
                        opponent.car_collision
                        or not aligned
                        or abs(lateral) > maximum_lateral
                        or abs(longitudinal) > approach_distance * 1.4
                        or follower.outside_limits
                    )
                    if invalid:
                        follower.overtake_candidates.discard(opponent_id)
                    elif longitudinal <= -completion_distance:
                        follower.overtake_candidates.discard(opponent_id)
                        follower.overtake_cooldowns[
                            opponent_id
                        ] = OVERTAKE_COOLDOWN_FRAMES
                        follower.overtakes += 1
                        follower.overtake_reward += OVERTAKE_REWARD
                        # The immediate addition ensures a pass on the final
                        # generation frame participates in selection.
                        follower.fitness += OVERTAKE_REWARD
                    continue

                follower_speed = follower.velocity.dot(heading)
                opponent_speed = opponent.velocity.dot(heading)
                can_arm = (
                    opponent_id not in follower.overtake_cooldowns
                    and aligned
                    and completion_distance < longitudinal <= approach_distance
                    and abs(lateral) <= maximum_lateral * 0.70
                    and follower_speed > opponent_speed + 0.02
                    and not opponent.car_collision
                    and not follower.outside_limits
                )
                if can_arm:
                    follower.overtake_candidates.add(opponent_id)

    @staticmethod
    def training_control_percentages(car):
        """Return bounded live pedal values for the training telemetry."""
        return (
            int(round(clamp(car.throttle_input, 0.0, 1.0) * 100)),
            int(round(clamp(car.brake_input, 0.0, 1.0) * 100)),
        )

    def training(self, events, dt):
        minus_rect = pygame.Rect(CANVAS_W + 143, 130, 24, 24)
        plus_rect = pygame.Rect(CANVAS_W + 218, 130, 24, 24)
        save_algorithm_rect = pygame.Rect(CANVAS_W + 18, 632, 114, 36)
        save_brain_rect = pygame.Rect(CANVAS_W + 142, 632, 116, 36)
        for event in events:
            if self.handle_camera_zoom_event(event):
                continue
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.mode = "menu"
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key in (pygame.K_EQUALS, pygame.K_KP_PLUS):
                    self.change_population(1)
                elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    self.change_population(-1)
                elif event.key == pygame.K_r:
                    self.reset_training(True)
                elif event.key == pygame.K_s:
                    champion = max(self.cars, key=lambda c: c.fitness)
                    self.best_brain = champion.brain
                    self.open_name_dialog(
                        "brain", "Export trained brain",
                        f"Champion Gen {self.generation:03}",
                        self.best_brain,
                    )
                elif event.key == pygame.K_a:
                    self.open_name_dialog(
                        "algorithm", "Save training algorithm",
                        self.algorithm_path.stem.replace("_", " "),
                        self.algorithm_source,
                    )
                elif event.key == pygame.K_w:
                    self.rain_level = 0.0 if self.rain_level else 1.0
                    self.notice("Training event: wet track" if self.rain_level else "Training event: dry track")
                elif event.key == pygame.K_p and self.cars:
                    max(self.cars, key=lambda c: c.fitness).puncture = True
                    self.notice("Training event: leader puncture")
                elif event.key == pygame.K_LEFTBRACKET:
                    self.adjust_camera_zoom(-1)
                elif event.key == pygame.K_RIGHTBRACKET:
                    self.adjust_camera_zoom(1)
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if minus_rect.collidepoint(event.pos):
                    self.change_population(-1)
                elif plus_rect.collidepoint(event.pos):
                    self.change_population(1)
                elif save_algorithm_rect.collidepoint(event.pos):
                    self.open_name_dialog(
                        "algorithm", "Save training algorithm",
                        self.algorithm_path.stem.replace("_", " "),
                        self.algorithm_source,
                    )
                elif save_brain_rect.collidepoint(event.pos):
                    champion = max(self.cars, key=lambda c: c.fitness)
                    self.best_brain = champion.brain
                    self.open_name_dialog(
                        "brain", "Export trained brain",
                        f"Champion Gen {self.generation:03}",
                        self.best_brain,
                    )
        if not self.paused:
            self.session_time += dt / 1000
            self.advance_training_cars()
            if self.session_time > self.training_duration or not any(c.alive for c in self.cars):
                self.reset_training(True)
        ranked = sorted(self.cars, key=lambda c: c.fitness, reverse=True)
        champion = ranked[0]
        self.follow = self.cars.index(champion)
        self.screen.fill(GRASS)
        self.screen.set_clip(pygame.Rect(0, 0, CANVAS_W, HEIGHT))
        camera_offset, camera_scale = self.camera_transform(champion.position)
        self.track.draw(self.screen, camera_offset, camera_scale)
        for car in reversed(ranked):
            car.draw(self.screen, car is champion, camera_offset, camera_scale)
        self.screen.set_clip(None)
        self.draw_minimap(self.cars, champion)
        self.draw_camera_zoom_control()
        training_mode = (
            "contact/draft • grid"
            if self.training_racecraft
            else "ghost • shared start"
        )
        self.panel(
            "Evolution Lab",
            f"{self.training_generation} • {training_mode}",
        )
        x = CANVAS_W + 18
        self.pill("Generation", f"{self.generation:03}", x, 105, 115)
        pygame.draw.rect(self.screen, (16, 32, 27), (x + 125, 105, 115, 54), border_radius=8)
        self.text("AGENTS", (x + 136, 113), "small", MUTED)
        pygame.draw.rect(self.screen, (31, 55, 46), minus_rect, border_radius=5)
        pygame.draw.rect(self.screen, (31, 55, 46), plus_rect, border_radius=5)
        self.text("−", (minus_rect.x + 7, minus_rect.y + 2), "body", WHITE)
        self.text("+", (plus_rect.x + 6, plus_rect.y + 1), "body", WHITE)
        count_surface = self.fonts["mono"].render(f"{self.population:02}", True, YELLOW)
        self.screen.blit(count_surface, count_surface.get_rect(center=(x + 183, 142)))
        pygame.draw.rect(self.screen, (15, 30, 25), (x, 171, 240, 44), border_radius=7)
        self.text("SESSION", (x + 11, 180), "small", MUTED)
        self.text(f"{self.session_time:05.1f} / {self.training_duration:4.0f}s", (x + 103, 181), "mono", WHITE)
        progress = clamp(self.session_time / self.training_duration, 0, 1)
        pygame.draw.rect(self.screen, (39, 57, 50), (x + 11, 204, 218, 3), border_radius=2)
        pygame.draw.rect(self.screen, CYAN, (x + 11, 204, int(218 * progress), 3), border_radius=2)
        energy = (
            f" • BAT {champion.battery:3.0f}%"
            if champion.generation == "Hybrid" else " • FULL ICE POWER"
        )
        self.text(
            f"{self.track.name[:20]}{energy}",
            (x, 220), "small", CYAN,
        )
        self.text(
            f"{champion.speed_kph:3.0f} KM/H  •  G{champion.gear}/8"
            f"  •  {champion.rpm:05.0f} RPM",
            (x, 243), "small", YELLOW,
        )
        throttle_percent, brake_percent = (
            self.training_control_percentages(champion)
        )
        pygame.draw.rect(
            self.screen, (14, 28, 23), (x, 264, 240, 42),
            border_radius=7,
        )
        for column, label, value, color in (
            (0, "THROTTLE", throttle_percent, (34, 197, 94)),
            (1, "BRAKE", brake_percent, RED),
        ):
            control_x = x + 8 + column * 116
            self.text(
                f"{label} {value:3d}%",
                (control_x, 269), "tiny",
                color if value else MUTED,
            )
            pygame.draw.rect(
                self.screen, (38, 52, 47),
                (control_x, 290, 106, 7), border_radius=4,
            )
            if value:
                pygame.draw.rect(
                    self.screen, color,
                    (control_x, 290, max(2, round(106 * value / 100)), 7),
                    border_radius=4,
                )

        # Full-width steering demographic. Zero remains fixed at the centre;
        # the live steering command grows left or right from that marker.
        steering_value = clamp(champion.steering_input, -1.0, 1.0)
        steering_percent = int(round(steering_value * 100))
        steering_card = pygame.Rect(x, 310, 240, 39)
        steering_bar = pygame.Rect(x + 8, 333, 224, 8)
        steering_centre = steering_bar.centerx
        steering_width = round(
            steering_bar.width / 2 * abs(steering_value)
        )
        pygame.draw.rect(
            self.screen, (14, 28, 23), steering_card, border_radius=7
        )
        self.text("L", (x + 8, 314), "tiny", MUTED)
        self.text("R", (x + 224, 314), "tiny", MUTED)
        steering_label = self.fonts["tiny"].render(
            f"STEERING {steering_percent:+4d}%",
            True,
            CYAN if steering_value else MUTED,
        )
        self.screen.blit(
            steering_label,
            steering_label.get_rect(center=(x + 120, 320)),
        )
        pygame.draw.rect(
            self.screen, (38, 52, 47), steering_bar, border_radius=4
        )
        if steering_width:
            steering_fill = pygame.Rect(
                (
                    steering_centre - steering_width
                    if steering_value < 0 else steering_centre
                ),
                steering_bar.y,
                steering_width,
                steering_bar.height,
            )
            pygame.draw.rect(
                self.screen,
                (249, 115, 22) if steering_value < 0 else CYAN,
                steering_fill,
                border_radius=4,
            )
        pygame.draw.line(
            self.screen, WHITE,
            (steering_centre, steering_bar.y - 2),
            (steering_centre, steering_bar.bottom + 2),
            2,
        )
        self.text(
            "RK / DRIVER      OVT  FITNESS  KM/H G",
            (x, 355), "tiny", MUTED,
        )
        for i, car in enumerate(ranked[:10]):
            y = 370 + i * 24
            selected = car is champion
            pygame.draw.rect(
                self.screen,
                (22, 43, 35) if selected else (12, 25, 21),
                (x, y, 240, 21), border_radius=6,
            )
            pygame.draw.rect(
                self.screen, car.color,
                (x + 7, y + 4, 4, 13), border_radius=2,
            )
            self.text(f"{i+1:>2}", (x + 17, y + 2), "small", MUTED)
            self.text(car.name[:8], (x + 39, y + 2), "small", WHITE)
            self.text(
                f"{car.overtakes:>2}", (x + 103, y + 2), "small",
                YELLOW if car.overtakes else MUTED,
            )
            self.text(
                f"{car.fitness:5.0f}", (x + 126, y + 2), "small",
                CYAN if selected else MUTED,
            )
            self.text(
                f"{car.speed_kph:3.0f} G{car.gear}",
                (x + 181, y + 2), "small",
                YELLOW if selected else MUTED,
            )
        for rect, label, color in (
            (save_algorithm_rect, "SAVE CODE", (30, 57, 47)),
            (save_brain_rect, "SAVE BRAIN", CYAN),
        ):
            pygame.draw.rect(self.screen, color, rect, border_radius=7)
            rendered = self.fonts["small"].render(
                label, True, INK if color == CYAN else WHITE
            )
            self.screen.blit(rendered, rendered.get_rect(center=rect.center))
        state = "PAUSED" if self.paused else "TRAINING"
        self.text(
            f"● {state}", (x, 673), "small",
            YELLOW if self.paused else (84, 220, 140),
        )
        self.footer_hint(("[−/+] Agents  •  Wheel/[ ] Zoom", "[A] Code  •  [S] Brain"))

    def resolve_collisions(self, apply_damage=True):
        collisions = 0
        for i, a in enumerate(self.cars):
            for b in self.cars[i + 1:]:
                manifold = self.car_collision_manifold(a, b)
                if manifold is None:
                    continue
                normal, overlap = manifold
                collisions += 1
                a.car_collision = True
                b.car_collision = True
                # Add clearance and a separating impulse so side-by-side cars
                # do not remain interlocked after the original impact.
                correction = normal * (overlap / 2 + 0.08)
                a.position -= correction
                b.position += correction
                closing_speed = (b.velocity - a.velocity).dot(normal)
                impact = max(0.0, -closing_speed)
                separation_impulse = normal * max(impact * 0.58, 0.025)
                a.velocity -= separation_impulse
                b.velocity += separation_impulse
                if not apply_damage:
                    # Training cars keep full health, but persistent contact
                    # must still lose fitness so avoidance can evolve.
                    contact_penalty = 4.0 + impact * 6.0
                    a.collision_penalty += contact_penalty
                    b.collision_penalty += contact_penalty
                    # Ensure a final-frame contact affects selection before
                    # the next physics update recalculates total fitness.
                    a.fitness -= contact_penalty
                    b.fitness -= contact_penalty
                    a.collision_count += 1
                    b.collision_count += 1
                if impact > 0:
                    if apply_damage:
                        damage = impact * 12.0
                        a.health = max(0.0, a.health - damage)
                        b.health = max(0.0, b.health - damage)
                        a.alive = a.health > 0
                        b.alive = b.health > 0
                if impact > 0.55:
                    self.log_event(
                        "crash", f"{a.name} and {b.name} collided", "high",
                        self.cars.index(a),
                    )
        return collisions

    @staticmethod
    def car_collision_manifold(a, b):
        """Return the minimum separating axis for two physical F1 boxes."""
        delta = b.position - a.position
        if delta.length_squared() > (CAR_LENGTH_M * 1.25) ** 2:
            return None
        a_forward = Vector2(1, 0).rotate(a.angle)
        b_forward = Vector2(1, 0).rotate(b.angle)
        axes = (
            a_forward,
            Vector2(-a_forward.y, a_forward.x),
            b_forward,
            Vector2(-b_forward.y, b_forward.x),
        )
        minimum_overlap = float("inf")
        collision_normal = axes[0]
        for axis in axes:
            axis = axis.normalize()
            a_radius = (
                CAR_LENGTH_M / 2 * abs(a_forward.dot(axis))
                + CAR_WIDTH_M / 2
                * abs(Vector2(-a_forward.y, a_forward.x).dot(axis))
            )
            b_radius = (
                CAR_LENGTH_M / 2 * abs(b_forward.dot(axis))
                + CAR_WIDTH_M / 2
                * abs(Vector2(-b_forward.y, b_forward.x).dot(axis))
            )
            overlap = a_radius + b_radius - abs(delta.dot(axis))
            if overlap <= 0:
                return None
            if overlap < minimum_overlap:
                minimum_overlap = overlap
                collision_normal = axis
        if delta.dot(collision_normal) < 0:
            collision_normal *= -1
        return collision_normal, minimum_overlap

    def update_slipstreams(self):
        """Find cars drafting within three car lengths directly ahead."""
        maximum_distance = CAR_LENGTH_M * 3.0
        minimum_distance = CAR_LENGTH_M * 0.75
        for follower in self.cars:
            follower.slipstream = 0.0
            follower.drafting_car = ""
            if not follower.alive or follower.finish_time is not None:
                continue
            heading = Vector2(1, 0).rotate(follower.angle)
            best_strength = 0.0
            for leader in self.cars:
                if (
                    leader is follower
                    or not leader.alive
                    or leader.finish_time is not None
                ):
                    continue
                relative = leader.position - follower.position
                longitudinal = relative.dot(heading)
                if not minimum_distance < longitudinal <= maximum_distance:
                    continue
                lateral = abs(relative.cross(heading))
                if lateral > CAR_WIDTH_M * 1.3:
                    continue
                leader_heading = Vector2(1, 0).rotate(leader.angle)
                if heading.dot(leader_heading) < 0.94:
                    continue
                strength = clamp(
                    (maximum_distance - longitudinal)
                    / (maximum_distance - minimum_distance),
                    0.0, 1.0,
                )
                strength *= 1.0 - lateral / (CAR_WIDTH_M * 1.3)
                if strength > best_strength:
                    best_strength = strength
                    follower.drafting_car = leader.name
            follower.slipstream = best_strength

    def update_opponent_vectors(self):
        """Publish the three nearest rivals in each car's local coordinates."""
        maximum_squared = OPPONENT_SENSOR_RANGE_M ** 2
        for follower in self.cars:
            heading = Vector2(1, 0).rotate(follower.angle)
            normal = Vector2(-heading.y, heading.x)
            nearby = []
            for opponent in self.cars:
                if (
                    opponent is follower
                    or not opponent.alive
                    or opponent.finish_time is not None
                ):
                    continue
                relative = opponent.position - follower.position
                distance_squared = relative.length_squared()
                if distance_squared > maximum_squared:
                    continue
                relative_velocity = opponent.velocity - follower.velocity
                nearby.append((
                    distance_squared,
                    clamp(
                        relative.dot(heading) / OPPONENT_SENSOR_RANGE_M,
                        -1.0, 1.0,
                    ),
                    clamp(
                        relative.dot(normal) / OPPONENT_SENSOR_RANGE_M,
                        -1.0, 1.0,
                    ),
                    clamp(
                        relative_velocity.dot(heading)
                        / REFERENCE_TOP_SPEED,
                        -1.0, 1.0,
                    ),
                    clamp(
                        relative_velocity.dot(normal)
                        / REFERENCE_TOP_SPEED,
                        -1.0, 1.0,
                    ),
                ))
            nearby.sort(key=lambda item: item[0])
            values = []
            for opponent in nearby[:3]:
                values.extend(opponent[1:])
            values.extend([0.0] * (12 - len(values)))
            follower.opponent_data = tuple(values)
            present_count = min(3, len(nearby))
            follower.opponent_presence = tuple(
                [1.0] * present_count + [0.0] * (3 - present_count)
            )

    def update_race_aggression(self):
        """Increase risk-taking as a car falls down the order or loses touch."""
        active = [
            car for car in self.cars
            if car.alive and car.finish_time is None
        ]
        ranked = sorted(active, key=lambda car: car.score, reverse=True)
        field_size = max(1, len(ranked))
        if not ranked:
            return
        leader_score = ranked[0].score
        lap_length = max(self.track.measured_length_m, 1.0)
        target_laps = max(float(getattr(self, "target_laps", 1)), 1.0)
        for index, car in enumerate(ranked):
            previous = ranked[index - 1] if index else car
            gap_to_leader = max(0.0, leader_score - car.score)
            gap_to_next = (
                max(0.0, previous.score - car.score)
                if index else 0.0
            )
            position_deficit = (
                index / (field_size - 1)
                if field_size > 1 else 0.0
            )
            leader_gap_pressure = clamp(
                gap_to_leader / (lap_length * 0.18),
                0.0, 1.0,
            )
            next_gap_pressure = clamp(
                gap_to_next / (lap_length * 0.045),
                0.0, 1.0,
            )
            close_opportunity = (
                clamp(1.0 - gap_to_next / 65.0, 0.0, 1.0)
                * position_deficit
            )
            race_progress = clamp(
                car.lap / target_laps, 0.0, 1.0
            )
            car.race_position = index + 1
            car.field_size = field_size
            car.position_deficit = position_deficit
            car.gap_to_leader_m = gap_to_leader
            car.gap_to_next_m = gap_to_next
            car.race_aggression = clamp(
                (position_deficit * 0.48)
                + (leader_gap_pressure * 0.22)
                + (next_gap_pressure * 0.15)
                + (close_opportunity * 0.10)
                + (race_progress * position_deficit * 0.05),
                0.0, 1.0,
            )

    def update_race_awareness(self, assist_passing=True):
        """Publish traffic data; optionally provide the legacy passing assist."""
        self.update_race_aggression()
        self.update_opponent_vectors()
        maximum_distance = CAR_LENGTH_M * 5.0
        passing_trigger = CAR_LENGTH_M * 4.0
        for follower in self.cars:
            follower.car_ahead = 0.0
            follower.car_ahead_distance = 1.0
            follower.car_ahead_side = 0.0
            follower.closing_speed = 0.0
            if not follower.alive or follower.finish_time is not None:
                follower.passing = False
                follower.passing_side = 0.0
                follower.passing_target = None
                continue

            if not assist_passing:
                follower.passing = False
                follower.passing_side = 0.0
                follower.passing_target = None

            heading = Vector2(1, 0).rotate(follower.angle)
            normal = Vector2(-heading.y, heading.x)
            track_width = self.track.width_at_segment(
                follower.track_segment
            )

            target = follower.passing_target
            if (
                not any(car is target for car in self.cars)
                or target is follower
                or not getattr(target, "alive", False)
                or target.finish_time is not None
            ):
                target = None

            if target is not None:
                relative = target.position - follower.position
                longitudinal = relative.dot(heading)
                lateral = relative.dot(normal)
                aligned = heading.dot(
                    Vector2(1, 0).rotate(target.angle)
                ) >= 0.82
                if (
                    longitudinal < -CAR_LENGTH_M * 1.25
                    or longitudinal > maximum_distance
                    or abs(lateral) > track_width
                    or not aligned
                ):
                    target = None

            closest = None
            closest_data = None
            if target is None:
                for leader in self.cars:
                    if (
                        leader is follower
                        or not leader.alive
                        or leader.finish_time is not None
                    ):
                        continue
                    relative = leader.position - follower.position
                    longitudinal = relative.dot(heading)
                    if not 0.0 < longitudinal <= maximum_distance:
                        continue
                    lateral = relative.dot(normal)
                    if abs(lateral) > track_width * 0.65:
                        continue
                    leader_heading = Vector2(1, 0).rotate(leader.angle)
                    if heading.dot(leader_heading) < 0.82:
                        continue
                    if (
                        closest_data is None
                        or longitudinal < closest_data[0]
                    ):
                        closest = leader
                        closest_data = (longitudinal, lateral)
                target = closest

            if target is None:
                follower.passing = False
                follower.passing_side = 0.0
                follower.passing_target = None
                continue

            relative = target.position - follower.position
            longitudinal = relative.dot(heading)
            lateral = relative.dot(normal)
            follower.car_ahead = float(longitudinal > 0.0)
            follower.car_ahead_distance = clamp(
                max(0.0, longitudinal) / maximum_distance, 0.0, 1.0
            )
            follower.car_ahead_side = clamp(
                lateral / max(track_width / 2, 1.0), -1.0, 1.0
            )
            follower_speed = follower.velocity.dot(heading)
            leader_speed = target.velocity.dot(heading)
            follower.closing_speed = clamp(
                (follower_speed - leader_speed) / 1.67, -1.0, 1.0
            )

            # In racecraft training these are perception-only signals. The
            # controller must decide whether and where to overtake.
            if not assist_passing:
                continue

            if follower.passing_target is target:
                follower.passing = True
                continue
            if (
                longitudinal <= passing_trigger
                and follower.closing_speed > 0.035
            ):
                sensor_values = follower.sensors(self.track)
                left_room = sensor_values[1] + sensor_values[0] * 0.35
                right_room = sensor_values[3] + sensor_values[4] * 0.35
                for nearby in self.cars:
                    if nearby is follower or nearby is target:
                        continue
                    nearby_relative = nearby.position - follower.position
                    nearby_longitudinal = nearby_relative.dot(heading)
                    if not (
                        -CAR_LENGTH_M * 1.5
                        < nearby_longitudinal
                        < CAR_LENGTH_M * 2.5
                    ):
                        continue
                    nearby_lateral = nearby_relative.dot(normal)
                    occupancy = clamp(
                        1.0 - abs(nearby_longitudinal)
                        / (CAR_LENGTH_M * 2.5),
                        0.0, 1.0,
                    )
                    if nearby_lateral >= 0:
                        right_room -= occupancy
                    else:
                        left_room -= occupancy
                if lateral > CAR_WIDTH_M * 0.40:
                    follower.passing_side = -1.0
                elif lateral < -CAR_WIDTH_M * 0.40:
                    follower.passing_side = 1.0
                else:
                    follower.passing_side = (
                        1.0 if right_room >= left_room else -1.0
                    )
                follower.passing = True
                follower.passing_target = target
            else:
                follower.passing = False
                follower.passing_side = 0.0
                follower.passing_target = None

    @staticmethod
    def format_lap_time(seconds):
        if seconds is None:
            return "--:--.---"
        minutes = int(seconds // 60)
        return f"{minutes}:{seconds - minutes * 60:06.3f}"

    def start_hotlap(self):
        brain = self.load_brain_choice(self.hotlap_brain)
        self.hotlap_car = spawn_car(
            self.track, brain, CYAN, "HOTLAP AI", 0,
        )
        self.hotlap_car.brain_name = self.brain_label(self.hotlap_brain)
        self.hotlap_car.generation = self.hotlap_generation
        self.hotlap_car.battery = (
            100.0 if self.hotlap_generation == "Hybrid" else 0.0
        )
        self.hotlap_car.fuel = 20.0
        self.hotlap_car.tyre = "Soft"
        self.hotlap_car.pit_box_index = 0
        self.cars = [self.hotlap_car]
        self.hotlap_time = 0.0
        self.hotlap_splits = []
        self.hotlap_finished = False
        self.paused = False
        self.rain_level = 0.0
        self.camera_zoom = DEFAULT_CAMERA_ZOOM

    def hotlap_setup(self, events):
        track_rect = pygame.Rect(850, 210, 360, 46)
        brain_rect = pygame.Rect(850, 310, 360, 58)
        previous_rect = pygame.Rect(850, 385, 70, 42)
        next_rect = pygame.Rect(940, 385, 70, 42)
        era_rect = pygame.Rect(850, 445, 360, 42)
        start_rect = pygame.Rect(850, 510, 360, 62)
        for event in events:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.mode = "menu"
                elif event.key in (pygame.K_LEFT, pygame.K_UP):
                    self.hotlap_brain = self.cycle_brain(self.hotlap_brain, -1)
                elif event.key in (pygame.K_RIGHT, pygame.K_DOWN):
                    self.hotlap_brain = self.cycle_brain(self.hotlap_brain, 1)
                elif event.key in (pygame.K_RETURN, pygame.K_SPACE):
                    self.start_hotlap()
                    self.mode = "hotlap"
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if track_rect.collidepoint(event.pos):
                    self.cycle_track(
                        -1 if event.pos[0] < track_rect.centerx else 1
                    )
                elif brain_rect.collidepoint(event.pos) or next_rect.collidepoint(event.pos):
                    self.hotlap_brain = self.cycle_brain(self.hotlap_brain, 1)
                elif previous_rect.collidepoint(event.pos):
                    self.hotlap_brain = self.cycle_brain(self.hotlap_brain, -1)
                elif era_rect.collidepoint(event.pos):
                    self.hotlap_generation = (
                        "ICE"
                        if self.hotlap_generation == "Hybrid"
                        else "Hybrid"
                    )
                elif start_rect.collidepoint(event.pos):
                    self.start_hotlap()
                    self.mode = "hotlap"

        self.draw_app_background()
        self.text("TWO-LAP HOTLAP", (55, 35), "small", CYAN)
        self.text("One brain. One circuit. Two timed laps.", (55, 62), "title")
        self.text(
            "The clock starts from rest and stops when lap two is completed.",
            (58, 119), "body", MUTED,
        )
        preview = pygame.Rect(55, 165, 740, 500)
        self.glass_card(preview, accent=YELLOW, radius=18)
        self.track.draw_preview(self.screen, preview.inflate(-40, -40))
        self.text(self.track.name, (82, 190), "h2", YELLOW)
        self.text(
            f"{self.track.lap_length_m / 1000:.3f} km  •  2 laps",
            (84, 224), "small", WHITE,
        )
        self.glass_card(
            pygame.Rect(825, 165, 410, 500),
            accent=CYAN,
            radius=18,
        )
        self.text("CIRCUIT", (850, 185), "small", MUTED)
        pygame.draw.rect(
            self.screen, UI_SURFACE_HOVER, track_rect, border_radius=10
        )
        pygame.draw.rect(
            self.screen, UI_BORDER, track_rect, 1, border_radius=10
        )
        self.text(
            f"‹  {self.track_label()[:26]}  ›",
            (track_rect.x + 18, track_rect.y + 14), "mono", CYAN,
        )
        self.text("AI BRAIN", (850, 270), "small", MUTED)
        pygame.draw.rect(
            self.screen, UI_SURFACE_HOVER, brain_rect, border_radius=10
        )
        pygame.draw.rect(
            self.screen, UI_BORDER, brain_rect, 1, border_radius=10
        )
        self.text(
            self.brain_label(self.hotlap_brain)[:30],
            (brain_rect.x + 18, brain_rect.y + 18), "mono", WHITE,
        )
        for rect, label in ((previous_rect, "‹"), (next_rect, "›")):
            pygame.draw.rect(
                self.screen, UI_SURFACE_HOVER, rect, border_radius=9
            )
            rendered = self.fonts["h2"].render(label, True, CYAN)
            self.screen.blit(rendered, rendered.get_rect(center=rect.center))
        self.text(
            f"{len(self.brain_choices())} selectable brain source(s)",
            (1030, 398), "small", MUTED,
        )
        pygame.draw.rect(
            self.screen, UI_SURFACE_HOVER, era_rect, border_radius=10
        )
        self.text("POWERTRAIN", (era_rect.x + 16, era_rect.y + 13), "small", MUTED)
        self.text(
            f"‹  {self.hotlap_generation.upper()}  ›",
            (era_rect.x + 205, era_rect.y + 12), "small", YELLOW,
        )
        pygame.draw.rect(
            self.screen, UI_SHADOW, start_rect.move(0, 5),
            border_radius=11,
        )
        pygame.draw.rect(self.screen, CYAN, start_rect, border_radius=11)
        label = self.fonts["mono"].render("START TWO-LAP RUN", True, INK)
        self.screen.blit(label, label.get_rect(center=start_rect.center))
        self.text(
            "Soft tyres • 20 kg fuel • dry circuit",
            (850, 595), "small", MUTED,
        )
        self.text(
            "[Esc] Back  •  Click circuit  •  [←/→] Choose brain",
            (55, 704), "small", MUTED,
        )

    def advance_hotlap(self, dt):
        car = self.hotlap_car
        if car is None or self.paused or self.hotlap_finished:
            return
        self.hotlap_time += dt / 1000
        previous_lap = car.lap
        car.slipstream = 0.0
        car.drafting_car = ""
        car.update(self.track, rain=0.0)
        if car.lap > previous_lap:
            previous_total = sum(self.hotlap_splits)
            self.hotlap_splits.append(self.hotlap_time - previous_total)
        if car.lap >= 2:
            self.hotlap_finished = True
            car.finish_time = self.hotlap_time
            car.velocity.update(0, 0)
        elif not car.alive:
            self.hotlap_finished = True

    def hotlap(self, events, dt):
        for event in events:
            if self.handle_camera_zoom_event(event):
                continue
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.mode = "menu"
                elif event.key == pygame.K_SPACE and not self.hotlap_finished:
                    self.paused = not self.paused
                elif event.key == pygame.K_r:
                    self.start_hotlap()
                elif event.key == pygame.K_LEFTBRACKET:
                    self.adjust_camera_zoom(-1)
                elif event.key == pygame.K_RIGHTBRACKET:
                    self.adjust_camera_zoom(1)

        car = self.hotlap_car
        if car is None:
            self.start_hotlap()
            car = self.hotlap_car
        self.advance_hotlap(dt)

        self.screen.fill(GRASS)
        self.screen.set_clip(pygame.Rect(0, 0, CANVAS_W, HEIGHT))
        camera_offset, camera_scale = self.camera_transform(car.position)
        self.track.draw(self.screen, camera_offset, camera_scale)
        car.draw(self.screen, True, camera_offset, camera_scale)
        self.screen.set_clip(None)
        self.draw_minimap([car], car)
        self.draw_camera_zoom_control()

        self.panel(
            "Hotlap Complete" if self.hotlap_finished else "Two-Lap Hotlap",
            car.brain_name,
        )
        x = CANVAS_W + 18
        self.pill("Lap", f"{min(car.lap + 1, 2)} / 2", x, 108, 112, YELLOW)
        self.pill("Timer", self.format_lap_time(self.hotlap_time), x + 122, 108, 118)
        self.text("LAP TIMES", (x, 190), "small", MUTED)
        for index in range(2):
            y = 220 + index * 60
            pygame.draw.rect(self.screen, (15, 31, 26), (x, y, 240, 48), border_radius=7)
            self.text(f"LAP {index + 1}", (x + 12, y + 15), "small", WHITE)
            value = (
                self.format_lap_time(self.hotlap_splits[index])
                if index < len(self.hotlap_splits) else "--:--.---"
            )
            self.text(value, (x + 108, y + 14), "mono", CYAN)
        self.text("CAR STATE", (x, 360), "small", MUTED)
        self.text(f"Fuel       {car.fuel:5.1f} kg", (x, 392), "mono", WHITE)
        self.text(f"Tyre wear  {car.tyre_wear:5.1f} %", (x, 422), "mono", WHITE)
        battery = (
            f"{car.battery:5.1f}% {'OT' if car.overtake_active else ''}"
            if car.generation == "Hybrid" else "  ICE"
        )
        self.text(f"Battery    {battery}", (x, 452), "mono", CYAN)
        self.text(f"Limits     {car.track_limits:5}", (x, 482), "mono", WHITE)
        if self.hotlap_finished:
            pygame.draw.rect(self.screen, (24, 56, 46), (x, 510, 240, 92), border_radius=9)
            result = (
                self.format_lap_time(self.hotlap_time)
                if car.alive else "DNF"
            )
            self.text("TWO-LAP TIME", (x + 14, 526), "small", YELLOW)
            self.text(result, (x + 14, 557), "h2", WHITE)
        state = "PAUSED" if self.paused else ("FINISHED" if self.hotlap_finished else "TIMED RUN")
        self.text(f"● {state}", (x, 640), "small", YELLOW if self.paused else CYAN)
        self.footer_hint(("[R] Restart  •  Wheel/[ ] Zoom", "[Space] Pause  •  [Esc] Back"))

    def race_setup(self, events):
        """Configure the complete grid before a race can be launched."""
        count = self.race_settings["cars"]
        setup = {
            "track": pygame.Rect(775, 82, 450, 38),
            "cars_minus": pygame.Rect(894, 151, 34, 34),
            "cars_plus": pygame.Rect(1050, 151, 34, 34),
            "laps_minus": pygame.Rect(894, 202, 34, 34),
            "laps_plus": pygame.Rect(1050, 202, 34, 34),
            "weather": pygame.Rect(894, 253, 190, 38),
            "generation": pygame.Rect(894, 303, 190, 38),
            "teams": pygame.Rect(894, 353, 190, 38),
            "name": pygame.Rect(808, 448, 276, 38),
            "team_name": pygame.Rect(1095, 448, 130, 38),
            "tyre": pygame.Rect(808, 505, 130, 38),
            "fuel_minus": pygame.Rect(956, 505, 34, 38),
            "fuel_plus": pygame.Rect(1050, 505, 34, 38),
            "color": pygame.Rect(1095, 505, 130, 38),
            "grid_up": pygame.Rect(995, 562, 105, 38),
            "grid_down": pygame.Rect(1110, 562, 115, 38),
            "brain": pygame.Rect(808, 620, 417, 38),
            "start": pygame.Rect(930, 675, 295, 55),
        }
        roster_rects = []
        for i in range(count):
            column, row = i // 10, i % 10
            roster_rects.append(pygame.Rect(52 + column * 350, 169 + row * 48, 328, 40))

        for event in events:
            if event.type == pygame.TEXTINPUT and self.editing_name:
                if self.editing_name == "driver":
                    self.replace_race_name_selection(event.text)
                else:
                    index = self.selected_entry // 2
                    self.team_names[index] = (self.team_names[index] + event.text)[:16]
            elif event.type == pygame.KEYDOWN:
                command = bool(event.mod & (pygame.KMOD_CTRL | pygame.KMOD_META))
                shift = bool(event.mod & pygame.KMOD_SHIFT)
                if self.editing_name == "driver":
                    selection = self.race_name_selection()
                    if command and event.key == pygame.K_a:
                        self.race_name_anchor = 0
                        self.race_name_cursor = len(self.race_name_value)
                    elif command and event.key == pygame.K_c and selection:
                        self.editor_set_clipboard(
                            self.race_name_value[selection[0]:selection[1]]
                        )
                    elif command and event.key == pygame.K_x and selection:
                        self.editor_set_clipboard(
                            self.race_name_value[selection[0]:selection[1]]
                        )
                        self.replace_race_name_selection("")
                    elif command and event.key == pygame.K_v:
                        self.replace_race_name_selection(
                            self.editor_get_clipboard()
                        )
                    elif event.key == pygame.K_ESCAPE:
                        self.finish_race_name_edit(False)
                    elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                        self.finish_race_name_edit(True)
                    elif event.key == pygame.K_BACKSPACE:
                        if selection:
                            self.replace_race_name_selection("")
                        elif self.race_name_cursor > 0:
                            self.race_name_anchor = self.race_name_cursor - 1
                            self.replace_race_name_selection("")
                    elif event.key == pygame.K_DELETE:
                        if selection:
                            self.replace_race_name_selection("")
                        elif self.race_name_cursor < len(self.race_name_value):
                            self.race_name_anchor = self.race_name_cursor + 1
                            self.replace_race_name_selection("")
                    elif event.key in (pygame.K_LEFT, pygame.K_RIGHT):
                        direction = -1 if event.key == pygame.K_LEFT else 1
                        if selection and not shift:
                            target = (
                                selection[0] if direction < 0 else selection[1]
                            )
                        else:
                            target = clamp(
                                self.race_name_cursor + direction,
                                0, len(self.race_name_value),
                            )
                        if shift and self.race_name_anchor is None:
                            self.race_name_anchor = self.race_name_cursor
                        elif not shift:
                            self.race_name_anchor = None
                        self.race_name_cursor = target
                    elif event.key in (pygame.K_HOME, pygame.K_END):
                        target = (
                            0 if event.key == pygame.K_HOME
                            else len(self.race_name_value)
                        )
                        if shift and self.race_name_anchor is None:
                            self.race_name_anchor = self.race_name_cursor
                        elif not shift:
                            self.race_name_anchor = None
                        self.race_name_cursor = target
                elif event.key == pygame.K_ESCAPE:
                    if self.editing_name == "team":
                        self.editing_name = None
                        pygame.key.stop_text_input()
                    else:
                        self.mode = "menu"
                elif (
                    event.key == pygame.K_BACKSPACE
                    and self.editing_name == "team"
                ):
                    index = self.selected_entry // 2
                    self.team_names[index] = self.team_names[index][:-1]
                elif (
                    event.key in (pygame.K_RETURN, pygame.K_KP_ENTER)
                    and self.editing_name == "team"
                ):
                    self.editing_name = None
                    pygame.key.stop_text_input()
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                roster_hit = next((i for i, rect in enumerate(roster_rects) if rect.collidepoint(event.pos)), None)
                entry = self.race_entries[self.selected_entry]
                if setup["name"].collidepoint(event.pos):
                    if (
                        self.editing_name == "driver"
                        and self.race_name_target is entry
                    ):
                        char_width = max(
                            self.fonts["mono"].size("M")[0], 1
                        )
                        self.race_name_cursor = clamp(
                            round(
                                (event.pos[0] - setup["name"].x - 12)
                                / char_width
                            ),
                            0, len(self.race_name_value),
                        )
                        self.race_name_anchor = None
                    else:
                        if self.editing_name == "driver":
                            self.finish_race_name_edit(True)
                        self.start_race_name_edit(entry)
                elif roster_hit is not None:
                    if self.editing_name == "driver":
                        self.finish_race_name_edit(True)
                    elif self.editing_name == "team":
                        pygame.key.stop_text_input()
                    self.selected_entry = roster_hit
                    self.editing_name = None
                    if getattr(event, "clicks", 1) >= 2:
                        self.start_race_name_edit(
                            self.race_entries[self.selected_entry]
                        )
                else:
                    if self.editing_name == "driver":
                        self.finish_race_name_edit(True)
                    elif self.editing_name == "team":
                        self.editing_name = None
                        pygame.key.stop_text_input()
                if setup["name"].collidepoint(event.pos) or roster_hit is not None:
                    continue
                if setup["track"].collidepoint(event.pos):
                    self.cycle_track(
                        -1 if event.pos[0] < setup["track"].centerx else 1
                    )
                elif setup["cars_minus"].collidepoint(event.pos):
                    self.race_settings["cars"] = max(2, count - 1)
                    self.selected_entry = min(self.selected_entry, self.race_settings["cars"] - 1)
                elif setup["cars_plus"].collidepoint(event.pos):
                    self.race_settings["cars"] = min(20, count + 1)
                elif setup["laps_minus"].collidepoint(event.pos):
                    self.race_settings["laps"] = max(1, self.race_settings["laps"] - 1)
                elif setup["laps_plus"].collidepoint(event.pos):
                    self.race_settings["laps"] = min(50, self.race_settings["laps"] + 1)
                elif setup["weather"].collidepoint(event.pos):
                    choices = ["Dry", "Wet", "Changing"]
                    self.race_settings["weather"] = choices[(choices.index(self.race_settings["weather"]) + 1) % len(choices)]
                elif setup["generation"].collidepoint(event.pos):
                    self.race_settings["generation"] = "ICE" if self.race_settings["generation"] == "Hybrid" else "Hybrid"
                elif setup["teams"].collidepoint(event.pos):
                    self.race_settings["teams"] = not self.race_settings["teams"]
                elif setup["team_name"].collidepoint(event.pos) and self.race_settings["teams"]:
                    self.editing_name = "team"
                    pygame.key.start_text_input()
                elif setup["tyre"].collidepoint(event.pos):
                    entry = self.race_entries[self.selected_entry]
                    tyres = ["Soft", "Medium", "Hard", "Wet"]
                    entry["tyre"] = tyres[(tyres.index(entry["tyre"]) + 1) % len(tyres)]
                elif setup["fuel_minus"].collidepoint(event.pos):
                    self.race_entries[self.selected_entry]["fuel"] = max(5, self.race_entries[self.selected_entry]["fuel"] - 5)
                elif setup["fuel_plus"].collidepoint(event.pos):
                    self.race_entries[self.selected_entry]["fuel"] = min(110, self.race_entries[self.selected_entry]["fuel"] + 5)
                elif setup["color"].collidepoint(event.pos):
                    entry = self.race_entries[self.selected_entry]
                    entry["color"] = (entry["color"] + 1) % len(COLORS)
                elif setup["brain"].collidepoint(event.pos):
                    entry = self.race_entries[self.selected_entry]
                    entry["brain"] = self.cycle_brain(
                        entry.get("brain", "__session__")
                    )
                elif setup["grid_up"].collidepoint(event.pos) and self.selected_entry > 0:
                    i = self.selected_entry
                    self.race_entries[i - 1], self.race_entries[i] = self.race_entries[i], self.race_entries[i - 1]
                    self.selected_entry -= 1
                elif setup["grid_down"].collidepoint(event.pos) and self.selected_entry < count - 1:
                    i = self.selected_entry
                    self.race_entries[i + 1], self.race_entries[i] = self.race_entries[i], self.race_entries[i + 1]
                    self.selected_entry += 1
                elif setup["start"].collidepoint(event.pos):
                    trained = bool(list(BRAIN_DIR.glob("*.json"))) or self.generation > 0
                    if trained:
                        self.start_race()
                        self.mode = "race"
                    else:
                        self.notice("Train at least one generation or export a champion first")

        self.draw_app_background()
        self.text("RACE WEEKEND", (52, 30), "small", CYAN)
        self.text("Build the starting grid", (52, 57), "title")
        self.text(
            "Click to configure • double-click a driver to rename",
            (55, 112), "body", MUTED,
        )
        pygame.draw.rect(
            self.screen, UI_SURFACE_RAISED,
            setup["track"], border_radius=9,
        )
        pygame.draw.rect(
            self.screen, UI_BORDER,
            setup["track"], 1, border_radius=9,
        )
        self.text("CIRCUIT", (setup["track"].x + 14, setup["track"].y + 11), "small", MUTED)
        self.text(
            f"‹  {self.track_label()[:25]}  ›",
            (setup["track"].x + 105, setup["track"].y + 10), "small", CYAN,
        )
        self.text("CLICK TO CHANGE", (setup["track"].right - 127, setup["track"].y + 11), "small", MUTED)
        self.text("STARTING GRID", (52, 145), "small", YELLOW)
        for i, rect in enumerate(roster_rects):
            entry = self.race_entries[i]
            selected = i == self.selected_entry
            pygame.draw.rect(
                self.screen,
                UI_SURFACE_HOVER if selected else UI_SURFACE,
                rect,
                border_radius=9,
            )
            pygame.draw.rect(
                self.screen,
                COLORS[entry["color"]] if selected else UI_BORDER,
                rect,
                1,
                border_radius=9,
            )
            pygame.draw.rect(self.screen, COLORS[entry["color"]], (rect.x + 7, rect.y + 7, 5, 26), border_radius=2)
            self.text(f"{i+1:>2}", (rect.x + 20, rect.y + 10), "mono", MUTED)
            display_name = (
                self.race_name_value
                if self.editing_name == "driver"
                and self.race_name_target is entry
                else entry["name"]
            )
            self.text(display_name, (rect.x + 54, rect.y + 9), "mono", WHITE)
            self.text(entry["tyre"][0], (rect.right - 55, rect.y + 10), "mono", YELLOW)

        self.glass_card(
            pygame.Rect(775, 132, 450, 278),
            accent=CYAN,
            radius=16,
        )
        self.text("SESSION", (800, 151), "small", YELLOW)
        for label, y, value, minus, plus in (
            ("CARS", 168, str(self.race_settings["cars"]), "cars_minus", "cars_plus"),
            ("LAPS", 219, str(self.race_settings["laps"]), "laps_minus", "laps_plus"),
        ):
            self.text(label, (800, y), "small", MUTED)
            for key, symbol in ((minus, "−"), (plus, "+")):
                pygame.draw.rect(self.screen, (31, 55, 46), setup[key], border_radius=5)
                self.text(symbol, (setup[key].x + 10, setup[key].y + 4), "body", WHITE)
            self.text(value, (973, y - 1), "mono", WHITE)
        for label, key, y in (
            ("WEATHER", "weather", 253), ("ERA", "generation", 303), ("TEAMS", "teams", 353),
        ):
            self.text(label, (800, y + 11), "small", MUTED)
            pygame.draw.rect(self.screen, (27, 52, 43), setup[key], border_radius=6)
            value = self.race_settings[key]
            if isinstance(value, bool):
                value = "PAIRED" if value else "INDIVIDUAL"
            self.text(value.upper(), (setup[key].x + 13, setup[key].y + 10), "small", CYAN)

        entry = self.race_entries[self.selected_entry]
        self.text(f"CAR {self.selected_entry + 1:02} CONFIGURATION", (775, 427), "small", YELLOW)
        for key in ("name", "team_name", "tyre", "color"):
            pygame.draw.rect(self.screen, (15, 31, 26), setup[key], border_radius=6)
        name_editing = (
            self.editing_name == "driver"
            and self.race_name_target is entry
        )
        if name_editing:
            pygame.draw.rect(
                self.screen, CYAN, setup["name"], 2, border_radius=6
            )
            selection = self.race_name_selection()
            char_width = max(self.fonts["mono"].size("M")[0], 1)
            if selection:
                pygame.draw.rect(
                    self.screen, (32, 91, 103),
                    (
                        setup["name"].x + 12 + selection[0] * char_width,
                        setup["name"].y + 8,
                        max(
                            char_width,
                            (selection[1] - selection[0]) * char_width,
                        ),
                        23,
                    ),
                    border_radius=2,
                )
            name_value = self.race_name_value
        else:
            name_value = entry["name"] or "UNNAMED"
        self.text(name_value, (setup["name"].x + 12, 459), "mono", WHITE)
        if name_editing and pygame.time.get_ticks() % 1000 < 550:
            caret_x = (
                setup["name"].x + 12
                + self.race_name_cursor
                * max(self.fonts["mono"].size("M")[0], 1)
            )
            pygame.draw.line(
                self.screen, YELLOW,
                (caret_x, setup["name"].y + 7),
                (caret_x, setup["name"].bottom - 7),
                2,
            )
        self.text(
            "ENTER CONFIRMS • ESC CANCELS"
            if name_editing else "CLICK NAME TO RENAME",
            (setup["name"].x + 3, 488), "small",
            CYAN if name_editing else MUTED,
        )
        team_value = self.team_names[self.selected_entry // 2] if self.race_settings["teams"] else "NO TEAM"
        self.text(team_value[:13], (setup["team_name"].x + 8, 460), "small", MUTED if not self.race_settings["teams"] else WHITE)
        self.text(entry["tyre"].upper(), (setup["tyre"].x + 12, 516), "small", YELLOW)
        self.text(f"{entry['fuel']:3}kg", (997, 516), "small", WHITE)
        for key, symbol in (("fuel_minus", "−"), ("fuel_plus", "+")):
            pygame.draw.rect(self.screen, (31, 55, 46), setup[key], border_radius=5)
            self.text(symbol, (setup[key].x + 10, setup[key].y + 5), "body", WHITE)
        pygame.draw.rect(self.screen, COLORS[entry["color"]], (setup["color"].x + 10, setup["color"].y + 9, 110, 20), border_radius=5)
        self.text("PIT: CONTROLLER CODE", (808, 573), "small", CYAN)
        for key, label in (("grid_up", "GRID UP"), ("grid_down", "GRID DOWN")):
            pygame.draw.rect(self.screen, (23, 44, 37), setup[key], border_radius=6)
            self.text(label, (setup[key].x + 15, setup[key].y + 11), "small", MUTED)
        pygame.draw.rect(self.screen, (15, 31, 26), setup["brain"], border_radius=6)
        self.text("AI BRAIN", (setup["brain"].x + 12, setup["brain"].y + 11), "small", MUTED)
        self.text(
            self.brain_label(entry.get("brain", "__session__"))[:25],
            (setup["brain"].x + 105, setup["brain"].y + 10),
            "small", CYAN,
        )
        self.text("CLICK TO CHANGE", (setup["brain"].right - 125, setup["brain"].y + 11), "small", MUTED)

        trained = bool(list(BRAIN_DIR.glob("*.json"))) or self.generation > 0
        pygame.draw.rect(self.screen, CYAN if trained else (56, 76, 69), setup["start"], border_radius=9)
        self.text("START RACE" if trained else "TRAINED AI REQUIRED", (978 if trained else 962, 694), "mono", INK if trained else MUTED)
        self.text("[Esc] Back to paddock", (52, 704), "small", MUTED)

    def race(self, events, dt):
        for event in events:
            if self.handle_camera_zoom_event(event):
                continue
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.mode = "menu"
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key in (pygame.K_LEFT, pygame.K_RIGHT):
                    self.metric = (
                        self.metric
                        + (1 if event.key == pygame.K_RIGHT else -1)
                    ) % 9
                elif event.key in (pygame.K_r, pygame.K_s):
                    self.save_replay()
                elif event.key == pygame.K_y:
                    self.flag_state = "YELLOW"
                    self.flag_until = pygame.time.get_ticks() + 10000
                    self.log_event("yellow", "Yellow flag deployed", "high")
                elif event.key == pygame.K_c:
                    self.flag_state = "SAFETY CAR"
                    self.flag_until = pygame.time.get_ticks() + 20000
                    self.log_event("safety_car", "Safety car phase started", "high")
                elif event.key == pygame.K_w and self.race_settings["weather"] == "Changing":
                    self.rain_level = 0.0 if self.rain_level else 1.0
                    self.log_event("weather", "Manual weather event triggered", "medium")
                elif event.key == pygame.K_p:
                    self.cars[self.follow].puncture = True
                    self.log_event("puncture", f"{self.cars[self.follow].name} punctured", "high", self.follow)
                elif event.key == pygame.K_LEFTBRACKET:
                    self.adjust_camera_zoom(-1)
                elif event.key == pygame.K_RIGHTBRACKET:
                    self.adjust_camera_zoom(1)
            elif event.type == pygame.MOUSEBUTTONDOWN and event.pos[0] >= CANVAS_W:
                if 94 <= event.pos[1] <= 137:
                    self.event_camera = not self.event_camera
                    self.camera_until = 0
                else:
                    ranked = sorted(
                        self.cars, key=lambda c: c.score, reverse=True
                    )
                    row = self.race_tower_row(
                        event.pos[1], len(ranked)
                    )
                    if row is not None:
                        self.follow = self.cars.index(ranked[row])
                        self.event_camera = False
        if not self.paused and self.advance_race_countdown(dt):
            pass
        elif not self.paused:
            self.race_lights_out_flash = max(
                0.0, self.race_lights_out_flash - dt / 1000
            )
            self.session_time += dt / 1000
            if self.flag_state != "GREEN" and pygame.time.get_ticks() >= self.flag_until:
                self.flag_state = "GREEN"
                self.log_event("green", "Track returned to green", "medium")
            leader_lap = max(car.lap for car in self.cars)
            if (
                self.race_settings["weather"] == "Changing"
                and leader_lap > 0 and leader_lap % 2 == 0
                and leader_lap != self.last_weather_lap
            ):
                self.last_weather_lap = leader_lap
                self.rain_level = 1.0 if random.random() <= self.weather_forecast else 0.0
                state = "wet" if self.rain_level else "dry"
                self.log_event("weather", f"Conditions changed to {state}", "medium")
                self.weather_forecast = random.uniform(.15, .85)
                self.weather_popup_until = pygame.time.get_ticks() + 5000
            self.update_race_awareness()
            self.update_slipstreams()
            for car in self.cars:
                if car.finish_time is None:
                    was_punctured = car.puncture
                    old_stops = car.pitstops
                    car.update(self.track, rain=self.rain_level)
                    if car.puncture and not was_punctured:
                        self.log_event("puncture", f"{car.name} suffered a puncture", "high", self.cars.index(car))
                    if car.pitstops > old_stops:
                        self.log_event("pitstop", f"{car.name} completed a pit stop", "medium", self.cars.index(car))
                    speed_cap = 3.5 if self.flag_state == "SAFETY CAR" else (5.8 if self.flag_state == "YELLOW" else None)
                    if speed_cap and car.velocity.length() > speed_cap:
                        car.velocity.scale_to_length(speed_cap)
                    if car.lap >= self.target_laps:
                        car.finish_time = self.session_time
                        self.log_event("finish", f"{car.name} finished", "medium")
            self.resolve_collisions()
            self.replay_tick += 1
            if self.replay_tick % 6 == 0:
                self.capture_replay_frame()
            if self.event_camera and pygame.time.get_ticks() >= self.camera_until:
                self.follow = random.randrange(len(self.cars))
                self.camera_until = pygame.time.get_ticks() + 30000
        ranked = sorted(self.cars, key=lambda c: (c.finish_time is not None, c.score), reverse=True)
        if all(c.finish_time is not None or not c.alive for c in self.cars):
            self.draw_results(ranked)
            return
        self.screen.fill(GRASS)
        self.screen.set_clip(pygame.Rect(0, 0, CANVAS_W, HEIGHT))
        focused_car = self.cars[self.follow]
        camera_offset, camera_scale = self.camera_transform(focused_car.position)
        self.track.draw(self.screen, camera_offset, camera_scale)
        for car in reversed(ranked):
            car.draw(
                self.screen, self.cars.index(car) == self.follow,
                camera_offset, camera_scale,
            )
        self.screen.set_clip(None)
        self.draw_minimap(self.cars, focused_car)
        self.draw_camera_zoom_control()
        self.draw_tower(ranked)
        if pygame.time.get_ticks() < self.weather_popup_until:
            popup = pygame.Rect(285, 30, 430, 82)
            pygame.draw.rect(self.screen, (8, 20, 25), popup, border_radius=10)
            pygame.draw.rect(self.screen, CYAN, popup, 2, border_radius=10)
            self.text("NEXT WEATHER WINDOW", (popup.x + 20, popup.y + 15), "small", CYAN)
            self.text(
                f"WET {self.weather_forecast*100:02.0f}%   •   DRY {(1-self.weather_forecast)*100:02.0f}%",
                (popup.x + 20, popup.y + 42), "h2", WHITE,
            )
        if self.event_log:
            event = self.event_log[-1]
            self.text(f"{event['type'].upper()}  {event['message']}", (20, HEIGHT - 30), "small", WHITE)
        if self.race_countdown > 0 or self.race_lights_out_flash > 0:
            self.draw_start_lights()

    def draw_start_lights(self):
        card = pygame.Rect(CANVAS_W // 2 - 235, 34, 470, 148)
        shadow = pygame.Surface(card.size, pygame.SRCALPHA)
        shadow.fill((4, 8, 7, 224))
        self.screen.blit(shadow, card)
        pygame.draw.rect(self.screen, (62, 72, 68), card, 2, border_radius=14)
        active = self.race_countdown > 0
        lit_count = (
            min(5, max(1, int(5.0 - self.race_countdown) + 1))
            if active else 0
        )
        title = (
            f"RACE START  •  {max(1, math.ceil(self.race_countdown))}"
            if active else "LIGHTS OUT"
        )
        title_color = WHITE if active else CYAN
        label = self.fonts["h2"].render(title, True, title_color)
        self.screen.blit(label, label.get_rect(center=(card.centerx, card.y + 29)))
        start_x = card.centerx - 164
        for index in range(5):
            pod = pygame.Rect(start_x + index * 82, card.y + 55, 66, 72)
            pygame.draw.rect(
                self.screen, (10, 13, 13), pod, border_radius=13
            )
            pygame.draw.rect(
                self.screen, (48, 55, 52), pod, 2, border_radius=13
            )
            light_color = (
                (255, 35, 45)
                if index < lit_count
                else ((25, 45, 39) if not active else (72, 18, 23))
            )
            pygame.draw.circle(
                self.screen, light_color, pod.center, 21
            )
            if index < lit_count:
                pygame.draw.circle(
                    self.screen, (255, 126, 116), pod.center, 11
                )

    @staticmethod
    def race_tower_row(mouse_y, car_count):
        """Return the integer timetable row under a logical mouse position."""
        row = int((float(mouse_y) - 190.0) // 48.0)
        return row if 0 <= row < min(10, car_count) else None

    def draw_tower(self, ranked):
        x = CANVAS_W
        camera_label = "AUTO CAM" if self.event_camera else "MANUAL CAM"
        self.panel("Race Control", f"{self.flag_state}  •  {camera_label}")
        self.text(f"LAP {min(max(c.lap for c in self.cars)+1, self.target_laps)} / {self.target_laps}", (x + 20, 101), "h2")
        self.text(f"{self.session_time:07.2f}", (x + 178, 108), "mono", YELLOW)
        labels = (
            "INTERVAL", "GAP TO LEADER", "TYRE / AGE",
            "PIT STOPS", "CONDITION", "BATTERY / OVERTAKE",
            "FUEL / PIT CALL", "SPEED / GEAR / RPM",
            "AGGRESSION / RISK",
        )
        metric_card = pygame.Rect(x + 12, 143, PANEL - 24, 34)
        pygame.draw.rect(
            self.screen, UI_SURFACE_RAISED,
            metric_card, border_radius=9,
        )
        pygame.draw.rect(
            self.screen, UI_BORDER,
            metric_card, 1, border_radius=9,
        )
        metric_text = self.fonts["small"].render(f"‹   {labels[self.metric]}   ›", True, CYAN)
        self.screen.blit(metric_text, metric_text.get_rect(center=(x + PANEL // 2, 160)))
        leader = ranked[0]
        for i, car in enumerate(ranked[:10]):
            y = 190 + i * 48
            focused = self.cars.index(car) == self.follow
            row_rect = pygame.Rect(x + 10, y, PANEL - 20, 41)
            pygame.draw.rect(
                self.screen,
                UI_SURFACE_HOVER if focused else UI_SURFACE,
                row_rect,
                border_radius=9,
            )
            pygame.draw.rect(
                self.screen,
                car.color if focused else UI_BORDER,
                row_rect,
                1,
                border_radius=9,
            )
            pygame.draw.rect(self.screen, car.color, (x + 15, y + 7, 5, 27), border_radius=2)
            self.text(f"{i+1:>2}", (x + 27, y + 10), "mono", WHITE if focused else MUTED)
            self.text(car.name, (x + 57, y + 6), "mono", WHITE)
            if self.metric == 0:
                previous = ranked[i - 1] if i else leader
                value = "LEADER" if i == 0 else f"+{previous.score-car.score:4.0f} m"
            elif self.metric == 1:
                value = "LEADER" if i == 0 else f"+{leader.score-car.score:4.0f} m"
            elif self.metric == 2:
                value = f"{car.tyre[0]} {car.tyre_laps}L/{car.tyre_wear:2.0f}%"
            elif self.metric == 3:
                value = f"{car.pitstops} STOP{'S' if car.pitstops != 1 else ''}"
            elif self.metric == 4:
                value = (
                    f"{car.health:3.0f}% DRAFT"
                    if car.slipstream > 0.05 else f"{car.health:3.0f}%"
                )
            elif self.metric == 5:
                if car.generation == "Hybrid":
                    pygame.draw.rect(
                        self.screen, (35, 57, 49),
                        (x + 148, y + 8, 86, 7), border_radius=3,
                    )
                    pygame.draw.rect(
                        self.screen,
                        YELLOW if car.overtake_active else CYAN,
                        (
                            x + 148, y + 8,
                            int(86 * clamp(car.battery / 100.0, 0.0, 1.0)),
                            7,
                        ),
                        border_radius=3,
                    )
                    value = (
                        f"{car.battery:3.0f}%  "
                        f"{'OVERTAKE' if car.overtake_active else 'READY'}"
                    )
                else:
                    value = "ICE  •  NO BATTERY"
            elif self.metric == 6:
                value = f"{car.fuel:4.1f}kg  {'PIT' if car.pit_requested else 'RUN'}"
            elif self.metric == 7:
                value = (
                    f"{car.speed_kph:3.0f}km/h "
                    f"G{car.gear} {car.rpm / 1000:04.1f}k"
                )
            else:
                risk = (
                    "OVERCOMMIT"
                    if car.aggression_error > 0.15
                    else (
                        "HESITATE"
                        if car.aggression_error < -0.15
                        else "PUSH"
                    )
                )
                value = f"AGG {car.race_aggression * 100:3.0f}%  {risk}"
            self.text(value, (x + 148, y + 22), "small", CYAN if focused else MUTED)
        if len(ranked) > 10:
            self.text(f"+ {len(ranked)-10} MORE CARS", (x + 20, 675), "small", MUTED)
        self.footer_hint(("[←/→] Metric  •  Wheel/[ ] Zoom", "[Y/C/W/P] Events  •  [Esc] Back"))

    def draw_results(self, ranked):
        self.draw_app_background()
        self.text("RACE CLASSIFICATION", (374, 42), "title")
        self.text(f"{self.track.name.upper()}  •  {self.target_laps} LAPS", (475, 98), "small", CYAN)
        podium = [c for c in ranked if c.finish_time is not None][:3]
        places = [(185, 185, (184, 115, 51)), (455, 135, (255, 197, 45)), (725, 205, (192, 202, 210))]
        order = [1, 0, 2]
        for box, index in zip(places, order):
            if index < len(podium):
                x, y, medal = box
                car = podium[index]
                self.glass_card(
                    pygame.Rect(x, y, 240, 176),
                    accent=medal,
                    radius=16,
                )
                pygame.draw.rect(
                    self.screen, medal, (x, y, 240, 5),
                    border_radius=3,
                )
                self.text(f"P{index+1}", (x + 100, y + 25), "h2", medal)
                name = self.fonts["h2"].render(car.name, True, car.color)
                self.screen.blit(name, name.get_rect(center=(x + 120, y + 88)))
                timing = self.fonts["mono"].render(f"{car.finish_time:.2f}s", True, WHITE)
                self.screen.blit(timing, timing.get_rect(center=(x + 120, y + 133)))
        table_x = 985
        self.text("FULL RESULTS", (table_x, 155), "small", YELLOW)
        winner_time = podium[0].finish_time if podium else None
        for i, car in enumerate(ranked[:8]):
            y = 185 + i * 42
            row = pygame.Rect(table_x, y, 235, 34)
            pygame.draw.rect(
                self.screen, UI_SURFACE, row, border_radius=8
            )
            pygame.draw.rect(
                self.screen, UI_BORDER, row, 1, border_radius=8
            )
            self.text(f"{i+1:>2}  {car.name}", (table_x + 10, y + 8), "mono", car.color)
            if car.finish_time is None:
                value = "DNF"
            elif i == 0 or winner_time is None:
                value = f"{car.finish_time:.1f}s"
            else:
                value = f"+{car.finish_time-winner_time:.1f}s"
            self.text(value, (table_x + 170, y + 9), "small", MUTED)
        replay = "REPLAY SAVED" if self.replay_saved else "[R] Save replay"
        self.text(f"{replay}   •   [Esc] Return to paddock", (445, 690), "body", CYAN if self.replay_saved else MUTED)

    def run(self):
        running = True
        while running:
            dt = self.clock.tick(FPS)
            raw_events = pygame.event.get()
            for event in raw_events:
                if event.type == pygame.QUIT:
                    running = False
                elif (
                    event.type == pygame.KEYDOWN
                    and event.key == pygame.K_l
                    and self.mode != "name_dialog"
                    and not event.mod & (
                        pygame.KMOD_CTRL | pygame.KMOD_ALT | pygame.KMOD_GUI
                    )
                ):
                    self.show_fps = not self.show_fps
            events = self.translate_events(raw_events)
            if self.mode == "menu":
                self.menu(events)
            elif self.mode == "editor":
                self.editor(events)
            elif self.mode == "algorithm":
                self.algorithm(events)
            elif self.mode == "training":
                self.training(events, dt)
            elif self.mode == "race_setup":
                self.race_setup(events)
            elif self.mode == "race":
                self.race(events, dt)
            elif self.mode == "hotlap_setup":
                self.hotlap_setup(events)
            elif self.mode == "hotlap":
                self.hotlap(events, dt)
            elif self.mode == "name_dialog":
                self.draw_name_dialog(events)
            if self.show_fps:
                self.draw_fps_counter()
            if pygame.time.get_ticks() < self.message_until:
                box = self.fonts["body"].render(self.message, True, INK)
                rect = box.get_rect(
                    center=(WIDTH // 2, HEIGHT - 35)
                ).inflate(38, 18)
                pygame.draw.rect(
                    self.screen, UI_SHADOW, rect.move(0, 4),
                    border_radius=11,
                )
                pygame.draw.rect(
                    self.screen, YELLOW, rect, border_radius=11
                )
                self.screen.blit(box, box.get_rect(center=rect.center))
            self.present()
        pygame.quit()


if __name__ == "__main__":
    Game().run()
