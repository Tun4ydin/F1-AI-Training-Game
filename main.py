from __future__ import annotations

import json
import math
import random
import re
from bisect import bisect_right
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pygame
from pygame import Vector2
from safe_algorithm import AlgorithmError, SafeAlgorithm

try:
    import pygame_gui
except ImportError:
    pygame_gui = None

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
BASE_ENGINE_ACCELERATION = 0.0086
LOW_SPEED_TRACTION = 0.52
FULL_TRACTION_SPEED = 0.60
HYBRID_COMBUSTION_SHARE = 0.80
HYBRID_ELECTRIC_SHARE = 0.20
HYBRID_DRS_ELECTRIC_BONUS = 0.30
HYBRID_TOTAL_POWER_SCALE = 1.1875
HYBRID_REGEN_RATE = 0.46
HYBRID_RECHARGE_RATE = 5.5 / FPS
HYBRID_RECHARGE_COMBUSTION_SHARE = 0.70
ROLLING_RESISTANCE = 0.00035
AERO_DRAG_COEFFICIENT = 0.00056
ICE_DRS_DRAG_MULTIPLIER = 0.30
DRS_MAX_GAP_SECONDS = 1.0
# Steering is expressed as degrees of yaw per simulation frame. These values
# give full-lock inputs more authority through medium- and high-speed corners
# without changing the controller's normalized -1..1 steering contract.
STEERING_BASE_YAW = 3.68
STEERING_SPEED_YAW = 0.79
STEERING_UNDERSTEER_LOSS = 0.12
LATERAL_GRIP_RECOVERY = 0.24
OPPONENT_SENSOR_RANGE_M = 40.0
OVERTAKE_REWARD = 150.0
OVERTAKE_COOLDOWN_FRAMES = FPS * 3
LOW_SPEED_DNF_KPH = 10.0
LOW_SPEED_DNF_SECONDS = 20.0
FINISH_BRAKE_INPUT = 0.10
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
_CAR_MASTER_CACHE = {}
_CAR_ROTATION_CACHE = OrderedDict()
CAR_ROTATION_STEP_DEGREES = 3
CAR_ROTATION_CACHE_LIMIT = 3072
CAR_MASTER_PATHS = {
    "ICE": ROOT / "assets" / "cars" / "ice_2000s_master.png",
    "Hybrid": ROOT / "assets" / "cars" / "hybrid_2022_master.png",
}


def clamp(value, low, high):
    return max(low, min(high, value))


def load_car_master(generation):
    """Load and tightly crop one transparent era-specific source sprite."""
    era = "Hybrid" if generation == "Hybrid" else "ICE"
    cached = _CAR_MASTER_CACHE.get(era)
    if cached is not None:
        return cached
    image = pygame.image.load(str(CAR_MASTER_PATHS[era]))
    if pygame.display.get_surface() is not None:
        image = image.convert_alpha()
    bounds = image.get_bounding_rect(min_alpha=8)
    if bounds.width and bounds.height:
        image = image.subsurface(bounds).copy()
    _CAR_MASTER_CACHE[era] = image
    return image


def build_car_sprite(generation, color, scale):
    """Build one recolored, correctly scaled sprite from a master PNG."""
    era = "Hybrid" if generation == "Hybrid" else "ICE"
    cache_key = (era, tuple(color), round(float(scale), 3))
    cached = _CAR_SPRITE_CACHE.get(cache_key)
    if cached is not None:
        return cache_key, cached

    logical_length = max(18, round(CAR_LENGTH_M * scale))
    logical_width = max(8, round(CAR_WIDTH_M * scale))
    master = load_car_master(era)
    body = pygame.transform.smoothscale(
        master, (logical_length, logical_width)
    )
    # The masters use neutral white bodywork. Multiplication retains their
    # painted highlights, carbon fibre, tyres and mechanical detail while
    # assigning each entry its chosen livery color.
    livery = tuple(
        min(255, round(channel * 0.78 + 55))
        for channel in color
    )
    body.fill((*livery, 255), special_flags=pygame.BLEND_RGBA_MULT)
    padding = max(5, round(scale * 0.70))
    sprite = pygame.Surface(
        (
            logical_length + padding * 2,
            logical_width + padding * 2,
        ),
        pygame.SRCALPHA,
    )
    sprite.blit(body, (padding, padding))
    _CAR_SPRITE_CACHE[cache_key] = sprite
    return cache_key, sprite


def rotated_car_sprite(generation, color, scale, angle):
    """Return one cached livery, rotation and shadow composite."""
    base_key, base = build_car_sprite(generation, color, scale)
    rotation = (
        round(float(angle) / CAR_ROTATION_STEP_DEGREES)
        * CAR_ROTATION_STEP_DEGREES
    ) % 360
    cache_key = (*base_key, rotation)
    cached = _CAR_ROTATION_CACHE.pop(cache_key, None)
    if cached is not None:
        _CAR_ROTATION_CACHE[cache_key] = cached
        return cached

    rotated = pygame.transform.rotozoom(base, -rotation, 1.0)
    composite = pygame.Surface(
        (rotated.get_width() + 8, rotated.get_height() + 8),
        pygame.SRCALPHA,
    )
    shadow = rotated.copy()
    shadow.fill(
        (18, 22, 22, 96), special_flags=pygame.BLEND_RGBA_MULT
    )
    composite.blit(shadow, (6, 7))
    composite.blit(rotated, (4, 4))
    _CAR_ROTATION_CACHE[cache_key] = composite
    while len(_CAR_ROTATION_CACHE) > CAR_ROTATION_CACHE_LIMIT:
        _CAR_ROTATION_CACHE.popitem(last=False)
    return composite


def prewarm_car_sprites(scale=DEFAULT_CAMERA_ZOOM):
    """Prepare every standard livery at the normal driving-camera scale."""
    for generation in ("ICE", "Hybrid"):
        for color in COLORS:
            build_car_sprite(generation, color, scale)


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
    TARGET_SPLINE_SAMPLES = 240
    MIN_SAMPLES_PER_SECTION = 4

    def __init__(
        self, points=None, name="Starter Ring", kerb_points=None, features=None,
        geometry="spline", declared_length_m=None, road_width_m=ROAD_W,
        road_widths_m=None, pitlane_points=None,
        grass_widths_m=None, pitlane_road_widths_m=None,
        pitlane_grass_widths_m=None,
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
        default_grass_width = float(
            (features or {}).get("border_margin", BORDER_W)
        )
        supplied_grass_widths = list(grass_widths_m or [])
        self.grass_widths_m = [
            clamp(
                float(supplied_grass_widths[i])
                if i < len(supplied_grass_widths)
                else default_grass_width,
                max(self.road_widths_m[i] + 4.0, 16.0), 80.0,
            )
            for i in range(len(self.points))
        ]
        self.kerb_width_m = self.road_width_m + 2.0
        self.samples_per_section = (
            1
            if geometry == "sampled"
            else self.spline_samples_per_section(len(self.points))
        )
        self.kerb_points = set(kerb_points) if kerb_points is not None else self._automatic_kerbs()
        default_sectors = [
            len(self.points) // 4, len(self.points) // 2, len(self.points) * 3 // 4,
        ] if len(self.points) >= 8 else []
        base_features = {
            "start_finish": 0,
            "pit_start_finish": None,
            "sectors": default_sectors,
            "pit_entry": None,
            "pit_exit": None,
            "pit_boxes": [],
            "drs_detection": None,
            "drs_entry": None,
            "drs_exit": None,
            "border_margin": BORDER_W,
        }
        self.features = {**base_features, **(features or {})}
        self.pitlane_points = [
            Vector2(point) for point in (pitlane_points or [])
        ]
        supplied_pit_widths = list(pitlane_road_widths_m or [])
        supplied_pit_grass = list(pitlane_grass_widths_m or [])
        self.pitlane_road_widths_m = [
            clamp(
                float(supplied_pit_widths[i])
                if i < len(supplied_pit_widths)
                else PITLANE_WIDTH_M,
                4.0, 18.0,
            )
            for i in range(len(self.pitlane_points))
        ]
        self.pitlane_grass_widths_m = [
            clamp(
                float(supplied_pit_grass[i])
                if i < len(supplied_pit_grass)
                else max(PITLANE_WIDTH_M + 2.0, 16.0),
                max(self.pitlane_road_widths_m[i] + 2.0, 8.0), 60.0,
            )
            for i in range(len(self.pitlane_points))
        ]
        (
            self.centerline,
            self.kerb_segments,
            self.centerline_widths_m,
            self.centerline_grass_widths_m,
        ) = self._build_spline()
        self.pitlane_centerline = self._build_pitlane()
        self.pitlane_centerline_widths_m = self._pitlane_sample_widths(
            self.pitlane_road_widths_m, PITLANE_WIDTH_M
        )
        self.pitlane_centerline_grass_widths_m = self._pitlane_sample_widths(
            self.pitlane_grass_widths_m, PITLANE_WIDTH_M + 2.0
        )
        self._pit_box_positions_cache = None
        self._ribbon_cache = {}
        self._open_ribbon_cache = {}
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

    @classmethod
    def spline_samples_per_section(cls, point_count):
        """Avoid oversampling tracks whose editor nodes are already dense."""
        if point_count <= 0:
            return cls.SAMPLES_PER_SECTION
        return int(clamp(
            math.ceil(cls.TARGET_SPLINE_SAMPLES / point_count),
            cls.MIN_SAMPLES_PER_SECTION,
            cls.SAMPLES_PER_SECTION,
        ))

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
        # Surface probes and AI raycasts call ``is_in_pitlane`` thousands of
        # times per frame.  Keep a separate index for the open pit road so
        # those probes only inspect the segment(s) in their current cell.
        self.pitlane_spatial_segments = {}
        self.pitlane_bounds = None
        if len(self.pitlane_centerline) < 2:
            return
        maximum_half_width = max(
            self.pitlane_centerline_widths_m,
            default=PITLANE_WIDTH_M,
        ) / 2
        # A 50 m cell is useful for a large main circuit, but it is far too
        # coarse for compact pit roads. On small maps it places nearly every
        # pit segment in one bucket, making each ray/wheel containment probe
        # scan the whole pitlane. Keep pit cells close to the road's scale.
        self.pitlane_spatial_cell_m = clamp(
            maximum_half_width * 3.0, 8.0, 18.0
        )
        self.pitlane_bounds = (
            min(point.x for point in self.pitlane_centerline)
            - maximum_half_width,
            min(point.y for point in self.pitlane_centerline)
            - maximum_half_width,
            max(point.x for point in self.pitlane_centerline)
            + maximum_half_width,
            max(point.y for point in self.pitlane_centerline)
            + maximum_half_width,
        )
        for index in range(len(self.pitlane_centerline) - 1):
            start = self.pitlane_centerline[index]
            end = self.pitlane_centerline[index + 1]
            segment_half_width = max(
                self.pitlane_centerline_widths_m[index],
                self.pitlane_centerline_widths_m[index + 1],
            ) / 2
            minimum_x = math.floor(
                (min(start.x, end.x) - segment_half_width)
                / self.pitlane_spatial_cell_m
            )
            maximum_x = math.floor(
                (max(start.x, end.x) + segment_half_width)
                / self.pitlane_spatial_cell_m
            )
            minimum_y = math.floor(
                (min(start.y, end.y) - segment_half_width)
                / self.pitlane_spatial_cell_m
            )
            maximum_y = math.floor(
                (max(start.y, end.y) + segment_half_width)
                / self.pitlane_spatial_cell_m
            )
            for cell_x in range(minimum_x, maximum_x + 1):
                for cell_y in range(minimum_y, maximum_y + 1):
                    self.pitlane_spatial_segments.setdefault(
                        (cell_x, cell_y), set()
                    ).add(index)

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
                list(self.grass_widths_m),
            )
        if count < 4:
            return (
                [p.copy() for p in self.points],
                [False] * count,
                list(self.road_widths_m),
                list(self.grass_widths_m),
            )
        curve, kerbs, widths, grass_widths = [], [], [], []
        for i in range(count):
            p0 = self.points[(i - 1) % count]
            p1 = self.points[i]
            p2 = self.points[(i + 1) % count]
            p3 = self.points[(i + 2) % count]
            for sample in range(self.samples_per_section):
                t = sample / self.samples_per_section
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
                grass_widths.append(
                    self.grass_widths_m[i]
                    + (
                        self.grass_widths_m[(i + 1) % count]
                        - self.grass_widths_m[i]
                    ) * t
                )
        return curve, kerbs, widths, grass_widths

    def set_kerb_points(self, indexes):
        self.kerb_points = set(indexes)
        (
            self.centerline,
            self.kerb_segments,
            self.centerline_widths_m,
            self.centerline_grass_widths_m,
        ) = self._build_spline()

    def _main_tangent_at_control(self, control_index):
        """Return a smooth forward tangent at a main-track control node."""
        spline_index = (
            int(control_index) * self.samples_per_section
        ) % len(self.centerline)
        previous = self.centerline[spline_index - 1]
        following = self.centerline[
            (spline_index + 1) % len(self.centerline)
        ]
        tangent = following - previous
        return (
            tangent.normalize()
            if tangent.length_squared() else Vector2(1, 0)
        )

    def _pitlane_edge_anchor(self, control_index, toward, is_entry):
        """Return the local asphalt edge facing an authored pit-road node."""
        spline_index = (
            int(control_index) * self.samples_per_section
        ) % len(self.centerline)
        centre = self.centerline[spline_index]
        tangent = self._main_tangent_at_control(control_index)
        normal = Vector2(-tangent.y, tangent.x)
        toward_edge = Vector2(toward) - centre
        side = 1.0 if toward_edge.dot(normal) >= 0.0 else -1.0
        half_width = self.centerline_widths_m[spline_index] / 2
        return centre + normal * side * half_width

    @staticmethod
    def _pit_transition_curve(
        start, end, start_tangent, end_tangent
    ):
        """Build a tangent-matched cubic connector without sharp wedges."""
        start, end = Vector2(start), Vector2(end)
        distance = start.distance_to(end)
        if distance <= 1e-6:
            return [start]
        start_tangent = (
            Vector2(start_tangent).normalize()
            if Vector2(start_tangent).length_squared()
            else (end - start).normalize()
        )
        end_tangent = (
            Vector2(end_tangent).normalize()
            if Vector2(end_tangent).length_squared()
            else (end - start).normalize()
        )
        handle = clamp(distance * 0.34, 3.0, 22.0)
        control_1 = start + start_tangent * handle
        control_2 = end - end_tangent * handle
        segments = int(clamp(math.ceil(distance / 3.5), 7, 18))
        curve = []
        for index in range(segments + 1):
            t = index / segments
            inverse = 1.0 - t
            curve.append(
                start * inverse ** 3
                + control_1 * (3.0 * inverse ** 2 * t)
                + control_2 * (3.0 * inverse * t ** 2)
                + end * t ** 3
            )
        return curve

    def _build_pitlane(self):
        """Connect pit nodes with smooth entry and exit transition curves."""
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
        if len(self.pitlane_points) >= 2:
            entry_direction = (
                self.pitlane_points[1] - self.pitlane_points[0]
            )
            exit_direction = (
                self.pitlane_points[-1] - self.pitlane_points[-2]
            )
        else:
            entry_direction = exit_anchor - self.pitlane_points[0]
            exit_direction = entry_direction.copy()
        entry_curve = self._pit_transition_curve(
            entry_anchor, self.pitlane_points[0],
            self._main_tangent_at_control(entry), entry_direction,
        )
        exit_curve = self._pit_transition_curve(
            self.pitlane_points[-1], exit_anchor,
            exit_direction, self._main_tangent_at_control(exit_index),
        )
        return [
            *entry_curve[:-1],
            *(point.copy() for point in self.pitlane_points),
            *exit_curve[1:],
        ]

    def _pitlane_sample_widths(self, control_widths, fallback):
        """Smooth authored pit-node widths across the rendered centreline."""
        if not self.pitlane_centerline:
            return []
        if not self.pitlane_points or not control_widths:
            return [float(fallback)] * len(self.pitlane_centerline)
        if len(self.pitlane_points) == 1:
            return [float(control_widths[0])] * len(self.pitlane_centerline)
        sampled = []
        for point in self.pitlane_centerline:
            best_distance = float("inf")
            best_width = float(control_widths[0])
            for index in range(len(self.pitlane_points) - 1):
                distance, _, ratio = seg_distance(
                    point, self.pitlane_points[index],
                    self.pitlane_points[index + 1],
                )
                if distance < best_distance:
                    best_distance = distance
                    best_width = (
                        control_widths[index]
                        + (
                            control_widths[index + 1]
                            - control_widths[index]
                        ) * ratio
                    )
            sampled.append(float(best_width))
        return sampled

    def open_ribbon_edges(self, points, widths):
        """Build and cache joined edges for an open variable-width road."""
        cache_key = (
            tuple((round(p.x, 6), round(p.y, 6)) for p in points),
            tuple(round(float(width), 6) for width in widths),
        )
        cached = self._open_ribbon_cache.get(cache_key)
        if cached is not None:
            return cached
        count = len(points)
        if count < 2:
            return [], []
        left, right = [], []
        for index, point in enumerate(points):
            previous = points[max(0, index - 1)]
            following = points[min(count - 1, index + 1)]
            tangent = following - previous
            if tangent.length_squared() <= 1e-12:
                tangent = Vector2(1, 0)
            tangent = tangent.normalize()
            normal = Vector2(-tangent.y, tangent.x)
            half_width = max(0.5, float(widths[index]) / 2)
            left.append(point + normal * half_width)
            right.append(point - normal * half_width)
        result = (left, right)
        self._open_ribbon_cache[cache_key] = result
        return result

    def pitlane_nearest(self, point, local_only=False):
        """Return distance and position on the open pitlane polyline.

        ``local_only`` is for containment queries.  Every pit segment is
        indexed with its full road half-width, so a point that can be inside
        the pitlane is guaranteed to find that segment in its own cell.
        General callers retain the exact full-polyline nearest-point result.
        """
        point = Vector2(point)
        best = (float("inf"), Vector2(), None, 0.0)
        if local_only:
            cell_size = getattr(
                self, "pitlane_spatial_cell_m", self.spatial_cell_m
            )
            cell = (
                math.floor(point.x / cell_size),
                math.floor(point.y / cell_size),
            )
            candidates = self.pitlane_spatial_segments.get(cell, ())
        else:
            candidates = range(max(0, len(self.pitlane_centerline) - 1))
        for index in candidates:
            distance, nearest, ratio = seg_distance(
                point,
                self.pitlane_centerline[index],
                self.pitlane_centerline[index + 1],
            )
            if distance < best[0]:
                best = (distance, nearest, index, ratio)
        return best

    def pitlane_point_ahead(self, point, lookahead_m=14.0):
        """Return a forward target along the open pit-lane centreline."""
        if len(self.pitlane_centerline) < 2:
            return None
        _, nearest, segment, _ = self.pitlane_nearest(point)
        if segment is None:
            return None
        target = nearest
        remaining = max(0.0, float(lookahead_m))
        for index in range(segment, len(self.pitlane_centerline) - 1):
            endpoint = self.pitlane_centerline[index + 1]
            segment_length = target.distance_to(endpoint)
            if segment_length >= remaining and segment_length > 1e-9:
                return target.lerp(endpoint, remaining / segment_length)
            remaining -= segment_length
            target = endpoint
        return self.pitlane_centerline[-1].copy()

    def is_in_pitlane(self, point):
        if len(self.pitlane_centerline) < 2:
            return False
        point = Vector2(point)
        if self.pitlane_bounds is not None:
            left, top, right, bottom = self.pitlane_bounds
            if not left <= point.x <= right or not top <= point.y <= bottom:
                return False
        distance, _, segment, ratio = self.pitlane_nearest(
            point, local_only=True
        )
        if segment is None:
            return False
        widths = self.pitlane_centerline_widths_m
        width = widths[segment] + (
            widths[min(segment + 1, len(widths) - 1)] - widths[segment]
        ) * ratio
        return distance <= width / 2

    def timing_line(self, pitlane=False):
        """Return timing-line centre, forward tangent, and half width."""
        if pitlane:
            marker = self.features.get("pit_start_finish")
            points = self.pitlane_points
            if marker is None or not points:
                return None
            index = int(marker)
            if not 0 <= index < len(points):
                return None
            centre = points[index]
            previous = points[max(0, index - 1)]
            following = points[min(len(points) - 1, index + 1)]
            tangent = following - previous
            half_width = PITLANE_WIDTH_M / 2
        else:
            if not self.points or not self.centerline:
                return None
            control_index = (
                int(self.features.get("start_finish", 0))
                % len(self.points)
            )
            index = control_index * self.samples_per_section
            centre = self.centerline[index]
            following = self.centerline[
                (index + 1) % len(self.centerline)
            ]
            tangent = following - centre
            half_width = self.centerline_widths_m[index] / 2
        if not tangent.length_squared():
            return None
        return centre, tangent.normalize(), half_width

    def feature_line(self, feature):
        """Return a full-width line for a main-track control-node feature."""
        marker = self.features.get(feature)
        if marker is None or not self.points or not self.centerline:
            return None
        control_index = int(marker)
        if not 0 <= control_index < len(self.points):
            return None
        index = control_index * self.samples_per_section
        centre = self.centerline[index]
        tangent = self._main_tangent_at_control(control_index)
        return centre, tangent, self.centerline_widths_m[index] / 2

    def crossed_timing_line(self, previous, current, pitlane=False):
        """Detect a forward centre-point crossing of a timing line."""
        timing_line = self.timing_line(pitlane)
        if timing_line is None:
            return False
        centre, tangent, half_width = timing_line
        previous = Vector2(previous)
        current = Vector2(current)
        before = (previous - centre).dot(tangent)
        after = (current - centre).dot(tangent)
        if before >= -1e-4 or after < 0.0:
            return False
        distance = after - before
        if distance <= 1e-9:
            return False
        crossing = previous.lerp(current, -before / distance)
        normal = Vector2(-tangent.y, tangent.x)
        return abs((crossing - centre).dot(normal)) <= (
            half_width + CAR_WIDTH_M / 2
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

    def nearest(self, point, fallback=True):
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
            if not fallback:
                return best
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
        # Surface probes include thousands of ray samples per frame. A point
        # with no nearby indexed segment is definitively outside the circuit,
        # so it must not fall back to scanning the entire centerline.
        distance, _, index, t = self.nearest(point, fallback=False)
        local_width = self.width_at_segment(index, t)
        if distance <= local_width / 2:
            return "asphalt"
        if distance <= (local_width + 2.0) / 2 and self.kerb_segments[index]:
            return "kerb"
        current_grass = self.centerline_grass_widths_m[index]
        following_grass = self.centerline_grass_widths_m[
            (index + 1) % len(self.centerline_grass_widths_m)
        ]
        grass_width = current_grass + (
            following_grass - current_grass
        ) * t
        if distance <= grass_width / 2:
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
        pit_points = (
            self._screen_points(
                self.pitlane_centerline, offset, scale
            )
            if len(self.pitlane_centerline) >= 2 else []
        )

        # Every visual layer is derived from the same mitered ribbon vertices.
        # This removes the triangular gaps produced by independent thick-line
        # caps at sharp or variable-width nodes.
        border_widths = [
            max(width + 4.0, grass_width)
            for width, grass_width in zip(
                self.centerline_widths_m,
                self.centerline_grass_widths_m,
            )
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
        if pit_points:
            pit_border_left, pit_border_right = self.open_ribbon_edges(
                self.pitlane_centerline,
                self.pitlane_centerline_grass_widths_m,
            )
            pit_border_left_px = self._screen_points(
                pit_border_left, offset, scale
            )
            pit_border_right_px = self._screen_points(
                pit_border_right, offset, scale
            )
            for index in range(len(pit_points) - 1):
                pygame.draw.polygon(
                    screen, (17, 43, 28),
                    (
                        pit_border_left_px[index],
                        pit_border_left_px[index + 1],
                        pit_border_right_px[index + 1],
                        pit_border_right_px[index],
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
        if pit_points:
            # Asphalt is the final road layer so it opens a clean, seamless
            # mouth through the main-track edge at pit entry and exit.
            pit_road_left, pit_road_right = self.open_ribbon_edges(
                self.pitlane_centerline,
                self.pitlane_centerline_widths_m,
            )
            pit_road_left_px = self._screen_points(
                pit_road_left, offset, scale
            )
            pit_road_right_px = self._screen_points(
                pit_road_right, offset, scale
            )
            for index in range(len(pit_points) - 1):
                pygame.draw.polygon(
                    screen, (67, 70, 74),
                    (
                        pit_road_left_px[index],
                        pit_road_left_px[index + 1],
                        pit_road_right_px[index + 1],
                        pit_road_right_px[index],
                    ),
                )
            pygame.draw.aalines(
                screen, (160, 164, 164), False, pit_road_left_px
            )
            pygame.draw.aalines(
                screen, (160, 164, 164), False, pit_road_right_px
            )
        # Main and pit timing lines share the same directional crossing model.
        timing_line = self.timing_line()
        if timing_line is not None:
            centre, tangent, half_width = timing_line
            p = centre * scale + offset
            normal = Vector2(-tangent.y, tangent.x)
            pygame.draw.line(
                screen, YELLOW,
                p - normal * half_width * scale,
                p + normal * half_width * scale,
                max(1, int(scale)),
            )
        pit_timing_line = self.timing_line(pitlane=True)
        if pit_timing_line is not None:
            centre, tangent, half_width = pit_timing_line
            p = centre * scale + offset
            normal = Vector2(-tangent.y, tangent.x)
            pygame.draw.line(
                screen, (255, 214, 80),
                p - normal * half_width * scale,
                p + normal * half_width * scale,
                max(1, int(scale)),
            )
        drs_colors = {
            "drs_detection": (255, 196, 61),
            "drs_entry": (38, 211, 126),
            "drs_exit": (239, 82, 92),
        }
        for feature, color in drs_colors.items():
            feature_line = self.feature_line(feature)
            if feature_line is None:
                continue
            centre, tangent, half_width = feature_line
            p = centre * scale + offset
            normal = Vector2(-tangent.y, tangent.x)
            pygame.draw.line(
                screen, color,
                p - normal * half_width * scale,
                p + normal * half_width * scale,
                max(2, int(1.4 * scale)),
            )
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
            "grass_widths_m": self.grass_widths_m,
            "pitlane_road_widths_m": self.pitlane_road_widths_m,
            "pitlane_grass_widths_m": self.pitlane_grass_widths_m,
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
            data.get("grass_widths_m"),
            data.get("pitlane_road_widths_m"),
            data.get("pitlane_grass_widths_m"),
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
        "overtake_active", "recharge_active", "off_track", "car_collision",
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
                result = self.program.run(state, self.parameters)
                if len(result) == 6:
                    # Brains compiled by the pre-Recharge parser remain valid.
                    return (*result[:4], 0.0, *result[4:])
                return result
            except (AlgorithmError, ArithmeticError, KeyError, TypeError, ValueError):
                # A bad runtime calculation stops this agent safely, never the game.
                return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1
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
        recharge = (
            state.get("is_hybrid", 0) >= 0.5
            and state.get("battery", 1.0) <= 0.20
        )
        return (
            steer, throttle, brake, float(overtake),
            float(recharge), float(pit_request), pit_tyre,
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

    @staticmethod
    def compile_source(source):
        """Compile current controllers with native Recharge output support."""
        return SafeAlgorithm(source)

    @classmethod
    def load_file(cls, path):
        try:
            data = json.loads(Path(path).read_text())
            if data.get("source"):
                stored_source = data["source"]
                source = cls.migrate_legacy_source(stored_source)
                program = cls.compile_source(source)
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
    race_distance_m: float | None = None
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
    retirement_time: float | None = None
    retirement_reason: str = ""
    low_speed_seconds: float = 0.0
    removed_from_track: bool = False
    starting_position: int = 1
    previous_progress: float = 0.0
    previous_progress_m: float | None = None
    outside_limits: bool = False
    pit_timer: float = 0.0
    pit_requested: bool = False
    pit_entry_committed: bool = False
    pit_exit_straight_frames: float = 0.0
    pit_exit_guidance_frames: float = 0.0
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
    recharge_active: bool = False
    drs_eligible: bool = False
    drs_active: bool = False
    drs_in_zone: bool = False
    drs_gap_seconds: float = float("inf")
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
    timing_history: list = field(
        default_factory=list, repr=False, compare=False
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

    def advance_after_finish(self, track, dt=1.0):
        """Follow the circuit under 10% braking, then leave the track."""
        if self.removed_from_track:
            return
        speed = self.velocity.length()
        braking_delta = FINISH_BRAKE_INPUT * 0.028 * dt
        remaining_speed = max(0.0, speed - braking_delta)
        if remaining_speed <= 1e-6:
            self.velocity.update(0.0, 0.0)
            self.removed_from_track = True
            return
        progress_m = track.progress_metres(self.position)
        target = track.point_at_distance(
            progress_m + clamp(remaining_speed * 28.0, 12.0, 26.0)
        )
        desired = target - self.position
        current_direction = (
            self.velocity.normalize()
            if self.velocity.length_squared()
            else Vector2(1, 0).rotate(self.angle)
        )
        if desired.length_squared():
            blended = current_direction.lerp(
                desired.normalize(), clamp(0.10 * dt, 0.0, 0.35)
            )
            if blended.length_squared():
                current_direction = blended.normalize()
        self.velocity = current_direction * remaining_speed
        self.angle = math.degrees(
            math.atan2(current_direction.y, current_direction.x)
        )
        self.brake_input = FINISH_BRAKE_INPUT
        self.throttle_input = 0.0
        self.steering_input = 0.0
        self.position += self.velocity * dt

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
                clamp(
                    self.battery_regen / HYBRID_REGEN_RATE,
                    0.0, 1.0,
                )
                if self.generation == "Hybrid" else 0.0
            ),
            float(self.generation == "Hybrid"),
            float(self.overtake_active),
            float(self.recharge_active),
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

    def pit_guidance(self, track, steer, throttle, brake):
        """Guide a committed stop through pit entry, lane, and exit."""
        if len(track.pitlane_centerline) < 2:
            self.pit_entry_committed = False
            self.pit_exit_straight_frames = 0.0
            self.pit_exit_guidance_frames = 0.0
            return steer, throttle, brake
        target = None
        entry_distance = float("inf")
        recovering_from_runoff = False
        straight_exit = False

        def pit_exit_target():
            """Aim from PIT OUT toward the following authored track node."""
            exit_node = track.features.get("pit_exit")
            if exit_node is None or not track.points:
                return track.pitlane_centerline[-1]
            # PIT OUT is normally already behind the car when merge guidance
            # begins. Using its distance plus a fixed offset can still leave
            # the target behind on long editor sections. Anchor guidance to
            # the next actual track node instead.
            next_node = (
                int(exit_node) + 1
            ) % len(track.points)
            next_segment = (
                next_node
            ) * track.samples_per_section
            next_progress_m = track.cumulative_lengths_m[next_segment]
            guidance_elapsed = max(
                0.0, 90.0 - self.pit_exit_guidance_frames
            )
            return track.point_at_distance(
                next_progress_m + guidance_elapsed * 0.55
            )

        # Once the exit is armed it owns steering even while the car's centre
        # is still inside the pit-lane surface. Previously the in-pit branch
        # reset this timer every frame, delaying zero-steer until it was too
        # late and allowing the nearest-track lookup to select a right-hand
        # section of circuit.
        if self.pit_exit_straight_frames > 0.0:
            self.pit_exit_straight_frames = max(
                0.0, self.pit_exit_straight_frames - 1.0
            )
            self.pit_entry_committed = False
            straight_exit = True
            target = None
        elif self.pit_exit_guidance_frames > 0.0:
            self.pit_exit_guidance_frames = max(
                0.0, self.pit_exit_guidance_frames - 1.0
            )
            self.pit_entry_committed = False
            target = pit_exit_target()
        elif self.in_pitlane:
            self.pit_entry_committed = False
            if self.position.distance_to(
                track.pitlane_centerline[-1]
            ) <= 18.0:
                # Begin the straight phase before leaving the pit ribbon. Arm
                # this only once; the active phase has priority next frame.
                self.pit_exit_straight_frames = FPS * 0.5
                self.pit_exit_guidance_frames = 90.0
                straight_exit = True
                target = None
            else:
                target = track.pitlane_point_ahead(
                    self.position,
                    clamp(self.velocity.length() * 32.0, 10.0, 24.0),
                )
        elif self.pit_requested:
            entry = track.features.get("pit_entry")
            if entry is None or not track.points:
                return steer, throttle, brake
            entry_segment = (
                int(entry) % len(track.points)
            ) * track.samples_per_section
            entry_progress_m = track.cumulative_lengths_m[entry_segment]
            current_progress_m = track.progress_metres(self.position)
            entry_distance = (
                entry_progress_m - current_progress_m
            ) % max(track.measured_length_m, 1e-9)
            if entry_distance <= 130.0:
                self.pit_entry_committed = True
            if not self.pit_entry_committed:
                return steer, throttle, brake
            if entry_distance <= 130.0:
                lookahead = clamp(
                    self.velocity.length() * 20.0, 14.0, 34.0
                )
                main_target = track.point_at_distance(
                    current_progress_m
                    + min(lookahead, max(entry_distance, 0.0))
                )
                merge = clamp(
                    (55.0 - entry_distance) / 45.0, 0.0, 1.0
                )
                pit_target = track.pitlane_centerline[0].lerp(
                    track.pitlane_centerline[1],
                    clamp((merge - 0.62) / 0.38, 0.0, 1.0),
                )
                target = main_target.lerp(pit_target, merge)
            else:
                # The commitment stays active if contact or a mistake carries
                # the car just beyond the entry. Aim it back at the pit road
                # instead of abandoning the stop for an entire extra lap.
                target = track.pitlane_centerline[
                    min(1, len(track.pitlane_centerline) - 1)
                ]
        else:
            self.pit_entry_committed = False
            return steer, throttle, brake

        recovering_from_runoff = (
            (
                self.pit_entry_committed or self.in_pitlane
                or self.pit_exit_straight_frames > 0.0
                or self.pit_exit_guidance_frames > 0.0
            )
            and track.surface(self.position)
            not in ("asphalt", "kerb", "pitlane")
        )
        if recovering_from_runoff:
            # Once a committed car leaves the paved merge, stop following the
            # normal racing-line blend and aim directly into the pit route.
            target = (
                None
                if straight_exit
                else pit_exit_target()
                if self.pit_exit_guidance_frames > 0.0
                else track.pitlane_point_ahead(self.position, 8.0)
                if self.in_pitlane
                else track.pitlane_centerline[
                    min(1, len(track.pitlane_centerline) - 1)
                ]
            )

        speed = self.velocity.length()
        target_speed = PIT_SPEED_LIMIT_MPF * (
            0.82 if self.in_pitlane else 0.95
        )
        if speed > target_speed:
            throttle = 0.0
            brake = max(
                brake,
                clamp(
                    (speed - target_speed)
                    / max(PIT_SPEED_LIMIT_MPF, 1e-9),
                    0.18, 1.0,
                ),
            )
        elif straight_exit:
            brake = 0.0
            throttle = clamp(max(throttle, 0.42), 0.0, 0.58)
        elif recovering_from_runoff:
            brake = 0.0
            throttle = clamp(max(throttle, 0.62), 0.0, 0.72)
        elif self.in_pitlane:
            brake = 0.0
            throttle = max(throttle, 0.34)
        else:
            brake = 0.0
            throttle = clamp(max(throttle, 0.34), 0.0, 0.55)

        if straight_exit:
            # Hold the pit-exit tangent for exactly half a second before
            # searching for and blending toward the circuit centreline.
            steer = 0.0
        elif target is not None:
            desired = Vector2(target) - self.position
            if desired.length_squared() <= 1e-9 and self.in_pitlane:
                desired = (
                    track.pitlane_centerline[-1]
                    - track.pitlane_centerline[-2]
                )
            if desired.length_squared():
                desired = desired.normalize()
                heading = Vector2(1, 0).rotate(self.angle)
                steering_error = math.atan2(
                    heading.cross(desired), heading.dot(desired)
                ) / math.pi
                steer = clamp(
                    steering_error * (
                        3.45 if recovering_from_runoff else
                        2.75 if (
                            self.in_pitlane
                            or self.pit_exit_guidance_frames > 0.0
                        ) else 2.35
                    )
                    - self.angular_velocity * 0.10,
                    -1.0, 1.0,
                )
        return steer, throttle, brake

    def update(self, track, dt=1.0, rain=0.0, damage_enabled=True):
        if not self.alive:
            return
        if self.pit_timer > 0:
            self.pit_timer -= dt
            self.overtake_active = False
            self.drs_active = False
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
            recharge_request, pit_request, pit_tyre,
        ) = self.brain.think(controller_inputs)
        steer = clamp(steer, -1.0, 1.0)
        throttle = clamp(throttle, 0.0, 1.0)
        previous_angle = self.angle
        pit_available = bool(track.pit_box_positions())
        emergency_pit = pit_available and (
            self.tyre_wear >= 90.0 or self.puncture
        )
        # A request is a commitment and remains active until service. This
        # prevents a controller's noisy threshold from cancelling pit entry.
        self.pit_requested = bool(
            self.pit_requested
            or (pit_available and pit_request >= 0.5)
            or emergency_pit
        )
        self.requested_tyre = int(clamp(pit_tyre, 0, 3))
        self.brake_input = clamp(brake, 0.0, 1.0)
        self.steering_input = steer
        self.throttle_input = throttle
        sensor_values = controller_inputs[:Brain.DRIVING_INPUTS]
        heading_error = sensor_values[5]
        forward = Vector2(1, 0).rotate(self.angle)
        normal = Vector2(-forward.y, forward.x)
        pit_transition = bool(
            self.pit_entry_committed or self.in_pitlane
            or self.pit_exit_straight_frames > 0.0
            or self.pit_exit_guidance_frames > 0.0
        )

        def wheel_surface_at(position):
            surface_name = track.surface(position)
            if not pit_transition:
                return surface_name
            pit_distance = track.pitlane_nearest(position)[0]
            if pit_distance <= PITLANE_WIDTH_M / 2 + 1.25:
                return "pitlane"
            if (
                surface_name == "wall"
                and pit_distance
                <= PITLANE_WIDTH_M / 2 + CAR_LENGTH_M
            ):
                # The paved entry/exit merge is the union between the main
                # road and pit road. Never wall-snap a partially merged car.
                return "grass"
            return surface_name

        wheel_surfaces = [
            wheel_surface_at(
                self.position + forward * longitudinal + normal * lateral
            )
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
        steer, throttle, brake = self.pit_guidance(
            track, steer, throttle, self.brake_input
        )
        self.brake_input = clamp(brake, 0.0, 1.0)
        self.steering_input = steer
        self.throttle_input = throttle
        if self.fuel <= 0:
            throttle = 0.0
        is_hybrid = self.generation == "Hybrid"
        self.recharge_active = bool(
            is_hybrid
            and recharge_request >= 0.5
            and self.battery < 100.0
        )
        hybrid_drs_deployment = float(
            is_hybrid and self.drs_eligible
            and not self.recharge_active
            and not self.in_pitlane and self.battery > 0.0
        )
        self.drs_active = bool(
            hybrid_drs_deployment
            or (
                not is_hybrid and self.drs_eligible
                and self.drs_in_zone and not self.in_pitlane
            )
        )
        self.overtake_active = bool(
            is_hybrid
            and not self.recharge_active
            and overtake_request >= 0.5
            and self.battery > 0.0
            and throttle > 0.0
            and self.brake_input < 0.05
        )
        deployment = 1.0 if self.overtake_active else 0.0
        electric_deployment = max(
            deployment, hybrid_drs_deployment
        )
        # Normal deployment is 80% combustion / 20% electric. DRS eligibility
        # forces deployment and adds another 10% electric output (80/30).
        if is_hybrid and self.recharge_active:
            power_ratio = (
                HYBRID_TOTAL_POWER_SCALE
                * HYBRID_RECHARGE_COMBUSTION_SHARE
            )
        elif is_hybrid:
            power_ratio = (
                HYBRID_TOTAL_POWER_SCALE
                * (
                    HYBRID_COMBUSTION_SHARE
                    + HYBRID_ELECTRIC_SHARE * electric_deployment
                    + HYBRID_DRS_ELECTRIC_BONUS
                    * hybrid_drs_deployment
                )
            )
        else:
            power_ratio = 1.0
        if is_hybrid:
            # Strong MGU-K recovery: full-speed heavy braking can restore
            # roughly 9.6 battery percentage points per second at 60 Hz.
            # Speed scaling prevents stationary or near-stationary charging.
            speed_factor = clamp(speed / 1.0, 0.0, 1.0)
            self.battery_regen = (
                self.brake_input ** 0.85
                * speed_factor
                * HYBRID_REGEN_RATE
                * dt
            )
            if self.recharge_active:
                # Throttle is normalized to 0.0–1.0, making this equivalent
                # to recharge_rate * (throttle_applied_percent / 100).
                # Recharge therefore produces no energy at zero throttle.
                throttle_applied = clamp(throttle, 0.0, 1.0)
                self.battery_regen += (
                    HYBRID_RECHARGE_RATE
                    * throttle_applied
                    * dt
                )
            battery_drain = (
                electric_deployment * 0.08
                + hybrid_drs_deployment * 0.04
            ) * throttle * dt
            self.battery = clamp(
                self.battery + self.battery_regen - battery_drain,
                0.0, 100.0,
            )
        else:
            self.battery = 0.0
            self.battery_regen = 0.0
            self.overtake_active = False
            self.recharge_active = False
        hybrid_speed = (
            1.48
            if self.recharge_active else
            1.55 + 0.12 * electric_deployment
            + 0.06 * hybrid_drs_deployment
        )
        ice_speed = 1.72 if self.drs_active else 1.67
        max_speed = (
            (hybrid_speed if is_hybrid else ice_speed)
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
        # Understeer remains useful telemetry but cannot overwhelm a full
        # steering command. The front axle keeps at least 88% authority.
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
            steer
            * (STEERING_BASE_YAW + speed * STEERING_SPEED_YAW)
            * grip * tyre_grip * aero_grip
            * (
                1.0
                - self.understeer * STEERING_UNDERSTEER_LOSS
                + self.oversteer * 0.20
            )
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
            grip * tyre_grip * LATERAL_GRIP_RECOVERY
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
            remaining_speed = current_speed - cornering_scrub
            # pygame-ce considers vectors below roughly 1e-6 to have zero
            # length inside scale_to_length. Snap that physically negligible
            # residual to rest before asking SDL to normalize it.
            if remaining_speed <= 1e-6:
                self.velocity.update(0.0, 0.0)
            else:
                self.velocity.scale_to_length(remaining_speed)

        # Rolling resistance plus quadratic aerodynamic drag means a small
        # maintenance throttle is no longer enough to hold maximum speed.
        current_speed = self.velocity.length()
        coast_drag = min(
            current_speed,
            (
                ROLLING_RESISTANCE
                + current_speed ** 2 * AERO_DRAG_COEFFICIENT
                * (
                    ICE_DRS_DRAG_MULTIPLIER
                    if (
                        self.generation == "ICE"
                        and self.drs_active
                    )
                    else 1.0
                )
            ) * dt,
        )
        if current_speed > 1e-9 and coast_drag > 0:
            remaining_speed = current_speed - coast_drag
            if remaining_speed <= 1e-6:
                self.velocity.update(0.0, 0.0)
            else:
                self.velocity.scale_to_length(remaining_speed)
        if self.velocity.length() > max_speed:
            self.velocity.scale_to_length(max_speed)
        if "grass" in wheel_surfaces:
            if pit_transition:
                # Keep enough momentum for the committed pit guidance to pull
                # the car back onto the paved merge instead of beaching it.
                self.dirty = max(self.dirty, 60)
                self.velocity *= 0.992 ** dt
            else:
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
        pit_recovery = (
            pit_transition
            and track.pitlane_nearest(self.position)[0]
            <= PITLANE_WIDTH_M / 2 + CAR_LENGTH_M * 1.5
        )
        if surface == "wall" and pit_recovery:
            # Preserve position continuity while the committed guidance steers
            # the car back into the merge corridor.
            self.velocity *= 0.72
        elif surface == "wall":
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
        previous_position = self.position.copy()
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
        if self.pit_requested and self.in_pitlane and pit_boxes:
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
        raw_progress_delta_m = (
            progress_m - self.previous_progress_m + track.measured_length_m / 2
        ) % track.measured_length_m - track.measured_length_m / 2
        movement = self.position - previous_position
        physical_step_m = movement.length()
        maximum_progress = max(0.05, physical_step_m * 1.5)
        # At crossings and close parallel sections, the global nearest-line
        # query can jump to a distant part of the lap. Reject impossible
        # topology jumps and fall back to actual forward vehicle movement.
        if abs(raw_progress_delta_m) > maximum_progress:
            progress_delta_m = clamp(
                movement.dot(forward),
                -physical_step_m, physical_step_m,
            )
        else:
            progress_delta_m = raw_progress_delta_m
        if self.race_distance_m is None:
            start_segment = (
                int(track.features.get("start_finish", 0))
                % len(track.points)
            ) * track.samples_per_section
            start_progress_m = track.cumulative_lengths_m[start_segment]
            relative_progress_m = (
                progress_m - start_progress_m
            ) % track.measured_length_m
            if self.lap == 0 and relative_progress_m > (
                track.measured_length_m / 2
            ):
                relative_progress_m -= track.measured_length_m
            self.race_distance_m = (
                self.lap * track.measured_length_m
                + relative_progress_m
            )
        self.race_distance_m += progress_delta_m
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

        def crossed_feature(feature):
            marker = track.features.get(feature)
            return (
                marker is not None
                and crossed(
                    (int(marker) % len(track.points))
                    * track.samples_per_section
                )
            )

        if crossed_feature("drs_detection"):
            self.drs_eligible = (
                self.drs_gap_seconds < DRS_MAX_GAP_SECONDS
            )
            if not self.drs_eligible:
                self.drs_active = False
        if crossed_feature("drs_entry"):
            self.drs_in_zone = True
        if crossed_feature("drs_exit"):
            self.drs_in_zone = False
            if self.generation == "ICE":
                self.drs_active = False

        start_marker = (
            int(track.features.get("start_finish", 0)) % len(track.points)
        ) * track.samples_per_section
        sectors = [
            (int(index) % len(track.points)) * track.samples_per_section
            for index in track.features.get("sectors", [])
        ]
        sectors.sort(
            key=lambda marker: (
                marker - start_marker
            ) % len(track.centerline)
        )
        if self.checkpoint < len(sectors) and crossed(sectors[self.checkpoint]):
            self.checkpoint += 1
        crossed_main_finish = track.crossed_timing_line(
            previous_position, self.position
        )
        crossed_pit_finish = (
            self.in_pitlane
            and track.crossed_timing_line(
                previous_position, self.position, pitlane=True
            )
        )
        if (
            (crossed_main_finish or crossed_pit_finish)
            and (not sectors or self.checkpoint >= len(sectors))
        ):
            self.lap += 1
            self.tyre_laps += 1
            self.checkpoint = 0
            self.race_distance_m = max(
                self.race_distance_m,
                self.lap * track.measured_length_m,
            )
        self.previous_progress = progress
        # Running order is physical distance only. Penalties still affect
        # training fitness, but cannot make the timetable jump positions.
        self.score = self.race_distance_m
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
        """Blit one prepared era/livery/rotation sprite for this car."""
        try:
            sprite = rotated_car_sprite(
                self.generation, self.color, scale, self.angle
            )
        except (FileNotFoundError, OSError, pygame.error):
            # Keep source checkouts usable if an asset is accidentally absent.
            return self._draw_legacy_procedural(
                screen, focused, offset, scale
            )
        screen_position = self.position * scale + offset
        screen.blit(
            sprite, sprite.get_rect(center=screen_position)
        )
        if focused:
            pygame.draw.circle(
                screen, CYAN, screen_position,
                max(12, round(2.2 * scale)), 2,
            )

    def _draw_legacy_procedural(
        self, screen, focused=False, offset=Vector2(), scale=1.0
    ):
        """Fallback renderer used only when packaged sprite assets are absent."""
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
    # Race grids use two staggered columns. P2 is beside but slightly behind
    # P1; P3 occupies the next slot directly behind P1.
    if wide_start:
        row = offset
        target_distance = row * (CAR_LENGTH_M + 4.0)
    else:
        row = offset // 2
        column = offset % 2
        row_spacing = CAR_LENGTH_M + 2.4
        column_stagger = 2.8
        target_distance = (
            row * row_spacing + column * column_stagger
        )
    start_control = (
        int(track.features.get("start_finish", 0)) % len(track.points)
    )
    current_index = start_control * track.samples_per_section
    current = track.centerline[current_index]
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
    grid_half_width = (
        CAR_WIDTH_M / 2 + .75
        if wide_start
        else clamp(
            track.road_width_m * .20,
            CAR_WIDTH_M / 2 + .55,
            max(
                CAR_WIDTH_M / 2 + .55,
                track.road_width_m / 2 - CAR_WIDTH_M / 2 - .75,
            ),
        )
    )
    position = start + normal * (-grid_half_width if offset % 2 else grid_half_width)
    car = Car(
        position.copy(), tangent.angle_to(Vector2(1, 0)) * -1,
        color, brain, name=name,
    )
    car.previous_progress = track.progress(car.position)
    car.previous_progress_m = track.progress_metres(car.position)
    start_progress_m = track.cumulative_lengths_m[
        start_control * track.samples_per_section
    ]
    relative_progress_m = (
        car.previous_progress_m - start_progress_m
    ) % track.measured_length_m
    if relative_progress_m > track.measured_length_m / 2:
        relative_progress_m -= track.measured_length_m
    car.race_distance_m = relative_progress_m
    car.score = relative_progress_m
    return car


class Button:
    def __init__(self, rect, title, subtitle=""):
        self.rect = pygame.Rect(rect)
        self.title = title
        self.subtitle = subtitle

    def draw(self, screen, fonts, mouse, draw_text=None):
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
        clean_title = self.title[len(number):].strip()
        if draw_text is None:
            number_surface = fonts["mono"].render(
                number, True, accent
            )
            screen.blit(
                number_surface,
                number_surface.get_rect(center=badge.center),
            )
            screen.blit(
                fonts["h2"].render(clean_title, True, WHITE),
                (self.rect.x + 82, self.rect.y + 17),
            )
            screen.blit(
                fonts["small"].render(self.subtitle, True, MUTED),
                (self.rect.x + 82, self.rect.y + 52),
            )
        else:
            draw_text(
                number, badge.center, "mono", accent, anchor="center"
            )
            draw_text(
                clean_title,
                (self.rect.x + 82, self.rect.y + 17),
                "h2", WHITE,
            )
            draw_text(
                self.subtitle,
                (self.rect.x + 82, self.rect.y + 52),
                "small", MUTED,
            )
        arrow = pygame.Rect(self.rect.right - 48, self.rect.centery - 16, 32, 32)
        pygame.draw.rect(
            screen,
            accent if hover else UI_SURFACE,
            arrow,
            border_radius=10,
        )
        if draw_text is None:
            arrow_text = fonts["body"].render(
                ">", True, INK if hover else MUTED
            )
            screen.blit(
                arrow_text, arrow_text.get_rect(center=arrow.center)
            )
        else:
            draw_text(
                ">", arrow.center, "body",
                INK if hover else MUTED, anchor="center",
            )
        return hover


class Game:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Formula AI Lab")
        self.display_flags = pygame.RESIZABLE | pygame.DOUBLEBUF
        display_info = pygame.display.Info()
        initial_window_size = (
            min(DISPLAY_WIDTH, display_info.current_w),
            min(
                DISPLAY_HEIGHT,
                display_info.current_h,
                max(HEIGHT, round(display_info.current_h * 0.85)),
            ),
        )
        self.window = pygame.display.set_mode(
            initial_window_size, self.display_flags
        )
        self._system_clipboard_ready = False
        self.ensure_system_clipboard()
        try:
            prewarm_car_sprites()
        except (FileNotFoundError, OSError, pygame.error):
            # Car.draw retains a procedural fallback for incomplete packages.
            pass
        self.screen = pygame.Surface((WIDTH, HEIGHT)).convert()
        self._ui_background = None
        self._present_surface = None
        self._present_size = None
        self._minimap_cache = {}
        self._native_text_commands = []
        self._native_font_cache = {}
        self.native_ui_modes = {
            "menu", "editor", "algorithm", "race_setup",
            "hotlap_setup", "replay_setup", "name_dialog",
        }
        self.viewport_scale = 1.0
        self.viewport_rect = pygame.Rect(0, 0, WIDTH, HEIGHT)
        self.update_viewport()
        self.clock = pygame.time.Clock()
        self.show_fps = False
        self.font_specs = {
            "title": (
                "Inter, SF Pro Display, Avenir Next, Arial", 44, True
            ),
            "h2": (
                "Inter, SF Pro Display, Avenir Next, Arial", 25, True
            ),
            "body": (
                "Inter, SF Pro Text, Avenir Next, Arial", 18, False
            ),
            "small": (
                "Inter, SF Pro Text, Avenir Next, Arial", 14, False
            ),
            "tiny": (
                "Inter, SF Pro Text, Avenir Next, Arial", 12, False
            ),
            "mono": (
                "SF Mono, Menlo, Consolas, monospace", 16, True
            ),
        }
        self.fonts = {
            key: pygame.font.SysFont(family, size, bold=bold)
            for key, (family, size, bold) in self.font_specs.items()
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
        self.best_fitness = None
        self.best_generation = 0
        self.last_generation_fitness = None
        self.last_generation_improved = False
        self.paused = False
        self.session_time = 0.0
        self.follow = 0
        self.race_tower_page = 0
        self.message = ""
        self.message_until = 0
        self.metric = 0
        self.target_laps = 3
        self.editor_points = []
        self.editor_widths = []
        self.editor_grass_widths = []
        self.editor_pitlane_points = []
        self.editor_pitlane_widths = []
        self.editor_pitlane_grass_widths = []
        self.editor_width_node = None
        self.editor_width_dragging = False
        self.editor_selected_kind = None
        self.editor_selected_index = None
        self.editor_node_dragging = False
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
        self.selected_replay = None
        self.replay_data = None
        self.replay_track = None
        self.replay_cars = []
        self.replay_frame_times = []
        self.replay_time = 0.0
        self.replay_rate = 1.0
        self.replay_resume_rate = 1.0
        self.replay_follow = 0
        self.replay_tower_page = 0
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
        self.menu_ui_manager = None
        self.menu_ui_buttons = []
        self.menu_ui_targets = {}
        # The native card renderer carries subtitles, badges and richer hover
        # feedback than a stock widget while still using pygame_gui elsewhere
        # when a future screen needs its layout engine.

    def setup_menu_ui(self):
        """Build the first pygame_gui screen without affecting game drawing."""
        if pygame_gui is None:
            return
        theme = {
            "button": {
                "colours": {
                    "normal_bg": "#122724",
                    "hovered_bg": "#193731",
                    "active_bg": "#43E1BE",
                    "selected_bg": "#193731",
                    "disabled_bg": "#0D1C1B",
                    "normal_text": "#EFF7F4",
                    "hovered_text": "#43E1BE",
                    "active_text": "#04090C",
                    "selected_text": "#43E1BE",
                    "disabled_text": "#849994",
                    "normal_border": "#304E48",
                    "hovered_border": "#43E1BE",
                    "active_border": "#43E1BE",
                    "selected_border": "#43E1BE",
                    "disabled_border": "#203631",
                },
                "font": {
                    "name": "noto_sans",
                    "size": "18",
                    "bold": "1",
                },
                "misc": {
                    "shape": "rounded_rectangle",
                    "shape_corner_radius": "14",
                    "border_width": "1",
                    "shadow_width": "3",
                    "text_horiz_alignment": "left",
                    "text_horiz_alignment_padding": "24",
                },
            }
        }
        self.menu_ui_manager = pygame_gui.UIManager(
            (WIDTH, HEIGHT),
            theme,
            enable_live_theme_updates=False,
        )
        definitions = (
            ((70, 170, 535, 74), "01   TRACK STUDIO"),
            ((675, 170, 535, 74), "02   AI TRAINING"),
            ((70, 260, 535, 74), "03   RACE WEEKEND"),
            ((675, 260, 535, 74), "04   TWO-LAP HOTLAP"),
            ((70, 350, 1140, 64), "05   REPLAY THEATRE"),
        )
        for index, (rect, label) in enumerate(definitions):
            button = pygame_gui.elements.UIButton(
                relative_rect=pygame.Rect(rect),
                text=label,
                manager=self.menu_ui_manager,
                object_id=f"#workspace_{index + 1}",
            )
            self.menu_ui_buttons.append(button)
            self.menu_ui_targets[button] = index

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
        # Simulation screens can scale directly into this display-backed view,
        # avoiding a second full-resolution blit every frame.
        self._window_viewport_surface = self.window.subsurface(
            self.viewport_rect
        )
        self.window.fill(INK)

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
        """Present logical pixels and redraw standstill text natively."""
        viewport_size = self.viewport_rect.size
        native_ui = self.native_ui_rendering()
        if not native_ui:
            pygame.transform.scale(
                self.screen,
                viewport_size,
                self._window_viewport_surface,
            )
            pygame.display.flip()
            return

        self.window.fill(INK)
        if self._present_size != viewport_size:
            self._present_surface = pygame.Surface(viewport_size).convert()
            self._present_size = viewport_size
        pygame.transform.smoothscale(
            self.screen, viewport_size, self._present_surface
        )
        for value, position, font, color, anchor, clip in (
            self._native_text_commands
        ):
            self.draw_native_text(
                value, position, font, color, anchor, clip
            )
        # Compose the entire viewport off-screen before touching the macOS
        # display surface. Direct alpha-font blits to a Retina backbuffer can
        # occasionally become solid color rectangles on pygame/SDL.
        self.window.blit(self._present_surface, self.viewport_rect)
        pygame.display.flip()
        # Keep a complete logical frame for modal background snapshots. This
        # happens after presentation, so enlarged users only see native text
        # while the next frame can still copy the workspace.
        for command in self._native_text_commands:
            self.draw_logical_text_command(*command)

    def native_ui_rendering(self):
        """Return whether the current screen benefits from native text."""
        return (
            getattr(self, "mode", None) in self.native_ui_modes
            and self.viewport_scale > 1.001
        )

    def native_font(self, font):
        """Return a cached font rasterized for the physical viewport."""
        family, logical_size, bold = self.font_specs[font]
        physical_size = max(1, round(logical_size * self.viewport_scale))
        key = (font, physical_size)
        cached = self._native_font_cache.get(key)
        if cached is None:
            cached = pygame.font.SysFont(
                family, physical_size, bold=bold
            )
            self._native_font_cache[key] = cached
        return cached

    def draw_native_text(
        self, value, position, font, color, anchor="topleft",
        clip=None,
    ):
        """Draw native-density text into the composited viewport."""
        rendered = self.native_font(font).render(
            str(value), True, color
        )
        physical_position = (
            round(float(position[0]) * self.viewport_scale),
            round(float(position[1]) * self.viewport_scale),
        )
        rect = rendered.get_rect()
        setattr(rect, anchor, physical_position)
        previous_clip = self._present_surface.get_clip()
        if clip is not None:
            self._present_surface.set_clip(pygame.Rect(
                round(clip.x * self.viewport_scale),
                round(clip.y * self.viewport_scale),
                max(1, round(clip.width * self.viewport_scale)),
                max(1, round(clip.height * self.viewport_scale)),
            ))
        self._present_surface.blit(rendered, rect)
        if clip is not None:
            self._present_surface.set_clip(previous_clip)

    def draw_logical_text_command(
        self, value, position, font, color, anchor="topleft",
        clip=None,
    ):
        """Backfill logical text after native presentation for snapshots."""
        rendered = self.fonts[font].render(str(value), True, color)
        rect = rendered.get_rect()
        setattr(rect, anchor, position)
        previous_clip = self.screen.get_clip()
        if clip is not None:
            self.screen.set_clip(clip)
        self.screen.blit(rendered, rect)
        if clip is not None:
            self.screen.set_clip(previous_clip)

    def text(
        self, value, pos, font="body", color=WHITE,
        anchor="topleft",
    ):
        """Draw logical text or queue a native-density text command."""
        position = (float(pos[0]), float(pos[1]))
        clip = self.screen.get_clip()
        if clip == self.screen.get_rect():
            clip = None
        self._native_text_commands.append(
            (str(value), position, font, color, anchor, clip)
        )
        if self.native_ui_rendering():
            return
        rendered = self.fonts[font].render(str(value), True, color)
        rect = rendered.get_rect()
        setattr(rect, anchor, position)
        self.screen.blit(rendered, rect)

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

    def section_heading(
        self, label, rect, accent=CYAN, detail=""
    ):
        """Draw a quiet section divider used by standstill workspaces."""
        rect = pygame.Rect(rect)
        pygame.draw.circle(
            self.screen, accent, (rect.x + 4, rect.centery), 3
        )
        self.text(label.upper(), (rect.x + 15, rect.y), "tiny", accent)
        if detail:
            self.text(
                str(detail).upper(),
                (rect.right, rect.centery),
                "tiny", MUTED, anchor="midright",
            )
        line_start = rect.x + min(
            175, self.fonts["tiny"].size(label.upper())[0] + 28
        )
        if not detail and line_start < rect.right:
            pygame.draw.line(
                self.screen, UI_BORDER,
                (line_start, rect.centery),
                (rect.right, rect.centery),
            )

    def control_field(
        self, rect, label, value, accent=CYAN, arrows=False,
        disabled=False, active=False, value_font="small",
    ):
        """Draw a labeled, hover-aware selector or editable field."""
        rect = pygame.Rect(rect)
        mouse = self.logical_mouse_position()
        hovered = rect.collidepoint(mouse) and not disabled
        fill = (
            (10, 22, 22) if disabled else
            UI_SURFACE_HOVER if hovered or active else UI_SURFACE
        )
        border = (
            (35, 48, 45) if disabled else
            accent if active else
            UI_BORDER_BRIGHT if hovered else UI_BORDER
        )
        pygame.draw.rect(
            self.screen, UI_SHADOW, rect.move(0, 3),
            border_radius=10,
        )
        pygame.draw.rect(self.screen, fill, rect, border_radius=10)
        pygame.draw.rect(
            self.screen, border, rect, 1, border_radius=10
        )
        pygame.draw.rect(
            self.screen,
            (50, 64, 60) if disabled else accent,
            (rect.x, rect.y + 10, 3, max(12, rect.height - 20)),
            border_radius=2,
        )
        if rect.height >= 48:
            self.text(
                str(label).upper(), (rect.x + 14, rect.y + 7),
                "tiny", MUTED if not disabled else (70, 82, 78),
            )
            value_y = rect.y + 25
        else:
            self.text(
                str(label).upper(), (rect.x + 13, rect.y + 11),
                "tiny", MUTED if not disabled else (70, 82, 78),
            )
            value_y = rect.y + 10
        value_color = (
            (78, 91, 86) if disabled else accent if arrows else WHITE
        )
        if rect.height >= 48:
            self.text(
                str(value), (rect.x + 14, value_y),
                value_font, value_color,
            )
        else:
            value_width = self.fonts[value_font].size(str(value))[0]
            value_x = max(
                rect.x + 82,
                rect.right - value_width - 18,
            )
            self.text(
                str(value), (value_x, value_y),
                value_font, value_color,
            )
        if arrows:
            self.text("‹", (rect.x + 8, rect.centery - 10), "body", accent)
            self.text(
                "›", (rect.right - 19, rect.centery - 10),
                "body", accent,
            )
        return hovered

    def compact_stepper(
        self, label, value, minus_rect, plus_rect, accent=CYAN,
        label_x=None,
    ):
        """Draw one label/value row and its paired decrement controls."""
        minus_rect = pygame.Rect(minus_rect)
        plus_rect = pygame.Rect(plus_rect)
        self.text(
            str(label).upper(),
            (
                minus_rect.x - 94 if label_x is None else label_x,
                minus_rect.centery - 7,
            ),
            "tiny", MUTED,
        )
        self.text(
            str(value),
            (minus_rect.x - 11, minus_rect.centery),
            "mono", accent, anchor="midright",
        )
        mouse = self.logical_mouse_position()
        for rect, symbol in ((minus_rect, "−"), (plus_rect, "+")):
            hovered = rect.collidepoint(mouse)
            pygame.draw.rect(
                self.screen,
                accent if hovered else UI_SURFACE_HOVER,
                rect,
                border_radius=7,
            )
            pygame.draw.rect(
                self.screen,
                accent if hovered else UI_BORDER,
                rect, 1, border_radius=7,
            )
            self.text(
                symbol, rect.center, "body",
                INK if hovered else WHITE, anchor="center",
            )

    def action_button(
        self, rect, label, accent=CYAN, enabled=True,
        secondary=False, detail="",
    ):
        """Draw a consistent primary or secondary standstill action."""
        rect = pygame.Rect(rect)
        hovered = (
            enabled and rect.collidepoint(self.logical_mouse_position())
        )
        if secondary:
            fill = UI_SURFACE_HOVER if hovered else UI_SURFACE_RAISED
            border = accent if hovered else UI_BORDER
            label_color = accent if hovered else MUTED
        else:
            fill = (
                tuple(min(255, channel + 16) for channel in accent)
                if hovered else accent
            ) if enabled else (47, 66, 61)
            border = fill
            label_color = INK if enabled else MUTED
        pygame.draw.rect(
            self.screen, UI_SHADOW, rect.move(0, 4),
            border_radius=11,
        )
        pygame.draw.rect(self.screen, fill, rect, border_radius=11)
        pygame.draw.rect(
            self.screen, border, rect, 1, border_radius=11
        )
        label_y = rect.centery - (7 if detail else 0)
        self.text(
            str(label).upper(), (rect.centerx, label_y),
            "mono", label_color, anchor="center",
        )
        if detail:
            self.text(
                str(detail), (rect.centerx, rect.centery + 13),
                "tiny",
                MUTED if secondary or not enabled else (26, 72, 62),
                anchor="center",
            )
        return hovered

    def draw_fps_counter(self):
        """Draw a compact optional performance counter."""
        label = f"FPS {self.clock.get_fps():5.1f}"
        value_size = self.fonts["mono"].size(label)
        card = pygame.Rect(
            CANVAS_W - 14 - value_size[0],
            14, value_size[0], value_size[1],
        ).inflate(30, 12)
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
        self.text(
            label, (card.x + 20, card.centery),
            "mono", CYAN, anchor="midleft",
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

    def draw_minimap(
        self, cars, focused=None, rect=(16, 16, 220, 165), track=None
    ):
        """Draw the complete circuit and color-coded live car positions."""
        track = track or self.track
        rect = pygame.Rect(rect)
        cache_key = (id(track), rect.size, track.name)
        cached = self._minimap_cache.get(cache_key)
        if cached is None:
            cached = self._build_minimap_background(rect, track)
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

    def _build_minimap_background(self, rect, track=None):
        """Render static minimap geometry once for the active track."""
        track = track or self.track
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
            f"TRACK MAP  •  {track.name[:14].upper()}",
            True, CYAN,
        )
        overlay.blit(title, (10, 7))
        # Keep thick circuit strokes and car markers away from the card edge.
        map_rect = pygame.Rect(
            16, 32, rect.width - 32, rect.height - 47
        )
        track_count = len(track.centerline)
        scale, offset, projected_points = self.minimap_projection(
            track.centerline + track.pitlane_centerline,
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
                int(track.features.get("start_finish", 0))
                % max(len(track.points), 1)
            ) * track.samples_per_section
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
        self.text(
            "Give it a clear name so it is easy to find in selectors.",
            (390, 308), "small", MUTED,
        )
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
        save_label = "REPLACE" if dialog["replace"] else "SAVE"
        self.action_button(
            cancel, "Cancel", CYAN, True, secondary=True
        )
        self.action_button(save, save_label, CYAN)
        helper = dialog["error"] or "Letters, numbers, spaces, hyphens and underscores"
        color = (255, 170, 170) if dialog["error"] else MUTED
        self.text(helper, (390, 397), "small", color)
        self.text(
            f"{len(dialog['value'])} CHARACTERS",
            (field.right, 397), "tiny", MUTED, anchor="topright",
        )
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

    def consider_training_champion(self):
        """Promote only a generation winner that beats the all-time record."""
        if not self.cars:
            return None, False
        champion = max(self.cars, key=lambda car: car.fitness)
        champion_fitness = float(champion.fitness)
        previous_best = getattr(self, "best_fitness", None)
        improved = (
            previous_best is None
            or champion_fitness > previous_best
        )
        self.last_generation_fitness = champion_fitness
        self.last_generation_improved = improved
        if improved:
            self.best_brain = champion.brain
            self.best_fitness = champion_fitness
            self.best_generation = self.generation
        return champion, improved

    def reset_training(self, evolve=False):
        if evolve and self.cars:
            self.consider_training_champion()
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
            program = Brain.compile_source(self.algorithm_source)
            self.algorithm_sources[self.training_generation] = (
                self.algorithm_source
            )
            self.best_brain = self.create_training_seed(
                program, self.algorithm_source
            )
            self.generation = 0
            self.best_fitness = None
            self.best_generation = 0
            self.last_generation_fitness = None
            self.last_generation_improved = False
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
        if not self.ensure_system_clipboard():
            return
        try:
            pygame.scrap.put(pygame.SCRAP_TEXT, text.encode("utf-8") + b"\0")
        except (pygame.error, TypeError):
            self._system_clipboard_ready = False

    def editor_get_clipboard(self):
        if not self.ensure_system_clipboard():
            return self.algorithm_clipboard
        try:
            raw = pygame.scrap.get(pygame.SCRAP_TEXT)
            if raw:
                text = raw.rstrip(b"\0").decode("utf-8", errors="replace")
                return text.replace("\r\n", "\n").replace("\r", "\n")
        except (pygame.error, TypeError):
            self._system_clipboard_ready = False
        return self.algorithm_clipboard

    def ensure_system_clipboard(self):
        """Initialize SDL's clipboard after the display is available."""
        if getattr(self, "_system_clipboard_ready", False):
            return True
        try:
            pygame.scrap.init()
            self._system_clipboard_ready = True
        except (pygame.error, TypeError):
            self._system_clipboard_ready = False
        return self._system_clipboard_ready

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
            car.starting_position = i + 1
            car.team = self.team_names[i // 2] if self.race_settings["teams"] else ""
            car.pit_box_index = i // 2 if self.race_settings["teams"] else i
            car.timing_history = [(car.score, 0.0)]
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
        self.race_tower_page = 0
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
                    "color": list(car.color),
                    "lap": car.lap, "tyre": car.tyre, "wear": round(car.tyre_wear, 2),
                    "fuel": round(car.fuel, 2), "pit_requested": car.pit_requested,
                    "slipstream": round(car.slipstream, 3),
                    "generation": car.generation,
                    "battery": round(car.battery, 2),
                    "overtake": car.overtake_active,
                    "recharge": car.recharge_active,
                    "drs_eligible": car.drs_eligible,
                    "drs_active": car.drs_active,
                    "drs_gap_seconds": (
                        round(car.drs_gap_seconds, 3)
                        if math.isfinite(car.drs_gap_seconds) else None
                    ),
                    "brake": round(car.brake_input, 3),
                    "aggression": round(car.race_aggression, 3),
                    "aggression_error": round(car.aggression_error, 3),
                    "speed_kph": round(car.speed_kph, 1),
                    "gear": car.gear,
                    "rpm": round(car.rpm),
                    "health": round(car.health, 2), "pitstops": car.pitstops,
                    "starting_position": car.starting_position,
                    "finish_time": car.finish_time,
                    "retirement_time": car.retirement_time,
                    "retirement_reason": car.retirement_reason,
                    "removed_from_track": car.removed_from_track,
                    "team": car.team, "brain": car.brain_name,
                }
                for car in self.cars
            ],
        })

    def save_replay(self):
        REPLAY_DIR.mkdir(parents=True, exist_ok=True)
        if self.cars:
            # Include the exact save moment and guarantee that even a replay
            # saved during the start sequence contains a drawable frame.
            self.capture_replay_frame()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        track_slug = component_filename(self.track.name, "")
        path = REPLAY_DIR / f"{track_slug}_{timestamp}.json"
        path.write_text(json.dumps({
            "version": 1,
            "track": self.track.name,
            "track_file": self.selected_track,
            "settings": self.race_settings,
            "teams": self.team_names if self.race_settings["teams"] else [],
            "events": self.event_log,
            "frames": self.replay_frames,
        }, indent=2))
        self.replay_saved = True
        self.notice(f"Replay saved: {path.name}")
        return path

    def replay_choices(self):
        """Return replay JSON filenames, newest first."""
        try:
            paths = sorted(
                REPLAY_DIR.glob("*.json"),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
        except OSError:
            return []
        return [path.name for path in paths]

    def cycle_replay(self, amount=1):
        choices = self.replay_choices()
        if not choices:
            self.selected_replay = None
            return None
        try:
            index = choices.index(self.selected_replay)
        except ValueError:
            index = -1 if amount > 0 else 0
        self.selected_replay = choices[(index + amount) % len(choices)]
        return self.selected_replay

    def replay_track_from_data(self, data):
        """Resolve the circuit referenced by a replay, with legacy fallback."""
        track_file = data.get("track_file")
        if isinstance(track_file, str):
            candidate = TRACK_DIR / Path(track_file).name
            if candidate.exists():
                return Track.load(candidate), False
        track_name = str(data.get("track", ""))
        for candidate in TRACK_DIR.glob("*.json"):
            try:
                metadata = json.loads(candidate.read_text())
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                continue
            if str(metadata.get("name", "")) == track_name:
                return Track.load(candidate), False
        return self.track, True

    @staticmethod
    def replay_angle_lerp(first, second, amount):
        delta = (float(second) - float(first) + 180.0) % 360.0 - 180.0
        return float(first) + delta * amount

    def start_selected_replay(self):
        choices = self.replay_choices()
        if self.selected_replay not in choices:
            self.selected_replay = choices[0] if choices else None
        if self.selected_replay is None:
            self.notice("No replay JSON files are available")
            return False
        path = REPLAY_DIR / Path(self.selected_replay).name
        try:
            data = json.loads(path.read_text())
            frames = data.get("frames")
            if not isinstance(frames, list) or not frames:
                raise ValueError("Replay contains no frames")
            if not all(
                isinstance(frame, dict)
                and isinstance(frame.get("cars"), list)
                and bool(frame["cars"])
                for frame in frames
            ):
                raise ValueError("Replay frame data is invalid")
            track, used_fallback = self.replay_track_from_data(data)
            frame_times = [float(frame.get("time", 0.0)) for frame in frames]
            if any(
                right < left
                for left, right in zip(frame_times, frame_times[1:])
            ):
                raise ValueError("Replay timestamps are not ordered")
        except (
            OSError, TypeError, ValueError, KeyError,
            json.JSONDecodeError,
        ) as error:
            self.notice(f"Replay could not be loaded: {error}")
            return False

        self.replay_data = data
        self.replay_track = track
        self.replay_frame_times = frame_times
        self.replay_time = frame_times[0]
        self.replay_rate = 1.0
        self.replay_resume_rate = 1.0
        first_cars = frames[0]["cars"]
        self.replay_cars = []
        for index, record in enumerate(first_cars):
            raw_color = record.get("color", COLORS[index % len(COLORS)])
            try:
                color = tuple(
                    int(clamp(float(channel), 0, 255))
                    for channel in raw_color[:3]
                )
                if len(color) != 3:
                    raise ValueError
            except (TypeError, ValueError):
                color = COLORS[index % len(COLORS)]
            self.replay_cars.append(Car(
                Vector2(
                    float(record.get("x", 0.0)),
                    float(record.get("y", 0.0)),
                ),
                float(record.get("angle", 0.0)),
                color,
                Brain(),
                name=str(record.get("name", f"Driver {index + 1}")),
                generation=str(record.get("generation", "ICE")),
            ))
        focus = int(frames[0].get("focus", 0))
        self.replay_follow = int(clamp(
            focus, 0, max(len(self.replay_cars) - 1, 0)
        ))
        self.replay_tower_page = self.replay_follow // 10
        self.camera_zoom = DEFAULT_CAMERA_ZOOM
        self.apply_replay_time()
        self.mode = "replay"
        if used_fallback:
            self.notice(
                "Recorded circuit was not found; using the active circuit"
            )
        return True

    def replay_frame_pair(self):
        """Return timeline frames surrounding the current replay time."""
        frames = self.replay_data["frames"]
        right = bisect_right(self.replay_frame_times, self.replay_time)
        left = int(clamp(right - 1, 0, len(frames) - 1))
        right = int(clamp(right, left, len(frames) - 1))
        start_time = self.replay_frame_times[left]
        end_time = self.replay_frame_times[right]
        amount = (
            clamp(
                (self.replay_time - start_time) / (end_time - start_time),
                0.0, 1.0,
            )
            if end_time > start_time else 0.0
        )
        return frames[left], frames[right], amount

    def apply_replay_time(self):
        """Interpolate JSON snapshots into drawable car objects."""
        if not self.replay_data or not self.replay_cars:
            return
        first_frame, second_frame, amount = self.replay_frame_pair()
        first_cars = first_frame["cars"]
        second_cars = second_frame["cars"]
        for index, car in enumerate(self.replay_cars):
            first = first_cars[min(index, len(first_cars) - 1)]
            second = second_cars[min(index, len(second_cars) - 1)]
            car.position.update(
                float(first.get("x", 0.0))
                + (float(second.get("x", 0.0)) - float(first.get("x", 0.0)))
                * amount,
                float(first.get("y", 0.0))
                + (float(second.get("y", 0.0)) - float(first.get("y", 0.0)))
                * amount,
            )
            car.angle = self.replay_angle_lerp(
                first.get("angle", 0.0), second.get("angle", 0.0), amount
            )
            state = second if amount >= 0.5 else first
            car.name = str(state.get("name", car.name))
            car.generation = str(state.get("generation", car.generation))
            car.lap = int(state.get("lap", 0))
            car.tyre = str(state.get("tyre", "Medium"))
            car.tyre_wear = float(state.get("wear", 0.0))
            car.fuel = float(state.get("fuel", 0.0))
            car.pit_requested = bool(state.get("pit_requested", False))
            car.slipstream = float(state.get("slipstream", 0.0))
            car.battery = float(state.get("battery", 0.0))
            car.overtake_active = bool(state.get("overtake", False))
            car.recharge_active = bool(state.get("recharge", False))
            car.drs_eligible = bool(state.get("drs_eligible", False))
            car.drs_active = bool(state.get("drs_active", False))
            drs_gap = state.get("drs_gap_seconds")
            car.drs_gap_seconds = (
                float(drs_gap) if drs_gap is not None else float("inf")
            )
            car.brake_input = float(state.get("brake", 0.0))
            car.race_aggression = float(state.get("aggression", 0.0))
            car.aggression_error = float(
                state.get("aggression_error", 0.0)
            )
            car.gear = int(state.get("gear", 1))
            car.rpm = float(state.get("rpm", IDLE_ENGINE_RPM))
            car.health = float(state.get("health", 100.0))
            car.pitstops = int(state.get("pitstops", 0))
            car.starting_position = int(
                state.get("starting_position", index + 1)
            )
            car.finish_time = state.get("finish_time")
            car.retirement_time = state.get("retirement_time")
            car.retirement_reason = str(state.get("retirement_reason", ""))
            car.removed_from_track = bool(
                state.get("removed_from_track", False)
            )
            car.alive = car.health > 0.0 and car.retirement_time is None
            car.team = str(state.get("team", ""))
            car.brain_name = str(state.get("brain", "CURRENT SESSION"))
            speed = float(state.get("speed_kph", 0.0)) / (FPS * 3.6)
            car.velocity = Vector2(1, 0).rotate(car.angle) * speed
            progress = self.replay_track.progress_metres(car.position)
            car.race_distance_m = (
                car.lap * self.replay_track.measured_length_m + progress
            )
            car.score = car.race_distance_m

    def replay_seek(self, seconds):
        if not self.replay_frame_times:
            return
        self.replay_time = clamp(
            self.replay_time + float(seconds),
            self.replay_frame_times[0],
            self.replay_frame_times[-1],
        )
        self.apply_replay_time()

    def set_replay_transport(self, direction):
        """Select rewind/pause/fast-forward, accelerating on repeat presses."""
        direction = int(clamp(direction, -1, 1))
        if direction == 0:
            if self.replay_rate:
                self.replay_resume_rate = self.replay_rate
            self.replay_rate = 0.0
            return
        if self.replay_rate * direction > 0:
            rate = min(8.0, abs(self.replay_rate) * 2.0)
        else:
            rate = 1.0
        self.replay_rate = direction * rate
        self.replay_resume_rate = self.replay_rate

    def toggle_replay_pause(self):
        if self.replay_rate:
            self.set_replay_transport(0)
        else:
            self.replay_rate = self.replay_resume_rate or 1.0

    def replay_ranked_cars(self):
        return sorted(
            self.replay_cars, key=self.race_order_key, reverse=True
        )

    def sync_replay_tower_page(self, ranked):
        if not ranked or not self.replay_cars:
            self.replay_tower_page = 0
            return
        followed = self.replay_cars[int(clamp(
            self.replay_follow, 0, len(self.replay_cars) - 1
        ))]
        rank_index = next(
            (index for index, car in enumerate(ranked) if car is followed), 0
        )
        self.replay_tower_page = rank_index // 10

    def change_replay_focus(self, ranked, direction):
        if not ranked or not self.replay_cars:
            return
        followed = self.replay_cars[int(clamp(
            self.replay_follow, 0, len(self.replay_cars) - 1
        ))]
        rank_index = next(
            (index for index, car in enumerate(ranked) if car is followed), 0
        )
        rank_index = int(clamp(
            rank_index + int(direction), 0, len(ranked) - 1
        ))
        selected = ranked[rank_index]
        self.replay_follow = next(
            index for index, car in enumerate(self.replay_cars)
            if car is selected
        )
        self.replay_tower_page = rank_index // 10

    @staticmethod
    def replay_setup_rects():
        return {
            "previous": pygame.Rect(104, 392, 58, 52),
            "next": pygame.Rect(1118, 392, 58, 52),
            "start": pygame.Rect(420, 548, 440, 62),
        }

    def replay_setup(self, events):
        self.draw_app_background()
        self.text("REPLAY THEATRE", (70, 55), "title", WHITE)
        self.text(
            "LOAD A RACE RECORDING  /  DIRECT EVERY CAMERA",
            (74, 112), "mono", CYAN,
        )
        choices = self.replay_choices()
        if choices and self.selected_replay not in choices:
            self.selected_replay = choices[0]
        rects = self.replay_setup_rects()
        card = pygame.Rect(70, 175, 1140, 320)
        self.glass_card(card, accent=UI_VIOLET, radius=18)
        self.section_heading(
            "Replay file", (100, 202, 1080, 28), UI_VIOLET,
            f"{len(choices)} recording{'s' if len(choices) != 1 else ''}",
        )
        if self.selected_replay:
            path = REPLAY_DIR / self.selected_replay
            try:
                size_mb = path.stat().st_size / (1024 * 1024)
            except OSError:
                size_mb = 0.0
            self.text(
                self.selected_replay,
                (WIDTH // 2, 315), "h2", WHITE, anchor="center",
            )
            self.text(
                f"JSON RACE RECORDING  •  {size_mb:.1f} MB",
                (WIDTH // 2, 357), "small", MUTED, anchor="center",
            )
        else:
            self.text(
                "NO REPLAY FILES FOUND",
                (WIDTH // 2, 325), "h2", MUTED, anchor="center",
            )
            self.text(
                "Save a completed or running race with R, then return here.",
                (WIDTH // 2, 366), "small", MUTED, anchor="center",
            )
        for key, label in (("previous", "<"), ("next", ">")):
            self.action_button(
                rects[key], label, UI_VIOLET, bool(choices), secondary=True
            )
        self.action_button(
            rects["start"], "Open replay", UI_VIOLET, bool(choices),
            detail="Interpolated playback • ranked driver cameras",
        )
        self.text(
            "LEFT/RIGHT selects  •  ENTER opens  •  ESC returns",
            (WIDTH // 2, 667), "small", MUTED, anchor="center",
        )
        for event in events:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.mode = "menu"
                elif event.key == pygame.K_LEFT:
                    self.cycle_replay(-1)
                elif event.key == pygame.K_RIGHT:
                    self.cycle_replay(1)
                elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                    self.start_selected_replay()
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if rects["previous"].collidepoint(event.pos):
                    self.cycle_replay(-1)
                elif rects["next"].collidepoint(event.pos):
                    self.cycle_replay(1)
                elif rects["start"].collidepoint(event.pos):
                    self.start_selected_replay()

    @staticmethod
    def replay_transport_rects():
        return {
            "rewind": pygame.Rect(26, HEIGHT - 62, 82, 38),
            "pause": pygame.Rect(116, HEIGHT - 62, 82, 38),
            "forward": pygame.Rect(206, HEIGHT - 62, 82, 38),
            "timeline": pygame.Rect(322, HEIGHT - 49, CANVAS_W - 350, 12),
        }

    @staticmethod
    def format_replay_time(seconds):
        seconds = max(0.0, float(seconds))
        minutes, seconds = divmod(seconds, 60.0)
        hours, minutes = divmod(int(minutes), 60)
        if hours:
            return f"{hours:d}:{minutes:02d}:{seconds:04.1f}"
        return f"{minutes:02d}:{seconds:04.1f}"

    def replay_timeline_fraction(self):
        if len(self.replay_frame_times) < 2:
            return 0.0
        start = self.replay_frame_times[0]
        duration = self.replay_frame_times[-1] - start
        return clamp((self.replay_time - start) / max(duration, 1e-9), 0, 1)

    def draw_replay_transport(self):
        rects = self.replay_transport_rects()
        card = pygame.Rect(16, HEIGHT - 72, CANVAS_W - 32, 58)
        pygame.draw.rect(
            self.screen, (5, 13, 16), card, border_radius=12
        )
        pygame.draw.rect(
            self.screen, UI_BORDER_BRIGHT, card, 1, border_radius=12
        )
        mouse = self.logical_mouse_position()
        labels = {
            "rewind": "J  REW",
            "pause": "K  PLAY" if self.replay_rate == 0 else "K  PAUSE",
            "forward": "L  FWD",
        }
        for key in ("rewind", "pause", "forward"):
            rect = rects[key]
            hovered = rect.collidepoint(mouse)
            active = (
                key == "rewind" and self.replay_rate < 0
                or key == "pause" and self.replay_rate == 0
                or key == "forward" and self.replay_rate > 0
            )
            pygame.draw.rect(
                self.screen,
                UI_VIOLET if hovered else UI_SURFACE_HOVER if active
                else UI_SURFACE_RAISED,
                rect, border_radius=8,
            )
            pygame.draw.rect(
                self.screen,
                UI_VIOLET if hovered or active else UI_BORDER,
                rect, 1, border_radius=8,
            )
            self.text(
                labels[key], rect.center, "tiny",
                INK if hovered else WHITE, anchor="center",
            )
        timeline = rects["timeline"]
        pygame.draw.rect(
            self.screen, (35, 52, 51), timeline, border_radius=6
        )
        fraction = self.replay_timeline_fraction()
        fill = timeline.copy()
        fill.width = round(timeline.width * fraction)
        if fill.width:
            pygame.draw.rect(
                self.screen, UI_VIOLET, fill, border_radius=6
            )
        knob_x = timeline.x + round(timeline.width * fraction)
        pygame.draw.circle(
            self.screen, WHITE, (knob_x, timeline.centery), 7
        )
        duration = self.replay_frame_times[-1]
        rate = "PAUSED" if self.replay_rate == 0 else f"{self.replay_rate:+g}x"
        self.text(
            (
                f"{self.format_replay_time(self.replay_time)}  /  "
                f"{self.format_replay_time(duration)}   {rate}"
            ),
            (timeline.centerx, timeline.y - 18), "tiny", WHITE,
            anchor="center",
        )

    def draw_replay_tower(self, ranked):
        x = CANVAS_W
        self.panel("Replay Director", "JSON PLAYBACK  •  MANUAL CAM")
        settings = self.replay_data.get("settings", {})
        target_laps = int(settings.get("laps", 0))
        focused = self.replay_cars[self.replay_follow]
        lap_label = f"LAP {focused.lap + 1}"
        if target_laps:
            lap_label += f" / {target_laps}"
        self.text(lap_label, (x + 20, 101), "h2", WHITE)
        self.text(
            self.format_replay_time(self.replay_time),
            (x + 170, 108), "mono", YELLOW,
        )
        page, start, end = self.race_tower_page_bounds(
            len(ranked), self.replay_tower_page
        )
        self.replay_tower_page = page
        self.text(
            f"POSITIONS {start + 1}–{end}",
            (x + 20, 128), "tiny", MUTED,
        )
        metric_card = pygame.Rect(x + 12, 143, PANEL - 24, 34)
        pygame.draw.rect(
            self.screen, UI_SURFACE_RAISED, metric_card, border_radius=9
        )
        pygame.draw.rect(
            self.screen, UI_BORDER, metric_card, 1, border_radius=9
        )
        self.text(
            "LAP  •  SPEED  •  GEAR  •  ENERGY",
            metric_card.center, "tiny", CYAN, anchor="center",
        )
        for row, car in enumerate(ranked[start:end]):
            rank_index = start + row
            y = 190 + row * 48
            selected = self.replay_cars[self.replay_follow] is car
            dnf = car.retirement_time is not None
            row_rect = pygame.Rect(x + 10, y, PANEL - 20, 41)
            pygame.draw.rect(
                self.screen,
                (37, 40, 41) if dnf else
                UI_SURFACE_HOVER if selected else UI_SURFACE,
                row_rect, border_radius=9,
            )
            pygame.draw.rect(
                self.screen,
                car.color if selected else UI_BORDER,
                row_rect, 1, border_radius=9,
            )
            pygame.draw.rect(
                self.screen, car.color,
                (x + 15, y + 7, 5, 27), border_radius=2,
            )
            self.text(
                f"{rank_index + 1:>2}", (x + 27, y + 10), "mono",
                WHITE if selected else MUTED,
            )
            self.text(car.name[:10], (x + 57, y + 6), "mono", WHITE)
            if dnf:
                value = "DNF"
            elif car.finish_time is not None:
                value = "FINISHED"
            elif car.generation == "Hybrid":
                energy = (
                    "REGEN" if car.recharge_active else
                    "DEPLOY" if car.overtake_active or car.drs_active else
                    f"{car.battery:.0f}%"
                )
                value = (
                    f"L{car.lap + 1}  {car.speed_kph:.0f}  "
                    f"G{car.gear}  {energy}"
                )
            else:
                value = f"L{car.lap + 1}  {car.speed_kph:.0f}km/h  G{car.gear}"
            self.text(
                value, (x + 148, y + 22), "small",
                CYAN if selected else MUTED,
            )
        if len(ranked) > 10:
            self.text(
                f"PAGE {page + 1}/{(len(ranked) - 1) // 10 + 1}",
                (x + 20, 675), "small", MUTED,
            )
        self.footer_hint((
            "UP/DOWN Driver  •  Click rows",
            "J/K/L Transport  •  ESC Library",
        ))

    def replay(self, events, dt):
        if not self.replay_data or not self.replay_cars:
            self.mode = "replay_setup"
            return
        for event in events:
            if self.handle_camera_zoom_event(event):
                continue
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.mode = "replay_setup"
                elif event.key == pygame.K_SPACE:
                    self.toggle_replay_pause()
                elif event.key == pygame.K_j:
                    self.set_replay_transport(-1)
                elif event.key == pygame.K_k:
                    self.set_replay_transport(0)
                elif event.key == pygame.K_l:
                    self.set_replay_transport(1)
                elif event.key == pygame.K_LEFT:
                    self.replay_seek(-5.0)
                elif event.key == pygame.K_RIGHT:
                    self.replay_seek(5.0)
                elif event.key == pygame.K_HOME:
                    self.replay_time = self.replay_frame_times[0]
                elif event.key == pygame.K_END:
                    self.replay_time = self.replay_frame_times[-1]
                elif event.key in (pygame.K_UP, pygame.K_DOWN):
                    self.change_replay_focus(
                        self.replay_ranked_cars(),
                        -1 if event.key == pygame.K_UP else 1,
                    )
                elif event.key == pygame.K_LEFTBRACKET:
                    self.adjust_camera_zoom(-1)
                elif event.key == pygame.K_RIGHTBRACKET:
                    self.adjust_camera_zoom(1)
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                rects = self.replay_transport_rects()
                if rects["rewind"].collidepoint(event.pos):
                    self.set_replay_transport(-1)
                elif rects["pause"].collidepoint(event.pos):
                    self.toggle_replay_pause()
                elif rects["forward"].collidepoint(event.pos):
                    self.set_replay_transport(1)
                elif rects["timeline"].inflate(0, 18).collidepoint(event.pos):
                    fraction = clamp(
                        (event.pos[0] - rects["timeline"].x)
                        / rects["timeline"].width,
                        0.0, 1.0,
                    )
                    start = self.replay_frame_times[0]
                    self.replay_time = start + fraction * (
                        self.replay_frame_times[-1] - start
                    )
                elif event.pos[0] >= CANVAS_W:
                    ranked = self.replay_ranked_cars()
                    page, start, end = self.race_tower_page_bounds(
                        len(ranked), self.replay_tower_page
                    )
                    self.replay_tower_page = page
                    row = self.race_tower_row(event.pos[1], end - start)
                    if row is not None:
                        selected = ranked[start + row]
                        self.replay_follow = next(
                            index for index, car in enumerate(self.replay_cars)
                            if car is selected
                        )

        if self.replay_rate:
            start = self.replay_frame_times[0]
            end = self.replay_frame_times[-1]
            next_time = self.replay_time + dt / 1000.0 * self.replay_rate
            self.replay_time = clamp(next_time, start, end)
            if next_time <= start or next_time >= end:
                self.replay_resume_rate = self.replay_rate
                self.replay_rate = 0.0
        self.apply_replay_time()
        ranked = self.replay_ranked_cars()
        self.sync_replay_tower_page(ranked)
        focused = self.replay_cars[self.replay_follow]
        self.screen.fill(GRASS)
        self.screen.set_clip(pygame.Rect(0, 0, CANVAS_W, HEIGHT))
        camera_offset, camera_scale = self.camera_transform(focused.position)
        self.replay_track.draw(self.screen, camera_offset, camera_scale)
        for car in reversed(ranked):
            if not car.removed_from_track:
                car.draw(
                    self.screen, car is focused,
                    camera_offset, camera_scale,
                )
        self.screen.set_clip(None)
        self.draw_minimap(
            [car for car in self.replay_cars if not car.removed_from_track],
            focused, track=self.replay_track,
        )
        self.draw_camera_zoom_control()
        self.draw_replay_transport()
        self.draw_replay_tower(ranked)

    def menu(self, events):
        self.draw_app_background()
        if self.menu_ui_manager is not None:
            for event in events:
                self.menu_ui_manager.process_events(event)
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
            Button((70, 170, 535, 74), "1  Track Studio", "Draw and save a closed circuit"),
            Button((675, 170, 535, 74), "2  AI Training", "Ghost agents share one start point"),
            Button((70, 260, 535, 74), "3  Race Weekend", "Physical collisions, drafting and mixed brains"),
            Button((675, 260, 535, 74), "4  Two-Lap Hotlap", "Choose one AI brain and record its time"),
            Button((70, 350, 1140, 64), "5  Replay Theatre", "Load recorded race JSON with full transport and driver cameras"),
        ]
        mouse = self.logical_mouse_position()
        if self.menu_ui_manager is None:
            for card in cards:
                card.draw(
                    self.screen, self.fonts, mouse, self.text
                )
        preview = pygame.Rect(70, 430, 1140, 200)
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
        if self.menu_ui_manager is not None:
            self.menu_ui_manager.update(
                max(self.clock.get_time() / 1000.0, 0.001)
            )
            self.menu_ui_manager.draw_ui(self.screen)
        for event in events:
            target = None
            if event.type == pygame.KEYDOWN and event.key in (
            pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4,
            pygame.K_5,
            ):
                target = event.key - pygame.K_1
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                pygame.event.post(pygame.event.Event(pygame.QUIT))
            elif (
                self.menu_ui_manager is not None
                and event.type == pygame_gui.UI_BUTTON_PRESSED
            ):
                target = self.menu_ui_targets.get(event.ui_element)
            if (
                self.menu_ui_manager is None
                and event.type == pygame.MOUSEBUTTONDOWN
                and event.button == 1
            ):
                target = next((i for i, c in enumerate(cards) if c.rect.collidepoint(event.pos)), None)
            if target == 0:
                self.editor_points = [p.copy() for p in self.track.points]
                self.editor_widths = list(self.track.road_widths_m)
                self.editor_grass_widths = list(
                    self.track.grass_widths_m
                )
                self.editor_pitlane_points = [
                    point.copy() for point in self.track.pitlane_points
                ]
                self.editor_pitlane_widths = list(
                    self.track.pitlane_road_widths_m
                )
                self.editor_pitlane_grass_widths = list(
                    self.track.pitlane_grass_widths_m
                )
                self.editor_width_node = None
                self.editor_width_dragging = False
                self.editor_selected_kind = None
                self.editor_selected_index = None
                self.editor_node_dragging = False
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
            elif target == 4:
                choices = self.replay_choices()
                if choices and self.selected_replay not in choices:
                    self.selected_replay = choices[0]
                self.mode = "replay_setup"

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
            docs.append((
                "battery/regen/overtake/recharge_active", YELLOW
            ))
        else:
            docs.append(("ICE: constant 100% power", YELLOW))
        docs.extend([
            ("OUTPUTS", CYAN),
            ("steering: -1 left / +1 right", WHITE),
            ("throttle / brake: 0.0 .. 1.0", WHITE),
        ])
        if self.training_generation == "Hybrid":
            docs.append(("overtake: 0 off / 1 deploy", WHITE))
            docs.append((
                "recharge: 0 off / 1 charge; 70% ICE, up to +5.5%/s",
                WHITE,
            ))
            docs.append((
                "recharge gain scales with throttle; 0 throttle = 0 gain",
                WHITE,
            ))
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
        docs = []
        docs_width = docs_rect.width - 48
        for raw_line, color in self.language_reference_docs():
            # Keep the reference readable at the logical resolution. Long
            # sensor names wrap instead of disappearing behind the scrollbar.
            words = raw_line.split()
            wrapped = []
            current = ""
            for word in words:
                candidate = f"{current} {word}".strip()
                if (
                    current
                    and self.fonts["small"].size(candidate)[0] > docs_width
                ):
                    wrapped.append(current)
                    current = f"  {word}"
                else:
                    current = candidate
            wrapped.append(current)
            docs.extend(
                (line, color if index == 0 else MUTED)
                for index, line in enumerate(wrapped)
            )
        docs_content_rect = pygame.Rect(
            docs_rect.x + 14, docs_rect.y + 47,
            docs_rect.width - 34, docs_rect.height - 61,
        )
        docs_line_height = 18
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
                        Brain.compile_source(self.algorithm_source)
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
        self.control_field(
            brain_rect, "Base brain",
            self.training_brain_label()[:27],
            CYAN, arrows=True,
        )
        self.control_field(
            racecraft_rect, "Traffic",
            "ON" if self.training_racecraft else "OFF",
            UI_ORANGE if self.training_racecraft else MUTED,
            active=self.training_racecraft,
        )
        self.control_field(
            era_rect, "Powertrain",
            self.training_generation.upper(), YELLOW,
        )
        self.control_field(
            track_rect, "Circuit", self.track_label()[:21],
            UI_BLUE, arrows=True,
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
                "small", color,
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
            self.text(
                (
                    f"Ln {current_line + 1}, Col {current_col + 1}"
                    "  •  Ctrl/Cmd+C/X/V  •  Ctrl/Cmd+S save"
                ),
                (55, 701), "small", MUTED,
            )
        self.action_button(
            reload_rect, "Reload", CYAN, self.algorithm_path.exists(),
            secondary=True,
        )
        self.action_button(
            start_rect, "Validate & train", CYAN, True,
            detail="Ctrl/Cmd + Enter",
        )

    def fit_editor_track(self):
        """Keep authored coordinates in metres; cameras handle screen fitting."""
        return Track(
            [point.copy() for point in self.editor_points],
            "Custom Circuit", self.editor_kerbs, self.editor_features,
            self.editor_geometry, self.editor_declared_length, self.editor_road_width,
            self.editor_widths,
            [point.copy() for point in self.editor_pitlane_points],
            self.editor_grass_widths,
            self.editor_pitlane_widths,
            self.editor_pitlane_grass_widths,
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
        self.editor_pitlane_widths.append(PITLANE_WIDTH_M)
        self.editor_pitlane_grass_widths.append(
            max(PITLANE_WIDTH_M + 2.0, 16.0)
        )
        return True

    def select_editor_node(self, screen_position):
        """Select the closest main or pit node under a right click."""
        route = self.editor_node_at(screen_position)
        pit = self.editor_pitlane_node_at(screen_position)
        if route is None and pit is None:
            self.editor_selected_kind = None
            self.editor_selected_index = None
            self.editor_width_node = None
            return False
        choose_pit = pit is not None
        if route is not None and pit is not None:
            click = Vector2(screen_position)
            route_screen = (
                self.editor_points[route] - self.editor_camera
            ) * self.editor_zoom
            pit_screen = (
                self.editor_pitlane_points[pit] - self.editor_camera
            ) * self.editor_zoom
            choose_pit = (
                pit_screen.distance_squared_to(click)
                <= route_screen.distance_squared_to(click)
            )
        self.editor_selected_kind = "pit" if choose_pit else "route"
        self.editor_selected_index = pit if choose_pit else route
        self.editor_width_node = None if choose_pit else route
        return True

    def adjust_selected_editor_width(self, width_kind, delta):
        """Adjust road or surrounding-grass width on the selected node."""
        index = self.editor_selected_index
        if index is None:
            self.notice("Right-click a route or pit-road node first")
            return
        if self.editor_selected_kind == "pit":
            if not 0 <= index < len(self.editor_pitlane_points):
                return
            if width_kind == "road":
                self.editor_pitlane_widths[index] = clamp(
                    self.editor_pitlane_widths[index] + delta, 4.0, 18.0
                )
                self.editor_pitlane_grass_widths[index] = max(
                    self.editor_pitlane_grass_widths[index],
                    self.editor_pitlane_widths[index] + 2.0,
                )
            else:
                self.editor_pitlane_grass_widths[index] = clamp(
                    self.editor_pitlane_grass_widths[index] + delta,
                    max(self.editor_pitlane_widths[index] + 2.0, 8.0),
                    60.0,
                )
        elif self.editor_selected_kind == "route":
            if not 0 <= index < len(self.editor_points):
                return
            if width_kind == "road":
                self.adjust_editor_node_width(index, delta)
                self.editor_grass_widths[index] = max(
                    self.editor_grass_widths[index],
                    self.editor_widths[index] + 4.0,
                )
            else:
                self.editor_grass_widths[index] = clamp(
                    self.editor_grass_widths[index] + delta,
                    max(self.editor_widths[index] + 4.0, 16.0), 80.0,
                )
        self.editor_declared_length = None

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
        self.editor_grass_widths = [
            max(grass_width, road_width + 4.0)
            for grass_width, road_width in zip(
                self.editor_grass_widths, self.editor_widths
            )
        ]
        if self.editor_widths:
            self.editor_road_width = sum(self.editor_widths) / len(self.editor_widths)
        else:
            self.editor_road_width = clamp(
                self.editor_road_width + delta, 6.0, 24.0
            )

    def adjust_all_editor_grass_widths(self, delta):
        self.editor_grass_widths = [
            clamp(
                width + delta,
                max(self.editor_widths[index] + 4.0, 16.0), 80.0,
            )
            for index, width in enumerate(self.editor_grass_widths)
        ]
        current = float(
            self.editor_features.get("border_margin", BORDER_W)
        )
        self.editor_features["border_margin"] = clamp(
            current + delta, 16.0, 80.0
        )

    def delete_editor_node(self, index):
        """Delete a route node and shift every later node into its index."""
        if not 0 <= index < len(self.editor_points):
            return False
        self.editor_points.pop(index)
        if index < len(self.editor_widths):
            self.editor_widths.pop(index)
        if index < len(self.editor_grass_widths):
            self.editor_grass_widths.pop(index)
        point_count = len(self.editor_points)

        self.editor_kerbs = {
            kerb_index - 1 if kerb_index > index else kerb_index
            for kerb_index in self.editor_kerbs
            if kerb_index != index and kerb_index < point_count + 1
        }
        self.editor_features["sectors"] = [
            sector_index - 1 if sector_index > index else sector_index
            for sector_index in self.editor_features.get("sectors", [])
            if sector_index != index and sector_index < point_count + 1
        ]
        for feature in (
            "start_finish", "pit_entry", "pit_exit",
            "drs_detection", "drs_entry", "drs_exit",
        ):
            feature_index = self.editor_features.get(feature)
            if feature_index is None:
                continue
            if not point_count:
                self.editor_features[feature] = (
                    0 if feature == "start_finish" else None
                )
            elif feature_index > index:
                self.editor_features[feature] = feature_index - 1
            elif feature_index == index:
                # The following node now owns the deleted node's position.
                self.editor_features[feature] = min(index, point_count - 1)

        self.editor_width_node = None
        self.editor_width_dragging = False
        if self.editor_selected_kind == "route":
            self.editor_selected_kind = None
            self.editor_selected_index = None
            self.editor_node_dragging = False
        self.editor_declared_length = None
        if (
            not self.editor_manual_kerbs
            and len(self.editor_points) >= 3
        ):
            self.editor_kerbs = Track(
                self.editor_points, geometry=self.editor_geometry,
                road_widths_m=self.editor_widths,
                grass_widths_m=self.editor_grass_widths,
            ).kerb_points
        if not self.editor_pitlane_ready():
            self.editor_pitlane_points.clear()
            self.editor_pitlane_widths.clear()
            self.editor_pitlane_grass_widths.clear()
            self.editor_features["pit_boxes"] = []
            self.editor_features["pit_start_finish"] = None
        return True

    def delete_editor_pitlane_node(self, index):
        """Delete a pit-road node and reindex its timing and box markers."""
        if not 0 <= index < len(self.editor_pitlane_points):
            return False
        self.editor_pitlane_points.pop(index)
        if index < len(self.editor_pitlane_widths):
            self.editor_pitlane_widths.pop(index)
        if index < len(self.editor_pitlane_grass_widths):
            self.editor_pitlane_grass_widths.pop(index)
        point_count = len(self.editor_pitlane_points)
        self.editor_features["pit_boxes"] = [
            box_index - 1 if box_index > index else box_index
            for box_index in self.editor_features.get("pit_boxes", [])
            if box_index != index
            and box_index < point_count + 1
        ]
        finish_index = self.editor_features.get("pit_start_finish")
        if finish_index is not None:
            if not point_count:
                self.editor_features["pit_start_finish"] = None
            elif finish_index > index:
                self.editor_features["pit_start_finish"] = finish_index - 1
            elif finish_index == index:
                self.editor_features["pit_start_finish"] = min(
                    index, point_count - 1
                )
        self.editor_declared_length = None
        if self.editor_selected_kind == "pit":
            self.editor_selected_kind = None
            self.editor_selected_index = None
            self.editor_node_dragging = False
        return True

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
        pit_finish_rect = pygame.Rect(x, 374, 114, 30)
        delete_rect = pygame.Rect(x + 122, 374, 114, 30)
        drs_tool_rects = {
            "drs_detection": pygame.Rect(x, 414, 76, 28),
            "drs_entry": pygame.Rect(x + 81, 414, 76, 28),
            "drs_exit": pygame.Rect(x + 162, 414, 76, 28),
        }
        minus_border = pygame.Rect(x + 145, 466, 38, 28)
        plus_border = pygame.Rect(x + 198, 466, 38, 28)
        minus_road = pygame.Rect(x + 145, 503, 38, 28)
        plus_road = pygame.Rect(x + 198, 503, 38, 28)
        selected_road_minus = pygame.Rect(x + 148, 575, 32, 26)
        selected_road_plus = pygame.Rect(x + 198, 575, 32, 26)
        selected_grass_minus = pygame.Rect(x + 148, 611, 32, 26)
        selected_grass_plus = pygame.Rect(x + 198, 611, 32, 26)

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
                    self.editor_zoom = clamp(
                        self.editor_zoom * (1.20 ** event.y), .28, 6.0
                    )
                    self.editor_camera = world_at_cursor - cursor / self.editor_zoom
            elif (
                event.type == pygame.MOUSEMOTION
                and event.buttons[2]
                and self.editor_node_dragging
                and event.pos[0] < CANVAS_W
            ):
                world = (
                    self.editor_camera
                    + Vector2(event.pos) / self.editor_zoom
                )
                index = self.editor_selected_index
                if self.editor_selected_kind == "route" and (
                    index is not None
                    and 0 <= index < len(self.editor_points)
                ):
                    self.editor_points[index] = world
                    self.editor_declared_length = None
                elif self.editor_selected_kind == "pit" and (
                    index is not None
                    and 0 <= index < len(self.editor_pitlane_points)
                ):
                    self.editor_pitlane_points[index] = world
                    self.editor_declared_length = None
            elif event.type == pygame.MOUSEMOTION and event.buttons[1] and event.pos[0] < CANVAS_W:
                self.editor_camera -= Vector2(event.rel) / self.editor_zoom
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 3:
                if event.pos[0] < CANVAS_W:
                    self.editor_node_dragging = self.select_editor_node(
                        event.pos
                    )
                    if self.editor_node_dragging:
                        kind = (
                            "P" if self.editor_selected_kind == "pit"
                            else ""
                        )
                        self.notice(
                            f"Selected {kind}{self.editor_selected_index + 1}"
                            " • drag to move"
                        )
                continue
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                selected = next((key for key, rect in tool_rects.items() if rect.collidepoint(event.pos)), None)
                selected_drs = next(
                    (
                        key for key, rect in drs_tool_rects.items()
                        if rect.collidepoint(event.pos)
                    ),
                    None,
                )
                if selected_drs:
                    self.editor_tool = selected_drs
                    continue
                if pit_finish_rect.collidepoint(event.pos):
                    if self.editor_pitlane_points:
                        self.editor_tool = "pit_finish"
                    else:
                        self.notice("Add at least one PIT ROAD node first")
                    continue
                if delete_rect.collidepoint(event.pos):
                    self.editor_tool = "delete"
                    continue
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
                    self.adjust_all_editor_grass_widths(-2.0)
                    continue
                if plus_border.collidepoint(event.pos):
                    self.adjust_all_editor_grass_widths(2.0)
                    continue
                if minus_road.collidepoint(event.pos):
                    self.adjust_all_editor_widths(-0.5)
                    continue
                if plus_road.collidepoint(event.pos):
                    self.adjust_all_editor_widths(0.5)
                    continue
                if selected_road_minus.collidepoint(event.pos):
                    self.adjust_selected_editor_width("road", -0.5)
                    continue
                if selected_road_plus.collidepoint(event.pos):
                    self.adjust_selected_editor_width("road", 0.5)
                    continue
                if selected_grass_minus.collidepoint(event.pos):
                    self.adjust_selected_editor_width("grass", -1.0)
                    continue
                if selected_grass_plus.collidepoint(event.pos):
                    self.adjust_selected_editor_width("grass", 1.0)
                    continue
                if event.pos[0] < CANVAS_W:
                    world = self.editor_camera + Vector2(event.pos) / self.editor_zoom
                    if self.editor_tool == "delete":
                        nearest_route = self.editor_node_at(event.pos)
                        nearest_pit = self.editor_pitlane_node_at(event.pos)
                        if nearest_route is None and nearest_pit is None:
                            self.notice("Select a route or pit-road node to delete")
                            continue
                        delete_pit = nearest_pit is not None
                        if nearest_route is not None and nearest_pit is not None:
                            route_screen = (
                                self.editor_points[nearest_route]
                                - self.editor_camera
                            ) * self.editor_zoom
                            pit_screen = (
                                self.editor_pitlane_points[nearest_pit]
                                - self.editor_camera
                            ) * self.editor_zoom
                            click = Vector2(event.pos)
                            delete_pit = (
                                pit_screen.distance_squared_to(click)
                                <= route_screen.distance_squared_to(click)
                            )
                        if delete_pit:
                            deleted_number = nearest_pit + 1
                            self.delete_editor_pitlane_node(nearest_pit)
                            self.notice(
                                f"Deleted pit-road node P{deleted_number}; "
                                "later pit nodes shifted"
                            )
                        else:
                            deleted_number = nearest_route + 1
                            self.delete_editor_node(nearest_route)
                            self.notice(
                                f"Deleted node {deleted_number}; later nodes shifted"
                            )
                        continue
                    if self.editor_tool == "pit_finish":
                        nearest_pit = self.editor_pitlane_node_at(event.pos)
                        if nearest_pit is None:
                            self.notice(
                                "Pit timing line must be placed on a PIT ROAD node"
                            )
                        else:
                            self.editor_features["pit_start_finish"] = (
                                nearest_pit
                            )
                            self.notice(
                                f"Pit timing line set at P{nearest_pit + 1}"
                            )
                        continue
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
                            self.editor_grass_widths.append(
                                max(
                                    self.editor_road_width + 4.0,
                                    float(self.editor_features.get(
                                        "border_margin", BORDER_W
                                    )),
                                )
                            )
                            self.editor_width_node = len(self.editor_points) - 1
                            self.editor_declared_length = None
                            if not self.editor_manual_kerbs and len(self.editor_points) >= 3:
                                self.editor_kerbs = Track(
                                    self.editor_points, geometry=self.editor_geometry,
                                    road_widths_m=self.editor_widths,
                                    grass_widths_m=self.editor_grass_widths,
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
                            elif self.editor_tool in (
                                "pit_entry", "pit_exit",
                                "drs_detection", "drs_entry", "drs_exit",
                            ):
                                self.editor_features[self.editor_tool] = nearest
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                self.editor_width_dragging = False
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 3:
                if (
                    self.editor_node_dragging
                    and not self.editor_manual_kerbs
                    and len(self.editor_points) >= 3
                ):
                    self.editor_kerbs = Track(
                        self.editor_points, geometry=self.editor_geometry,
                        road_widths_m=self.editor_widths,
                        grass_widths_m=self.editor_grass_widths,
                    ).kerb_points
                self.editor_node_dragging = False
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
                elif event.key == pygame.K_9:
                    self.editor_tool = "delete"
                elif event.key == pygame.K_0:
                    if self.editor_pitlane_points:
                        self.editor_tool = "pit_finish"
                    else:
                        self.notice("Add at least one PIT ROAD node first")
                elif (
                    event.key == pygame.K_DELETE
                    and self.editor_width_node is not None
                ):
                    deleted_number = self.editor_width_node + 1
                    self.delete_editor_node(self.editor_width_node)
                    self.notice(
                        f"Deleted node {deleted_number}; later nodes shifted"
                    )
                elif event.key == pygame.K_BACKSPACE:
                    if self.editor_tool in (
                        "pitlane", "pit_box", "pit_finish"
                    ):
                        if self.editor_pitlane_points:
                            self.delete_editor_pitlane_node(
                                len(self.editor_pitlane_points) - 1
                            )
                    elif self.editor_points:
                        self.delete_editor_node(
                            len(self.editor_points) - 1
                        )
                elif event.key == pygame.K_c:
                    self.editor_points.clear()
                    self.editor_widths.clear()
                    self.editor_grass_widths.clear()
                    self.editor_pitlane_points.clear()
                    self.editor_pitlane_widths.clear()
                    self.editor_pitlane_grass_widths.clear()
                    self.editor_width_node = None
                    self.editor_width_dragging = False
                    self.editor_selected_kind = None
                    self.editor_selected_index = None
                    self.editor_node_dragging = False
                    self.editor_kerbs.clear()
                    self.editor_features = {
                        "start_finish": 0, "pit_start_finish": None,
                        "sectors": [], "pit_entry": None, "pit_exit": None,
                        "pit_boxes": [], "drs_detection": None,
                        "drs_entry": None, "drs_exit": None,
                        "border_margin": BORDER_W,
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
                    elif (
                        any(
                            self.editor_features.get(feature) is not None
                            for feature in (
                                "drs_detection", "drs_entry", "drs_exit"
                            )
                        )
                        and not all(
                            self.editor_features.get(feature) is not None
                            for feature in (
                                "drs_detection", "drs_entry", "drs_exit"
                            )
                        )
                    ):
                        self.notice(
                            "DRS requires DET, IN and OUT nodes"
                        )
                    elif (
                        self.editor_features.get("drs_entry") is not None
                        and self.editor_features.get("drs_entry")
                        == self.editor_features.get("drs_exit")
                    ):
                        self.notice(
                            "DRS IN and DRS OUT must use different nodes"
                        )
                    elif event.key == pygame.K_s:
                        self.open_name_dialog(
                            "track", "Save track",
                            self.track.name or "Custom Circuit",
                        )
                    else:
                        self.track = self.fit_editor_track()
                        self.selected_track = None
                        self.mode = "menu"

        editor_mouse = Vector2(self.logical_mouse_position())
        mouse_on_editor = (
            0 <= editor_mouse.x < CANVAS_W
            and 0 <= editor_mouse.y < HEIGHT
        )
        mouse_world = (
            self.editor_camera + editor_mouse / self.editor_zoom
            if mouse_on_editor else None
        )

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
            self.editor_grass_widths,
            self.editor_pitlane_widths,
            self.editor_pitlane_grass_widths,
        ) if screen_points else None
        if temp and len(screen_points) >= 2:
            temp.draw(self.screen, -self.editor_camera * self.editor_zoom, self.editor_zoom)
        pitlane_screen_points = [
            (point - self.editor_camera) * self.editor_zoom
            for point in self.editor_pitlane_points
        ]
        for index, point in enumerate(pitlane_screen_points):
            is_box = index in self.editor_features.get("pit_boxes", [])
            is_pit_finish = (
                index == self.editor_features.get("pit_start_finish")
            )
            pygame.draw.circle(
                self.screen,
                YELLOW if is_pit_finish else
                (168, 85, 247) if is_box else (249, 115, 22),
                point, 7 if is_pit_finish else 6 if is_box else 5,
            )
            pygame.draw.circle(self.screen, WHITE, point, 7, 1)
            if (
                self.editor_selected_kind == "pit"
                and index == self.editor_selected_index
            ):
                pygame.draw.circle(
                    self.screen, YELLOW, point, 12, 2
                )
            self.text(
                f"P{index + 1}{' LAP' if is_pit_finish else ''}",
                point + Vector2(7, -16), "tiny",
                YELLOW if is_pit_finish else WHITE,
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
            "drs_detection": (255, 196, 61),
            "drs_entry": (38, 211, 126),
            "drs_exit": (239, 82, 92),
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
                or i == self.editor_features.get("drs_detection")
                or i == self.editor_features.get("drs_entry")
                or i == self.editor_features.get("drs_exit")
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
                    self.editor_selected_kind == "route"
                    and i == self.editor_selected_index
                ):
                    pygame.draw.circle(
                        self.screen, YELLOW, point, radius + 7, 2
                    )
            if (
                self.editor_geometry != "sampled"
                or i % 30 == 0
                or is_feature
                or i == self.editor_width_node
            ):
                self.text(str(i + 1), point + Vector2(6, -15), "small", WHITE)
        if mouse_world is not None:
            pygame.draw.line(
                self.screen, (120, 210, 175),
                (editor_mouse.x - 7, editor_mouse.y),
                (editor_mouse.x + 7, editor_mouse.y), 1,
            )
            pygame.draw.line(
                self.screen, (120, 210, 175),
                (editor_mouse.x, editor_mouse.y - 7),
                (editor_mouse.x, editor_mouse.y + 7), 1,
            )
            coordinate_text = (
                f"X {mouse_world.x:.1f} m   Y {mouse_world.y:.1f} m"
            )
            coordinate_size = self.fonts["tiny"].size(coordinate_text)
            coordinate_rect = pygame.Rect(
                0, 0, coordinate_size[0], coordinate_size[1]
            )
            coordinate_rect.topleft = (
                editor_mouse.x + 14, editor_mouse.y + 14
            )
            coordinate_rect.inflate_ip(16, 10)
            coordinate_rect.clamp_ip(
                pygame.Rect(8, 8, CANVAS_W - 16, HEIGHT - 16)
            )
            pygame.draw.rect(
                self.screen, (8, 24, 20), coordinate_rect,
                border_radius=6,
            )
            pygame.draw.rect(
                self.screen, (64, 135, 108), coordinate_rect, 1,
                border_radius=6,
            )
            self.text(
                coordinate_text, coordinate_rect.center,
                "tiny", WHITE, anchor="center",
            )
        self.screen.set_clip(None)

        self.panel("Track Studio", f"World editor • {temp.lap_length_m / 1000:.3f} km" if temp else "World editor")
        self.pill("Points", f"{len(self.editor_points)} / 8", x, 108, 112, YELLOW)
        self.pill("Zoom", f"{self.editor_zoom:.2f}x", x + 122, 108, 114)
        self.section_heading(
            "Authoring tools", (x, 169, 236, 16),
            CYAN, self.editor_tool.replace("_", " "),
        )
        pit_finish_active = self.editor_tool == "pit_finish"
        pit_finish_available = bool(self.editor_pitlane_points)
        pygame.draw.rect(
            self.screen,
            (91, 76, 31) if pit_finish_active else
            (35, 31, 22) if pit_finish_available else (17, 23, 21),
            pit_finish_rect, border_radius=6,
        )
        pygame.draw.rect(
            self.screen,
            YELLOW if pit_finish_active else
            (95, 83, 45) if pit_finish_available else (35, 40, 38),
            pit_finish_rect, 1, border_radius=6,
        )
        self.text(
            "0  PIT TIMING",
            (pit_finish_rect.x + 13, pit_finish_rect.y + 8), "tiny",
            WHITE if pit_finish_active else
            MUTED if pit_finish_available else (73, 82, 78),
        )
        delete_active = self.editor_tool == "delete"
        pygame.draw.rect(
            self.screen,
            (104, 40, 45) if delete_active else (35, 25, 26),
            delete_rect, border_radius=6,
        )
        pygame.draw.rect(
            self.screen, RED if delete_active else (83, 52, 54),
            delete_rect, 1, border_radius=6,
        )
        self.text(
            "9  DELETE", (delete_rect.x + 26, delete_rect.y + 8), "tiny",
            WHITE if delete_active else MUTED,
        )
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
        drs_labels = {
            "drs_detection": "DRS DET",
            "drs_entry": "DRS IN",
            "drs_exit": "DRS OUT",
        }
        for key, rect in drs_tool_rects.items():
            active = self.editor_tool == key
            pygame.draw.rect(
                self.screen,
                (41, 78, 63) if active else (15, 30, 25),
                rect, border_radius=6,
            )
            pygame.draw.rect(
                self.screen, CYAN if active else (37, 58, 50),
                rect, 1, border_radius=6,
            )
            self.text(
                drs_labels[key], (rect.x + 8, rect.y + 7), "tiny",
                WHITE if active else MUTED,
            )
        self.section_heading(
            "Global geometry", (x, 449, 236, 16), YELLOW
        )
        self.compact_stepper(
            "Grass",
            f"{int(self.editor_features.get('border_margin', BORDER_W))}m",
            minus_border, plus_border, YELLOW, label_x=x + 12,
        )
        self.compact_stepper(
            "Road", f"{self.editor_road_width:.1f}m",
            minus_road, plus_road, YELLOW, label_x=x + 12,
        )
        pygame.draw.rect(
            self.screen, (14, 28, 23), (x, 543, 236, 121),
            border_radius=10,
        )
        pygame.draw.rect(
            self.screen, UI_BORDER, (x, 543, 236, 121), 1,
            border_radius=10,
        )
        selected_index = self.editor_selected_index
        selected_kind = self.editor_selected_kind
        if selected_kind == "route" and selected_index is not None:
            selected_label = f"NODE {selected_index + 1}"
            selected_road = self.editor_widths[selected_index]
            selected_grass = self.editor_grass_widths[selected_index]
        elif selected_kind == "pit" and selected_index is not None:
            selected_label = f"PIT P{selected_index + 1}"
            selected_road = self.editor_pitlane_widths[selected_index]
            selected_grass = self.editor_pitlane_grass_widths[selected_index]
        else:
            selected_label = "NONE"
            selected_road = None
            selected_grass = None
        self.section_heading(
            "Selected node", (x + 12, 555, 212, 16),
            CYAN, selected_label,
        )
        self.text(
            f"Road  {selected_road:.1f} m"
            if selected_road is not None else "Road  —",
            (x + 12, 582), "small", WHITE,
        )
        self.text(
            f"Grass {selected_grass:.1f} m"
            if selected_grass is not None else "Grass —",
            (x + 12, 618), "small", WHITE,
        )
        mouse = self.logical_mouse_position()
        for rect, symbol in (
            (selected_road_minus, "−"), (selected_road_plus, "+"),
            (selected_grass_minus, "−"), (selected_grass_plus, "+"),
        ):
            hovered = rect.collidepoint(mouse)
            pygame.draw.rect(
                self.screen,
                CYAN if hovered else (28, 51, 43),
                rect, border_radius=6,
            )
            pygame.draw.rect(
                self.screen,
                CYAN if hovered else UI_BORDER,
                rect, 1, border_radius=6,
            )
            self.text(
                symbol, rect.center, "body",
                INK if hovered else WHITE, anchor="center",
            )
        self.text(
            "Right-drag moves • +/- changes width",
            (x + 12, 646), "tiny", MUTED,
        )
        self.footer_hint((
            "[Esc] Paddock  •  [0] Pit line  •  [9] Delete",
            "[S] Save  •  [Enter] Apply",
        ))

    def advance_training_cars(self, dt=1.0):
        """Advance one training step with the selected interaction rules."""
        collisions = 0
        if self.training_racecraft:
            # Racecraft training exposes traffic but does not choose passes.
            # The user's controller must interpret the sensors and steer.
            self.update_race_awareness(assist_passing=False)
            self.update_slipstreams()
            self.update_drs_gaps(self.cars)
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
                car.drs_gap_seconds = float("inf")
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

    @staticmethod
    def training_hybrid_energy_state(car):
        """Return the live Hybrid mode and electrical deployment percentage."""
        if car.recharge_active:
            return "RECHARGE", "CHARGING", (77, 171, 247), 0
        if car.battery_regen > 0.001:
            return "REGEN", "HARVEST", (77, 171, 247), 0
        if car.drs_active:
            return "M.O.M.", "+30% ELEC", (34, 197, 94), 30
        if car.overtake_active:
            return "DEPLOY", "+20% ELEC", YELLOW, 20
        return "READY", "0% ELEC", MUTED, 0

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
                    self.open_name_dialog(
                        "brain", "Export all-time best brain",
                        f"Champion Gen {self.best_generation:03}",
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
                    self.open_name_dialog(
                        "brain", "Export all-time best brain",
                        f"Champion Gen {self.best_generation:03}",
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
        hybrid_training = champion.generation == "Hybrid"
        telemetry_shift = 27 if hybrid_training else 0
        if hybrid_training:
            energy_card = pygame.Rect(x, 217, 240, 47)
            pygame.draw.rect(
                self.screen, (14, 28, 23), energy_card,
                border_radius=7,
            )
            mode, detail, mode_color, deployment_percent = (
                self.training_hybrid_energy_state(champion)
            )
            battery_ratio = clamp(champion.battery / 100.0, 0.0, 1.0)
            battery_color = (
                RED if battery_ratio < 0.25
                else YELLOW if battery_ratio < 0.50
                else CYAN
            )
            self.text("BATTERY", (x + 8, 222), "tiny", MUTED)
            self.text(
                f"{champion.battery:3.0f}%",
                (x + 96, 221), "small", battery_color,
            )
            battery_bar = pygame.Rect(x + 8, 246, 136, 8)
            pygame.draw.rect(
                self.screen, (38, 52, 47), battery_bar,
                border_radius=4,
            )
            if battery_ratio > 0:
                pygame.draw.rect(
                    self.screen, battery_color,
                    (
                        battery_bar.x, battery_bar.y,
                        max(2, round(battery_bar.width * battery_ratio)),
                        battery_bar.height,
                    ),
                    border_radius=4,
                )
            mode_card = pygame.Rect(x + 152, 222, 80, 34)
            pygame.draw.rect(
                self.screen,
                tuple(round(channel * 0.22) for channel in mode_color),
                mode_card, border_radius=6,
            )
            pygame.draw.rect(
                self.screen, mode_color, mode_card, 1,
                border_radius=6,
            )
            self.text(
                mode, (mode_card.centerx, mode_card.y + 3),
                "tiny", mode_color, anchor="midtop",
            )
            self.text(
                detail, (mode_card.centerx, mode_card.y + 18),
                "tiny", WHITE if deployment_percent else mode_color,
                anchor="midtop",
            )
        else:
            self.text(
                f"{self.track.name[:20]} • FULL ICE POWER",
                (x, 220), "small", CYAN,
            )
        self.text(
            f"{champion.speed_kph:3.0f} KM/H  •  G{champion.gear}/8"
            f"  •  {champion.rpm:05.0f} RPM",
            (x, 243 + telemetry_shift), "small", YELLOW,
        )
        throttle_percent, brake_percent = (
            self.training_control_percentages(champion)
        )
        control_y = 264 + telemetry_shift
        pygame.draw.rect(
            self.screen, (14, 28, 23), (x, control_y, 240, 42),
            border_radius=7,
        )
        for column, label, value, color in (
            (0, "THROTTLE", throttle_percent, (34, 197, 94)),
            (1, "BRAKE", brake_percent, RED),
        ):
            control_x = x + 8 + column * 116
            self.text(
                f"{label} {value:3d}%",
                (control_x, control_y + 5), "tiny",
                color if value else MUTED,
            )
            pygame.draw.rect(
                self.screen, (38, 52, 47),
                (control_x, control_y + 26, 106, 7), border_radius=4,
            )
            if value:
                pygame.draw.rect(
                    self.screen, color,
                    (
                        control_x, control_y + 26,
                        max(2, round(106 * value / 100)), 7,
                    ),
                    border_radius=4,
                )

        # Full-width steering demographic. Zero remains fixed at the centre;
        # the live steering command grows left or right from that marker.
        steering_value = clamp(champion.steering_input, -1.0, 1.0)
        steering_percent = int(round(steering_value * 100))
        steering_y = 310 + telemetry_shift
        steering_card = pygame.Rect(x, steering_y, 240, 39)
        steering_bar = pygame.Rect(x + 8, steering_y + 23, 224, 8)
        steering_centre = steering_bar.centerx
        steering_width = round(
            steering_bar.width / 2 * abs(steering_value)
        )
        pygame.draw.rect(
            self.screen, (14, 28, 23), steering_card, border_radius=7
        )
        self.text("L", (x + 8, steering_y + 4), "tiny", MUTED)
        self.text("R", (x + 224, steering_y + 4), "tiny", MUTED)
        steering_label = self.fonts["tiny"].render(
            f"STEERING {steering_percent:+4d}%",
            True,
            CYAN if steering_value else MUTED,
        )
        self.screen.blit(
            steering_label,
            steering_label.get_rect(center=(x + 120, steering_y + 10)),
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
            (x, 355 + telemetry_shift), "tiny", MUTED,
        )
        ranking_y = 370 + telemetry_shift
        ranking_spacing = 23 if hybrid_training else 24
        for i, car in enumerate(ranked[:10]):
            y = ranking_y + i * ranking_spacing
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
            (save_brain_rect, "SAVE BEST", CYAN),
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
        best_fitness = getattr(self, "best_fitness", None)
        record = "BEST —" if best_fitness is None else (
            f"BEST {best_fitness:.0f} • G{self.best_generation:03}"
        )
        self.text(record, (x + 91, 673), "small", CYAN)
        self.footer_hint(("[−/+] Agents  •  Wheel/[ ] Zoom", "[A] Code  •  [S] Brain"))

    def resolve_collisions(self, apply_damage=True):
        collisions = 0
        for i, a in enumerate(self.cars):
            for b in self.cars[i + 1:]:
                if (
                    not a.alive or not b.alive
                    or a.removed_from_track or b.removed_from_track
                    or a.finish_time is not None
                    or b.finish_time is not None
                    or a.pit_timer > 0 or b.pit_timer > 0
                    or a.in_pitlane or b.in_pitlane
                ):
                    # The pit lane is non-colliding: cars can queue and overlap
                    # without the race-track separation solver pushing them
                    # through the narrow lane or away from their pit boxes.
                    continue
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
        pygame.draw.rect(
            self.screen, (9, 21, 22),
            (preview.x + 22, preview.y + 20, 310, 82),
            border_radius=12,
        )
        self.text(self.track.name, (82, 190), "h2", YELLOW)
        self.text(
            f"{self.track.lap_length_m / 1000:.3f} km  •  2 timed laps",
            (84, 224), "small", WHITE,
        )
        self.text(
            "LIVE CIRCUIT PREVIEW", (preview.x + 24, preview.bottom - 39),
            "tiny", MUTED,
        )
        self.glass_card(
            pygame.Rect(825, 165, 410, 500),
            accent=CYAN,
            radius=18,
        )
        self.section_heading(
            "Run configuration", (850, 184, 360, 16),
            CYAN, "3 choices",
        )
        self.control_field(
            track_rect, "01  Circuit", self.track_label()[:26],
            YELLOW, arrows=True, value_font="mono",
        )
        self.text(
            "02  AI BRAIN", (850, 283), "tiny", MUTED,
        )
        self.control_field(
            brain_rect, "Controller source",
            self.brain_label(self.hotlap_brain)[:30],
            CYAN, value_font="mono",
        )
        for rect, label in ((previous_rect, "‹"), (next_rect, "›")):
            hovered = rect.collidepoint(self.logical_mouse_position())
            pygame.draw.rect(
                self.screen, CYAN if hovered else UI_SURFACE_HOVER,
                rect, border_radius=9,
            )
            pygame.draw.rect(
                self.screen, CYAN if hovered else UI_BORDER,
                rect, 1, border_radius=9,
            )
            self.text(
                label, rect.center, "h2",
                INK if hovered else CYAN, anchor="center",
            )
        self.text(
            f"{len(self.brain_choices())} selectable brain source(s)",
            (1032, 398), "tiny", MUTED,
        )
        self.control_field(
            era_rect, "03  Powertrain",
            self.hotlap_generation.upper(), YELLOW, arrows=True,
        )
        self.action_button(
            start_rect, "Start two-lap run", CYAN, True,
            detail="Enter / Space",
        )
        self.text(
            "FIXED CONDITIONS", (850, 598), "tiny", CYAN,
        )
        self.text(
            "Soft tyres  •  20 kg fuel  •  dry circuit",
            (850, 619), "small", WHITE,
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
        if car.generation == "Hybrid":
            battery = (
                f"{car.battery:5.1f}% "
                f"{'RECHARGE' if car.recharge_active else 'M.O.M.' if car.drs_active else 'OT' if car.overtake_active else ''}"
            )
            battery_color = (
                (77, 171, 247) if car.recharge_active else
                (34, 197, 94) if car.drs_active else CYAN
            )
        else:
            battery = "DRS"
            battery_color = (
                (34, 197, 94) if car.drs_active else
                (77, 171, 247) if car.drs_eligible else
                YELLOW
            )
        self.text(
            f"Battery    {battery}", (x, 452), "mono", battery_color
        )
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
        self.control_field(
            setup["track"], "Circuit", self.track_label()[:25],
            UI_BLUE, arrows=True,
        )
        self.section_heading(
            "Starting grid", (52, 142, 678, 16),
            YELLOW, f"{count} cars",
        )
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
        self.section_heading(
            "Session", (800, 151, 400, 16), CYAN,
            "Race rules",
        )
        for label, value, minus, plus in (
            (
                "Cars", str(self.race_settings["cars"]),
                "cars_minus", "cars_plus",
            ),
            (
                "Laps", str(self.race_settings["laps"]),
                "laps_minus", "laps_plus",
            ),
        ):
            self.compact_stepper(
                label, value, setup[minus], setup[plus], CYAN
            )
        for label, key, accent in (
            ("Weather", "weather", UI_BLUE),
            ("Power", "generation", YELLOW),
            ("Teams", "teams", UI_VIOLET),
        ):
            value = self.race_settings[key]
            if isinstance(value, bool):
                value = "PAIRED" if value else "INDIVIDUAL"
            self.control_field(
                setup[key], label, value.upper(), accent,
            )

        entry = self.race_entries[self.selected_entry]
        self.glass_card(
            pygame.Rect(775, 418, 450, 246),
            accent=COLORS[entry["color"]],
            radius=16,
        )
        self.section_heading(
            f"Car {self.selected_entry + 1:02} configuration",
            (800, 428, 400, 16),
            COLORS[entry["color"]], "Selected entry",
        )
        for key in ("name", "team_name", "tyre", "color"):
            pygame.draw.rect(
                self.screen,
                UI_SURFACE_HOVER
                if setup[key].collidepoint(self.logical_mouse_position())
                else (10, 24, 23),
                setup[key], border_radius=7,
            )
            pygame.draw.rect(
                self.screen,
                UI_BORDER_BRIGHT
                if setup[key].collidepoint(self.logical_mouse_position())
                else UI_BORDER,
                setup[key], 1, border_radius=7,
            )
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
            if name_editing else "CLICK DRIVER NAME TO RENAME",
            (setup["name"].x + 3, 488), "tiny",
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
        self.text("PIT STRATEGY: CONTROLLER CODE", (808, 573), "tiny", CYAN)
        for key, label in (("grid_up", "GRID UP"), ("grid_down", "GRID DOWN")):
            self.action_button(
                setup[key], label, CYAN, True, secondary=True,
            )
        self.control_field(
            setup["brain"], "AI brain",
            self.brain_label(
                entry.get("brain", "__session__")
            )[:25],
            CYAN, arrows=True,
        )

        trained = bool(list(BRAIN_DIR.glob("*.json"))) or self.generation > 0
        self.action_button(
            setup["start"],
            "Start race" if trained else "Trained AI required",
            CYAN, trained,
            detail="5-light start sequence" if trained else "",
        )
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
                elif event.key in (pygame.K_UP, pygame.K_DOWN):
                    ranked = sorted(
                        self.cars, key=self.race_order_key, reverse=True
                    )
                    self.change_race_focus(
                        ranked,
                        -1 if event.key == pygame.K_UP else 1,
                    )
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
                        self.cars, key=self.race_order_key, reverse=True
                    )
                    page, page_start, page_end = (
                        self.race_tower_page_bounds(
                            len(ranked),
                            getattr(self, "race_tower_page", 0),
                        )
                    )
                    self.race_tower_page = page
                    row = self.race_tower_row(
                        event.pos[1], page_end - page_start
                    )
                    if row is not None:
                        self.follow = self.cars.index(
                            ranked[page_start + row]
                        )
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
            self.update_drs_gaps(self.cars)
            for car in self.cars:
                if car.removed_from_track:
                    continue
                if car.finish_time is not None:
                    car.advance_after_finish(self.track)
                    continue
                if not car.alive:
                    self.retire_race_car(car, "DAMAGE")
                    continue
                was_punctured = car.puncture
                old_stops = car.pitstops
                car.update(self.track, rain=self.rain_level)
                self.record_race_timing_sample(car)
                if car.puncture and not was_punctured:
                    self.log_event("puncture", f"{car.name} suffered a puncture", "high", self.cars.index(car))
                if car.pitstops > old_stops:
                    self.log_event("pitstop", f"{car.name} completed a pit stop", "medium", self.cars.index(car))
                speed_cap = 3.5 if self.flag_state == "SAFETY CAR" else (5.8 if self.flag_state == "YELLOW" else None)
                if speed_cap and car.velocity.length() > speed_cap:
                    car.velocity.scale_to_length(speed_cap)
                if car.lap >= self.target_laps:
                    car.finish_time = self.session_time
                    car.low_speed_seconds = 0.0
                    car.throttle_input = 0.0
                    car.brake_input = FINISH_BRAKE_INPUT
                    self.log_event("finish", f"{car.name} finished", "medium")
                    continue
                if car.speed_kph <= LOW_SPEED_DNF_KPH:
                    car.low_speed_seconds += dt / 1000.0
                else:
                    car.low_speed_seconds = 0.0
                if car.low_speed_seconds >= LOW_SPEED_DNF_SECONDS:
                    self.retire_race_car(car, "LOW SPEED")
            self.resolve_collisions()
            for car in self.cars:
                if (
                    not car.alive
                    and car.finish_time is None
                    and not car.removed_from_track
                ):
                    self.retire_race_car(car, "DAMAGE")
            self.replay_tick += 1
            if self.replay_tick % 6 == 0:
                self.capture_replay_frame()
            if self.event_camera and pygame.time.get_ticks() >= self.camera_until:
                visible_indexes = [
                    index for index, car in enumerate(self.cars)
                    if not car.removed_from_track
                ]
                if visible_indexes:
                    self.follow = random.choice(visible_indexes)
                self.camera_until = pygame.time.get_ticks() + 30000
        ranked = sorted(
            self.cars, key=self.race_order_key, reverse=True
        )
        if all(c.removed_from_track for c in self.cars):
            self.draw_results(ranked)
            return
        self.screen.fill(GRASS)
        self.screen.set_clip(pygame.Rect(0, 0, CANVAS_W, HEIGHT))
        visible_cars = [
            car for car in self.cars if not car.removed_from_track
        ]
        if self.cars[self.follow].removed_from_track and visible_cars:
            self.follow = self.cars.index(visible_cars[0])
        self.sync_race_tower_page(ranked)
        focused_car = self.cars[self.follow]
        camera_offset, camera_scale = self.camera_transform(focused_car.position)
        self.track.draw(self.screen, camera_offset, camera_scale)
        for car in reversed(ranked):
            if car.removed_from_track:
                continue
            car.draw(
                self.screen, self.cars.index(car) == self.follow,
                camera_offset, camera_scale,
            )
        self.screen.set_clip(None)
        self.draw_minimap(visible_cars, focused_car)
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

    def record_race_timing_sample(self, car):
        """Store sparse distance/time samples for live timing interpolation."""
        history = car.timing_history
        if not history:
            history.append((car.score, self.session_time))
            return
        if car.score < history[-1][0] + 1.0:
            return
        history.append((car.score, self.session_time))
        minimum_distance = (
            car.score - self.track.measured_length_m * 1.25
        )
        first_kept = bisect_right(
            history, (minimum_distance, float("inf"))
        )
        if first_kept > 1:
            del history[:first_kept - 1]

    def retire_race_car(self, car, reason="LOW SPEED"):
        """Classify a DNF and remove the car from the physical circuit."""
        if car.finish_time is not None or car.removed_from_track:
            return
        car.alive = False
        car.retirement_time = self.session_time
        car.retirement_reason = reason
        car.removed_from_track = True
        car.velocity.update(0.0, 0.0)
        car.throttle_input = 0.0
        car.brake_input = 0.0
        car.overtake_active = False
        car.drs_active = False
        car.recharge_active = False
        self.log_event(
            "dnf", f"{car.name} retired: {reason.lower()}", "high",
            self.cars.index(car),
        )

    @staticmethod
    def crossing_time_at_distance(car, distance_m):
        """Interpolate when a car crossed an absolute race distance."""
        history = car.timing_history
        if not history or distance_m < history[0][0]:
            return None
        index = bisect_right(history, (distance_m, float("inf")))
        if index == 0:
            return None
        if index >= len(history):
            last_distance, last_time = history[-1]
            return (
                last_time
                if abs(last_distance - distance_m) <= 1e-6
                else None
            )
        left_distance, left_time = history[index - 1]
        right_distance, right_time = history[index]
        span = right_distance - left_distance
        if span <= 1e-9:
            return right_time
        fraction = clamp(
            (distance_m - left_distance) / span, 0.0, 1.0
        )
        return left_time + (right_time - left_time) * fraction

    def race_gap_seconds(self, ahead, behind):
        """Return the live time gap from `behind` to `ahead`."""
        if (
            ahead.finish_time is not None
            and behind.finish_time is not None
        ):
            return max(0.0, behind.finish_time - ahead.finish_time)
        crossing_time = self.crossing_time_at_distance(
            ahead, behind.score
        )
        if crossing_time is not None:
            return max(0.0, self.session_time - crossing_time)
        distance_gap = max(0.0, ahead.score - behind.score)
        effective_speed_mps = max(
            (
                ahead.velocity.length()
                + behind.velocity.length()
            ) * FPS / 2,
            25.0,
        )
        return distance_gap / effective_speed_mps

    def update_drs_gaps(self, cars=None):
        """Publish the live time gap to the next car for DRS detection."""
        field = [
            car for car in (cars if cars is not None else self.cars)
            if car.alive and car.finish_time is None
        ]
        ranked = sorted(field, key=self.race_order_key, reverse=True)
        for index, car in enumerate(ranked):
            car.drs_gap_seconds = (
                self.race_gap_seconds(ranked[index - 1], car)
                if index > 0 else float("inf")
            )

    @staticmethod
    def format_race_gap(gap_seconds):
        return (
            f"+{gap_seconds:5.3f}s"
            if gap_seconds < 100.0
            else f"+{gap_seconds:5.1f}s"
        )

    @staticmethod
    def race_order_key(car):
        """Sort running cars by distance and finishers by finish time."""
        if car.finish_time is not None:
            return 1, -car.finish_time
        return 0, car.score

    @staticmethod
    def race_tower_row(mouse_y, car_count):
        """Return the integer timetable row under a logical mouse position."""
        row = int((float(mouse_y) - 190.0) // 48.0)
        return row if 0 <= row < min(10, car_count) else None

    @staticmethod
    def race_tower_page_bounds(car_count, page):
        """Clamp a timetable page and return its ranked slice."""
        car_count = max(0, int(car_count))
        maximum_page = max(0, (car_count - 1) // 10)
        page = int(clamp(int(page), 0, maximum_page))
        start = page * 10
        return page, start, min(start + 10, car_count)

    def sync_race_tower_page(self, ranked):
        """Keep the ten-driver timetable page attached to the focused car."""
        if not ranked or not self.cars:
            self.race_tower_page = 0
            return
        followed = (
            self.cars[self.follow]
            if 0 <= self.follow < len(self.cars)
            else ranked[0]
        )
        rank_index = next(
            (
                index for index, car in enumerate(ranked)
                if car is followed
            ),
            0,
        )
        self.race_tower_page = rank_index // 10

    def change_race_focus(self, ranked, direction):
        """Move the manual camera to the adjacent classified driver."""
        if not ranked or not self.cars:
            return
        followed = (
            self.cars[self.follow]
            if 0 <= self.follow < len(self.cars)
            else ranked[0]
        )
        rank_index = next(
            (
                index for index, car in enumerate(ranked)
                if car is followed
            ),
            0,
        )
        rank_index = int(clamp(
            rank_index + int(direction), 0, len(ranked) - 1
        ))
        selected = ranked[rank_index]
        self.follow = next(
            index for index, car in enumerate(self.cars)
            if car is selected
        )
        self.race_tower_page = rank_index // 10
        self.event_camera = False

    def draw_checkered_border(self, rect, square=5):
        """Draw a compact chequered classification border around a row."""
        colors = (WHITE, (35, 40, 42))
        columns = max(1, math.ceil(rect.width / square))
        rows = max(1, math.ceil(rect.height / square))
        for column in range(columns):
            width = min(square, rect.right - (rect.x + column * square))
            for y, phase in ((rect.y, 0), (rect.bottom - square, 1)):
                pygame.draw.rect(
                    self.screen, colors[(column + phase) % 2],
                    (rect.x + column * square, y, width, square),
                )
        for row in range(1, rows - 1):
            height = min(square, rect.bottom - (rect.y + row * square))
            for x, phase in ((rect.x, 0), (rect.right - square, 1)):
                pygame.draw.rect(
                    self.screen, colors[(row + phase) % 2],
                    (x, rect.y + row * square, square, height),
                )

    def draw_tower(self, ranked):
        x = CANVAS_W
        camera_label = "AUTO CAM" if self.event_camera else "MANUAL CAM"
        self.panel("Race Control", f"{self.flag_state}  •  {camera_label}")
        self.text(f"LAP {min(max(c.lap for c in self.cars)+1, self.target_laps)} / {self.target_laps}", (x + 20, 101), "h2")
        self.text(f"{self.session_time:07.2f}", (x + 178, 108), "mono", YELLOW)
        page, page_start, page_end = self.race_tower_page_bounds(
            len(ranked), getattr(self, "race_tower_page", 0)
        )
        self.race_tower_page = page
        self.text(
            f"POSITIONS {page_start + 1}–{page_end}",
            (x + 20, 128), "tiny", MUTED,
        )
        labels = (
            "INTERVAL (SECONDS)", "GAP TO LEADER (SECONDS)", "TYRE / AGE",
            "PIT STOPS", "CONDITION", "BATTERY / ENERGY MODE",
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
        for row, car in enumerate(ranked[page_start:page_end]):
            rank_index = page_start + row
            y = 190 + row * 48
            focused = self.cars.index(car) == self.follow
            dnf = (
                car.finish_time is None
                and car.retirement_time is not None
            )
            finished = car.finish_time is not None
            row_rect = pygame.Rect(x + 10, y, PANEL - 20, 41)
            pygame.draw.rect(
                self.screen,
                (37, 40, 41) if dnf else
                UI_SURFACE_HOVER if focused else UI_SURFACE,
                row_rect,
                border_radius=9,
            )
            pygame.draw.rect(
                self.screen,
                (100, 104, 106) if dnf else
                car.color if focused else UI_BORDER,
                row_rect,
                1,
                border_radius=9,
            )
            if finished:
                self.draw_checkered_border(row_rect)
            row_color = (125, 129, 131) if dnf else car.color
            pygame.draw.rect(self.screen, row_color, (x + 15, y + 7, 5, 27), border_radius=2)
            self.text(
                f"{rank_index+1:>2}", (x + 27, y + 10), "mono",
                (145, 149, 151) if dnf else WHITE if focused else MUTED,
            )
            self.text(
                car.name, (x + 57, y + 6), "mono",
                (145, 149, 151) if dnf else WHITE,
            )
            value_color = (
                (145, 149, 151) if dnf else
                CYAN if focused else MUTED
            )
            if self.metric == 0:
                previous = (
                    ranked[rank_index - 1]
                    if rank_index else leader
                )
                value = (
                    "LEADER" if rank_index == 0
                    else self.format_race_gap(
                        self.race_gap_seconds(previous, car)
                    )
                )
            elif self.metric == 1:
                value = (
                    "LEADER" if rank_index == 0
                    else self.format_race_gap(
                        self.race_gap_seconds(leader, car)
                    )
                )
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
                        (77, 171, 247)
                        if car.recharge_active else
                        (34, 197, 94)
                        if car.drs_active else
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
                        f"{'RECHARGE' if car.recharge_active else 'M.O.M.' if car.drs_active else 'OVERTAKE' if car.overtake_active else 'M.O.M. READY' if car.drs_eligible else 'READY'}"
                    )
                else:
                    value = "DRS"
                    value_color = (
                        (34, 197, 94)
                        if car.drs_active else
                        (77, 171, 247)
                        if car.drs_eligible else
                        YELLOW
                    )
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
            if dnf:
                value = "DNF"
            elif finished:
                value = "FINISHED"
            self.text(
                value, (x + 148, y + 22), "small", value_color
            )
        if len(ranked) > 10:
            self.text(
                f"PAGE {page + 1}/{(len(ranked) - 1) // 10 + 1}",
                (x + 20, 675), "small", MUTED,
            )
        self.footer_hint((
            "UP/DOWN Driver  •  LEFT/RIGHT Data",
            "Wheel/[ ] Zoom  •  [Esc] Back",
        ))

    def draw_results(self, ranked):
        self.draw_app_background()
        self.text("RACE CLASSIFICATION", (390, 34), "title")
        self.text(
            f"{self.track.name.upper()}  •  {self.target_laps} LAPS",
            (500, 87), "small", CYAN,
        )
        finishers = sorted(
            (car for car in self.cars if car.finish_time is not None),
            key=lambda car: car.finish_time,
        )
        dnfs = sorted(
            (
                car for car in self.cars
                if car.finish_time is None
            ),
            key=lambda car: car.score,
            reverse=True,
        )
        classification = finishers + dnfs
        table = pygame.Rect(70, 125, 1140, 555)
        self.glass_card(table, accent=YELLOW, radius=15)
        columns = {
            "pos": 90, "driver": 175, "status": 465,
            "time": 690, "pits": 935, "grid": 1060,
        }
        self.text("POS", (columns["pos"], 143), "small", MUTED)
        self.text("DRIVER", (columns["driver"], 143), "small", MUTED)
        self.text("CLASSIFICATION", (columns["status"], 143), "small", MUTED)
        self.text("TOTAL TIME", (columns["time"], 143), "small", MUTED)
        self.text("PIT STOPS", (columns["pits"], 143), "small", MUTED)
        self.text("START", (columns["grid"], 143), "small", MUTED)
        pygame.draw.line(
            self.screen, UI_BORDER, (84, 165), (1196, 165), 1
        )
        row_height = min(
            25, max(20, int(495 / max(len(classification), 1)))
        )
        for index, car in enumerate(classification):
            y = 172 + index * row_height
            row = pygame.Rect(84, y, 1112, row_height - 3)
            dnf = car.finish_time is None
            pygame.draw.rect(
                self.screen,
                (38, 41, 42) if dnf else UI_SURFACE,
                row, border_radius=5,
            )
            if not dnf:
                self.draw_checkered_border(row, square=4)
            text_color = (137, 141, 143) if dnf else WHITE
            place = "DNF" if dnf else f"P{index + 1}"
            status = (
                car.retirement_reason or "DNF"
                if dnf else "FINISHED"
            )
            total_time = (
                car.retirement_time
                if dnf else car.finish_time
            )
            self.text(place, (columns["pos"], y + 3), "mono", text_color)
            self.text(
                car.name, (columns["driver"], y + 3), "mono",
                (125, 129, 131) if dnf else car.color,
            )
            self.text(status, (columns["status"], y + 3), "small", text_color)
            self.text(
                f"{total_time:8.3f}s" if total_time is not None else "—",
                (columns["time"], y + 3), "mono", text_color,
            )
            self.text(
                str(car.pitstops), (columns["pits"] + 28, y + 3),
                "mono", text_color,
            )
            self.text(
                f"P{car.starting_position}",
                (columns["grid"] + 18, y + 3), "mono", text_color,
            )
        replay = "REPLAY SAVED" if self.replay_saved else "[R] Save replay"
        self.text(
            f"{replay}   •   [Esc] Return to paddock",
            (445, 704), "body", CYAN if self.replay_saved else MUTED,
        )

    def run(self):
        running = True
        while running:
            dt = self.clock.tick(FPS)
            self._native_text_commands.clear()
            raw_events = pygame.event.get()
            for event in raw_events:
                if event.type == pygame.QUIT:
                    running = False
                elif (
                    event.type == pygame.KEYDOWN
                    and (
                        event.key == pygame.K_F3
                        or event.key == pygame.K_l
                        and self.mode != "replay"
                    )
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
            elif self.mode == "replay_setup":
                self.replay_setup(events)
            elif self.mode == "replay":
                self.replay(events, dt)
            elif self.mode == "name_dialog":
                self.draw_name_dialog(events)
            if self.show_fps:
                self.draw_fps_counter()
            if pygame.time.get_ticks() < self.message_until:
                message_size = self.fonts["body"].size(self.message)
                rect = pygame.Rect(
                    0, 0, message_size[0], message_size[1]
                )
                rect.center = (WIDTH // 2, HEIGHT - 35)
                rect.inflate_ip(38, 18)
                pygame.draw.rect(
                    self.screen, UI_SHADOW, rect.move(0, 4),
                    border_radius=11,
                )
                pygame.draw.rect(
                    self.screen, YELLOW, rect, border_radius=11
                )
                self.text(
                    self.message, rect.center, "body", INK,
                    anchor="center",
                )
            self.present()
        pygame.quit()


if __name__ == "__main__":
    Game().run()
