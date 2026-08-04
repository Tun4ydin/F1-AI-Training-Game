import os
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame
from pygame import Vector2

from main import (
    Brain, CAR_LENGTH_M, CAR_WIDTH_M, COLORS, DEFAULT_CAMERA_ZOOM,
    HYBRID_RECHARGE_RATE, Car, Game, OVERTAKE_REWARD, Track,
    component_filename, safe_component_name, spawn_car,
)
from safe_algorithm import SafeAlgorithm


ROOT = Path(__file__).resolve().parents[1]


class RaceModeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.track = Track.load(
            ROOT / "saved_data" / "tracks" / "spa_francorchamps.json"
        )

    def test_training_agents_are_ghosts_at_the_exact_same_spawn(self):
        game = object.__new__(Game)
        game.track = self.track
        game.population = 6
        game.best_brain = Brain()
        game.generation = 0
        game.cars = []
        game.training_generation = "ICE"
        game.reset_training()
        positions = {(round(car.position.x, 6), round(car.position.y, 6)) for car in game.cars}
        self.assertEqual(len(positions), 1)
        self.assertTrue(all(car.generation == "ICE" for car in game.cars))
        self.assertTrue(all(car.battery == 0.0 for car in game.cars))

    def test_training_retains_all_time_brain_until_fitness_is_beaten(self):
        game = object.__new__(Game)
        game.track = self.track
        game.population = 3
        game.training_generation = "ICE"
        game.training_racecraft = False
        game.generation = 3
        held_brain = Brain(config={"aggression": 0.41})
        tied_brain = Brain(config={"aggression": 0.72})
        game.best_brain = held_brain
        game.best_fitness = 100.0
        game.best_generation = 2
        game.last_generation_fitness = None
        game.last_generation_improved = False
        game.cars = [
            Car(Vector2(), 0, COLORS[0], tied_brain, fitness=100.0),
            Car(Vector2(), 0, COLORS[1], Brain(), fitness=80.0),
        ]

        game.reset_training(evolve=True)

        self.assertIs(game.best_brain, held_brain)
        self.assertEqual(game.best_fitness, 100.0)
        self.assertFalse(game.last_generation_improved)
        self.assertIs(game.cars[0].brain, held_brain)

        improved_brain = Brain(config={"aggression": 0.93})
        game.cars[1].brain = improved_brain
        game.cars[1].fitness = 125.0
        game.reset_training(evolve=True)

        self.assertIs(game.best_brain, improved_brain)
        self.assertEqual(game.best_fitness, 125.0)
        self.assertEqual(game.best_generation, 4)
        self.assertTrue(game.last_generation_improved)
        self.assertIs(game.cars[0].brain, improved_brain)

    def test_high_resolution_viewport_preserves_logical_coordinates(self):
        game = object.__new__(Game)
        game.window = pygame.Surface((1920, 1080))
        game.update_viewport()
        self.assertEqual(game.viewport_rect.height, 1080)
        self.assertGreater(game.viewport_rect.width, 1800)
        logical = game.logical_position(game.viewport_rect.center)
        self.assertAlmostEqual(logical[0], 640.0, delta=0.5)
        self.assertAlmostEqual(logical[1], 380.0, delta=0.5)

    def test_fractional_race_timetable_click_returns_integer_row(self):
        row = Game.race_tower_row(238.5, 12)
        self.assertIs(type(row), int)
        self.assertEqual(row, 1)
        self.assertIsNone(Game.race_tower_row(189.9, 12))
        self.assertIsNone(Game.race_tower_row(670.0, 12))
        self.assertIsNone(Game.race_tower_row(286.0, 2))

    def test_race_camera_follows_classification_across_tower_pages(self):
        game = object.__new__(Game)
        game.cars = [
            Car(
                Vector2(index, 0), 0, COLORS[index % len(COLORS)],
                Brain(), name=f"Driver {index + 1}",
            )
            for index in range(20)
        ]
        # The classification order deliberately differs from the cars list.
        ranked = list(reversed(game.cars))
        game.follow = game.cars.index(ranked[9])
        game.race_tower_page = 0
        game.event_camera = True

        game.change_race_focus(ranked, 1)

        self.assertIs(game.cars[game.follow], ranked[10])
        self.assertEqual(game.race_tower_page, 1)
        self.assertFalse(game.event_camera)

        game.change_race_focus(ranked, -1)

        self.assertIs(game.cars[game.follow], ranked[9])
        self.assertEqual(game.race_tower_page, 0)

    def test_race_tower_page_bounds_cover_second_ten_drivers(self):
        self.assertEqual(Game.race_tower_page_bounds(20, 1), (1, 10, 20))
        self.assertEqual(Game.race_tower_page_bounds(16, 1), (1, 10, 16))
        self.assertEqual(Game.race_tower_page_bounds(8, 3), (0, 0, 8))

    def test_minimap_projection_preserves_aspect_and_stays_in_bounds(self):
        rect = pygame.Rect(12, 28, 196, 126)
        points = [
            Vector2(-50, -25), Vector2(150, -25),
            Vector2(150, 75), Vector2(-50, 75),
        ]
        scale, _, projected = Game.minimap_projection(points, rect)
        self.assertAlmostEqual(scale, 0.98)
        self.assertAlmostEqual(
            projected[0].distance_to(projected[1]), 200 * scale
        )
        self.assertAlmostEqual(
            projected[1].distance_to(projected[2]), 100 * scale
        )
        self.assertTrue(all(
            rect.left <= point.x <= rect.right
            and rect.top <= point.y <= rect.bottom
            for point in projected
        ))

    def test_pitlane_containment_uses_local_spatial_segments(self):
        track = self.track
        self.assertGreaterEqual(len(track.pitlane_centerline), 2)
        segment = len(track.pitlane_centerline) // 2
        point = track.pitlane_centerline[segment].copy()
        exact = track.pitlane_nearest(point)
        local = track.pitlane_nearest(point, local_only=True)
        self.assertAlmostEqual(local[0], exact[0], places=7)
        self.assertEqual(local[2], exact[2])
        self.assertTrue(track.is_in_pitlane(point))

        cell = (
            int(point.x // track.spatial_cell_m),
            int(point.y // track.spatial_cell_m),
        )
        candidates = track.pitlane_spatial_segments[cell]
        self.assertLess(
            len(candidates), len(track.pitlane_centerline) - 1
        )

    def test_dense_spline_tracks_use_adaptive_sampling(self):
        oval = Track.load(
            ROOT / "saved_data" / "tracks" / "oval_002.json"
        )
        self.assertEqual(Track.spline_samples_per_section(12), 14)
        self.assertEqual(oval.samples_per_section, 4)
        self.assertEqual(len(oval.centerline), len(oval.points) * 4)
        self.assertLess(len(oval.centerline), 300)

    def test_replay_interpolates_position_and_wrapped_heading(self):
        game = object.__new__(Game)
        game.replay_track = Track(
            [(0, 0), (100, 0), (100, 100), (0, 100)],
            geometry="sampled",
        )
        game.replay_frame_times = [0.0, 1.0]
        game.replay_time = 0.5
        game.replay_data = {
            "frames": [
                {
                    "time": 0.0,
                    "cars": [{
                        "name": "Replay Car", "x": 0.0, "y": 0.0,
                        "angle": 350.0, "lap": 0, "speed_kph": 100.0,
                    }],
                },
                {
                    "time": 1.0,
                    "cars": [{
                        "name": "Replay Car", "x": 10.0, "y": 20.0,
                        "angle": 10.0, "lap": 0, "speed_kph": 120.0,
                    }],
                },
            ],
        }
        game.replay_cars = [
            Car(Vector2(), 0, COLORS[0], Brain(), name="Replay Car")
        ]

        game.apply_replay_time()

        self.assertEqual(game.replay_cars[0].position, Vector2(5, 10))
        self.assertAlmostEqual(game.replay_cars[0].angle % 360, 0.0)
        self.assertAlmostEqual(game.replay_cars[0].speed_kph, 120.0)

    def test_replay_transport_accelerates_and_reverses(self):
        game = object.__new__(Game)
        game.replay_rate = 1.0
        game.replay_resume_rate = 1.0

        game.set_replay_transport(-1)
        self.assertEqual(game.replay_rate, -1.0)
        game.set_replay_transport(-1)
        self.assertEqual(game.replay_rate, -2.0)
        game.toggle_replay_pause()
        self.assertEqual(game.replay_rate, 0.0)
        game.toggle_replay_pause()
        self.assertEqual(game.replay_rate, -2.0)
        game.set_replay_transport(1)
        self.assertEqual(game.replay_rate, 1.0)

    def test_replay_camera_crosses_from_driver_ten_to_eleven(self):
        game = object.__new__(Game)
        game.replay_cars = [
            Car(
                Vector2(index, 0), 0,
                COLORS[index % len(COLORS)], Brain(),
                name=f"Replay {index + 1}", score=20 - index,
            )
            for index in range(20)
        ]
        ranked = game.replay_ranked_cars()
        game.replay_follow = game.replay_cars.index(ranked[9])
        game.replay_tower_page = 0

        game.change_replay_focus(ranked, 1)

        self.assertIs(game.replay_cars[game.replay_follow], ranked[10])
        self.assertEqual(game.replay_tower_page, 1)

    def test_racecraft_training_uses_grid_drafting_and_collisions(self):
        game = object.__new__(Game)
        game.track = self.track
        game.population = 6
        game.best_brain = Brain()
        game.generation = 0
        game.cars = []
        game.training_generation = "ICE"
        game.training_racecraft = True
        game.rain_level = 0.0
        game.event_log = []
        game.event_camera = False
        game.session_time = 0.0
        game.follow = 0
        game.reset_training()
        positions = {
            (round(car.position.x, 4), round(car.position.y, 4))
            for car in game.cars
        }
        self.assertGreater(len(positions), 1)
        minimum_gap = min(
            first.position.distance_to(second.position)
            for index, first in enumerate(game.cars)
            for second in game.cars[index + 1:]
        )
        self.assertGreater(minimum_gap, CAR_LENGTH_M + 3.0)
        game.update_slipstreams()
        self.assertFalse(any(car.slipstream > 0 for car in game.cars))
        heading = Vector2(1, 0).rotate(game.cars[0].angle)
        game.cars[1].position = (
            game.cars[0].position + heading * CAR_LENGTH_M * 2
        )
        game.cars[1].angle = game.cars[0].angle
        game.update_slipstreams()
        self.assertGreater(game.cars[0].slipstream, 0.0)

        game.cars[1].position = game.cars[0].position.copy()
        game.cars[1].angle = game.cars[0].angle
        self.assertGreaterEqual(game.advance_training_cars(), 1)
        self.assertIsNone(
            game.car_collision_manifold(game.cars[0], game.cars[1])
        )
        self.assertTrue(game.cars[0].car_collision)
        self.assertTrue(game.cars[1].car_collision)
        state = dict(zip(
            Brain.INPUT_NAMES,
            game.cars[0].controller_inputs(self.track),
        ))
        self.assertEqual(state["car_collision"], 1.0)
        game.cars[0].update(self.track)
        self.assertFalse(game.cars[0].car_collision)

    def test_oriented_collision_boxes_separate_and_exchange_momentum(self):
        game = object.__new__(Game)
        a = Car(Vector2(0, 0), 0, COLORS[0], Brain(), name="A")
        b = Car(Vector2(CAR_LENGTH_M - 0.2, 0), 0, COLORS[1], Brain(), name="B")
        a.velocity = Vector2(1.0, 0)
        b.velocity = Vector2(0, 0)
        game.cars = [a, b]
        game.event_log = []
        game.event_camera = False
        game.session_time = 0.0
        self.assertIsNotNone(game.car_collision_manifold(a, b))
        self.assertEqual(game.resolve_collisions(), 1)
        self.assertIsNone(game.car_collision_manifold(a, b))
        self.assertLess(a.velocity.x, 1.0)
        self.assertGreater(b.velocity.x, 0.0)
        self.assertLess(a.health, 100.0)

    def test_training_impacts_keep_full_health(self):
        game = object.__new__(Game)
        a = Car(Vector2(0, 0), 0, COLORS[0], Brain(), name="A")
        b = Car(
            Vector2(CAR_LENGTH_M - 0.2, 0), 0,
            COLORS[1], Brain(), name="B",
        )
        a.velocity = Vector2(1.0, 0)
        game.cars = [a, b]
        game.event_log = []
        game.event_camera = False
        game.session_time = 0.0
        self.assertEqual(game.resolve_collisions(apply_damage=False), 1)
        self.assertEqual(a.health, 100.0)
        self.assertEqual(b.health, 100.0)
        self.assertTrue(a.car_collision)
        self.assertTrue(b.car_collision)
        self.assertEqual(a.collision_count, 1)
        self.assertEqual(b.collision_count, 1)
        self.assertGreater(a.collision_penalty, 0.0)
        self.assertGreater(b.collision_penalty, 0.0)

        wall_car = spawn_car(
            self.track, Brain(), COLORS[2], "Wall Training Car"
        )
        tangent = self.track.tangent(wall_car.position)
        normal = Vector2(-tangent.y, tangent.x)
        wall_car.position += normal * 20.0
        wall_car.velocity = normal
        self.assertEqual(self.track.surface(wall_car.position), "wall")
        wall_car.update(self.track, damage_enabled=False)
        self.assertEqual(wall_car.health, 100.0)
        self.assertTrue(wall_car.alive)

    def test_clean_overtake_gets_large_reward_without_reward_farming(self):
        track = Track(
            [(0, 0), (1000, 0), (1000, 150), (0, 150)],
            "Overtake Reward Test", geometry="sampled", road_width_m=8.5,
            road_widths_m=[8.5] * 4,
        )
        follower = Car(
            Vector2(100, 0), 0, COLORS[0], Brain(), name="Follower"
        )
        leader = Car(
            Vector2(112, 0), 0, COLORS[1], Brain(), name="Leader"
        )
        follower.velocity = Vector2(1.1, 0)
        leader.velocity = Vector2(0.7, 0)
        game = object.__new__(Game)
        game.track = track
        game.cars = [follower, leader]

        game.update_training_overtakes()
        self.assertIn(id(leader), follower.overtake_candidates)
        follower.position = Vector2(120, 0)
        game.update_training_overtakes()
        self.assertEqual(follower.overtakes, 1)
        self.assertEqual(follower.overtake_reward, OVERTAKE_REWARD)
        self.assertEqual(follower.fitness, OVERTAKE_REWARD)

        # Reversing the order during the cooldown cannot repeatedly farm the
        # same rival for fitness.
        follower.position = Vector2(100, 0)
        game.update_training_overtakes()
        follower.position = Vector2(120, 0)
        game.update_training_overtakes()
        self.assertEqual(follower.overtakes, 1)
        self.assertEqual(follower.overtake_reward, OVERTAKE_REWARD)

    def test_collision_assisted_overtake_is_not_rewarded(self):
        track = Track(
            [(0, 0), (1000, 0), (1000, 150), (0, 150)],
            "Contact Pass Test", geometry="sampled", road_width_m=8.5,
            road_widths_m=[8.5] * 4,
        )
        follower = Car(
            Vector2(100, 0), 0, COLORS[0], Brain(), name="Follower"
        )
        leader = Car(
            Vector2(112, 0), 0, COLORS[1], Brain(), name="Leader"
        )
        follower.velocity = Vector2(1.1, 0)
        leader.velocity = Vector2(0.7, 0)
        game = object.__new__(Game)
        game.track = track
        game.cars = [follower, leader]

        game.update_training_overtakes()
        follower.position = Vector2(120, 0)
        follower.car_collision = True
        leader.car_collision = True
        game.update_training_overtakes()
        self.assertEqual(follower.overtakes, 0)
        self.assertEqual(follower.overtake_reward, 0.0)

    def test_slipstream_reaches_three_car_lengths_in_same_lane(self):
        game = object.__new__(Game)
        follower = Car(Vector2(0, 0), 0, COLORS[0], Brain(), name="Follower")
        leader = Car(Vector2(CAR_LENGTH_M * 2, 0), 0, COLORS[1], Brain(), name="Leader")
        game.cars = [follower, leader]
        game.update_slipstreams()
        self.assertGreater(follower.slipstream, 0.0)
        self.assertEqual(follower.drafting_car, "Leader")
        leader.position.x = CAR_LENGTH_M * 3.1
        game.update_slipstreams()
        self.assertEqual(follower.slipstream, 0.0)

    def test_faster_starter_car_moves_side_passes_and_recentres(self):
        track = Track(
            [(0, 0), (1000, 0), (1000, 150), (0, 150)],
            "Pass Test", geometry="sampled", road_width_m=8.5,
            road_widths_m=[8.5] * 4,
        )
        fast_source = (
            ROOT / "saved_data" / "algorithms" / "ice_controller.fai"
        ).read_text()
        slow_source = "steering = 0.0\nthrottle = 0.0\n"
        fast = Car(
            Vector2(100, 0), 0, COLORS[0],
            Brain(
                program=SafeAlgorithm(fast_source),
                source=fast_source,
            ),
            name="Fast",
        )
        slow = Car(
            Vector2(116, 0), 0, COLORS[1],
            Brain(
                program=SafeAlgorithm(slow_source),
                source=slow_source,
            ),
            name="Slow",
        )
        fast.velocity = Vector2(1.15, 0)
        slow.velocity = Vector2(0.65, 0)
        for car in (fast, slow):
            car.previous_progress = track.progress(car.position)
            car.previous_progress_m = track.progress_metres(car.position)

        game = object.__new__(Game)
        game.track = track
        game.cars = [fast, slow]
        game.event_log = []
        game.event_camera = False
        game.session_time = 0.0
        collisions = 0
        maximum_offset = 0.0
        started_pass = False
        for _ in range(180):
            game.update_race_awareness()
            game.update_slipstreams()
            started_pass = started_pass or fast.passing
            fast.update(track, damage_enabled=False)
            slow.update(track, damage_enabled=False)
            slow.velocity = Vector2(0.65, 0)
            collisions += game.resolve_collisions(apply_damage=False)
            maximum_offset = max(maximum_offset, abs(fast.position.y))

        self.assertTrue(started_pass)
        # A pass may include one brief box-edge touch, but cars must not stay
        # interlocked and repeatedly collide while the maneuver completes.
        self.assertLessEqual(collisions, 1)
        self.assertGreater(maximum_offset, CAR_WIDTH_M * 0.75)
        self.assertGreater(fast.position.x, slow.position.x + CAR_LENGTH_M)
        self.assertLess(abs(fast.position.y), 1.25)
        self.assertFalse(fast.passing)

    def test_racecraft_controller_passes_from_raw_awareness_without_assist(self):
        track = Track(
            [(0, 0), (1000, 0), (1000, 150), (0, 150)],
            "Unassisted Pass Test", geometry="sampled", road_width_m=8.5,
            road_widths_m=[8.5] * 4,
        )
        fast_source = (
            ROOT / "saved_data" / "algorithms" / "ice_controller.fai"
        ).read_text()
        slow_source = "steering = 0.0\nthrottle = 0.0\n"
        fast = Car(
            Vector2(100, 0), 0, COLORS[0],
            Brain(
                program=SafeAlgorithm(fast_source),
                source=fast_source,
            ),
            name="Learning Fast",
        )
        slow = Car(
            Vector2(116, 0), 0, COLORS[1],
            Brain(
                program=SafeAlgorithm(slow_source),
                source=slow_source,
            ),
            name="Traffic",
        )
        fast.velocity = Vector2(1.15, 0)
        slow.velocity = Vector2(0.65, 0)
        for car in (fast, slow):
            car.previous_progress = track.progress(car.position)
            car.previous_progress_m = track.progress_metres(car.position)

        game = object.__new__(Game)
        game.track = track
        game.cars = [fast, slow]
        game.event_log = []
        game.event_camera = False
        game.session_time = 0.0
        collisions = 0
        maximum_offset = 0.0
        saw_opponent = False
        for _ in range(240):
            game.update_race_awareness(assist_passing=False)
            game.update_slipstreams()
            self.assertFalse(fast.passing)
            self.assertEqual(fast.passing_side, 0.0)
            saw_opponent = saw_opponent or fast.opponent_presence[0] == 1.0
            fast.update(track, damage_enabled=False)
            slow.update(track, damage_enabled=False)
            slow.velocity = Vector2(0.65, 0)
            collisions += game.resolve_collisions(apply_damage=False)
            game.update_training_overtakes()
            maximum_offset = max(maximum_offset, abs(fast.position.y))

        self.assertTrue(saw_opponent)
        self.assertLessEqual(collisions, 1)
        self.assertEqual(fast.overtakes, 1)
        self.assertEqual(fast.overtake_reward, OVERTAKE_REWARD)
        self.assertGreater(maximum_offset, CAR_WIDTH_M)
        self.assertGreater(fast.position.x, slow.position.x + CAR_LENGTH_M)
        self.assertLess(abs(fast.position.y), 1.25)

    def test_high_load_understeer_and_low_grip_oversteer_are_reported(self):
        source = "steering = 1.0\nthrottle = 1.0\n"
        program = SafeAlgorithm(source)
        understeering = spawn_car(
            self.track,
            Brain(program=program, source=source),
            COLORS[0],
        )
        heading = Vector2(1, 0).rotate(understeering.angle)
        understeering.velocity = heading * 1.6
        understeering.update(self.track)
        self.assertGreater(understeering.understeer, 0.01)

        oversteering = spawn_car(
            self.track,
            Brain(program=program, source=source),
            COLORS[1],
        )
        tangent = self.track.tangent(oversteering.position)
        normal = Vector2(-tangent.y, tangent.x)
        oversteering.position += normal * 8.0
        oversteering.velocity = tangent
        self.assertEqual(self.track.surface(oversteering.position), "grass")
        oversteering.update(self.track, damage_enabled=False)
        self.assertGreater(oversteering.oversteer, 0.01)

    def test_understeer_no_longer_overwhelms_steering_authority(self):
        source = "steering = 1.0\nthrottle = 0.0\n"
        program = SafeAlgorithm(source)
        normal = spawn_car(
            self.track,
            Brain(program=program, source=source),
            COLORS[0],
        )
        affected = spawn_car(
            self.track,
            Brain(program=program, source=source),
            COLORS[1],
        )
        heading = Vector2(1, 0).rotate(normal.angle)
        normal.velocity = heading * 1.2
        affected.velocity = heading * 1.2
        affected.understeer = 1.0
        normal_start = normal.angle
        affected_start = affected.angle
        normal.update(self.track)
        affected.update(self.track)
        normal_turn = abs(normal.angle - normal_start)
        affected_turn = abs(affected.angle - affected_start)
        self.assertGreater(affected_turn, normal_turn * 0.84)
        self.assertLess(affected.understeer, 0.8)

    def test_car_rotates_quickly_and_recovers_lateral_slide(self):
        turning_source = "steering = 1.0\nthrottle = 0.0\n"
        turning = spawn_car(
            self.track,
            Brain(
                program=SafeAlgorithm(turning_source),
                source=turning_source,
            ),
            COLORS[0],
        )
        heading = Vector2(1, 0).rotate(turning.angle)
        turning.velocity = heading
        starting_angle = turning.angle
        turning.update(self.track)
        # Full lock at racing speed should now deliver the more agile
        # steering response requested for tight and medium-speed corners.
        self.assertGreater(abs(turning.angle - starting_angle), 4.2)

        settled_source = "steering = 0.0\nthrottle = 0.0\n"
        settled = spawn_car(
            self.track,
            Brain(
                program=SafeAlgorithm(settled_source),
                source=settled_source,
            ),
            COLORS[1],
        )
        heading = Vector2(1, 0).rotate(settled.angle)
        normal = Vector2(-heading.y, heading.x)
        settled.velocity = heading + normal * 0.5
        settled.update(self.track)
        self.assertLess(abs(settled.velocity.dot(normal)), 0.4)

    def test_draft_adds_acceleration_but_reduces_steering_response(self):
        source = "steering = 1.0\nthrottle = 1.0\n"
        program = SafeAlgorithm(source)
        normal = spawn_car(
            self.track, Brain(program=program, source=source), COLORS[0]
        )
        drafted = spawn_car(
            self.track, Brain(program=program, source=source), COLORS[1]
        )
        initial_angle = normal.angle
        drafted.slipstream = 1.0
        normal.update(self.track)
        drafted.update(self.track)
        self.assertGreater(drafted.velocity.length(), normal.velocity.length())
        self.assertLess(
            abs(drafted.angle - initial_angle),
            abs(normal.angle - initial_angle),
        )

    def test_hybrid_has_seventy_percent_base_power_and_deploys_remaining_power(self):
        ice_source = "steering = 0.0\nthrottle = 1.0\n"
        boost_source = ice_source + "overtake = 1.0\n"
        ice = spawn_car(
            self.track,
            Brain(program=SafeAlgorithm(ice_source), source=ice_source),
            COLORS[0],
        )
        hybrid = spawn_car(
            self.track,
            Brain(program=SafeAlgorithm(ice_source), source=ice_source),
            COLORS[1],
        )
        boosted = spawn_car(
            self.track,
            Brain(program=SafeAlgorithm(boost_source), source=boost_source),
            COLORS[2],
        )
        ice.generation = "ICE"
        hybrid.generation = "Hybrid"
        boosted.generation = "Hybrid"
        ice.update(self.track)
        hybrid.update(self.track)
        boosted.update(self.track)
        self.assertAlmostEqual(
            hybrid.velocity.length() / ice.velocity.length(),
            0.70,
            delta=0.01,
        )
        self.assertAlmostEqual(
            boosted.velocity.length(), ice.velocity.length(), delta=0.001
        )
        self.assertTrue(boosted.overtake_active)
        self.assertLess(boosted.battery, 100.0)
        self.assertEqual(ice.battery, 0.0)

    def test_hybrid_regeneration_scales_with_braking(self):
        source = (
            "steering = 0.0\n"
            "throttle = 0.0\n"
            "brake = 1.0\n"
        )
        car = spawn_car(
            self.track,
            Brain(program=SafeAlgorithm(source), source=source),
            COLORS[0],
        )
        car.generation = "Hybrid"
        car.battery = 50.0
        car.velocity = Vector2(1, 0).rotate(car.angle)
        previous_speed = car.velocity.length()
        car.update(self.track)
        self.assertGreater(car.battery, 50.0)
        self.assertGreater(car.battery_regen, 0.0)
        self.assertLess(car.velocity.length(), previous_speed)

    def test_tiny_lateral_velocity_stops_without_vector_scale_error(self):
        source = (
            "steering = 1.0\n"
            "throttle = 0.0\n"
            "brake = 0.0\n"
        )
        car = spawn_car(
            self.track,
            Brain(program=SafeAlgorithm(source), source=source),
            COLORS[0],
        )
        heading = Vector2(1, 0).rotate(car.angle)
        normal = Vector2(-heading.y, heading.x)
        car.velocity = heading * 8e-7 + normal * 8e-7

        car.update(self.track, damage_enabled=False)

        self.assertEqual(car.velocity, Vector2())

    def test_hybrid_recharge_scales_with_throttle_and_stops_at_zero(self):
        def recharge_car(throttle):
            source = (
                "steering = 0.0\n"
                f"throttle = {throttle}\n"
                "brake = 0.0\n"
                "recharge = 1.0\n"
            )
            car = spawn_car(
                self.track,
                Brain(program=SafeAlgorithm(source), source=source),
                COLORS[0],
            )
            car.generation = "Hybrid"
            car.battery = 20.0
            car.update(self.track)
            return car

        idle = recharge_car(0.0)
        half_throttle = recharge_car(0.5)
        full_throttle = recharge_car(1.0)

        self.assertTrue(idle.recharge_active)
        self.assertEqual(idle.battery_regen, 0.0)
        self.assertEqual(idle.battery, 20.0)
        self.assertAlmostEqual(
            half_throttle.battery_regen,
            HYBRID_RECHARGE_RATE * 0.5,
        )
        self.assertAlmostEqual(
            full_throttle.battery_regen,
            HYBRID_RECHARGE_RATE,
        )

    def test_ice_and_hybrid_use_distinct_era_silhouettes(self):
        ice_surface = pygame.Surface((260, 160), pygame.SRCALPHA)
        hybrid_surface = pygame.Surface((260, 160), pygame.SRCALPHA)
        ice = Car(Vector2(16, 10), 0, COLORS[0], Brain(), generation="ICE")
        hybrid = Car(
            Vector2(16, 10), 0, COLORS[0], Brain(), generation="Hybrid"
        )
        ice.draw(ice_surface, scale=8.0)
        hybrid.draw(hybrid_surface, scale=8.0)
        self.assertNotEqual(
            pygame.image.tobytes(ice_surface, "RGBA"),
            pygame.image.tobytes(hybrid_surface, "RGBA"),
        )
        self.assertNotEqual(
            pygame.mask.from_surface(ice_surface).count(),
            pygame.mask.from_surface(hybrid_surface).count(),
        )

    def test_car_sprite_rotation_is_reused_between_frames(self):
        from main import _CAR_ROTATION_CACHE, _CAR_SPRITE_CACHE

        _CAR_ROTATION_CACHE.clear()
        _CAR_SPRITE_CACHE.clear()
        surface = pygame.Surface((260, 160), pygame.SRCALPHA)
        car = Car(
            Vector2(16, 10), 13.0, (101, 149, 211), Brain(),
            generation="Hybrid",
        )
        car.draw(surface, scale=8.0)
        first_rotation_count = len(_CAR_ROTATION_CACHE)
        first_base_count = len(_CAR_SPRITE_CACHE)
        car.angle = 13.2
        car.draw(surface, scale=8.0)
        self.assertEqual(len(_CAR_ROTATION_CACHE), first_rotation_count)
        self.assertEqual(len(_CAR_SPRITE_CACHE), first_base_count)
        self.assertEqual(first_rotation_count, 1)
        self.assertEqual(first_base_count, 1)

    def test_default_camera_keeps_car_art_readable_at_true_scale(self):
        surface = pygame.Surface((260, 160), pygame.SRCALPHA)
        car = Car(
            Vector2(130 / DEFAULT_CAMERA_ZOOM, 80 / DEFAULT_CAMERA_ZOOM),
            0, COLORS[0], Brain(), generation="Hybrid",
        )
        car.draw(surface, scale=DEFAULT_CAMERA_ZOOM)
        bounds = pygame.mask.from_surface(surface).get_bounding_rects()
        union = bounds[0].unionall(bounds[1:])
        self.assertGreaterEqual(union.width, 42)
        self.assertGreaterEqual(union.height, 15)

    def test_algorithm_can_command_braking_and_hybrid_overtake(self):
        program = SafeAlgorithm(
            "steering = 0.0\n"
            "throttle = 0.8\n"
            "brake = 0.25\n"
            "overtake = battery > 0.2\n"
        )
        self.assertEqual(
            program.run({"battery": 0.8}, program.defaults()),
            (0.0, 0.8, 0.25, 1.0, 0.0, 1),
        )

    def test_ice_and_hybrid_controller_drafts_are_independent(self):
        game = object.__new__(Game)
        game.training_generation = "Hybrid"
        game.algorithm_source = "HYBRID DRAFT"
        game.algorithm_path = Path("hybrid.fai")
        game.algorithm_sources = {
            "Hybrid": "HYBRID TEMPLATE",
            "ICE": "ICE TEMPLATE",
        }
        game.algorithm_paths = {
            "Hybrid": Path("hybrid.fai"),
            "ICE": Path("ice.fai"),
        }
        game.algorithm_cursor = 0
        game.algorithm_anchor = None
        game.algorithm_undo = [("old", 0, None)]
        game.algorithm_redo = [("new", 0, None)]
        game.algorithm_scroll_line = 4
        game.algorithm_error = "error"
        game.switch_training_generation("ICE")
        self.assertEqual(game.algorithm_source, "ICE TEMPLATE")
        self.assertEqual(game.algorithm_sources["Hybrid"], "HYBRID DRAFT")
        game.algorithm_source = "ICE DRAFT"
        game.switch_training_generation("Hybrid")
        self.assertEqual(game.algorithm_source, "HYBRID DRAFT")
        self.assertEqual(game.algorithm_sources["ICE"], "ICE DRAFT")

    def test_algorithm_editor_copy_and_multiline_system_paste(self):
        game = object.__new__(Game)
        game.algorithm_clipboard = ""
        game._system_clipboard_ready = False
        with patch("main.pygame.scrap.init") as scrap_init, patch(
            "main.pygame.scrap.put"
        ) as scrap_put, patch(
            "main.pygame.scrap.get", return_value=b"brake = 0.2\r\nthrottle = 0.8\0"
        ):
            game.editor_set_clipboard("steering = -0.25")
            scrap_init.assert_called_once_with()
            scrap_put.assert_called_once_with(
                pygame.SCRAP_TEXT, b"steering = -0.25\0"
            )
            self.assertEqual(
                game.editor_get_clipboard(),
                "brake = 0.2\nthrottle = 0.8",
            )

    def test_algorithm_editor_clipboard_falls_back_when_unavailable(self):
        game = object.__new__(Game)
        game.algorithm_clipboard = ""
        game._system_clipboard_ready = False
        with patch("main.pygame.scrap.init", side_effect=pygame.error("unavailable")):
            game.editor_set_clipboard("overtake = 1")
            self.assertEqual(game.editor_get_clipboard(), "overtake = 1")

    def test_training_brain_selector_defaults_to_empty_and_lists_saves(self):
        source = (
            "gain = parameter(0.5, 0.0, 1.0)\n"
            "steering = 0.0\n"
            "throttle = gain\n"
        )
        with tempfile.TemporaryDirectory() as directory, patch(
            "main.BRAIN_DIR", Path(directory)
        ):
            saved_path = Brain(
                program=SafeAlgorithm(source),
                parameters={"gain": 0.8},
                source=source,
            ).save("Seed Brain")
            game = object.__new__(Game)
            game.training_base_brain = "__empty__"

            choices = game.training_brain_choices()
            self.assertEqual(choices[0], ("__empty__", "EMPTY BRAIN"))
            self.assertNotIn("__session__", dict(choices))
            self.assertIn(saved_path.name, dict(choices))
            self.assertEqual(game.training_brain_label(), "EMPTY BRAIN")

            game.cycle_training_brain()
            self.assertEqual(game.training_base_brain, saved_path.name)
            self.assertEqual(game.training_brain_label(), "SEED BRAIN")

    def test_saved_brain_seeds_compatible_training_parameters(self):
        saved_source = (
            "gain = parameter(0.5, 0.0, 1.0)\n"
            "old_value = parameter(0.4, 0.0, 1.0)\n"
            "steering = 0.0\n"
            "throttle = gain\n"
        )
        current_source = (
            "gain = parameter(0.2, 0.1, 0.6)\n"
            "fresh = parameter(0.3, 0.0, 1.0)\n"
            "steering = 0.0\n"
            "throttle = gain * fresh\n"
        )
        with tempfile.TemporaryDirectory() as directory, patch(
            "main.BRAIN_DIR", Path(directory)
        ):
            saved_path = Brain(
                config={"mutation": 0.07},
                program=SafeAlgorithm(saved_source),
                parameters={"gain": 0.95, "old_value": 0.9},
                source=saved_source,
            ).save("Experienced Seed")
            game = object.__new__(Game)
            game.training_base_brain = saved_path.name

            seed = game.create_training_seed(
                SafeAlgorithm(current_source), current_source
            )
            self.assertEqual(seed.source, current_source)
            self.assertAlmostEqual(seed.parameters["gain"], 0.6)
            self.assertAlmostEqual(seed.parameters["fresh"], 0.3)
            self.assertNotIn("old_value", seed.parameters)
            self.assertAlmostEqual(seed.config["mutation"], 0.07)

            game.training_base_brain = "__empty__"
            empty = game.create_training_seed(
                SafeAlgorithm(current_source), current_source
            )
            self.assertEqual(
                empty.parameters, {"gain": 0.2, "fresh": 0.3}
            )

    def test_race_assigns_the_selected_brain_to_each_entry(self):
        saved = sorted((ROOT / "saved_data" / "brains").glob("*.json"))
        if not saved:
            self.skipTest("No saved brain fixture")
        game = object.__new__(Game)
        game.track = self.track
        game.best_brain = Brain()
        game.race_settings = {
            "cars": 2, "laps": 1, "weather": "Dry",
            "generation": "Hybrid", "teams": False,
        }
        game.race_entries = [
            {"name": "Session", "color": 0, "tyre": "Soft", "fuel": 20, "brain": "__session__"},
            {"name": "Saved", "color": 1, "tyre": "Medium", "fuel": 20, "brain": saved[0].name},
        ]
        game.team_names = ["Team 1"]
        game.start_race()
        self.assertEqual(game.cars[0].brain_name, "CURRENT SESSION")
        self.assertEqual(game.cars[1].brain_name, saved[0].stem.replace("_", " ").upper())
        self.assertIsNot(game.cars[0].brain, game.cars[1].brain)
        self.assertEqual(game.race_countdown, 5.0)
        self.assertEqual(game.flag_state, "START SEQUENCE")

        positions = [car.position.copy() for car in game.cars]
        for expected in (4.0, 3.0, 2.0, 1.0):
            self.assertTrue(game.advance_race_countdown(1000))
            self.assertEqual(game.race_countdown, expected)
            self.assertEqual(
                [car.position for car in game.cars], positions
            )
        self.assertTrue(game.advance_race_countdown(1000))
        self.assertEqual(game.race_countdown, 0.0)
        self.assertEqual(game.flag_state, "GREEN")
        self.assertEqual(game.race_lights_out_flash, 1.0)
        self.assertIn("LIGHTS OUT", game.event_log[-1]["message"])
        self.assertFalse(game.advance_race_countdown(1000))

    def test_hotlap_clock_stops_exactly_after_second_lap(self):
        class FinishingCar:
            def __init__(self):
                self.lap = 1
                self.slipstream = 1.0
                self.drafting_car = "Other"
                self.alive = True
                self.finish_time = None
                self.velocity = Vector2(1, 0)

            def update(self, track, rain=0.0):
                self.lap = 2

        game = object.__new__(Game)
        game.track = self.track
        game.hotlap_car = FinishingCar()
        game.hotlap_time = 100.0
        game.hotlap_splits = [49.0]
        game.hotlap_finished = False
        game.paused = False
        game.advance_hotlap(16)
        self.assertTrue(game.hotlap_finished)
        self.assertAlmostEqual(game.hotlap_time, 100.016)
        self.assertEqual(len(game.hotlap_splits), 2)
        self.assertAlmostEqual(game.hotlap_splits[1], 51.016)
        self.assertEqual(game.hotlap_car.velocity.length(), 0.0)

    def test_component_names_are_sanitized_without_losing_readable_labels(self):
        self.assertEqual(
            safe_component_name("  My Spa: Test!  "),
            "My Spa Test",
        )
        self.assertEqual(component_filename("My Spa Test", ".json"), "my_spa_test.json")

    def test_named_tracks_and_brains_store_their_display_names(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            track_dir = root / "tracks"
            brain_dir = root / "brains"
            with patch("main.TRACK_DIR", track_dir), patch("main.BRAIN_DIR", brain_dir):
                track_path = Track.load(
                    ROOT / "saved_data" / "tracks" / "spa_francorchamps.json"
                ).save("Seven Kilometre Circuit")
                brain_path = Brain().save("Wet Race Brain")
            self.assertEqual(track_path.name, "seven_kilometre_circuit.json")
            self.assertEqual(brain_path.name, "wet_race_brain.json")
            self.assertEqual(json.loads(track_path.read_text())["name"], "Seven Kilometre Circuit")
            self.assertEqual(json.loads(brain_path.read_text())["name"], "Wet Race Brain")

    def test_track_selector_loads_the_selected_saved_track(self):
        game = object.__new__(Game)
        game.track = self.track
        game.selected_track = "spa_francorchamps.json"
        game.camera_zoom = 2.0
        choices = game.track_choices()
        self.assertGreaterEqual(len(choices), 2)
        selected = game.cycle_track(1)
        self.assertEqual(game.selected_track, selected)
        self.assertEqual(game.track.name, dict(choices)[selected])
        self.assertEqual(game.camera_zoom, DEFAULT_CAMERA_ZOOM)

    def test_name_dialog_edits_selection_and_guards_existing_files(self):
        game = object.__new__(Game)
        game.name_dialog = {
            "kind": "brain",
            "value": "Old Name",
            "cursor": 8,
            "anchor": 0,
            "return_mode": "training",
            "payload": Brain(),
            "error": "",
            "replace": False,
        }
        game.replace_name_dialog_selection("New Brain")
        self.assertEqual(game.name_dialog["value"], "New Brain")
        with tempfile.TemporaryDirectory() as directory:
            brain_dir = Path(directory)
            existing = brain_dir / "new_brain.json"
            existing.write_text("{}")
            with patch("main.BRAIN_DIR", brain_dir):
                self.assertIsNone(game.confirm_name_dialog())
            self.assertEqual(existing.read_text(), "{}")
        self.assertTrue(game.name_dialog["replace"])
        self.assertIn("already exists", game.name_dialog["error"])

    def test_race_car_name_can_be_replaced_confirmed_and_cancelled(self):
        game = object.__new__(Game)
        entry = {"name": "NOVA"}
        game.editing_name = None
        game.race_name_value = ""
        game.race_name_cursor = 0
        game.race_name_anchor = None
        game.race_name_target = None
        game.message = ""
        game.message_until = 0
        with (
            patch.object(pygame.key, "start_text_input"),
            patch.object(pygame.key, "stop_text_input"),
        ):
            game.start_race_name_edit(entry)
            self.assertEqual(game.race_name_selection(), (0, 4))
            game.replace_race_name_selection("Hamilton")
            game.finish_race_name_edit(True)
            self.assertEqual(entry["name"], "HAMILTON")

            game.start_race_name_edit(entry)
            game.replace_race_name_selection("Temporary")
            game.finish_race_name_edit(False)
            self.assertEqual(entry["name"], "HAMILTON")

            game.start_race_name_edit(entry)
            game.replace_race_name_selection("")
            game.finish_race_name_edit(True)
            self.assertEqual(entry["name"], "HAMILTON")
            self.assertIn("must have a name", game.message)


if __name__ == "__main__":
    unittest.main()
