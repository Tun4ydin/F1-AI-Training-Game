import os
import json
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame
from pygame import Vector2

from main import (
    GEAR_COUNT, MAX_ENGINE_RPM, PIT_SPEED_LIMIT_KPH, REFERENCE_TOP_SPEED,
    ROAD, Brain, COLORS, Game, Track, spawn_car,
)
from safe_algorithm import SafeAlgorithm


ROOT = Path(__file__).resolve().parents[1]


class TrainingRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.track = Track.load(
            ROOT / "saved_data" / "tracks" / "spa_francorchamps.json"
        )
        cls.controller_source = (
            ROOT / "saved_data" / "algorithms" / "custom_controller.fai"
        ).read_text()
        cls.ice_controller_source = (
            ROOT / "saved_data" / "algorithms" / "ice_controller.fai"
        ).read_text()
        cls.hybrid_controller_source = (
            ROOT / "saved_data" / "algorithms" / "hybrid_controller.fai"
        ).read_text()

    def simulate(self, source, frames):
        program = SafeAlgorithm(source)
        car = spawn_car(
            self.track,
            Brain(program=program, source=source),
            COLORS[0],
            "Test Agent",
        )
        for _ in range(frames):
            car.update(self.track)
        return car

    def test_spa_uses_real_metric_length(self):
        self.assertAlmostEqual(self.track.measured_length_m, 7004.0, delta=0.1)
        self.assertEqual(self.track.road_width_m, 8.5)
        self.assertTrue(
            all(width == 8.5 for width in self.track.road_widths_m)
        )
        self.assertAlmostEqual(self.track.progress_metres(self.track.centerline[0]), 0.0)

    def test_node_widths_interpolate_and_control_the_surface(self):
        track = Track(
            [(0, 0), (100, 0), (100, 100), (0, 100)],
            kerb_points=[],
            geometry="sampled",
            road_width_m=8.0,
            road_widths_m=[8.0, 10.0, 12.0, 14.0],
        )
        self.assertAlmostEqual(track.width_at_segment(0, 0.5), 9.0)
        self.assertEqual(track.surface((50, 4.4)), "asphalt")
        self.assertEqual(track.surface((50, 4.6)), "grass")

    def test_pitlane_connects_entry_nodes_and_is_a_driveable_surface(self):
        track = Track(
            [(0, 0), (100, 0), (100, 100), (0, 100)],
            features={
                "pit_entry": 0, "pit_exit": 1, "pit_boxes": [0, 1],
            },
            geometry="sampled",
            road_width_m=8.0,
            pitlane_points=[(25, 20), (75, 20)],
        )
        self.assertEqual(track.pitlane_centerline[0], Vector2(0, 4))
        self.assertEqual(track.pitlane_centerline[-1], Vector2(100, 4))
        self.assertEqual(
            track.pitlane_centerline[1:-1],
            [Vector2(25, 20), Vector2(75, 20)],
        )
        self.assertEqual(track.surface((50, 20)), "pitlane")
        self.assertTrue(track.is_in_pitlane((50, 20)))
        self.assertEqual(
            track.pit_box_positions(),
            [Vector2(25, 20), Vector2(75, 20)],
        )
        source = (
            "steering = 0.0\n"
            "throttle = 0.0\n"
            "pit_request = 1.0\n"
        )
        pit_car = spawn_car(
            track,
            Brain(program=SafeAlgorithm(source), source=source),
            COLORS[0],
        )
        pit_car.position = Vector2(25, 20)
        pit_car.update(track, damage_enabled=False)
        self.assertGreater(pit_car.pit_timer, 0.0)

        opposite = Track(
            [(0, 0), (100, 0), (100, 100), (0, 100)],
            features={"pit_entry": 0, "pit_exit": 1},
            geometry="sampled",
            road_width_m=8.0,
            road_widths_m=[10.0, 12.0, 8.0, 8.0],
            pitlane_points=[(25, -20), (75, -20)],
        )
        self.assertEqual(opposite.pitlane_centerline[0], Vector2(0, -5))
        self.assertEqual(opposite.pitlane_centerline[-1], Vector2(100, -6))

    def test_pitlane_enforces_eighty_kph_without_limiting_main_track(self):
        track = Track(
            [(0, 0), (100, 0), (100, 100), (0, 100)],
            features={"pit_entry": 0, "pit_exit": 1, "pit_boxes": []},
            geometry="sampled",
            road_width_m=8.0,
            pitlane_points=[(25, 20), (75, 20)],
        )
        source = "steering = 0.0\nthrottle = 0.0\n"
        brain = Brain(
            program=SafeAlgorithm(source),
            source=source,
        )
        pit_car = spawn_car(track, brain, COLORS[0])
        pit_car.position = Vector2(50, 20)
        pit_car.angle = 0.0
        pit_car.velocity = Vector2(1.0, 0.0)
        pit_car.update(track, damage_enabled=False)
        self.assertTrue(pit_car.in_pitlane)
        self.assertLessEqual(pit_car.speed_kph, PIT_SPEED_LIMIT_KPH + 0.01)

        track_car = spawn_car(track, brain, COLORS[1])
        track_car.position = Vector2(50, 0)
        track_car.angle = 0.0
        track_car.velocity = Vector2(1.0, 0.0)
        track_car.update(track, damage_enabled=False)
        self.assertFalse(track_car.in_pitlane)
        self.assertGreater(track_car.speed_kph, PIT_SPEED_LIMIT_KPH)

    def test_variable_width_ribbon_has_no_triangular_asphalt_gaps(self):
        track = Track(
            [(50, 50), (250, 55), (235, 220), (65, 205)],
            kerb_points=[1, 2],
            features={"start_finish": 0, "sectors": []},
            geometry="sampled",
            road_width_m=12.0,
            road_widths_m=[8.0, 22.0, 7.0, 18.0],
        )
        surface = pygame.Surface((300, 270))
        surface.fill((31, 92, 49))
        track.draw(surface)
        checked = 0
        for y in range(35, 235):
            for x in range(35, 270):
                point = Vector2(x, y)
                distance, _, segment, ratio = track.nearest(point)
                width = track.width_at_segment(segment, ratio)
                if (
                    distance <= width / 2 - 1.25
                    and point.distance_to(track.centerline[0]) > 14
                ):
                    pixel = surface.get_at((x, y))[:3]
                    self.assertLessEqual(
                        max(abs(pixel[i] - ROAD[i]) for i in range(3)),
                        25,
                    )
                    checked += 1
        self.assertGreater(checked, 4500)

    def test_eight_speed_gearbox_reaches_thirteen_thousand_rpm(self):
        car = spawn_car(self.track, Brain(), COLORS[0])
        heading = Vector2(1, 0).rotate(car.angle)
        car.velocity = heading * REFERENCE_TOP_SPEED
        car.update_drivetrain(1.0)
        self.assertEqual(car.gear, GEAR_COUNT)
        self.assertEqual(car.rpm, MAX_ENGINE_RPM)
        self.assertAlmostEqual(
            car.speed_kph, REFERENCE_TOP_SPEED * 60 * 3.6
        )

    def test_training_control_demographics_are_bounded_percentages(self):
        car = spawn_car(self.track, Brain(), COLORS[0])
        car.throttle_input = 0.735
        car.brake_input = 0.126
        self.assertEqual(
            Game.training_control_percentages(car), (74, 13)
        )
        car.throttle_input = 1.5
        car.brake_input = -0.5
        self.assertEqual(
            Game.training_control_percentages(car), (100, 0)
        )

    def test_training_hybrid_energy_infographic_states(self):
        car = spawn_car(self.track, Brain(), COLORS[0])
        car.generation = "Hybrid"
        self.assertEqual(
            Game.training_hybrid_energy_state(car)[:2],
            ("READY", "0% ELEC"),
        )
        car.overtake_active = True
        self.assertEqual(
            Game.training_hybrid_energy_state(car)[:2],
            ("DEPLOY", "+20% ELEC"),
        )
        car.drs_active = True
        self.assertEqual(
            Game.training_hybrid_energy_state(car)[:2],
            ("M.O.M.", "+30% ELEC"),
        )
        car.battery_regen = 0.1
        self.assertEqual(
            Game.training_hybrid_energy_state(car)[:2],
            ("REGEN", "HARVEST"),
        )
        car.recharge_active = True
        self.assertEqual(
            Game.training_hybrid_energy_state(car)[:2],
            ("RECHARGE", "CHARGING"),
        )

    def test_extended_ai_state_is_normalized_and_tracks_previous_action(self):
        self.assertEqual(set(Brain.INPUT_NAMES), SafeAlgorithm.INPUT_NAMES)
        source = (
            "steering = 0.25\n"
            "throttle = 0.75\n"
            "brake = 0.1\n"
        )
        car = self.simulate(source, 2)
        inputs = car.controller_inputs(self.track)
        self.assertEqual(len(inputs), len(Brain.INPUT_NAMES))
        state = dict(zip(Brain.INPUT_NAMES, inputs))
        self.assertEqual(len(state), len(Brain.INPUT_NAMES))
        for name in (
            "local_velocity_forward", "local_velocity_lateral",
            "angular_velocity", "traction", "tire_slip", "rpm", "gear",
            "ray_left_90", "ray_left_18", "ray_right_18", "ray_right_90",
            "waypoint_5_forward", "waypoint_5_right",
            "waypoint_40_forward", "waypoint_40_right",
            "opponent_1_present", "opponent_2_present",
            "opponent_3_present",
            "previous_steering", "previous_throttle", "previous_brake",
        ):
            self.assertGreaterEqual(state[name], -1.0, name)
            self.assertLessEqual(state[name], 1.0, name)
        self.assertAlmostEqual(state["previous_steering"], 0.25)
        self.assertAlmostEqual(state["previous_throttle"], 0.75)
        self.assertAlmostEqual(state["previous_brake"], 0.1)
        self.assertIn(int(state["gear_number"]), range(1, 9))
        self.assertLessEqual(state["rpm_value"], MAX_ENGINE_RPM)

    def test_three_nearest_opponents_are_local_position_velocity_vectors(self):
        follower = spawn_car(self.track, Brain(), COLORS[0])
        heading = Vector2(1, 0).rotate(follower.angle)
        normal = Vector2(-heading.y, heading.x)
        first = spawn_car(self.track, Brain(), COLORS[1])
        second = spawn_car(self.track, Brain(), COLORS[2])
        fourth = spawn_car(self.track, Brain(), COLORS[3])
        first.position = follower.position + heading * 12 + normal * 6
        second.position = follower.position - heading * 24
        fourth.position = follower.position + heading * 80
        first.velocity = heading * 0.5
        follower.velocity = heading * 0.2
        game = object.__new__(Game)
        game.cars = [follower, first, second, fourth]
        game.update_opponent_vectors()
        self.assertAlmostEqual(follower.opponent_data[0], 12 / 60)
        self.assertAlmostEqual(follower.opponent_data[1], 6 / 60)
        self.assertAlmostEqual(
            follower.opponent_data[2], 0.3 / REFERENCE_TOP_SPEED
        )
        self.assertAlmostEqual(follower.opponent_data[4], -24 / 60)
        self.assertEqual(follower.opponent_data[8:], (0.0,) * 4)
        self.assertEqual(follower.opponent_presence, (1.0, 1.0, 0.0))

    def test_variable_widths_round_trip_through_track_json(self):
        data = {
            "name": "Width Test",
            "points": [[0, 0], [100, 0], [100, 100], [0, 100]],
            "kerb_points": [],
            "features": {"border_margin": 30},
            "geometry": "sampled",
            "declared_length_m": 400,
            "road_width_m": 10,
            "road_widths_m": [8, 9, 11, 12],
        }
        with tempfile.NamedTemporaryFile("w", suffix=".json") as handle:
            json.dump(data, handle)
            handle.flush()
            loaded = Track.load(handle.name)
        self.assertEqual(loaded.road_widths_m, [8.0, 9.0, 11.0, 12.0])
        self.assertEqual(loaded.road_width_m, 10.0)

    def test_pitlane_round_trips_and_editor_requires_both_endpoints(self):
        data = {
            "name": "Pitlane Test",
            "points": [[0, 0], [100, 0], [100, 100], [0, 100]],
            "features": {
                "pit_entry": 0, "pit_exit": 1, "pit_boxes": [1],
            },
            "geometry": "sampled",
            "road_width_m": 8,
            "pitlane_points": [[25, 20], [75, 20]],
        }
        with tempfile.NamedTemporaryFile("w", suffix=".json") as handle:
            json.dump(data, handle)
            handle.flush()
            loaded = Track.load(handle.name)
        self.assertEqual(
            loaded.pitlane_points,
            [Vector2(25, 20), Vector2(75, 20)],
        )
        self.assertEqual(loaded.pit_box_positions(), [Vector2(75, 20)])

        game = object.__new__(Game)
        game.editor_points = [Vector2(point) for point in data["points"]]
        game.editor_features = {
            "pit_entry": None, "pit_exit": None, "pit_boxes": [],
        }
        game.editor_pitlane_points = []
        self.assertFalse(game.add_editor_pitlane_node((25, 20)))
        game.editor_features["pit_entry"] = 0
        self.assertFalse(game.add_editor_pitlane_node((25, 20)))
        game.editor_features["pit_exit"] = 1
        self.assertTrue(game.add_editor_pitlane_node((25, 20)))
        self.assertTrue(game.add_editor_pitlane_node((75, 20)))
        self.assertTrue(game.toggle_editor_pit_box(1))
        self.assertFalse(game.toggle_editor_pit_box(4))
        self.assertEqual(game.editor_features["pit_boxes"], [1])

    def test_editor_selects_and_adjusts_individual_node_width(self):
        game = object.__new__(Game)
        game.editor_points = [
            self.track.points[0].copy(),
            self.track.points[1].copy(),
        ]
        game.editor_widths = [8.0, 10.0]
        game.editor_road_width = 9.0
        game.editor_camera = game.editor_points[0].copy()
        game.editor_zoom = 1.0
        self.assertEqual(game.editor_node_at((0, 0)), 0)
        game.adjust_editor_node_width(0, 0.5)
        self.assertEqual(game.editor_widths, [8.5, 10.0])
        self.assertEqual(game.editor_road_width, 9.25)

    def test_spawn_initializes_progress_without_free_fitness(self):
        car = spawn_car(self.track, Brain(), COLORS[0], offset=12)
        self.assertIsNotNone(car.previous_progress_m)
        self.assertEqual(car.forward_distance_m, 0.0)
        self.assertEqual(car.fitness, 0.0)

    def test_starter_controller_drives_forward_instead_of_oscillating(self):
        car = self.simulate(self.controller_source, 900)
        self.assertGreater(car.forward_distance_m, 700.0)
        self.assertGreater(car.fitness, 650.0)
        self.assertEqual(car.track_limits, 0)

    def test_powertrain_templates_are_separate_and_hybrid_recovers(self):
        SafeAlgorithm(self.ice_controller_source)
        SafeAlgorithm(self.hybrid_controller_source)
        self.assertNotEqual(
            self.ice_controller_source, self.hybrid_controller_source
        )
        self.assertNotIn("overtake =", self.ice_controller_source)
        self.assertIn("overtake =", self.hybrid_controller_source)
        for source in (
            self.ice_controller_source, self.hybrid_controller_source
        ):
            for sensor in (
                "waypoint_5_right",
                "waypoint_20_right",
                "local_velocity_lateral",
                "angular_velocity",
                "tire_slip",
                "previous_steering",
                "opponent_1_velocity_forward",
                "opponent_1_present",
                "opponent_2_right",
                "opponent_3_right",
                "rpm",
                "gear",
            ):
                self.assertIn(sensor, source)

        ice_program = SafeAlgorithm(self.ice_controller_source)
        ice_car = spawn_car(
            self.track,
            Brain(
                program=ice_program,
                source=self.ice_controller_source,
            ),
            COLORS[1],
            "ICE Agile Starter",
        )
        ice_car.generation = "ICE"
        for _ in range(1800):
            ice_car.update(self.track)
        self.assertGreater(ice_car.forward_distance_m, 1200.0)
        self.assertTrue(ice_car.alive)

        program = SafeAlgorithm(self.hybrid_controller_source)
        car = spawn_car(
            self.track,
            Brain(program=program, source=self.hybrid_controller_source),
            COLORS[0],
            "Hybrid Recovery",
        )
        car.generation = "Hybrid"
        car.battery = 100.0
        for _ in range(1800):
            car.update(self.track)
        self.assertGreater(car.forward_distance_m, 1200.0)
        self.assertGreater(car.velocity.length(), 0.05)
        self.assertLess(car.battery, 100.0)
        self.assertTrue(car.alive)

    def test_hybrid_controller_latches_recharge_from_ten_to_forty_percent(self):
        program = SafeAlgorithm(self.hybrid_controller_source)
        parameters = program.defaults()
        base = {
            "is_hybrid": 1.0,
            "forward": 1.0,
            "speed": 0.7,
            "speed_kph": 280.0,
            "rpm_value": 11000.0,
            "traction": 1.0,
        }

        enters = program.run({
            **base,
            "battery": 0.09,
            "battery_percent": 9.0,
            "gear_number": 7.0,
        }, parameters)
        low_gear = program.run({
            **base,
            "battery": 0.09,
            "battery_percent": 9.0,
            "gear_number": 6.0,
        }, parameters)
        braking = program.run({
            **base,
            "battery": 0.09,
            "battery_percent": 9.0,
            "gear_number": 7.0,
            "forward": 0.4,
            "speed": 0.8,
        }, parameters)
        latched = program.run({
            **base,
            "battery": 0.30,
            "battery_percent": 30.0,
            "gear_number": 3.0,
            "recharge_active": 1.0,
        }, parameters)
        released = program.run({
            **base,
            "battery": 0.41,
            "battery_percent": 41.0,
            "gear_number": 8.0,
            "recharge_active": 1.0,
        }, parameters)

        self.assertEqual(enters[4], 1.0)
        self.assertEqual(low_gear[4], 0.0)
        self.assertGreater(braking[2], 0.0)
        self.assertEqual(braking[4], 0.0)
        self.assertEqual(latched[4], 1.0)
        self.assertEqual(released[4], 0.0)

    def test_fallback_brain_uses_the_same_steering_convention(self):
        car = spawn_car(self.track, Brain(), COLORS[0], "Fallback Agent")
        for _ in range(900):
            car.update(self.track)
        self.assertGreater(car.forward_distance_m, 500.0)
        self.assertGreater(car.fitness, 450.0)

    def test_legacy_starter_champions_are_migrated_on_load(self):
        legacy = (
            "# Formula AI Controller\n"
            "steer_strength = parameter(0.85, 0.20, 1.80)\n"
            "recovery_gain = parameter(0.75, 0.10, 1.50)\n"
            "left_space = left\n"
            "right_space = right\n"
            "open_side = left_space - right_space\n"
            "raw_steering = sign(open_side) * steer_strength "
            "- (heading_error * recovery_gain)\n"
            "steering = clamp(raw_steering, -1.0, 1.0)\n"
            "throttle = forward\n"
        )
        migrated = Brain.migrate_legacy_source(legacy)
        self.assertIn("open_side = right_space - left_space", migrated)
        self.assertIn(
            "(open_side * steer_strength) + (heading_error * recovery_gain)",
            migrated,
        )
        self.assertIn("pit_request =", migrated)

    def test_stalled_controller_cannot_win_on_start_position(self):
        stalled = self.simulate("steering = 1.0\nthrottle = 0.0\n", 900)
        moving = self.simulate(self.controller_source, 900)
        self.assertLess(stalled.fitness, 0.0)
        self.assertGreater(moving.fitness, stalled.fitness + 500.0)

    def test_old_controllers_default_to_no_pit_request(self):
        program = SafeAlgorithm("steering = 0.0\nthrottle = 1.0\n")
        self.assertEqual(
            program.run({}, program.defaults()),
            (0.0, 1.0, 0.0, 0.0, 0.0, 1),
        )

    def test_strategy_can_read_fuel_and_tyre_data(self):
        source = (
            "steering = 0.0\n"
            "throttle = fuel\n"
            "pit_request = 0.0\n"
            "pit_tyre = 1.0\n"
            "if tyre_wear >= 0.65 and fuel_kg > 10:\n"
            "    pit_request = 1.0\n"
            "    pit_tyre = 2.0\n"
        )
        program = SafeAlgorithm(source)
        decision = program.run(
            {"fuel": 0.4, "fuel_kg": 44, "tyre_wear": 0.7},
            program.defaults(),
        )
        self.assertEqual(decision, (0.0, 0.4, 0.0, 0.0, 1.0, 2))

    def test_car_publishes_normalized_race_state(self):
        car = spawn_car(self.track, Brain(), COLORS[0])
        car.tyre = "Soft"
        car.tyre_wear = 72.0
        car.tyre_laps = 4
        car.fuel = 44.0
        car.health = 81.0
        car.puncture = True
        car.generation = "Hybrid"
        car.battery = 63.0
        car.battery_regen = 0.0225
        car.overtake_active = True
        car.outside_limits = True
        car.car_collision = True
        car.understeer = 0.4
        car.oversteer = 0.25
        car.car_ahead = 1.0
        car.car_ahead_distance = 0.3
        car.car_ahead_side = -0.2
        car.closing_speed = 0.15
        car.passing = True
        car.passing_side = 1.0
        state = dict(zip(Brain.INPUT_NAMES, car.controller_inputs(self.track, 0.6)))
        self.assertAlmostEqual(state["tyre_wear"], 0.72)
        self.assertEqual(state["tyre_age"], 4.0)
        self.assertAlmostEqual(state["fuel"], 0.4)
        self.assertEqual(state["fuel_kg"], 44.0)
        self.assertAlmostEqual(state["health"], 0.81)
        self.assertEqual(state["puncture"], 1.0)
        self.assertEqual(state["rain"], 0.6)
        self.assertEqual(state["pit_available"], 1.0)
        self.assertEqual(state["tyre_soft"], 1.0)
        self.assertAlmostEqual(state["battery"], 0.63)
        self.assertEqual(state["battery_percent"], 63.0)
        self.assertAlmostEqual(state["regen"], 0.45)
        self.assertEqual(state["is_hybrid"], 1.0)
        self.assertEqual(state["overtake_active"], 1.0)
        self.assertEqual(state["off_track"], 1.0)
        self.assertEqual(state["car_collision"], 1.0)
        self.assertAlmostEqual(state["understeer"], 0.4)
        self.assertAlmostEqual(state["oversteer"], 0.25)
        self.assertEqual(state["car_ahead"], 1.0)
        self.assertAlmostEqual(state["car_ahead_distance"], 0.3)
        self.assertAlmostEqual(state["car_ahead_side"], -0.2)
        self.assertAlmostEqual(state["closing_speed"], 0.15)
        self.assertEqual(state["passing"], 1.0)
        self.assertEqual(state["passing_side"], 1.0)
        self.assertGreaterEqual(state["racing_line_offset"], -1.0)
        self.assertLessEqual(state["racing_line_offset"], 1.0)

    def test_algorithm_can_read_track_and_car_contact_inputs(self):
        program = SafeAlgorithm(
            "steering = car_collision\n"
            "throttle = 1.0 - off_track\n"
        )
        decision = program.run(
            {"off_track": 1.0, "car_collision": 1.0},
            program.defaults(),
        )
        self.assertEqual(decision, (1.0, 0.0, 0.0, 0.0, 0.0, 1))

    def test_pit_service_requires_controller_request_and_uses_requested_tyre(self):
        no_request = self.simulate("steering = 0.0\nthrottle = 0.0\n", 1)
        no_request.position = self.track.points[0].copy()
        no_request.tyre_wear = 70.0
        no_request.update(self.track)
        self.assertEqual(no_request.pit_timer, 0.0)

        strategy = (
            "steering = 0.0\n"
            "throttle = 0.0\n"
            "pit_request = tyre_wear >= 0.65\n"
            "pit_tyre = 2.0\n"
        )
        requested = self.simulate(strategy, 1)
        requested.position = self.track.points[0].copy()
        requested.tyre_wear = 70.0
        requested.update(self.track)
        self.assertGreater(requested.pit_timer, 0.0)
        for _ in range(120):
            requested.update(self.track)
        self.assertEqual(requested.pitstops, 1)
        self.assertEqual(requested.tyre, "Hard")
        self.assertEqual(requested.tyre_wear, 0.0)


if __name__ == "__main__":
    unittest.main()
