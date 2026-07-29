"""A tiny, deliberately restricted Python-like controller language.

This is an interpreter, not ``eval``/``exec``. User programs cannot import,
access attributes, index objects, open files, or call arbitrary functions.
"""
from __future__ import annotations

import ast
import math
import random
from dataclasses import dataclass


class AlgorithmError(ValueError):
    pass


@dataclass(frozen=True)
class Parameter:
    name: str
    default: float
    low: float
    high: float


def clamp(value, low, high):
    return max(low, min(high, value))


class SafeAlgorithm:
    INPUT_NAMES = {
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
    }
    OUTPUT_NAMES = {"steering", "throttle"}
    OPTIONAL_OUTPUTS = {
        "brake": 0.0,
        "overtake": 0.0,
        "recharge": 0.0,
        "pit_request": 0.0,
        "pit_tyre": 1.0,
    }
    SAFE_CALLS = {
        "abs": abs,
        "min": min,
        "max": max,
        "clamp": clamp,
        "sign": lambda value: (value > 0) - (value < 0),
        "sqrt": lambda value: math.sqrt(max(0, value)),
    }
    BIN_OPS = {
        ast.Add: lambda a, b: a + b,
        ast.Sub: lambda a, b: a - b,
        ast.Mult: lambda a, b: a * b,
        ast.Div: lambda a, b: a / b if abs(b) > 1e-9 else 0,
        ast.Mod: lambda a, b: a % b if abs(b) > 1e-9 else 0,
        ast.Pow: lambda a, b: a ** clamp(b, -4, 4),
    }
    CMP_OPS = {
        ast.Lt: lambda a, b: a < b,
        ast.LtE: lambda a, b: a <= b,
        ast.Gt: lambda a, b: a > b,
        ast.GtE: lambda a, b: a >= b,
        ast.Eq: lambda a, b: a == b,
        ast.NotEq: lambda a, b: a != b,
    }

    def __init__(self, source):
        self.source = source
        try:
            self.tree = ast.parse(source, mode="exec")
        except SyntaxError as exc:
            raise AlgorithmError(f"Line {exc.lineno}: {exc.msg}") from None
        if sum(1 for _ in ast.walk(self.tree)) > 4000:
            raise AlgorithmError(
                "Program is too large (maximum 4000 syntax nodes)"
            )
        self.parameters = self._validate()

    def _validate(self):
        parameters = {}
        assigned = set()

        def validate_expr(node):
            if isinstance(node, ast.Constant):
                if not isinstance(node.value, (int, float, bool)):
                    raise AlgorithmError(f"Line {node.lineno}: only numeric constants are allowed")
            elif isinstance(node, ast.Name):
                allowed = (
                    self.INPUT_NAMES
                    | set(self.OPTIONAL_OUTPUTS)
                    | set(parameters)
                    | assigned
                    | {"True", "False"}
                )
                if node.id not in allowed:
                    raise AlgorithmError(f"Line {node.lineno}: unknown variable '{node.id}'")
            elif isinstance(node, ast.BinOp) and type(node.op) in self.BIN_OPS:
                validate_expr(node.left)
                validate_expr(node.right)
            elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd, ast.Not)):
                validate_expr(node.operand)
            elif isinstance(node, ast.BoolOp) and isinstance(node.op, (ast.And, ast.Or)):
                for value in node.values:
                    validate_expr(value)
            elif isinstance(node, ast.Compare):
                validate_expr(node.left)
                for op, value in zip(node.ops, node.comparators):
                    if type(op) not in self.CMP_OPS:
                        raise AlgorithmError(f"Line {node.lineno}: comparison is not allowed")
                    validate_expr(value)
            elif isinstance(node, ast.IfExp):
                validate_expr(node.test)
                validate_expr(node.body)
                validate_expr(node.orelse)
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id not in self.SAFE_CALLS:
                    raise AlgorithmError(f"Line {node.lineno}: call to '{node.func.id}' is not allowed")
                if node.keywords:
                    raise AlgorithmError(f"Line {node.lineno}: keyword arguments are not allowed")
                for arg in node.args:
                    validate_expr(arg)
            else:
                raise AlgorithmError(f"Line {getattr(node, 'lineno', '?')}: unsafe or unsupported expression")

        def validate_statements(statements):
            for statement in statements:
                if isinstance(statement, ast.Assign) and len(statement.targets) == 1 and isinstance(statement.targets[0], ast.Name):
                    name = statement.targets[0].id
                    if name in self.INPUT_NAMES:
                        raise AlgorithmError(f"Line {statement.lineno}: sensor inputs are read-only")
                    if isinstance(statement.value, ast.Call) and isinstance(statement.value.func, ast.Name) and statement.value.func.id == "parameter":
                        if name in parameters or name in assigned:
                            raise AlgorithmError(f"Line {statement.lineno}: duplicate parameter '{name}'")
                        args = statement.value.args
                        if len(args) != 3 or not all(isinstance(arg, ast.Constant) and isinstance(arg.value, (int, float)) for arg in args):
                            raise AlgorithmError(f"Line {statement.lineno}: parameter(default, min, max) needs three numbers")
                        default, low, high = map(lambda arg: float(arg.value), args)
                        if low > default or default > high:
                            raise AlgorithmError(f"Line {statement.lineno}: parameter must satisfy min <= default <= max")
                        parameters[name] = Parameter(name, default, low, high)
                    else:
                        validate_expr(statement.value)
                        assigned.add(name)
                elif isinstance(statement, ast.If):
                    validate_expr(statement.test)
                    validate_statements(statement.body)
                    validate_statements(statement.orelse)
                elif isinstance(statement, ast.Pass):
                    continue
                else:
                    name = type(statement).__name__
                    raise AlgorithmError(f"Line {getattr(statement, 'lineno', '?')}: {name} is not allowed")

        validate_statements(self.tree.body)
        unconditional = {
            statement.targets[0].id
            for statement in self.tree.body
            if isinstance(statement, ast.Assign)
            and isinstance(statement.targets[0], ast.Name)
            and not (
                isinstance(statement.value, ast.Call)
                and isinstance(statement.value.func, ast.Name)
                and statement.value.func.id == "parameter"
            )
        }
        if not self.OUTPUT_NAMES.issubset(unconditional):
            missing = ", ".join(sorted(self.OUTPUT_NAMES - unconditional))
            raise AlgorithmError(f"Program must assign: {missing}")
        return parameters

    def defaults(self):
        return {name: parameter.default for name, parameter in self.parameters.items()}

    def mutate(self, values, amount=.16):
        result = dict(values)
        for name, parameter in self.parameters.items():
            current = result.get(name, parameter.default)
            span = parameter.high - parameter.low
            if random.random() < .55:
                current += random.gauss(0, span * amount)
            result[name] = clamp(current, parameter.low, parameter.high)
        return result

    def run(self, inputs, parameters):
        environment = {name: float(inputs.get(name, 0)) for name in self.INPUT_NAMES}
        environment.update(self.OPTIONAL_OUTPUTS)
        for name, spec in self.parameters.items():
            environment[name] = clamp(float(parameters.get(name, spec.default)), spec.low, spec.high)

        def evaluate(node):
            if isinstance(node, ast.Constant):
                return node.value
            if isinstance(node, ast.Name):
                return environment[node.id]
            if isinstance(node, ast.BinOp):
                return self.BIN_OPS[type(node.op)](evaluate(node.left), evaluate(node.right))
            if isinstance(node, ast.UnaryOp):
                value = evaluate(node.operand)
                return -value if isinstance(node.op, ast.USub) else (not value if isinstance(node.op, ast.Not) else value)
            if isinstance(node, ast.BoolOp):
                values = [bool(evaluate(value)) for value in node.values]
                return all(values) if isinstance(node.op, ast.And) else any(values)
            if isinstance(node, ast.Compare):
                left = evaluate(node.left)
                for op, comparator in zip(node.ops, node.comparators):
                    right = evaluate(comparator)
                    if not self.CMP_OPS[type(op)](left, right):
                        return False
                    left = right
                return True
            if isinstance(node, ast.IfExp):
                return evaluate(node.body if evaluate(node.test) else node.orelse)
            if isinstance(node, ast.Call):
                return self.SAFE_CALLS[node.func.id](*(evaluate(arg) for arg in node.args))
            raise AlgorithmError("Unsupported runtime expression")

        def execute(statements):
            for statement in statements:
                if isinstance(statement, ast.Assign):
                    name = statement.targets[0].id
                    if name in self.parameters:
                        continue
                    environment[name] = evaluate(statement.value)
                elif isinstance(statement, ast.If):
                    execute(statement.body if evaluate(statement.test) else statement.orelse)

        execute(self.tree.body)
        return (
            clamp(float(environment["steering"]), -1, 1),
            clamp(float(environment["throttle"]), 0, 1),
            clamp(float(environment["brake"]), 0, 1),
            clamp(float(environment["overtake"]), 0, 1),
            clamp(float(environment["recharge"]), 0, 1),
            clamp(float(environment["pit_request"]), 0, 1),
            int(round(clamp(float(environment["pit_tyre"]), 0, 3))),
        )
