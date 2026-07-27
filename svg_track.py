"""Import a closed SVG cubic path as a metre-calibrated Formula AI track."""
from __future__ import annotations

import json
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path

NUMBER_OR_COMMAND = re.compile(r"[A-Za-z]|[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?")


def cubic(a, b, c, d, t):
    inverse = 1 - t
    return (
        inverse ** 3 * a[0] + 3 * inverse ** 2 * t * b[0] + 3 * inverse * t ** 2 * c[0] + t ** 3 * d[0],
        inverse ** 3 * a[1] + 3 * inverse ** 2 * t * b[1] + 3 * inverse * t ** 2 * c[1] + t ** 3 * d[1],
    )


def parse_closed_cubic(path_data, samples_per_curve=20):
    """Parse the M/m, C/c, L/l and Z/z subset used by circuit centre lines."""
    tokens = NUMBER_OR_COMMAND.findall(path_data.replace(",", " "))
    points = []
    cursor = (0.0, 0.0)
    start = None
    command = None
    index = 0

    def number():
        nonlocal index
        value = float(tokens[index])
        index += 1
        return value

    while index < len(tokens):
        if tokens[index].isalpha():
            command = tokens[index]
            index += 1
        if command in ("m", "M"):
            x, y = number(), number()
            cursor = (cursor[0] + x, cursor[1] + y) if command == "m" else (x, y)
            start = cursor
            points.append(cursor)
            command = "l" if command == "m" else "L"
        elif command in ("l", "L"):
            x, y = number(), number()
            cursor = (cursor[0] + x, cursor[1] + y) if command == "l" else (x, y)
            points.append(cursor)
        elif command in ("c", "C"):
            values = [number() for _ in range(6)]
            if command == "c":
                control1 = (cursor[0] + values[0], cursor[1] + values[1])
                control2 = (cursor[0] + values[2], cursor[1] + values[3])
                endpoint = (cursor[0] + values[4], cursor[1] + values[5])
            else:
                control1 = (values[0], values[1])
                control2 = (values[2], values[3])
                endpoint = (values[4], values[5])
            origin = cursor
            points.extend(cubic(origin, control1, control2, endpoint, step / samples_per_curve)
                          for step in range(1, samples_per_curve + 1))
            cursor = endpoint
        elif command in ("z", "Z"):
            if start and points[-1] != start:
                points.append(start)
            command = None
        else:
            raise ValueError(f"Unsupported SVG path command: {command}")
    if len(points) < 4:
        raise ValueError("SVG path does not contain a usable closed circuit")
    if points[0] == points[-1]:
        points.pop()
    return points


def distance(a, b):
    return math.hypot(b[0] - a[0], b[1] - a[1])


def length(points):
    return sum(distance(point, points[(i + 1) % len(points)]) for i, point in enumerate(points))


def resample_closed(points, count):
    perimeter = length(points)
    spacing = perimeter / count
    result = [points[0]]
    segment = 0
    current = points[0]
    remaining = spacing
    while len(result) < count:
        target = points[(segment + 1) % len(points)]
        available = distance(current, target)
        if available + 1e-9 >= remaining:
            ratio = remaining / max(available, 1e-9)
            current = (
                current[0] + (target[0] - current[0]) * ratio,
                current[1] + (target[1] - current[1]) * ratio,
            )
            result.append(current)
            remaining = spacing
        else:
            remaining -= available
            segment = (segment + 1) % len(points)
            current = target
    return result


def import_svg(svg_path, output_path, lap_length_m, name="Imported Circuit", path_id=None):
    root = ET.parse(svg_path).getroot()
    namespace = "{http://www.w3.org/2000/svg}"
    paths = list(root.iter(namespace + "path"))
    if path_id:
        element = next((path for path in paths if path.get("id") == path_id), None)
    else:
        candidates = [path for path in paths if "fill:none" in path.get("style", "") and "z" in path.get("d", "").lower()]
        element = max(candidates, key=lambda path: len(path.get("d", "")), default=None)
    if element is None:
        raise ValueError("No suitable closed circuit path was found in the SVG")

    raw = parse_closed_cubic(element.get("d"))
    source_length = length(raw)
    scaled = [(x * lap_length_m / source_length, y * lap_length_m / source_length) for x, y in raw]
    sampled = resample_closed(scaled, 360)
    correction = lap_length_m / length(sampled)
    sampled = [(x * correction, y * correction) for x, y in sampled]
    minimum_x = min(x for x, _ in sampled)
    minimum_y = min(y for _, y in sampled)
    sampled = [(x - minimum_x, y - minimum_y) for x, y in sampled]

    kerbs = []
    for i, point in enumerate(sampled):
        previous = sampled[i - 1]
        following = sampled[(i + 1) % len(sampled)]
        incoming = math.atan2(point[1] - previous[1], point[0] - previous[0])
        outgoing = math.atan2(following[1] - point[1], following[0] - point[0])
        turn = abs((outgoing - incoming + math.pi) % (2 * math.pi) - math.pi)
        if math.degrees(turn) >= 4.0:
            kerbs.append(i)

    data = {
        "name": name,
        "points": [[round(x, 4), round(y, 4)] for x, y in sampled],
        "kerb_points": kerbs,
        "geometry": "sampled",
        "declared_length_m": float(lap_length_m),
        "road_width_m": 12.0,
        "features": {
            "start_finish": 0,
            "sectors": [90, 180, 270],
            "pit_entry": 340,
            "pit_exit": 18,
            "pit_boxes": [0, 2, 4, 6, 8, 10, 12, 14, 16, 18],
            "border_margin": 30.0,
        },
        "source": {
            "file": Path(svg_path).name,
            "path_id": element.get("id"),
            "attribution": "Will Pittenger, CC BY-SA 3.0; based on Google Earth data",
        },
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, indent=2))
    return data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("svg")
    parser.add_argument("output")
    parser.add_argument("--length", type=float, required=True)
    parser.add_argument("--name", default="Imported Circuit")
    parser.add_argument("--path-id")
    args = parser.parse_args()
    imported = import_svg(args.svg, args.output, args.length, args.name, args.path_id)
    print(f"Imported {imported['name']}: {imported['declared_length_m']:.0f} m")
