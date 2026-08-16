# Formula AI Lab — native C++ edition

This is a separate native C++/SDL2 + Dear ImGui port of the Pygame project. It does not
modify or embed Python. It opens the existing JSON tracks, brains, and replay
files from `../f1race/saved_data`, while all new saves go into this project's
own `saved_data` directory.

Implemented native workspaces:

- circuit browser/editor with variable node widths and JSON export;
- ICE/Hybrid algorithm editor with selection, system clipboard, undo/redo,
  independent drafts, and `.fai` export;
- population training with mutation, all-time champion retention, ghost or
  racecraft starts, telemetry, and brain JSON export;
- race weekend with 2–20 cars, collisions, drafting, Hybrid deployment/regen,
  five-light start, timing tower, minimap, and camera selection;
- two-lap hotlap timing;
- JSON replay browser/playback with pause, seek, rewind/fast-forward, and
  driver camera selection.

All interactive interface surfaces use Dear ImGui: workspace cards, saved-track
and brain selectors, ICE/Hybrid selection, the multiline algorithm editor,
track tools, replay picker, session controls, and live telemetry. SDL2 is kept
underneath for the circuit ribbon, selective kerbs, cars, and minimap. Dear
ImGui is vendored in `third_party/imgui`, so a build does not need network
access.

The native simulation now executes the same restricted Python-like `.fai`
language as the Pygame project. It supports trainable `parameter()` values,
assignments, nested `if`/`else`, arithmetic and boolean expressions, ternary
expressions, and the safe math helpers used by the bundled ICE and Hybrid
controllers. Saved source-based and legacy weight-based brain JSON files both
run natively.

The simulation uses world-space velocity and yaw rather than moving icons along
a route parameter. Track raycasts and look-ahead waypoints, three-car opponent
vectors, throttle/brake history, tyre grip and wear, fuel load, understeer,
oversteer, automatic eight-speed gearing, 13,000 RPM, Hybrid deployment and
regen, ICE DRS, pit-lane guidance and the 80 km/h limit all feed the controller.
Racecraft sessions add physical box collisions, damage in races (but not
training), drafting, passing awareness and overtake rewards. Race replays are
captured as Python-compatible JSON.

## Build

On this Mac, SDL2, CMake, and nlohmann-json are already installed:

```bash
cd f1race_cpp
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/formula_ai_lab_cpp
```

Run the non-graphical compatibility check with:

```bash
ctest --test-dir build --output-on-failure
```

## Controls

- Menu: `1`–`6` or click a workspace; `Esc` quits/returns.
- Any simulation: `Space` pause, Up/Down camera, `[`/`]` zoom.
- Track Studio: click empty space to add a node, left-drag an existing node to
  move it, hold the left button on a node and scroll to change its width,
  middle-drag to pan, and scroll normally to zoom. Right click removes a nearby
  node; `S` saves and `C` clears.
- Algorithm Lab: normal text input and mouse selection; `Ctrl/Cmd+A/C/X/V`,
  `Ctrl/Cmd+Z/Y`, `Ctrl/Cmd+S`; `Tab` inserts four spaces.
- Training: `R` evolves now, `S` saves the all-time champion.
- Race: `R`/`S` saves a replay, `Y` deploys yellow, `C` deploys the safety-car
  phase, `W` changes weather in a changing-weather event, and `P` punctures the
  focused car.
- Replay: `J/K/L` reverse/pause/forward and Left/Right seeks five seconds.

The SDL circuit canvas uses aspect-correct letterboxing independently from
Dear ImGui. ImGui always renders in real window coordinates, preventing the
double-scaling and clipped controls that occur on smaller or Retina displays.
