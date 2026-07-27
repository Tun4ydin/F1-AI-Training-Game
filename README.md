# Formula AI Lab

A playable Pygame prototype for building circuits, evolving racing agents, and
running AI races. It is intentionally data-driven: tracks and trained brains are
plain JSON files in `saved_data/`.

World geometry uses metres. The bundled Spa-Francorchamps reference is calibrated
to 7,004 m with an 8.5 m asphalt width. Cars use a 2.00 m maximum width, 3.40 m
wheelbase and 5.60 m overall icon/collision length. The editor grid is 100 m;
asphalt width and border distance are authored in metres. Training and races use
a zoomed follow-camera because a physically scaled F1 car is nearly invisible
in a whole-circuit view. The logical interface is smoothly rendered into a
resizable 1920×1080 display with aspect-correct letterboxing and mapped input.
ICE cars use an early-2000s silhouette with a slim nose, open cockpit and
grooved tyres; Hybrid cars use a 2022 silhouette with ground-effect bodywork,
larger covered wheels and a halo. Both are four-times antialiased vector
sprites with visible wings, suspension, cockpit/helmet, floor and livery
details. The 8× default follow camera is twice as close as the original view
without changing the cars' real dimensions or collision boxes.

## Run

Python 3.11–3.13 is recommended (Pygame may not yet publish wheels for newer
Python versions).

```bash
cd f1race
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python main.py
```

## Controls

- Menu: click a card or press `1`, `2`, `3`, or `4`
- Track editor: use tools `1`–`8` for route, kerb, sector, start/finish,
  pit entry, pit exit, pit road, and pit boxes. Pit Road remains locked until
  different Pit In and Pit Out route nodes exist. Its first authored node
  connects from the side of the Pit In node's asphalt facing the pit road;
  every later node extends the open road and reconnects to the facing asphalt
  edge at Pit Out. Node-specific main-road widths determine both connection
  points. Pit boxes can only be toggled on authored pit-road nodes.
  Cars on this six-metre-wide secondary asphalt are limited to 80 km/h.
  Mouse wheel zooms the large world canvas;
  middle-mouse drag pans. Hold the left mouse button on a route node and scroll
  to change that node's asphalt width in 0.5 m steps; widths interpolate between
  nodes and affect rendering, track limits, grip and AI sensors. `Backspace`
  undoes the route, `C` clears, `S` opens the named-track save dialog, and
  `Enter` applies the fitted circuit without exporting it. Variable-width
  asphalt, border and selective kerbs share a mitered ribbon mesh, preventing
  triangular gaps at imported or sharply changing nodes.
- Algorithm designer: write restricted Python-like controller code in the
  built-in editor. Click the Track selector to choose any saved circuit before
  training, and click Power to train an ICE or Hybrid field. `Ctrl+S` validates
  and opens the named-controller save dialog; `Ctrl+Enter` starts training on
  the selected circuit and powertrain.
  ICE and Hybrid maintain independent editor drafts and starter files:
  `ice_controller.fai` and `hybrid_controller.fai`. Switching Power changes
  both the editable template and its matching language reference. The Base Brain
  selector starts on Empty Brain, which uses the controller's declared parameter
  defaults. Selecting a saved brain seeds generation zero with its compatible
  learned parameter values; new or renamed parameters keep their current
  controller defaults. Click Racecraft to enable or disable racecraft training.
  Declare evolving values with `parameter(default, minimum, maximum)`.
  Controllers receive normalized tyre wear, remaining fuel, fuel kilograms,
  health, puncture, rain, slipstream strength, lap progress, pit availability,
  tyre-compound flags, normalized `battery`, `battery_percent`, regenerative
  braking rate `regen`, `is_hybrid`, `overtake_active`, `off_track`, and
  `car_collision`. The last two inputs are normalized contact-state flags:
  `1.0` means the car is fully outside the circuit or has just contacted
  another car. Racecraft also publishes normalized `understeer`, `oversteer`,
  signed `racing_line_offset`, `car_ahead`, `car_ahead_distance`,
  signed `car_ahead_side`, and `closing_speed`. In Racecraft training,
  `passing` and signed `passing_side` deliberately remain zero: they are
  compatibility assists for normal races, not answers supplied to training.
  The user's controller must decide when and where to pass. Additional dynamics
  inputs include normalized `local_velocity_forward`,
  `local_velocity_lateral`, `angular_velocity`, `traction`, `tire_slip`,
  `rpm`, and `gear`; raw display values are also available as `speed_kph`,
  `rpm_value`, and `gear_number`. Four extra side/near-forward rays extend the
  original five-ray set to nine rays. Normalized `waypoint_5_*`,
  `waypoint_10_*`, `waypoint_20_*`, and `waypoint_40_*` vectors describe the
  approaching centreline in the car's local frame. `opponent_1_*` through
  `opponent_3_*` publish the three nearest cars' local relative positions and
  velocities within 60 m, including cars beside or behind the agent.
  `opponent_1_present` through `opponent_3_present` distinguish a real car from
  an empty zero-padded slot. `previous_steering`, `previous_throttle`, and
  `previous_brake` allow user code to smooth its next action. Set optional
  `brake` and `overtake` outputs from `0` to `1`. Set `pit_request` to `1` to
  stop and select the next compound with
  `pit_tyre` (`0` Soft, `1` Medium, `2` Hard, `3` Wet).
  The editor supports mouse-drag and Shift selection, double-click word
  selection, `Ctrl/Cmd+A/C/X/V`, undo/redo, find, select-line, multiline
  Tab/Shift+Tab indentation, comment toggling, Home/End, Page Up/Down, and
  mouse-wheel scrolling.
- Training: click `-`/`+` or use the keyboard to change the live population,
  `Space` pauses, `R` evolves,
  `A` or Save Code exports the current algorithm, and `S` or Save Brain exports
  the current champion. The champion telemetry includes live green throttle and
  red brake percentage bars alongside speed, gear, and RPM. Fitness primarily
  measures real forward
  metres, while reversing, leaving the circuit, damage, and prolonged stalling
  reduce it; circling near the start cannot win a generation. Every training
  agent starts at the exact same position and remains a collision-free ghost.
  Training, hotlap, and race views include a fitted top-left circuit minimap.
  Cars use their assigned colors, overlapping agents form a compact dot
  cluster, and the currently followed car has a white outline.
  Exporting a champion opens a naming dialog before its brain JSON is written.
  With Racecraft enabled, agents instead start in a widely spaced staggered
  formation with one car per longitudinal slot, collide physically, and
  receive active slipstream/dirty-air behavior. The automatic passing planner
  is disabled in this mode, so fitness must select controllers that interpret
  the raw 360-degree opponent vectors themselves. Every contact adds a
  persistent fitness penalty. A clean move from fully behind to fully ahead
  awards `+150` fitness and increments the live `OVT` count; collision-assisted
  moves are rejected and each opponent has a cooldown to prevent reward
  farming. This lets a user train racecraft while the
  default mode remains deterministic ghost training. Health damage is disabled
  throughout training: wall and car impacts still affect motion and fitness,
  but cannot destroy an agent.
  High steering demand can create a mild, fast-recovering understeer state,
  while throttle and steering on reduced grip can create oversteer; both states
  are available to user code. A faster steering rack and stronger lateral tyre
  recovery let cars rotate into tight apexes sooner and settle more quickly on
  corner exit.
  Every car uses an automatic eight-speed gearbox with a 13,000 RPM redline.
  The champion card and population table show live speed, gear and RPM.
- Race: select an independent brain for each grid entry by clicking its AI Brain
  field, and click the Circuit selector to choose from every saved track. Cars
  use physical collision boxes and cannot ghost through each other.
  Starting a race runs a five-second, five-light sequence. Cars and the official
  session clock remain stopped until all red lights go out.
  Click a driver and then its Driver Name field to rename it, or double-click
  the driver directly in the grid. The rename field supports selection,
  `Ctrl/Cmd+A/C/X/V`, arrows, Home/End, Backspace/Delete, Enter to confirm, and
  Escape to cancel.
  Click a driver in the timing tower to follow it, `Space` pauses, left/right
  changes the timing metric—including Battery/Overtake and Speed/Gear/RPM
  telemetry—`R` saves the replay, and `Esc` returns.
- Training/race camera: `[` zooms out and `]` zooms in
- Hotlap: choose a saved or current-session brain and a saved circuit, then run
  exactly two timed laps with the selected ICE or Hybrid powertrain. `R`
  restarts, `Space` pauses, and the final panel reports both lap splits and the
  combined time.

The track, controller, and trained-brain save dialogs support text selection,
arrow/Home/End navigation, Backspace/Delete, and `Ctrl/Cmd+A/C/X/V`. Saving over
an existing component requires a second confirmation.

The bundled circuit is immediately usable. A custom circuit needs at least eight
points and is automatically closed. The first point is the start/finish line;
all other points also act as ordered mini-sector checkpoints.

Controller steering uses screen-space direction: `-1` turns left and `+1` turns
right. `heading_error` follows that same sign convention, so adding a positive
multiple of it steers the car back toward the racing-line tangent.

## SVG circuit import

Closed SVG paths containing move/line/cubic commands can be calibrated to a
known lap length:

```bash
python svg_track.py input.svg saved_data/tracks/circuit.json \
  --length 7004 --name "Circuit de Spa-Francorchamps" --path-id path2840
```

The bundled Spa geometry was derived from the supplied SVG by Will Pittenger,
licensed CC BY-SA 3.0 and described in its metadata as based on Google Earth
data.

## Current prototype scope

Implemented: spline-smoothed editable tracks, a zoomable/pannable world editor,
selective/manual apex kerbs, authored sectors/start/pit markers/border distance,
ordered anti-skip checkpoints, asphalt/kerb/grass traction, four-wheel
track-limit detection, grass slowdown and dirty tyres, ICE/hybrid power modes,
Hybrid 70% base power plus battery-backed 30% overtake deployment, regenerative
braking, fuel mass, tyre compounds/wear/punctures, wall and car collisions, safely
interpreted user-written controllers, mutation-based training for 1–50 cars,
champion import/export,
2–20 car races with per-car brain selection, physical oriented collision boxes,
three-car-length slipstream and dirty-air cornering loss, timing tower, camera
focus, laps, gaps, health, two-lap hotlaps, named component saves, saved-track
selectors for every event type, and results.

The ICE and Hybrid defaults are the Agile Waypoint v2 controllers. They combine
5/10/20/40 m waypoint anticipation with the nine-ray boundary fan, heading and
centreline correction, lateral-velocity/yaw damping, light understeer
compensation, oversteer counter-steering, previous-command smoothing, and
nearest-opponent closing speed. Their trainable parameters let evolution tune
turn-in, stability, braking and stint pace for custom circuits. The Hybrid
version adds predictive regenerative braking and deploys overtake power only
with suitable battery, grip, gear and corner load. Low-speed recovery always
releases the brake, preventing the first-corner stall.

Race Weekend includes a 2–20 car grid editor, editable driver/team names,
starting-grid reordering, generation, laps, dry/wet/changing weather, individual
colours, fuel and starting tyres. Rain changes dry/wet tyre behavior. The user
controller decides when to request a pit stop and which compound to fit; authored
pit boxes perform the requested service. Replay JSON contains car positions,
view focus, timing metric, weather and event history.
Replay car data also records powertrain, battery percentage, brake demand, and
overtake deployment, plus speed, gear and RPM.

## Native releases

Install `requirements-build.txt`, then run `python build_release.py`. PyInstaller
builds the native application for the current OS. The included GitHub Actions
workflow builds Windows, macOS, and Linux artifacts when a `v*` tag is pushed or
the workflow is started manually.

The 2027 championship calendar cannot be bundled accurately yet because its
official circuit calendar is future data. The track format supports adding
preloaded circuits later without changing the simulation.
