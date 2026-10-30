# Real-Time Hand Gesture Recognition

A webcam pipeline that detects a hand, extracts 21 landmarks, and classifies
the gesture with a small neural network — the "Tony Stark control panel" idea
from `context.md`.

```
webcam -> OpenCV -> MediaPipe landmarks -> normalization -> MLP -> gesture label
```

## Gestures

1. `f_sign`
2. `v_sign`
3. `pen_hold`
4. `palm_hand`
5. `fist`

## Setup

```bash
uv sync
```

The hand-landmarker model lives at `models/hand_landmarker.task`.

## Workflow

### 1. Collect a dataset

```bash
uv run python src/data/collect.py
```

- Press `1`–`5` to record the matching gesture (a 3-second countdown plays,
  then a ~3-second burst is recorded and stops automatically).
- Press `space` to stop a burst early, `q` to quit.
- Each burst gets a `clip_id`; samples land in
  `src/data/sessions/<session>/<gesture>.csv`.
- Record the same gesture several times, with your hand in different
  positions, distances and lighting. Aim for **many short clips** rather
  than one long recording.

### 2. Train the classifier

```bash
uv run python src/mlp/train.py
```

This reads every CSV, splits **by whole clips** (so near-duplicate frames
never leak between train and validation), trains the MLP and writes
`models/gesture_mlp.pt` (weights + label order + normalization stats).
It prints validation accuracy and a confusion matrix.

To evaluate on a completely unseen recording day, set `TEST_SESSION` in
`src/mlp/train.py` to that session's folder name.

### 3. Run it live

```bash
uv run python src/camera.py
```

Shows the hand skeleton plus the smoothed predicted gesture for **each**
hand. Press `q` to quit, `a` to arm/disarm real actions. Without a trained
checkpoint it still runs, just without predictions.

## Gesture → action control layer

The camera loop feeds detections into `src/control/`, which turns gestures
into events and events into actions:

```
identity (A) → state (C/D) → bindings (D) → actions (E) → backend
```

- **`identity.py`** keeps stable slots for up to two hands, matching them by
  wrist position so a MediaPipe reorder does not swap the hands' histories.
- **`state.py`** holds one tracker *per hand*: majority-vote smoothing plus
  hysteresis (a gesture must persist to appear and persist-absent to drop),
  emitting `ACTIVATED` / `HELD` / `DEACTIVATED` events.
- **`bindings.py`** maps events to actions. Two modes:
  - `trigger` — fires once when the gesture appears (e.g. fist → press `D`);
  - `hold` — active the whole time (e.g. palm → move mouse, fist → hold LMB),
    with `release` called when it ends.
- **`actions.py`** describes what to do; **`backends.py`** does it
  (`null` = dry-run, `pynput` = X11, `ydotool` = Wayland).

Edits to bindings go in `src/control/config.toml` — no code changes needed.

### Chat macro (`send_random_line`)

Opens chat, pastes a random line from a CSV and sends it:

```toml
[[binding]]
gesture = "f_sign"
hand = "Right"
mode = "trigger"
  [binding.action]
  type = "send_random_line"
  csv_path = "src/control/test.csv"
  open_chat = "shift_enter"   # "shift_enter" = team/command chat | "enter" = normal
  send = "enter"              # "enter" | "shift_enter"
```

- `csv_path` — one phrase per line; commas inside a phrase are kept.
  Use `column = 0` if you want a specific column of a real CSV instead.
- `open_chat` — how to open the chat: `shift_enter` (all/team chat) or `enter`.
- `send` — how to submit after pasting.

Requires a clipboard tool (`wl-clipboard` on Wayland); see below.

### Safety

- `dry_run = true` in the config by default: actions are **logged, not
  executed**.
- The camera loop starts **disarmed**; press `a` to arm.
- On this machine the session is **Wayland**, where `pynput` is often
  ignored by the compositor. Use `ydotool` (needs its daemon) for real
  Wayland input.
- The chat macro needs a clipboard tool because `ydotool` can only type with
  a built-in US keymap (it would mangle Cyrillic / non-US layouts). Install
  one: `sudo pacman -S wl-clipboard`.

### Tests

```bash
uv run python tests/test_control.py
```

Covers identity, smoothing/hysteresis, trigger-vs-hold dispatch and the hand
filter — no webcam, torch or MediaPipe required.

## Project layout

| Path | Role |
|---|---|
| `src/hand_detection.py` | MediaPipe hand tracking (async) |
| `src/draw.py` | Skeleton / FPS / per-hand label overlays |
| `src/camera.py` | Live inference + control loop |
| `src/data/preprocess.py` | Feature extraction + queue helper (shared) |
| `src/data/data_collector.py` | Dataset recorder |
| `src/data/collect.py` | Recording UI |
| `src/mlp/model.py` | MLP architecture |
| `src/mlp/dataset.py` | CSV loading + grouped split |
| `src/mlp/train.py` | Training + evaluation |
| `src/mlp/classifier.py` | Stateless landmark → probabilities |
| `src/control/identity.py` | Stable hand slots |
| `src/control/state.py` | Smoothing, hysteresis, events |
| `src/control/pipeline.py` | Ties identity + state together |
| `src/control/bindings.py` | Event → action mapping |
| `src/control/actions.py` | Action descriptions |
| `src/control/backends.py` | Input backends (null/pynput/ydotool) |
| `src/control/config.toml` | Binding configuration |
| `tests/test_control.py` | Pure-logic tests |

## Design notes

- **Features** are translated to the wrist and scaled by hand size, so the
  model is robust to where the hand is and how far away it is.
- **One consumer** drains the MediaPipe result queue per frame
  (`drain_latest`), keeping only the freshest result and preventing the
  queue from growing.
- **Grouped splitting by `clip_id`** avoids the classic "98% validation,
  poor live accuracy" leakage trap.
- **Temporal smoothing** (majority vote + confidence threshold) stops the
  label from flickering.
