"""Tests for the pure control logic (identity + state).

Run from the repository root:

    uv run python tests/test_control.py

No webcam, no torch, no MediaPipe -- just sequences of coordinates and
label probabilities, so the state machine is verifiable in isolation.
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from control.identity import HandIdentity  # noqa: E402
from control.state import HandState, Phase  # noqa: E402

LABELS = ["f_sign", "v_sign", "pen_hold", "palm_hand", "fist"]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
class FakeLandmark:
    def __init__(self, x, y):
        self.x = x
        self.y = y


def fake_hand(wrist_x, wrist_y=0.5):
    """A 21-landmark hand whose wrist sits at (wrist_x, wrist_y)."""
    hand = [FakeLandmark(wrist_x, wrist_y)]
    hand += [FakeLandmark(wrist_x + 0.01 * i, wrist_y) for i in range(1, 21)]
    return hand


def probs(label, confidence=0.95):
    import numpy as np

    p = np.full(len(LABELS), (1.0 - confidence) / (len(LABELS) - 1), dtype="float32")
    p[LABELS.index(label)] = confidence
    return p


class FakeCategory:
    def __init__(self, name):
        self.category_name = name


def handedness(*names):
    return [[FakeCategory(n)] for n in names]


def feed(state, label, frames=1):
    """Feed a label for N frames and collect all events."""
    events = []
    for _ in range(frames):
        events += state.update(probs(label))
    return events


def phases(events, gesture=None):
    return [
        (e.gesture, e.phase) for e in events if gesture is None or e.gesture == gesture
    ]


# --------------------------------------------------------------------------- #
# state tests
# --------------------------------------------------------------------------- #
def test_trigger_fires_once_while_held():
    state = HandState(slot=0, labels=LABELS, confirm_frames=3, release_frames=3)
    # Confirm the gesture, then keep holding it.
    events = feed(state, "fist", frames=3)
    assert phases(events) == [("fist", Phase.ACTIVATED)], phases(events)

    # Holding must emit HELD, not ACTIVATED, so a "trigger" binding fires once.
    events = feed(state, "fist", frames=5)
    assert all(p is Phase.HELD for _, p in phases(events)), phases(events)
    assert not any(p is Phase.ACTIVATED for _, p in phases(events))
    print("ok  trigger fires once while held")


def test_hysteresis_ignores_single_noisy_frame():
    state = HandState(slot=0, labels=LABELS, confirm_frames=3, release_frames=3)
    feed(state, "fist", frames=3)
    assert state.stable == "fist"

    # One frame of a different gesture must not change the stable state.
    state.update(probs("palm_hand"))
    assert state.stable == "fist", state.stable
    print("ok  single noisy frame ignored")


def test_release_requires_sustained_absence():
    # A larger window lets the vote genuinely fall below min_confidence:
    # alternating between two labels means neither reaches a majority.
    state = HandState(
        slot=0, labels=LABELS, smooth_window=4,
        min_confidence=0.6, confirm_frames=3, release_frames=3,
    )
    feed(state, "fist", frames=5)
    assert state.stable == "fist"

    # Alternate two other gestures: no single class reaches 60% of the window.
    frames_until_release = 0
    released = False
    for i in range(20):
        state.update(probs("palm_hand" if i % 2 == 0 else "pen_hold"))
        frames_until_release += 1
        if state.stable is None:
            released = True
            break

    assert released, "never released after sustained low confidence"
    assert frames_until_release >= state.release_frames, frames_until_release
    assert state.stable is None
    print("ok  release requires sustained absence")


def test_switch_emits_deactivate_then_activate():
    state = HandState(
        slot=0, labels=LABELS, smooth_window=1,
        min_confidence=0.6, confirm_frames=2, release_frames=5,
    )
    feed(state, "fist", frames=2)
    events = feed(state, "palm_hand", frames=2)
    got = phases(events)
    assert ("fist", Phase.DEACTIVATED) in got, got
    assert ("palm_hand", Phase.ACTIVATED) in got, got
    assert state.stable == "palm_hand"
    print("ok  switching gesture emits deactivate then activate")


def test_pipeline_keeps_hands_independent():
    from control.pipeline import GesturePipeline

    class FakeClassifier:
        labels = LABELS

        def predict(self, hand):
            # The hand's x position encodes which gesture it is "making".
            x = hand[0].x
            return probs("fist" if x < 0.5 else "palm_hand")

    class FakeResult:
        def __init__(self, hands, hn):
            self.hand_landmarks = hands
            self.handedness = handedness(*hn)

    pipeline = GesturePipeline(labels=LABELS, smooth_window=1, confirm_frames=2)
    clf = FakeClassifier()

    left = fake_hand(0.2)
    right = fake_hand(0.8)

    for _ in range(3):
        readings = pipeline.update(FakeResult([left, right], ["Right", "Left"]), clf)

    by_slot = {r.slot: r.gesture for r in readings}
    assert by_slot[0] == "fist", by_slot
    assert by_slot[1] == "palm_hand", by_slot
    print("ok  two hands classified independently")


def test_dispatcher_trigger_fires_once_and_hold_releases():
    from control.actions import Action
    from control.backends import NullBackend
    from control.bindings import Binding, Dispatcher, Mode
    from control.state import Event, Phase

    calls = []

    class Spy(Action):
        def __init__(self, name):
            self.name = name

        def perform(self, ctx, backend):
            calls.append(("perform", self.name))

        def release(self, ctx, backend):
            calls.append(("release", self.name))

    class RightReading:
        slot = 0
        hand = "Right"
        landmarks = fake_hand(0.5)

    class LeftReading:
        slot = 1
        hand = "Left"
        landmarks = fake_hand(0.2)

    # Built inline (not from config.toml) so editing the config never breaks
    # the tests.
    bindings = [
        Binding("fist", Mode.TRIGGER, Spy("right_fist"), hand="Right"),
        Binding("fist", Mode.HOLD, Spy("left_fist"), hand="Left"),
    ]
    dispatcher = Dispatcher(bindings=bindings, backend=NullBackend())

    # --- Right hand: "fist" is a trigger binding ---------------------------
    # HELD frames must be ignored so the key is pressed only once.
    dispatcher.handle([Event(0, "Right", "fist", Phase.HELD)], RightReading(), (640, 480))
    assert calls == [], calls

    dispatcher.handle([Event(0, "Right", "fist", Phase.ACTIVATED)], RightReading(), (640, 480))
    assert calls == [("perform", "right_fist")], calls

    # A trigger binding must not call release on DEACTIVATED.
    dispatcher.handle([Event(0, "Right", "fist", Phase.DEACTIVATED)], RightReading(), (640, 480))
    assert calls == [("perform", "right_fist")], calls

    # --- Left hand: "fist" is a hold binding -------------------------------
    calls.clear()
    dispatcher.handle([Event(1, "Left", "fist", Phase.ACTIVATED)], LeftReading(), (640, 480))
    dispatcher.handle([Event(1, "Left", "fist", Phase.HELD)], LeftReading(), (640, 480))
    assert calls == [("perform", "left_fist"), ("perform", "left_fist")], calls

    calls.clear()
    dispatcher.handle([Event(1, "Left", "fist", Phase.DEACTIVATED)], LeftReading(), (640, 480))
    assert calls == [("release", "left_fist")], calls
    print("ok  dispatcher: trigger fires once, hold releases, hand filter works")


def test_pipeline_releases_hold_when_hand_disappears():
    from control.pipeline import GesturePipeline
    from control.state import Event, Phase

    class FakeClassifier:
        labels = LABELS

        def predict(self, hand):
            return probs("fist")

    class FakeResult:
        def __init__(self, hands, hn):
            self.hand_landmarks = hands
            self.handedness = handedness(*hn)

    pipeline = GesturePipeline(labels=LABELS, smooth_window=1, confirm_frames=1)
    clf = FakeClassifier()
    hand = fake_hand(0.5)

    pipeline.update(FakeResult([hand], ["Right"]), clf)
    # Hand leaves the frame: we must get a DEACTIVATED event so a `hold`
    # binding's action is released (otherwise the mouse button stays down).
    readings = pipeline.update(FakeResult([], []), clf)
    events = [e for r in readings for e in r.events]
    assert any(e.phase is Phase.DEACTIVATED and e.gesture == "fist" for e in events), events

    # A second missing frame must not repeat the release.
    assert pipeline.update(FakeResult([], []), clf) == []
    print("ok  disappearing hand releases held gestures")


def test_hold_key_presses_once_and_releases():
    from control.actions import ActionContext, build_action
    from control.backends import NullBackend
    from control.bindings import Mode

    log = []

    class RecordingBackend(NullBackend):
        def key_down(self, key):
            log.append(("down", key))

        def key_up(self, key):
            log.append(("up", key))

        def tap_key(self, key):
            log.append(("tap", key))

    ctx = ActionContext(landmarks=fake_hand(0.5), frame_size=(640, 480),
                        gesture="f_sign", hand="Right", slot=0)
    backend = RecordingBackend()

    # A hold-mode key action must press once, ignore repeats, then release.
    action = build_action({"type": "key", "key": "g"}, mode=Mode.HOLD)
    action.perform(ctx, backend)
    action.perform(ctx, backend)  # repeated HELD frames must not re-press
    action.release(ctx, backend)
    assert log == [("down", "g"), ("up", "g")], log

    # A trigger-mode key action taps once and never releases.
    log.clear()
    action = build_action({"type": "key", "key": "g"}, mode=Mode.TRIGGER)
    action.perform(ctx, backend)
    assert log == [("tap", "g")], log
    print("ok  hold key presses once and releases; trigger taps")


def test_mouse_move_emits_relative_deltas():
    from control.actions import ActionContext, MouseMoveAction
    from control.backends import NullBackend

    moves = []

    class RecordingBackend(NullBackend):
        def move_mouse_relative(self, dx, dy):
            moves.append((round(dx), round(dy)))

    def ctx_at(tip_x):
        hand = fake_hand(0.5)
        hand[8] = type(hand[8])(tip_x, 0.5)
        return ActionContext(landmarks=hand, frame_size=(640, 480),
                             gesture="pen_hold", hand="Left", slot=0)

    action = MouseMoveAction(sensitivity=1000.0, deadzone=0.0, smoothing=0.0)

    # First frame only establishes the reference point: no movement yet.
    action.perform(ctx_at(0.5), RecordingBackend())
    assert moves == [], moves

    # Moving the fingertip right must produce a positive dx.
    action.perform(ctx_at(0.6), RecordingBackend())
    assert moves and moves[-1][0] > 0, moves

    # A movement inside the deadzone must be ignored.
    moves.clear()
    quiet = MouseMoveAction(sensitivity=1000.0, deadzone=0.05, smoothing=0.0)
    quiet.perform(ctx_at(0.5), RecordingBackend())
    quiet.perform(ctx_at(0.51), RecordingBackend())
    assert moves == [], moves

    # The ydotool backend must refuse absolute positioning.
    from control.backends import YdotoolBackend
    try:
        YdotoolBackend(probe=False).move_mouse(10, 10)
    except NotImplementedError:
        pass
    else:
        raise AssertionError("ydotool absolute movement should be unsupported")
    print("ok  mouse move emits relative deltas with deadzone")


def test_config_bindings_load_as_requested():
    from control.bindings import Mode, load_config

    config = load_config(SRC_DIR / "control" / "config.toml")
    pen = [b for b in config.bindings if b.gesture == "pen_hold"]
    assert len(pen) == 1 and pen[0].hand == "Left" and pen[0].mode is Mode.HOLD, pen
    assert type(pen[0].action).__name__ == "MouseMoveAction"

    f = [b for b in config.bindings if b.gesture == "f_sign" and b.hand == "Right"]
    assert len(f) == 1 and f[0].mode is Mode.TRIGGER, f
    action = f[0].action
    assert type(action).__name__ == "SendRandomLineAction", action
    assert action.open_chat == "shift_enter" and action.send == "enter", action
    print("ok  config: left pen_hold -> mouse move, right f_sign -> chat macro")


def test_scroll_action_and_ydotool_keycodes():
    from control.actions import ActionContext, build_action
    from control.backends import NullBackend, _keycode

    # Pure keycode mapping (ydotool 1.x uses evdev codes, not names).
    assert _keycode("g") == 34
    assert _keycode("space") == 57
    assert _keycode("34") == 34

    logged = []

    class RecordingBackend(NullBackend):
        def scroll(self, amount):
            logged.append(amount)

    ctx = ActionContext(landmarks=fake_hand(0.5), frame_size=(640, 480),
                        gesture="v_sign", hand="Right", slot=0)
    action = build_action({"type": "scroll", "amount": 2}, mode="hold")
    action.perform(ctx, RecordingBackend())
    assert logged == [2], logged

    # Unknown key names must fail loudly rather than silently doing nothing.
    try:
        _keycode("ctrl+c")
    except ValueError:
        pass
    else:
        raise AssertionError("multi-key strings should be rejected")
    print("ok  scroll action + ydotool keycode mapping")


def test_send_random_line_macro_sequence():
    from control.actions import ActionContext, build_action
    from control.backends import NullBackend

    events = []

    class RecordingBackend(NullBackend):
        def tap_combo(self, keys):
            events.append(("combo", "+".join(keys)))

        def set_clipboard(self, text):
            events.append(("clipboard", text))

    ctx = ActionContext(landmarks=fake_hand(0.5), frame_size=(640, 480),
                        gesture="f_sign", hand="Right", slot=0)
    action = build_action({
        "type": "send_random_line",
        "csv_path": "src/control/test.csv",
        "open_chat": "shift_enter",
        "send": "enter",
    })
    # No real delays in tests.
    action.pre_delay = 0.0
    action.post_delay = 0.0

    action.perform(ctx, RecordingBackend())

    kinds = [e[0] for e in events]
    assert kinds == ["combo", "clipboard", "combo", "combo"], events
    assert events[0] == ("combo", "leftshift+enter"), events[0]  # open chat
    assert events[2] == ("combo", "leftctrl+v"), events[2]  # paste
    assert events[3] == ("combo", "enter"), events[3]  # send

    # The pasted text must actually be one of the CSV lines.
    from control.macros import load_lines

    assert events[1][1] in load_lines("src/control/test.csv"), events[1]

    # "enter" (normal chat) must send with a bare Enter.
    events.clear()
    action2 = build_action({
        "type": "send_random_line",
        "csv_path": "src/control/test.csv",
        "open_chat": "enter",
        "send": "enter",
    })
    action2.pre_delay = 0.0
    action2.post_delay = 0.0
    action2.perform(ctx, RecordingBackend())
    assert events[0] == ("combo", "enter"), events[0]
    print("ok  send_random_line macro: open -> clipboard -> paste -> send")


# --------------------------------------------------------------------------- #
# identity tests
# --------------------------------------------------------------------------- #
def test_two_hands_keep_stable_slots_when_reordered():
    # MediaPipe may return hands in a different order between frames; the
    # slots must follow the physical hands (near their old positions).
    identity = HandIdentity()
    left = fake_hand(0.2)
    right = fake_hand(0.8)

    first = identity.update([left, right], handedness("Right", "Left"))
    slots_first = {id(h): s for s, h, _ in first}
    assert slots_first[id(left)] == 0 and slots_first[id(right)] == 1

    # Same hands, reversed list order.
    second = identity.update([right, left], handedness("Left", "Right"))
    slots_second = {id(h): s for s, h, _ in second}
    assert slots_second[id(left)] == 0, "left hand changed slot"
    assert slots_second[id(right)] == 1, "right hand changed slot"
    print("ok  slots follow hands when MediaPipe reorders them")


def test_new_hand_gets_new_slot():
    identity = HandIdentity()
    identity.update([fake_hand(0.2)], handedness("Right"))
    result = identity.update([fake_hand(0.2), fake_hand(0.8)], handedness("Right", "Left"))
    slots = sorted(s for s, _, _ in result)
    assert slots == [0, 1], slots
    print("ok  a second hand gets its own slot")


def test_slot_expires_after_grace():
    identity = HandIdentity(grace_frames=3)
    identity.update([fake_hand(0.2)], handedness("Right"))
    for _ in range(4):
        identity.update([], handedness())
    # A hand back at the same place starts a fresh slot because the old one
    # expired.
    result = identity.update([fake_hand(0.2)], handedness("Right"))
    assert result[0][0] == 1, result
    print("ok  slot expires after grace period")


def test_layout_switcher_parses_and_switches():
    from control.layout import LayoutSwitcher, parse_layouts

    # The exact shape qdbus6 --literal prints for the a(sss) layout list.
    raw = (
        '[Argument: a(sss) {[Argument: (sss) "us", "", "English (US)"], '
        '[Argument: (sss) "ru", "", "Russian"]}]'
    )
    rows = parse_layouts(raw)
    assert rows == [("us", "", "English (US)"), ("ru", "", "Russian")], rows

    # Unparseable output must not crash -- just yields nothing.
    assert parse_layouts("garbage") == []
    assert parse_layouts("") == []

    # A missing qdbus client disables the switcher instead of raising.
    sw = LayoutSwitcher(target="us", binary="qdbus-does-not-exist")
    assert sw.enabled is False and sw.index is None
    assert sw.activate() is None  # no-op
    print("ok  layout switcher parses layout lists and degrades gracefully")


def test_ydotool_forces_latin_layout_for_keys():
    from control.backends import YdotoolBackend

    class FakeSwitcher:
        enabled = True
        index = 0

        def __init__(self):
            self.calls = []

        def activate(self):
            self.calls.append("activate")
            return 1  # previous layout index (ru)

        def restore(self, saved):
            self.calls.append(("restore", saved))

    backend = YdotoolBackend(probe=False, layout=None)  # no live D-Bus probe
    fake = FakeSwitcher()
    backend._switcher = fake

    sent = []
    backend._run = lambda *args: sent.append(args)  # don't touch the daemon

    # A tap switches to Latin, sends the key, then restores the old layout.
    backend.tap_key("g")
    assert sent == [("key", "34:1", "34:0")], sent
    assert fake.calls == ["activate", ("restore", 1)], fake.calls

    # A held key must keep Latin active until it is released.
    fake.calls.clear()
    backend.key_down("g")
    assert fake.calls == ["activate"], fake.calls
    backend.key_up("g")
    assert fake.calls == ["activate", ("restore", 1)], fake.calls

    # A combo (Ctrl+V) is one Latin-wrapped emission.
    fake.calls.clear()
    sent.clear()
    backend.tap_combo(["leftctrl", "v"])
    assert sent == [("key", "29:1", "47:1", "47:0", "29:0")], sent
    assert fake.calls == ["activate", ("restore", 1)], fake.calls
    print("ok  ydotool wraps key events in a Latin-layout switch")


def main() -> None:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
    print(f"\n{len(tests)} tests passed")


if __name__ == "__main__":
    main()
