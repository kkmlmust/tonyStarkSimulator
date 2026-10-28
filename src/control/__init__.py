"""Gesture → action control layer.

Pipeline::

    identity (A)  ->  state (C/D)  ->  bindings (D)  ->  actions (E)

* ``identity``  : stable slot ids for up to two hands.
* ``state``     : per-hand smoothing, hysteresis and gesture events.
* ``pipeline``  : ties A + C/D together for the camera loop.
* ``bindings``  : maps events to actions (trigger / hold).
* ``actions``   : descriptions of things to do.
* ``backends``  : how to actually do them (null / pynput / ydotool).
"""

from .bindings import Binding, ControlConfig, Dispatcher, Mode, load_config
from .identity import HandIdentity
from .pipeline import GesturePipeline, HandReading
from .state import Event, HandState, Phase

__all__ = [
    "Binding",
    "ControlConfig",
    "Dispatcher",
    "Mode",
    "load_config",
    "HandIdentity",
    "GesturePipeline",
    "HandReading",
    "Event",
    "HandState",
    "Phase",
]
