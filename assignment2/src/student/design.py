"""
Step 1 design choices (tracker pose, robot pose, calibration-body layout) and the
numbers that justify them. Kept in code so the notebook and the unit tests use the
same values.

Geometry (robot base coordinates, mm): the wrist centre reaches x in [500, 1100],
y in [-300, 300], z in [0, 600]; the end effector is 100 mm further and the calibration
markers up to 75 mm further still, so every marker of every reachable pose lies inside
the ENVELOPE: a cube of half-size 475 mm centred at (800, 0, 300).

Tracker: the viewing volume is a frustum along the camera's +z axis, 1000-2200 mm deep,
half-width 500 mm at the near face growing to 1000 mm at the far face. We aim the
camera at the envelope centre from a distance D = 1625 mm (camera above the robot,
looking down -z of the workspace). The envelope then spans depths 1150-2100 mm with
>= 87 mm of lateral margin at its near face and 100 mm of depth margin at the far
face, which covers the worst-case shift of the real robot base (0.05 rad x 1.57 m +
5 mm ~ 83 mm). The tracker has no jiggle, so this pose is exact.

Robot: F_rob_approx = identity (base at the workspace origin); the true base pose is
estimated as part of the calibration, so this choice is arbitrary.

Calibration body: 8 markers on the corners of a cube with circumradius 74.99 mm (the
largest allowed, 75 mm, minus a rounding margin). Symmetric, non-coplanar, maximum spread: the orientation noise of one
reading is sigma_b / sqrt(lambda_min) = 0.1 / sqrt(30000) = 5.8e-4 rad (see
_core.orientation_noise_per_reading); the position noise is 0.1 / sqrt(8) = 0.035 mm.
"""
from typing import List

import numpy as np

from data_types.nominal_types import vct3, Rot, Frame

ENVELOPE_CENTRE = np.array([800.0, 0.0, 300.0])
ENVELOPE_HALF = 475.0
TRACKER_DISTANCE = 1625.0
MBEE_RADIUS = 74.99      # the limit is ||a_k|| <= 75 mm; 74.99 keeps the corner coordinates (r/sqrt 3) inside it after rounding


def tracker_pose() -> Frame:
    """Camera at (800, 0, 300 + 1625) looking straight down (-z of the workspace)."""
    eye = ENVELOPE_CENTRE + np.array([0.0, 0.0, TRACKER_DISTANCE])
    R = np.array([[1.0, 0.0, 0.0],      # camera x = workspace x
                  [0.0, -1.0, 0.0],     # camera y = -workspace y
                  [0.0, 0.0, -1.0]])    # camera z (viewing axis) = -workspace z
    return Frame(Rot(matrix=R), vct3(eye))


def robot_pose() -> Frame:
    return Frame.eye()


def mbee_layout(radius: float = MBEE_RADIUS) -> List[vct3]:
    s = radius / np.sqrt(3.0)
    return [vct3(i * s, j * s, k * s) for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)]


def envelope_corners(F_rob: Frame = None) -> List[vct3]:
    """Corners of the marker envelope in the workspace (nominal robot pose)."""
    F_rob = F_rob if F_rob is not None else robot_pose()
    out = []
    for i in (-1, 1):
        for j in (-1, 1):
            for k in (-1, 1):
                out.append(F_rob * vct3(ENVELOPE_CENTRE + ENVELOPE_HALF * np.array([i, j, k])))
    return out
