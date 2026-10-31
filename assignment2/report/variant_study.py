"""
Pose-plan / time trade-off study for the report: the calibration pipeline with deliberately
different plans, each on several robot seeds (default 10), forward error on the fixed test grid.
Kept out of the graded notebook to keep its run time short.

    cd assignment2 && python report/variant_study.py [--seeds 10]      -> results/variants.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src import simulation                                   # noqa: E402
from src.simulation import clock, constants                  # noqa: E402
from src.student import metrics, workflow, kinematics, design  # noqa: E402

TEST_LEVELS = {0: (100.0, 300.0, 500.0), 1: (-200.0, 0.0, 200.0), 2: (100.0, 300.0, 500.0),
               3: (-1.3, 0.0, 1.3), 4: (-1.3, 0.0, 1.3), 5: (-2.0, 0.0, 2.0)}
TEST_GRID = [np.array([x, y, z, a, b, c]) for x in TEST_LEVELS[0] for y in TEST_LEVELS[1] for z in TEST_LEVELS[2]
             for a in TEST_LEVELS[3] for b in TEST_LEVELS[4] for c in TEST_LEVELS[5]]

VARIANTS = [
    ("default: wrist ±1,±2; h 75; 5×7 readings", {}),
    ("1 command × 34 readings", {"commands_per_pose": 1}),
    ("wrist ±0.5,±1 rad", {"wrist_sweep": 1.0}),
    ("wrist ±0.75,±1.5 rad", {"wrist_sweep": 1.5}),
    ("wrist ±1.15,±2.3 rad", {"wrist_sweep": 2.3}),
    ("slide step 150 mm", {"slide_step": 150.0}),
    ("5×2 = 10 readings per pose", {"readings_per_pose": 10}),
    ("5×20 = 100 readings per pose", {"readings_per_pose": 100}),
    ("wrist polynomial degree 2", {"n_poly": (2, 2, 2, 2, 2, 2)}),
    ("slide polynomial degree 1", {"n_poly": (1, 1, 1, 1, 1, 1)}),
]


def run_once(seed, plan_kwargs):
    kw = dict(plan_kwargs)
    n_poly = kw.pop("n_poly", (2, 2, 2, 1, 1, 1))
    clock.reset()
    rob = simulation.create_robot("rob", design.robot_pose(), seed=seed)
    trk = simulation.create_tracker("trk", design.tracker_pose(), seed=seed)
    mbee = simulation.create_robot_calibration_body(rob, design.mbee_layout())
    simulation.add_marker_bodies(trk, [mbee])
    offsets = workflow.measure_home_offsets(rob, n_homings=5)
    poses, info = workflow.plan_calibration_poses(rob, trk, design.mbee_layout(), desired_accuracy=constants.TARGET_ANG_ERR_RAD,
                                                  return_info=True, **kw)
    data = workflow.collect_calibration_data(rob, trk, mbee, poses, repeats=info["readings_per_command"])
    imp = kinematics.ImprovedKinematics(rob, workflow.estimate_base_frame(data, rob), joint_offsets=offsets, n_poly=n_poly)
    imp.fit(data)
    Ft = trk.nominal_pose()
    s = metrics.summarize_errors([metrics.frame_error(Ft.inv() * rob.actual_ee_pose(th), imp.predict_in_tracker(th)) for th in TEST_GRID])
    return clock.elapsed, len(poses), info["readings_per_pose"], s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "results", "variants.json"))
    a = ap.parse_args()
    seeds = list(range(1, a.seeds + 1))
    t0 = time.time()
    out = []
    for name, kw in VARIANTS:
        rows = [run_once(s, kw) for s in seeds]
        t = np.mean([r[0] for r in rows])
        rp, ap_ = np.array([r[3]["rms_pos"] for r in rows]), np.array([r[3]["rms_angle"] for r in rows])
        mp, ma = np.array([r[3]["max_pos"] for r in rows]), np.array([r[3]["max_angle"] for r in rows])
        rec = {"name": name, "t_calibration": float(t), "n_commands": rows[0][1], "readings": rows[0][2], "n_seeds": len(seeds),
               "rms_pos": float(rp.mean()), "max_pos": float(mp.max()), "rms_angle": float(ap_.mean()), "max_angle": float(ma.max()),
               "rms_angle_sem": float(ap_.std(ddof=1) / np.sqrt(len(seeds))), "rms_pos_sem": float(rp.std(ddof=1) / np.sqrt(len(seeds))),
               "max_angle_mean": float(ma.mean()), "max_pos_mean": float(mp.mean()),
               "frac_seeds_in_spec": float(np.mean((mp <= constants.TARGET_POS_ERR_MM) & (ma <= constants.TARGET_ANG_ERR_RAD)))}
        out.append(rec)
        print(f"{name:40s} t {t:5.0f} s | pos rms {rec['rms_pos']:.3f} max {rec['max_pos']:.3f} | "
              f"ang rms {rec['rms_angle']:.2e} ± {rec['rms_angle_sem']:.1e}  max(mean/worst) {rec['max_angle_mean']:.2e}/{rec['max_angle']:.2e} | "
              f"seeds in spec {100 * rec['frac_seeds_in_spec']:.0f}%")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump({"seeds": seeds, "variants": out, "wall_seconds": time.time() - t0}, open(a.out, "w"), indent=1)
    print(f"wrote {a.out} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
