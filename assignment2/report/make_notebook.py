"""Build notebooks/assignment2_solution.ipynb from the course template by filling in the TODO cells."""
import json
import sys
from pathlib import Path

TEMPLATE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/mnt/user-data/uploads/assignment2/notebooks/assignment2_template.ipynb")
OUT = Path(__file__).resolve().parents[1] / "notebooks" / "assignment2_solution.ipynb"

nb = json.load(open(TEMPLATE))
cells = nb["cells"]


def set_src(i, text):
    cells[i]["source"] = text.strip("\n").splitlines(keepends=True)
    if cells[i]["cell_type"] == "code":
        cells[i]["outputs"] = []
        cells[i]["execution_count"] = None


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True), "outputs": [], "execution_count": None}


# ---------------------------------------------------------------- cell 1: imports
set_src(1, '''
import sys, os, time, json
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "..")))

import numpy as np
from src import simulation, vct3, Rot, Frame
from src.simulation import clock, constants
from src.student import metrics, registration, workflow, kinematics, design

np.set_printoptions(precision=4, suppress=True)
clock.reset()
WALL_START = time.time()        # real run time must stay under 10 minutes when graded
''')

# ---------------------------------------------------------------- cell 3: Step 1 design
set_src(3, '''
# Our Step-1 design lives in src/student/design.py (the unit tests use the same values):
#  * tracker 1625 mm above the centre (800, 0, 300) of the 950 mm "marker envelope", looking
#    straight down: the whole envelope lies at depths 1150-2100 mm with >= 87 mm lateral and
#    100 mm depth margin, more than the ~83 mm worst-case shift of the real robot base;
#  * robot base at the workspace origin (its true pose is estimated anyway);
#  * calibration body: 8 markers on the corners of a 75 mm-circumradius cube -- the largest
#    allowed spread, symmetric and non-coplanar: 5.8e-4 rad / 0.035 mm per single reading.
F_tracker_approx = design.tracker_pose()   # Frame, tracker pose relative to the workspace (the camera looks along its local +z)
F_rob_approx = design.robot_pose()         # Frame, robot base pose relative to the workspace
mbee_local_positions = design.mbee_layout()  # 8 points, every ||a*_k|| = 75 mm

print("tracker pose:\\n", F_tracker_approx.R.matrix, F_tracker_approx.p)
print("calibration body:", [np.round(a.vec.ravel(), 1).tolist() for a in mbee_local_positions])
''')

# ---------------------------------------------------------------- cell 7: visibility check (add margin check)
set_src(7, '''
def mbee_markers_world(theta):
    F_ee_world = rob.F_rob * rob.fwd_kins(theta)
    return [F_ee_world * a for a in mbee_local_positions]

n_vis, n_tot = 0, 0
for x in (0, 300, 600):
    for y in (-300, 0, 300):
        for z in (0, 300, 600):
            for w in (-1.8, 0.0, 1.8):
                theta = np.array([x, y, z, w, -w, w])
                n_tot += 1
                n_vis += trk.is_visible(mbee_markers_world(theta))
print(f"calibration body visible at {n_vis}/{n_tot} corner/edge poses of the work volume")

# stronger check: the whole marker envelope (every marker of every reachable pose), shifted by the
# worst-case base placement error (0.05 rad x 1.57 m + 5 mm ~ 83 mm) in every direction
corners = design.envelope_corners(F_rob_approx)
shifts = [vct3(0, 0, 0)] + [vct3(*(85.0 * np.eye(3)[i] * s)) for i in range(3) for s in (1, -1)]
print("envelope visible under all worst-case base shifts:", all(trk.is_visible([c + d for c in corners]) for d in shifts))
''')

# ---------------------------------------------------------------- cell 15: Step 6 baseline (print with our summary)
set_src(15, '''
home_offsets = workflow.measure_home_offsets(rob, n_homings=5)   # 5 homings: random part 0.005 mm / 1e-4 rad -> /sqrt(5)
print("delta_theta_sys at home (from the encoders):", home_offsets)

center = np.array([300.0, 0.0, 300.0, 0.0, 0.0, 0.0])
baseline_poses = [center] + [center + np.eye(6)[k] * v for k in range(3) for v in (-250.0, 250.0)] \\
                 + [center + np.eye(6)[3] * 0.8, center + np.eye(6)[4] * 0.8, center + np.eye(6)[5] * 1.5]

baseline = workflow.collect_calibration_data(rob, trk, mbee_placed, baseline_poses, repeats=1)
F_X_baseline = workflow.estimate_base_frame(baseline, rob)
residuals = [metrics.frame_error(F_obs, F_X_baseline * rob.fwd_kins(th)) for th, F_obs in zip(baseline.thetas, baseline.frames)]
print(f"{len(baseline)} baseline samples in {baseline.t_collect:.1f}s")
print("nominal kinematics vs observations:", metrics.format_summary(metrics.summarize_errors(residuals)))
''')

# ---------------------------------------------------------------- cell 17: Step 7 plan + collect
set_src(17, '''
DESIRED_ACCURACY = constants.TARGET_ANG_ERR_RAD  # rad; the orientation target is the limiting one

# Method 1: home, then sweep each joint alone (9 stops per slide, +-1 / +-2 rad per wrist joint).
# The plan sizes itself from DESIRED_ACCURACY: 34 tracker readings per pose give tracker noise
# 1/10 of the orientation budget, taken as 5 re-commands x 7 readings (re-commanding costs only
# t_settle and re-rolls the joint repeatability noise); see workflow.plan_calibration_poses.
poses, plan_info = workflow.plan_calibration_poses(rob, trk, mbee_local_positions,
                                                   desired_accuracy=DESIRED_ACCURACY, return_info=True)
n_distinct = len({tuple(np.round(th, 6)) for th in poses})
print(f"plan: {len(poses)} commands over {n_distinct} distinct poses, {plan_info['commands_per_pose']} commands x "
      f"{plan_info['readings_per_command']} readings = {plan_info['readings_per_pose']} readings per pose")
print(f"      slide step {plan_info['slide_steps_mm'][0]:.0f} mm, wrist stops {plan_info['wrist_stops_rad']} rad, "
      f"predicted {plan_info['predicted_time']['total_s']:.0f} s "
      f"(motion {plan_info['predicted_time']['motion_s']:.0f} s + sampling {plan_info['predicted_time']['sampling_s']:.0f} s)")
print(f"      expected orientation noise per pose: {plan_info['expected_orientation_noise_rad']:.1e} rad")

t0 = clock.elapsed
sweeps = workflow.collect_calibration_data(rob, trk, mbee_placed, poses, repeats=plan_info["readings_per_command"])
print(f"{len(sweeps)} samples ({sum(sweeps.readings)} readings) collected in {sweeps.t_collect:.1f}s "
      f"(predicted {plan_info['predicted_time']['total_s']:.0f}s); clock now {clock.elapsed:.1f}s")

# the Step-6 baseline poses were already paid for: use them in the fit as well
data = workflow.merge_data(baseline, sweeps)
F_X_nom = workflow.estimate_base_frame(data, rob)
''')

# ---------------------------------------------------------------- cell 19: Step 8 fit
set_src(19, '''
# Parameterization: base twist (6) + axis tilts (2 x 5) + tool twist (6) + distortion polynomials
# (degree 2 on the slides, 1 on the wrist: the wrist ripple has slope <= 5e-5 rad/rad, a quadratic
# only fits noise). Joint offsets come from HomeRobot() and are not fitted. The base rotation
# about d_1 is frozen (fix_gauge=True): it is the model's one unobservable direction.
improved = kinematics.ImprovedKinematics(rob, F_X_nom, joint_offsets=home_offsets, n_poly=(2, 2, 2, 1, 1, 1))
fit_info = improved.fit(data)
print("fit:", {k: (round(v, 6) if isinstance(v, float) else v) for k, v in fit_info.items()})

rank = improved.jacobian_rank_report(data)
print("jacobian rank, all parameters:", rank["rank"], "/", rank["n_params"],
      "| weakest direction:", rank["weakest_directions"][0]["direction"])
print("jacobian rank, fitted parameters:", rank["free"]["rank"], "/", rank["free"]["n_params"],
      f"| condition number {rank['free']['condition_number']:.1f}")

t_calibration = clock.elapsed
print(f"total simulated calibration time: {t_calibration:.1f}s (soft budget {constants.TARGET_TIME_SEC:.0f}s)")

estimated, truth = improved.parameters_report(), rob.ground_truth_parameters()   # truth: debugging only
cmp = improved.compare_with_ground_truth(truth, trk.nominal_pose())
print("estimated axis angles (deg):", estimated["axis_angles_deg"], estimated["axis_angle_pairs"])
print("   true axis angles (deg):", cmp["axis_angles_deg_true"], "| error (mrad):", np.radians(cmp["axis_angles_deg_err"]) * 1e3)
print("estimated joint offsets:", estimated["joint_offset"], "| true (at home):", cmp["joint_offset_true_at_home"])
print("estimated joint scales :", estimated["joint_scale"], "| true:", truth["joint_scale"])
print("axis direction errors after removing the base gauge (mrad):", cmp["axis_direction_err_mrad"])
print("tool frame error (rad, mm):", cmp["tool_frame_err_rad_mm"], "| base frame error (rad, mm):", cmp["base_frame_err_rad_mm"])
''')

# ---------------------------------------------------------------- cell 21: Step 9 forward evaluation
set_src(21, '''
test_levels = {0: (100.0, 300.0, 500.0), 1: (-200.0, 0.0, 200.0), 2: (100.0, 300.0, 500.0),
               3: (-1.3, 0.0, 1.3), 4: (-1.3, 0.0, 1.3), 5: (-2.0, 0.0, 2.0)}
test_grid = [np.array([x, y, z, a, b, c])
             for x in test_levels[0] for y in test_levels[1] for z in test_levels[2]
             for a in test_levels[3] for b in test_levels[4] for c in test_levels[5]]
test_grid = [th for th in test_grid if trk.is_visible(mbee_markers_world(th))]
print(f"{len(test_grid)} visible test poses")

F_trk = trk.nominal_pose()

fwd_err_nominal, fwd_err_improved = [], []
for theta in test_grid:
    truth_pose = F_trk.inv() * rob.actual_ee_pose(theta)            # ground truth, reporting only
    fwd_err_nominal.append(metrics.frame_error(truth_pose, F_X_nom * rob.fwd_kins(theta)))
    fwd_err_improved.append(metrics.frame_error(truth_pose, improved.predict_in_tracker(theta)))

print("ImprovedFwdKins :", metrics.format_summary(metrics.summarize_errors(fwd_err_improved)))
print("nominal FwdKins :", metrics.format_summary(metrics.summarize_errors(fwd_err_nominal)))
''')

# ---------------------------------------------------------------- cell 23: Step 10 inverse evaluation
set_src(23, '''
rng = np.random.default_rng(0)
targets = [test_grid[i] for i in rng.choice(len(test_grid), size=min(40, len(test_grid)), replace=False)]
# same 40 targets, visited in a greedy nearest-neighbour order: the wrist moves at 0.1 rad/s, so the
# visiting order is what sets the testing time (the handout asks us to report that time)
targets = workflow.order_poses_by_motion_time(rob, targets)
t_test_start = clock.elapsed

inv_err_nominal, inv_err_improved = [], []
for theta_t in targets:
    F_target = F_trk.inv() * rob.actual_ee_pose(theta_t)   # a reachable target, in the tracker frame
    for kins, F_X, errs in ((rob, F_X_nom, inv_err_nominal), (improved, improved.base_frame(), inv_err_improved)):
        theta = kins.inv_kins(F_X.inv() * F_target, seed=theta_t)
        rob.move_joints(theta)
        actual = F_trk.inv() * mbee_placed.actual_placement           # ground truth, reporting only
        errs.append(metrics.frame_error(F_target, actual))

t_testing = clock.elapsed - t_test_start
print("ImprovedInvKins :", metrics.format_summary(metrics.summarize_errors(inv_err_improved)))
print("nominal InvKins :", metrics.format_summary(metrics.summarize_errors(inv_err_nominal)))
print(f"simulated testing time: {t_testing:.1f}s")
''')

# ---------------------------------------------------------------- cell 25: summary
set_src(25, '''
def row(name, s):
    ok = s["max_pos"] <= constants.TARGET_POS_ERR_MM and s["max_angle"] <= constants.TARGET_ANG_ERR_RAD
    return (f"| {name:16s} | {s['rms_pos']:.3f} | {s['max_pos']:.3f} | {s['rms_angle']:.2e} | {s['max_angle']:.2e} | "
            f"{100 * s['frac_within_target']:5.1f}% | {'yes' if ok else 'NO'} |")

summary = {
    "fwd_nominal": metrics.summarize_errors(fwd_err_nominal), "fwd_improved": metrics.summarize_errors(fwd_err_improved),
    "inv_nominal": metrics.summarize_errors(inv_err_nominal), "inv_improved": metrics.summarize_errors(inv_err_improved),
}
print(f"SEED = {SEED}; targets: {constants.TARGET_POS_ERR_MM} mm, {constants.TARGET_ANG_ERR_RAD} rad")
print("| model            | pos rms | pos max | ang rms | ang max | in spec | meets target |")
print("|------------------|---------|---------|---------|---------|---------|--------------|")
for name, key in (("nominal FwdKins", "fwd_nominal"), ("ImprovedFwdKins", "fwd_improved"),
                  ("nominal InvKins", "inv_nominal"), ("ImprovedInvKins", "inv_improved")):
    print(row(name, summary[key]))
print(f"\\nsimulated time: calibration {t_calibration:.1f} s "
      f"(homing {5 * rob.t_settle:.1f} s + baseline {baseline.t_collect:.1f} s + sweeps {sweeps.t_collect:.1f} s, "
      f"{len(poses)} commands, {sum(sweeps.readings) + sum(baseline.readings)} tracker readings), "
      f"testing {t_testing:.1f} s; real run time so far {time.time() - WALL_START:.0f} s")
''')

# ---------------------------------------------------------------- extra: multi-seed table + results dump
cells.append(md('''
## Repeat for other seeds

The handout and the README ask for results for more than one robot. `run_pipeline(seed)`
repeats Steps 2-10 on a fresh robot and tracker (the same code as above, nothing tuned to the seed)
and returns the summary table; the report takes its tables from `results/notebook_results.json`.
'''))
cells.append(code('''
def run_pipeline(seed, n_inv_targets=40, verbose=False):
    clock.reset()
    rob_s = simulation.create_robot("rob", F_rob_approx, seed=seed)
    trk_s = simulation.create_tracker("trk", F_tracker_approx, seed=seed)
    mbee_s = simulation.create_robot_calibration_body(rob_s, mbee_local_positions)
    simulation.add_marker_bodies(trk_s, [mbee_s])
    offsets = workflow.measure_home_offsets(rob_s, n_homings=5)
    base = workflow.collect_calibration_data(rob_s, trk_s, mbee_s, baseline_poses, repeats=1)
    poses_s, info_s = workflow.plan_calibration_poses(rob_s, trk_s, mbee_local_positions, desired_accuracy=DESIRED_ACCURACY, return_info=True)
    sw = workflow.collect_calibration_data(rob_s, trk_s, mbee_s, poses_s, repeats=info_s["readings_per_command"])
    d = workflow.merge_data(base, sw)
    FX0 = workflow.estimate_base_frame(d, rob_s)
    imp = kinematics.ImprovedKinematics(rob_s, FX0, joint_offsets=offsets, n_poly=(2, 2, 2, 1, 1, 1))
    fit_s = imp.fit(d)
    t_cal = clock.elapsed
    Ft = trk_s.nominal_pose()
    grid = [th for th in test_grid]   # same fixed grid (visibility does not depend on the seed)
    fn, fi = [], []
    for th in grid:
        tp = Ft.inv() * rob_s.actual_ee_pose(th)
        fn.append(metrics.frame_error(tp, FX0 * rob_s.fwd_kins(th)))
        fi.append(metrics.frame_error(tp, imp.predict_in_tracker(th)))
    r = np.random.default_rng(0)
    tg = workflow.order_poses_by_motion_time(rob_s, [grid[i] for i in r.choice(len(grid), size=min(n_inv_targets, len(grid)), replace=False)])
    t0 = clock.elapsed
    inn, ini = [], []
    for th_t in tg:
        Ftar = Ft.inv() * rob_s.actual_ee_pose(th_t)
        for kins, FX, errs in ((rob_s, FX0, inn), (imp, imp.base_frame(), ini)):
            rob_s.move_joints(kins.inv_kins(FX.inv() * Ftar, seed=th_t))
            errs.append(metrics.frame_error(Ftar, Ft.inv() * mbee_s.actual_placement))
    t_tst = clock.elapsed - t0
    cmp_s = imp.compare_with_ground_truth(rob_s.ground_truth_parameters(), Ft)
    out = {"seed": seed, "t_calibration": t_cal, "t_testing": t_tst, "t_baseline": base.t_collect, "t_sweeps": sw.t_collect,
           "n_commands": len(poses_s), "readings_per_pose": info_s["readings_per_pose"],
           "readings_per_command": info_s["readings_per_command"], "commands_per_pose": info_s["commands_per_pose"], "fit": fit_s,
           "fwd_nominal": metrics.summarize_errors(fn), "fwd_improved": metrics.summarize_errors(fi),
           "inv_nominal": metrics.summarize_errors(inn), "inv_improved": metrics.summarize_errors(ini),
           "axis_direction_err_mrad": cmp_s["axis_direction_err_mrad"].tolist(),
           "axis_angles_deg_err": cmp_s["axis_angles_deg_err"].tolist(), "joint_scale_err": cmp_s["joint_scale_err"].tolist(),
           "joint_offset_err": cmp_s["joint_offset_err"].tolist(), "tool_frame_err_rad_mm": list(cmp_s["tool_frame_err_rad_mm"]),
           "base_frame_err_rad_mm": list(cmp_s["base_frame_err_rad_mm"])}
    if verbose:
        print(f"seed {seed}: ImprovedFwdKins {metrics.format_summary(out['fwd_improved'])}")
        print(f"         ImprovedInvKins {metrics.format_summary(out['inv_improved'])}")
        print(f"         t_cal {t_cal:.0f} s, t_test {t_tst:.0f} s")
    return out

# the notebook's own run (Steps 6-10 above) is the first entry; seeds 2 and 3 are fresh runs of the same code
main_entry = {"seed": SEED, "t_calibration": t_calibration, "t_testing": t_testing, "t_baseline": baseline.t_collect,
              "t_sweeps": sweeps.t_collect, "n_commands": len(poses), "readings_per_pose": plan_info["readings_per_pose"],
              "readings_per_command": plan_info["readings_per_command"], "commands_per_pose": plan_info["commands_per_pose"],
              "fit": fit_info, "fwd_nominal": summary["fwd_nominal"], "fwd_improved": summary["fwd_improved"],
              "inv_nominal": summary["inv_nominal"], "inv_improved": summary["inv_improved"],
              "axis_direction_err_mrad": cmp["axis_direction_err_mrad"].tolist(), "axis_angles_deg_err": cmp["axis_angles_deg_err"].tolist(),
              "joint_scale_err": cmp["joint_scale_err"].tolist(), "joint_offset_err": cmp["joint_offset_err"].tolist(),
              "tool_frame_err_rad_mm": list(cmp["tool_frame_err_rad_mm"]), "base_frame_err_rad_mm": list(cmp["base_frame_err_rad_mm"])}
results = {"notebook_seed": SEED, "seeds": [main_entry], "plan_info": {k: v for k, v in plan_info.items() if k != "slide_stops"},
           "fit_info": fit_info, "rank": {"rank": rank["rank"], "n_params": rank["n_params"],
                                          "weakest": rank["weakest_directions"][0], "free_rank": rank["free"]["rank"],
                                          "free_n": rank["free"]["n_params"], "free_cond": rank["free"]["condition_number"],
                                          "singular_values": rank["singular_values"]},
           "params_vs_truth": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in cmp.items()},
           "test_grid_size": len(test_grid)}
for s in (2, 3):
    results["seeds"].append(run_pipeline(s, verbose=True))
print(f"\\nreal run time for the whole notebook: {time.time() - WALL_START:.0f} s")

os.makedirs("../results", exist_ok=True)
with open("../results/notebook_results.json", "w") as f:
    json.dump(results, f, indent=1, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else (float(o) if isinstance(o, np.floating) else str(o)))
print("saved ../results/notebook_results.json")
'''))


# ---------------------------------------------------------------- extra: pointer to the variant study
cells.append(md('''
## Pose-plan / time trade-off

The same pipeline with deliberately different plans (wrist sweep length, slide spacing, readings
per pose, re-commanding, polynomial degrees), each on 10 robot seeds, is run by
`report/variant_study.py` (kept out of this notebook to keep its run time short; it takes about
6 minutes) and written to `results/variants.json`. The report's trade-off table comes from there.
'''))

OUT.parent.mkdir(parents=True, exist_ok=True)
json.dump(nb, open(OUT, "w"), indent=1)
print("wrote", OUT, "with", len(cells), "cells")
