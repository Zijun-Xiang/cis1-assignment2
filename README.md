# CIS I (601.455/655) — Assignment 2: Kinematic Calibration of a Cartesian Robot

Johns Hopkins University, Computer-Integrated Surgery I, Fall 2026.

This repository holds our solution to Assignment 2: calibrating a simulated Cartesian robot + spherical wrist
with an optical tracker, so that the improved forward/inverse kinematics are accurate to 0.2 mm / 1 mrad over
the whole work volume in as little simulated time as possible.

| file | what it is |
|---|---|
| `Assignment2_Report.pdf` | the written report (design rationale, error model, identifiability, pose-plan/time trade-off, unit tests, results for 3 seeds) |
| `Assignment2_Code_Architecture.pdf` | a walkthrough of the code: architecture, pipeline, every file and function |
| `assignment2_solution.zip` | everything below, zipped |
| `assignment2/` | the solution files, laid out exactly as they sit inside the course repo's `assignments/assignment2/` folder |

## Results (course simulator, seed 1, 729-pose test grid)

| model | position max | orientation max |
|---|---|---|
| nominal kinematics | 16.1 mm | 56 mrad |
| **ImprovedFwdKins** | **0.118 mm** | **0.35 mrad** |
| **ImprovedInvKins** | **0.115 mm** | **0.49 mrad** |

Simulated calibration time 389 s (soft budget 900 s); all tests pass (53 = 23 provided + 30 ours); the notebook
runs in about 45 s of real time.

## How to run

The code needs the course framework (`src/simulation`, `src/robot.py`, the assignment-1 tracker classes and the
`data_types` / `uncertainty_networks` packages), which lives in the course repository
[LCSR-CIIS/CIS_I_Course](https://github.com/LCSR-CIIS/CIS_I_Course). Copy the contents of `assignment2/` over
`assignments/assignment2/` in a checkout of that repository, then:

```
cd assignments/assignment2
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m pytest tests -q                              # 53 tests
jupyter notebook notebooks/assignment2_solution.ipynb  # run all cells
python report/variant_study.py                         # optional 10-seed trade-off study (~6 min)
python report/make_report.py --authors "..."           # regenerates the report (needs pandoc)
```

See `assignment2/README_SOLUTION.md` for a file-by-file description.
