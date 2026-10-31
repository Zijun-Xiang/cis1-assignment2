# Assignment 2 — our solution: what is where

Everything the course template asked for is in place; the template files themselves are untouched.

| path | what it is |
|---|---|
| `src/student/metrics.py` | `frame_error`, `summarize_errors` (+ `format_summary`) |
| `src/student/registration.py` | own Arun registration, `average_frames`, `refine_marker_body` |
| `src/student/workflow.py` | `measure_home_offsets`, `plan_calibration_poses`, `collect_calibration_data`, `estimate_base_frame`, plus `merge_data`, `order_poses_by_motion_time`, `pose_is_visible`, `predicted_collection_time` |
| `src/student/kinematics.py` | `ImprovedKinematics` (parameterization, `fwd_kins`, `residuals`, `fit`, `jacobian_rank_report`, `parameters_report`, `compare_with_ground_truth`; `inv_kins` as provided) |
| `src/student/design.py` | Step-1 design values (tracker pose, robot pose, calibration-body layout, marker envelope) |
| `src/student/_core.py` | pure-numpy math behind kinematics/workflow (se(3), parameter model, Jacobian, least squares, rank analysis, plan sizing) |
| `src/student/_convert.py` | Frame/Rot/vct3 <-> numpy helpers |
| `tests/test_student.py` | the course's tests (unchanged), the three `TODO` tests, and our own (30 test functions) |
| `notebooks/assignment2_solution.ipynb` | the filled-in template, executed (outputs included); writes `results/notebook_results.json` |
| `report/Assignment2_Report.pdf` / `.docx` / `report.md` | the written report (generated from the results JSON) |
| `report/variant_study.py` | 10-seed pose-plan / time trade-off study -> `results/variants.json` (about 6 min) |
| `report/template.md`, `report/make_report.py` | report template + generator (`python report/make_report.py`, needs pandoc; falls back to Markdown + DOCX if LaTeX is missing) |
| `report/make_notebook.py` | builds the solution notebook from the course template (how it was made; not needed to run anything) |

## Run

```
cd assignment2
python -m pytest tests -q                              # 53 tests: 23 provided + 30 ours
jupyter notebook notebooks/assignment2_solution.ipynb  # run all cells (~1 min); writes results/notebook_results.json
python report/variant_study.py                         # optional, ~6 min; writes results/variants.json
python report/make_report.py --authors "Name 1, Name 2" --tests "53 passed"   # regenerates the report
```

## Before submitting

* put your names in the report (`--authors`) and check the test line printed by pytest (`--tests`);
* if you change the design, re-run the notebook and `make_report.py`; the report text is written to match the
  design in `design.py` / `plan_calibration_poses` (5 commands x 7 readings per pose, 75 mm slide step,
  +-1/+-2 rad wrist stops), so re-read Sections 4.1 and 6.5 if you change those numbers;
* the report's "Deviations from the template" paragraph lists every place the notebook differs from the template.
