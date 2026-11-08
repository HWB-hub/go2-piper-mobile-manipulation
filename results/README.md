# Result directory index

These directories record separate experiments, seeds, checkpoints, and asset versions. Do not combine sample counts without checking the original records.

| Directory pattern | Purpose and interpretation |
| --- | --- |
| `baseline_*` | Earlier baseline before the mounting geometry update. |
| `holdout_*` | Earlier four-category evaluation: spin, transition, curves, combined. |
| `stability_v8_fresh_*` | General v8 policy evaluations using the earlier mount. |
| `s_curve_base`, `s_curve_test`, `s_curve_video` | S-curve baseline, specialized policy and demonstration. |
| `walk_run_base`, `walk_run_v3_contact`, `walk_run_demo_1581` | Speed-transition baseline, contact analysis and demonstration. Flight is intermittent. |
| `ground_pick_stable_*` | Earlier staged ground-pick experiments and recheck/video. |
| `user_mount_adapt_*` | Adapted general policy evaluated after the mount update. |
| `user_mount_fresh_base_*`, `user_mount_fresh_adapt_*` | New-seed comparisons of base and adapted policies after the mount update. |
| `user_mount_spin`, `user_mount_transition`, `user_mount_serpentine`, `user_mount_walk_run` | Tests with updated geometry; S-curve and speed transition use specialized policies. |
| `user_mount_pick_*`, `user_mount_video_pick` | Ground-pick tests and recording with updated geometry. |
| `github_user_mount_*` | Updated-geometry demonstration recordings and associated runs. |

Consult the JSON data for exact settings and definitions. Mount geometry and policy checkpoint matter when interpreting results.
