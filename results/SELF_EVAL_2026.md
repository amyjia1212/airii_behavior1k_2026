# 2026 self-evaluation results (PI_BEHAVIOR checkpoints in this repo)

Self-evaluation on the **public test instances 0-9, 1 rollout each** (BEHAVIOR-1K v3.9.3-post1 `omnigibson.eval.eval`,
default 1.5x human-length timeout), logged to W&B (`b1k-2026/b1k-eval-stagefix`, `b1k-2026/b1k-eval`).
Policy settings: 280x280 inputs, soft inpainting, 20 flow steps, execute 13 / keep 4 of 30 actions in 10 env steps
(1.3x speed-up via cubic interpolation), gripper recovery 90/15. These are **not** the official hidden-test scores.

| Task | Name | Checkpoint | Q-score | Success rate | Rollouts |
|---:|---|---|---:|---:|---:|
| 50 | freeze_fruit | `b1k-2026-t50-70-72-85/step_049999` | 0.014 | 0.00 | 10 |
| 52 | sorting_bottles_cans_and_paper | `b1k-2026-t52/step_008000` | 0.075 | 0.00 | 10 |
| 53 | tidying_living_room | `b1k-2026-t52-53-73-66-86/step_049999` | 0.025 | 0.00 | 10 |
| 65 | unloading_the_car | `b1k-2026-t65-71-74-75/step_049999` | 0.000 | 0.00 | 10 |
| 66 | turning_out_all_lights_before_sleep | `b1k-2026-t52-53-73-66-86/step_049999` | 0.220 | 0.00 | 10 |
| 70 | thawing_frozen_food | `b1k-2026-t50-70-72-85/step_049999` | 0.267 | 0.00 | 10 |
| 71 | clean_your_rusty_garden_tools | `b1k-2026-t65-71-74-75/step_049999` | 0.080 | 0.00 | 10 |
| 72 | cook_a_frozen_pie | `b1k-2026-t50-70-72-85/step_049999` | 0.000 | 0.00 | 10 |
| 73 | organizing_school_stuff | `b1k-2026-t52-53-73-66-86/step_049999` | 0.133 | 0.00 | 10 |
| 74 | carrying_out_garden_furniture | `b1k-2026-t65-71-74-75/step_049999` | 0.050 | 0.00 | 10 |
| 75 | put_together_a_basic_pruning_kit | `b1k-2026-t65-71-74-75/step_049999` | 0.000 | 0.00 | 10 |
| 79 | polishing_shoes | `b1k-2026-t79-81-84-88/step_049999` | 0.067 | 0.00 | 10 |
| 81 | packing_meal_for_delivery | `b1k-2026-t81-ft10k/step_009999` | 0.083 | 0.00 | 10 |
| 84 | tidying_bathroom | `b1k-2026-t79-81-84-88/step_049999` | 0.200 | 0.00 | 10 |
| 85 | putting_dirty_dishes_in_sink | `b1k-2026-t50-70-72-85/step_049999` | 0.275 | 0.00 | 10 |
| 86 | make_gift_bags_for_baby_showers | `b1k-2026-t52-53-73-66-86/step_049999` | 0.017 | 0.00 | 10 |
| 88 | rearrange_your_room | `b1k-2026-t79-81-84-88/step_049999` | 0.067 | 0.00 | 10 |
| 90 | composting_waste | `b1k-2026-t90-91-93-96-97/step_099999` | n/a | n/a | not self-evaluated (no W&B eval run recovered) |
| 91 | store_produce | `b1k-2026-t90-91-93-96-97/step_099999` | n/a | n/a | not self-evaluated (no W&B eval run recovered) |
| 93 | clean_a_keyboard | `b1k-2026-t90-91-93-96-97/step_099999` | n/a | n/a | not self-evaluated (no W&B eval run recovered) |
| 94 | dispose_of_batteries | `b1k-2026-t94-95-98/step_049999` | 0.100 | 0.00 | 10 |
| 95 | cook_brussels_sprouts | `b1k-2026-t94-95-98/step_049999` | 0.000 | 0.00 | 10 |
| 96 | cook_broccolini | `b1k-2026-t90-91-93-96-97/step_099999` | n/a | n/a | not self-evaluated (no W&B eval run recovered) |
| 97 | setup_a_bar_for_a_cocktail_party | `b1k-2026-t90-91-93-96-97/step_099999` | n/a | n/a | not self-evaluated (no W&B eval run recovered) |
| 98 | laying_tile_floors | `b1k-2026-t94-95-98/step_049999` | 0.425 | 0.00 | 10 |

Mean over the 20 self-evaluated tasks: **Q-score 0.105**, success rate 0.000.

Run-to-run variance is large: re-running `putting_dirty_dishes_in_sink` (task 85) with the same checkpoint and settings on other pods gave
Q = 0.625 / 0.400 / 0.275 (runs `...-stagefix-control`, `...-stagefix-control-pod2`, `...-stagefix`), so differences of a few hundredths between
checkpoints are within noise.

Raw per-run summaries for every eval we ran: [`selfeval_runs_2026.json`](selfeval_runs_2026.json).
