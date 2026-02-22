# CA3_nav_starter
The goal of this assignment is to test and modify different motion planning algorithms by designing, running, and interpreting experiments. 
By the end, you should have a concrete, experimental understanding of:
1. How EKF-SLAM behaves under different trajectories and noise conditions
2. How PRM and RRT compare as sampling-based mtion planners in the same environment
3. Simple ways to improve PRM and RRT and when those improvements help.

## Files
Provided files & modifications:

* `environment.py`
  * Modified to add new obstacles
  * Changed radius size of obstacles
* `slam.py`
* `motion_planning.py`
  * Added `plan_improved_prm` and `plan_improved_rrt` functions for Part C
* `test_EKF-SLAM.py`
* `test_PathPlanning.py`

**Part A1 – EKF-SLAM Path Design**
* `test_greedy_SLAM.py` – Modified the loop in the `main()` function of test_EKF-SLAM.py to implement a greedy control strategy. The robot mainly drives forward without intentionally looping back to already visited paths. The robot moves in a large track around the given map.
* `test_loop_SLAM.py` – Modified the loop in the `main()` function to test_EKF_SLAM.py to implement a looping/revist contrl strategy. The robot tries to loop around and revist regions that its already visited (loop closure) to reduce uncertainty in landmark positions.

**Part A2 – Noise Sensitivity**
* `plot_landmark_uncertainty.py` – Helper script to generate plots and figures mapping landmark uncertainty. Code generated with the help of Copilot.
* `plot_uncertainty_metrics.py` – Helper script to generate plots and figures mapping robot uncertainty. Code generated with the help of Copilot.
These scripts were run as I varied the parameters for process noise and measurement noise in EKF-SLAM.

**Part B1 – Start/Goal Scenarios**
* `test_planning.py` – Modified start and end goal for PRM and RRT from `test_PathPlanning.py`
I modified obstacle locations and radius size to approach the open/narrow map configurations.

**Part B2 – Metrics and Single-Scenario Comparison**
* `scenario_metrics_comparison.py` – Helper script that runs PRM and RRT and generates a table of statistics for each run. Code to calculate and log table of statistics was generated with Copilot.

**Part B3 – Parameter Studies**
* `prm_test_parameters.py` – Test PRM algorithm with varying n_samples parameter. Measures success rate, path length, and planning time across different sample sizes. Summary of statistics and figures generated with Copilot.
* `rrt_test_parameters.py` – Test RRT algorithm with varying step_size and goal_sample_rate parameters. Measures success rate, path length, and planning time across different configurations. Summary of statistics and figures generated with Copilot.

**Part C - Improving PRM and RRT**
* `improved_prm_testing.py` – Added a new function in `motion_planning.py` to support an improved implementation of PRM. The function in this script calls the improved planning approach from `motion_planning.py`. The code to generate statistics was generated with Copilot.
* `improved_rrt_testing.py` – Added a new function in `motion_planning.py` to support an improved implementation of RRT. Runs this newly modified implementation and produces statistics as a comparison measure. Code to generate statistics was generated wtih Copilot.
