# This script was generated using Copilot
#!/usr/bin/env python3
"""
Compare PRM and RRT planning algorithms across multiple trials.
Measures success rate, path length, and planning effort.
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from environment import Environment
from motion_planning import PRM, RRT
from tabulate import tabulate


def calculate_path_length(path):
    """Calculate total path length as sum of Euclidean distances."""
    if path is None or len(path) < 2:
        return None
    return sum(np.linalg.norm(path[i+1] - path[i]) for i in range(len(path)-1))


def visualize_path(env, start, goal, path, algorithm_name, trial_number, save=True):
    """Visualize and optionally save the path."""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Set up plot
    ax.set_xlim(env.x_min, env.x_max)
    ax.set_ylim(env.y_min, env.y_max)
    ax.set_xlabel("x [m]", fontsize=12)
    ax.set_ylabel("y [m]", fontsize=12)
    ax.set_title(f"{algorithm_name} - First Successful Path (Trial {trial_number})", 
                 fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Draw obstacles
    for lm_id, lm_pos in env.landmarks.items():
        circle = Circle((lm_pos[0], lm_pos[1]), env.obstacle_radius,
                       color='gray', alpha=0.4, label='Obstacles' if lm_id == 0 else '')
        ax.add_patch(circle)
    
    # Draw path
    if path is not None and len(path) > 1:
        path_array = np.array(path)
        ax.plot(path_array[:, 0], path_array[:, 1], 'b-', linewidth=2, 
                label='Path', zorder=3)
        ax.plot(path_array[:, 0], path_array[:, 1], 'bo', markersize=4, 
                alpha=0.6, zorder=4)
    
    # Draw start and goal
    ax.plot(start[0], start[1], 'go', markersize=15, label='Start', zorder=5)
    ax.plot(goal[0], goal[1], 'r*', markersize=20, label='Goal', zorder=5)
    
    # Add path info
    if path is not None:
        path_length = calculate_path_length(path)
        info_text = f"Path Length: {path_length:.2f} m\nWaypoints: {len(path)}"
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    
    if save:
        filename = f"{algorithm_name.lower()}_first_success_trial_{trial_number}.png"
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {filename}")
    
    plt.close(fig)
    return fig


def run_prm_trial(env, start, goal, n_samples=300, k_neighbors=10, seed=None, time_limit=5.0):
    """Run a single PRM trial and collect metrics."""
    if seed is not None:
        np.random.seed(seed)
    
    prm = PRM(env, n_samples=n_samples, k_neighbors=k_neighbors)
    
    start_time = time.time()
    path = prm.plan(start, goal, visualize_callback=None)
    planning_time = time.time() - start_time
    
    # Success condition: path found AND within time limit
    success = (path is not None) and (planning_time <= time_limit)
    path_length = calculate_path_length(path) if path is not None else None
    
    return {
        'success': success,
        'path': path if success else None,  # Only return path if within time limit
        'path_length': path_length if success else None,
        'planning_time': planning_time,
        'nodes_sampled': n_samples,
        'nodes_in_path': len(path) if success else None,
        'timed_out': (path is not None) and (planning_time > time_limit)  # Track if we found path but timed out
    }


def run_rrt_trial(env, start, goal, max_iter=1000, step_size=0.5, 
                  goal_sample_rate=0.1, goal_tolerance=0.5, seed=None):
    """Run a single RRT trial and collect metrics."""
    if seed is not None:
        np.random.seed(seed)
    
    rrt = RRT(env, max_iter=max_iter, step_size=step_size, 
              goal_sample_rate=goal_sample_rate)
    
    start_time = time.time()
    path = rrt.plan(start, goal, goal_tolerance=goal_tolerance, 
                   visualize_callback=None)
    planning_time = time.time() - start_time
    
    success = path is not None
    path_length = calculate_path_length(path) if success else None
    iterations_used = len(rrt.nodes)  # Number of nodes = number of successful iterations
    
    return {
        'success': success,
        'path': path,
        'path_length': path_length,
        'planning_time': planning_time,
        'iterations': iterations_used,
        'nodes_in_path': len(path) if success else None
    }


def run_comparison(n_trials=30):
    """Run comparison between PRM and RRT."""
    print("\n" + "="*80)
    print("MOTION PLANNING ALGORITHM COMPARISON")
    print("="*80)
    print(f"\nRunning {n_trials} trials for each algorithm...")
    print("Environment: 20 circular obstacles")
    print("Start: [1.0, 1.0], Goal: [12.5, 17.5]")
    print("="*80 + "\n")
    
    # Setup common parameters
    env = Environment(seed=42)
    start = np.array([1.0, 1.0])
    goal = np.array([12.5, 17.5])
    
    # PRM parameters
    prm_n_samples = 300
    prm_k_neighbors = 10
    prm_time_limit = 1.0  # 5 second time limit for PRM
    
    # RRT parameters
    rrt_max_iter = 400
    rrt_step_size = 0.5
    rrt_goal_sample_rate = 0.1
    rrt_goal_tolerance = 0.5
    
    # Collect results
    prm_results = []
    rrt_results = []
    prm_first_success_saved = False
    rrt_first_success_saved = False
    
    print(f"Running PRM trials (time limit: {prm_time_limit}s)...")
    for i in range(n_trials):
        if (i + 1) % 10 == 0:
            print(f"  Trial {i+1}/{n_trials}")
        result = run_prm_trial(env, start, goal, 
                              n_samples=prm_n_samples,
                              k_neighbors=prm_k_neighbors,
                              time_limit=prm_time_limit,
                              seed=100 + i)
        prm_results.append(result)
        
        # Save first successful run
        if result['success'] and not prm_first_success_saved:
            print(f"\n  First PRM success at trial {i+1}! (time: {result['planning_time']:.3f}s)")
            visualize_path(env, start, goal, result['path'], "PRM", i+1, save=True)
            prm_first_success_saved = True
            print()
        elif result['timed_out'] and i < 5:  # Print first few timeouts for debugging
            print(f"  Trial {i+1}: Found path but exceeded time limit ({result['planning_time']:.3f}s)")
    
    print("\nRunning RRT trials...")
    for i in range(n_trials):
        if (i + 1) % 10 == 0:
            print(f"  Trial {i+1}/{n_trials}")
        result = run_rrt_trial(env, start, goal,
                              max_iter=rrt_max_iter,
                              step_size=rrt_step_size,
                              goal_sample_rate=rrt_goal_sample_rate,
                              goal_tolerance=rrt_goal_tolerance,
                              seed=100 + i)
        rrt_results.append(result)
        
        # Save first successful run
        if result['success'] and not rrt_first_success_saved:
            print(f"\n  First RRT success at trial {i+1}!")
            visualize_path(env, start, goal, result['path'], "RRT", i+1, save=True)
            rrt_first_success_saved = True
            print()
    
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80 + "\n")
    
    # Calculate statistics
    prm_stats = calculate_statistics(prm_results, "PRM", prm_n_samples)
    rrt_stats = calculate_statistics(rrt_results, "RRT", rrt_max_iter)
    
    # Add timeout statistics for PRM
    prm_timeouts = sum(1 for r in prm_results if r.get('timed_out', False))
    if prm_timeouts > 0:
        print(f"Note: PRM had {prm_timeouts} trials that found paths but exceeded {prm_time_limit}s time limit\n")
    
    # Print comparison table
    print_comparison_table(prm_stats, rrt_stats)
    
    # Print detailed statistics
    print_detailed_statistics(prm_stats, rrt_stats)
    
    return prm_results, rrt_results, prm_stats, rrt_stats


def calculate_statistics(results, algorithm_name, planning_parameter):
    """Calculate statistics from trial results."""
    n_trials = len(results)
    n_success = sum(1 for r in results if r['success'])
    success_rate = n_success / n_trials * 100
    
    # Extract successful trials only for length statistics
    successful_results = [r for r in results if r['success']]
    
    if successful_results:
        path_lengths = [r['path_length'] for r in successful_results]
        planning_times = [r['planning_time'] for r in successful_results]
        nodes_in_path = [r['nodes_in_path'] for r in successful_results]
        
        # Get iterations/nodes info
        if algorithm_name == "RRT":
            iterations = [r['iterations'] for r in successful_results]
            avg_iterations = np.mean(iterations)
            std_iterations = np.std(iterations)
        else:
            avg_iterations = planning_parameter
            std_iterations = 0
        
        stats = {
            'algorithm': algorithm_name,
            'n_trials': n_trials,
            'n_success': n_success,
            'success_rate': success_rate,
            'avg_path_length': np.mean(path_lengths),
            'std_path_length': np.std(path_lengths),
            'min_path_length': np.min(path_lengths),
            'max_path_length': np.max(path_lengths),
            'avg_planning_time': np.mean(planning_times),
            'std_planning_time': np.std(planning_times),
            'avg_nodes_in_path': np.mean(nodes_in_path),
            'avg_planning_effort': avg_iterations,
            'std_planning_effort': std_iterations
        }
    else:
        stats = {
            'algorithm': algorithm_name,
            'n_trials': n_trials,
            'n_success': 0,
            'success_rate': 0.0,
            'avg_path_length': None,
            'std_path_length': None,
            'min_path_length': None,
            'max_path_length': None,
            'avg_planning_time': None,
            'std_planning_time': None,
            'avg_nodes_in_path': None,
            'avg_planning_effort': None,
            'std_planning_effort': None
        }
    
    return stats


def print_comparison_table(prm_stats, rrt_stats):
    """Print comparison table using tabulate."""
    
    # Main comparison table
    headers = ["Metric", "PRM", "RRT"]
    
    def format_value(val, std=None):
        if val is None:
            return "N/A"
        if std is not None and std is not None:
            return f"{val:.2f} ± {std:.2f}"
        return f"{val:.2f}"
    
    table_data = [
        ["Success Rate (%)", 
         f"{prm_stats['success_rate']:.1f}", 
         f"{rrt_stats['success_rate']:.1f}"],
        
        ["Avg Path Length (m)", 
         format_value(prm_stats['avg_path_length'], prm_stats['std_path_length']),
         format_value(rrt_stats['avg_path_length'], rrt_stats['std_path_length'])],
        
        ["Min Path Length (m)",
         format_value(prm_stats['min_path_length']),
         format_value(rrt_stats['min_path_length'])],
        
        ["Max Path Length (m)",
         format_value(prm_stats['max_path_length']),
         format_value(rrt_stats['max_path_length'])],
        
        ["Avg Planning Time (s)",
         format_value(prm_stats['avg_planning_time'], prm_stats['std_planning_time']),
         format_value(rrt_stats['avg_planning_time'], rrt_stats['std_planning_time'])],
        
        ["Avg Waypoints in Path",
         format_value(prm_stats['avg_nodes_in_path']),
         format_value(rrt_stats['avg_nodes_in_path'])],
        
        ["Planning Effort*",
         f"{prm_stats['avg_planning_effort']:.0f} nodes sampled",
         format_value(rrt_stats['avg_planning_effort'], rrt_stats['std_planning_effort']) + " iterations"]
    ]
    
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    print("\n* Planning Effort: Number of nodes sampled (PRM) or iterations used (RRT)")


def print_detailed_statistics(prm_stats, rrt_stats):
    """Print additional detailed statistics."""
    print("\n" + "="*80)
    print("DETAILED STATISTICS")
    print("="*80)
    
    for stats in [prm_stats, rrt_stats]:
        print(f"\n{stats['algorithm']}:")
        print(f"  Trials: {stats['n_trials']}")
        print(f"  Successes: {stats['n_success']}/{stats['n_trials']} ({stats['success_rate']:.1f}%)")
        
        if stats['n_success'] > 0:
            print(f"  Path Length:")
            print(f"    Mean: {stats['avg_path_length']:.2f} m")
            print(f"    Std:  {stats['std_path_length']:.2f} m")
            print(f"    Min:  {stats['min_path_length']:.2f} m")
            print(f"    Max:  {stats['max_path_length']:.2f} m")
            print(f"  Planning Time:")
            print(f"    Mean: {stats['avg_planning_time']:.4f} s")
            print(f"    Std:  {stats['std_planning_time']:.4f} s")
            print(f"  Waypoints: {stats['avg_nodes_in_path']:.1f} (average)")
            
            if stats['algorithm'] == "RRT":
                print(f"  Iterations: {stats['avg_planning_effort']:.1f} ± {stats['std_planning_effort']:.1f}")
            else:
                print(f"  Nodes Sampled: {stats['avg_planning_effort']:.0f}")
        else:
            print("  No successful trials!")
    
    print("\n" + "="*80)


def main():
    """Main function."""
    # Number of trials for each algorithm
    N_TRIALS = 30
    
    # Run comparison
    prm_results, rrt_results, prm_stats, rrt_stats = run_comparison(N_TRIALS)
    
    # Summary
    print("\nSUMMARY:")
    print(f"  PRM: {prm_stats['success_rate']:.1f}% success, "
          f"avg length {prm_stats['avg_path_length']:.2f}m" if prm_stats['avg_path_length'] else "PRM: No successes")
    print(f"  RRT: {rrt_stats['success_rate']:.1f}% success, "
          f"avg length {rrt_stats['avg_path_length']:.2f}m" if rrt_stats['avg_path_length'] else "RRT: No successes")
    
    # Determine winner
    if prm_stats['success_rate'] > rrt_stats['success_rate']:
        print(f"\n✓ PRM had higher success rate (+{prm_stats['success_rate'] - rrt_stats['success_rate']:.1f}%)")
    elif rrt_stats['success_rate'] > prm_stats['success_rate']:
        print(f"\n✓ RRT had higher success rate (+{rrt_stats['success_rate'] - prm_stats['success_rate']:.1f}%)")
    else:
        print("\n  Both algorithms had equal success rates")
    
    if prm_stats['avg_path_length'] and rrt_stats['avg_path_length']:
        if prm_stats['avg_path_length'] < rrt_stats['avg_path_length']:
            diff = rrt_stats['avg_path_length'] - prm_stats['avg_path_length']
            print(f"✓ PRM found shorter paths on average (-{diff:.2f}m)")
        else:
            diff = prm_stats['avg_path_length'] - rrt_stats['avg_path_length']
            print(f"✓ RRT found shorter paths on average (-{diff:.2f}m)")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()