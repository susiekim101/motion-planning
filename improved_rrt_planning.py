#!/usr/bin/env python3
"""
Animated visualization of PRM and RRT planning algorithms.
Shows step-by-step how each algorithm builds its data structure and finds a path.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from environment import Environment
from motion_planning import RRT
from tabulate import tabulate


class RRTVisualizer:
    """Visualize RRT algorithm step-by-step."""
    
    def __init__(self, env, start, goal):
        self.env = env
        self.start = start
        self.goal = goal
        self.fig, self.ax = plt.subplots(figsize=(12, 10))
        plt.ion()
        
    def visualize(self, nodes, parents, iteration, rand_point, nearest_idx, new_node):
        """Callback for RRT visualization."""
        self.ax.clear()
        self.ax.set_title(f"RRT: Growing Tree (iteration {iteration})", 
                         fontsize=14, fontweight='bold')
        self.ax.set_xlim(self.env.x_min, self.env.x_max)
        self.ax.set_ylim(self.env.y_min, self.env.y_max)
        self.ax.set_xlabel("x [m]")
        self.ax.set_ylabel("y [m]")
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        
        # Draw obstacles
        for lm_pos in self.env.landmarks.values():
            circle = Circle((lm_pos[0], lm_pos[1]), self.env.obstacle_radius,
                          color='gray', alpha=0.4)
            self.ax.add_patch(circle)
        
        # Draw tree edges
        for i, parent_idx in parents.items():
            if parent_idx is not None:
                self.ax.plot([nodes[i][0], nodes[parent_idx][0]],
                           [nodes[i][1], nodes[parent_idx][1]],
                           'b-', linewidth=0.8, alpha=0.4)
        
        # Draw tree nodes
        if nodes:
            nodes_array = np.array(nodes)
            self.ax.plot(nodes_array[:, 0], nodes_array[:, 1], 'bo', 
                        markersize=3, alpha=0.6, label='Tree nodes')
        
        # Highlight the latest extension
        if new_node is not None and nearest_idx is not None:
            nearest_node = nodes[nearest_idx]
            # Draw random sample
            self.ax.plot(rand_point[0], rand_point[1], 'rx', markersize=10,
                        label='Random sample', alpha=0.7)
            # Draw nearest node
            self.ax.plot(nearest_node[0], nearest_node[1], 'yo', markersize=8,
                        label='Nearest node', zorder=5)
            # Draw new node
            self.ax.plot(new_node[0], new_node[1], 'go', markersize=8,
                        label='New node', zorder=5)
            # Draw extension line
            self.ax.plot([nearest_node[0], new_node[0]],
                       [nearest_node[1], new_node[1]],
                       'g-', linewidth=2, alpha=0.8, zorder=4)
        
        # Draw start and goal
        self.ax.plot(self.start[0], self.start[1], 'go', markersize=15, 
                    label='Start', zorder=7)
        self.ax.plot(self.goal[0], self.goal[1], 'r*', markersize=20, 
                    label='Goal', zorder=7)
        
        # Add info box
        info = f"Tree nodes: {len(nodes)}\nIteration: {iteration}"
        if new_node is not None:
            dist_to_goal = np.linalg.norm(new_node - self.goal)
            info += f"\nDist to goal: {dist_to_goal:.2f}m"
        
        self.ax.text(0.02, 0.98, info, transform=self.ax.transAxes,
                    fontsize=10, verticalalignment='top', family='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
        
        self.ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
        plt.pause(0.01)
    
    def close(self):
        plt.ioff()
        plt.close()

def run_rrt_statistical_analysis(n_trials=30, max_iter=2000, adaptive=False):
    """
    Run multiple RRT trials and generate statistical summary.
    
    Args:
        n_trials: Number of trials to run
        max_iter: Maximum iterations for RRT
        adaptive: Whether to use adaptive goal sampling
        
    Returns:
        Dictionary with statistics
    """
    print("\n" + "="*80)
    print(f"RRT STATISTICAL ANALYSIS (adaptive={adaptive})")
    print("="*80)
    print(f"Running {n_trials} trials...")
    print(f"Max iterations: {max_iter}")
    print("="*80 + "\n")
    
    # Setup
    env = Environment(seed=42)
    start = np.array([1.0, 1.0])
    goal = np.array([17.0, 17.0])
    
    # RRT parameters
    step_size = 0.5
    goal_sample_rate = 0.1
    goal_tolerance = 0.5
    
    # Store results
    results = []
    
    for trial in range(n_trials):
        if (trial + 1) % 10 == 0:
            print(f"  Trial {trial+1}/{n_trials}")
        
        # Set seed for reproducibility
        np.random.seed(2000 + trial)
        
        # Create new RRT instance
        if adaptive:
            rrt = RRT(env, max_iter=max_iter, step_size=step_size,
                     goal_sample_rate=0.05, adaptive_goal_bias=True,
                     final_goal_sample_rate=0.9)
        else:
            rrt = RRT(env, max_iter=max_iter, step_size=step_size,
                     goal_sample_rate=goal_sample_rate)
        
        # Time the planning
        start_time = time.time()
        path = rrt.plan(start, goal, goal_tolerance=goal_tolerance, 
                       visualize_callback=None)
        planning_time = time.time() - start_time
        
        # Determine success
        success = path is not None
        
        # Calculate path length
        if path is not None:
            path_length = sum(np.linalg.norm(path[i+1] - path[i]) 
                            for i in range(len(path)-1))
        else:
            path_length = None
        
        # Calculate iterations used
        iterations_used = len(rrt.nodes)
        
        # Store results
        results.append({
            'trial': trial + 1,
            'success': success,
            'path': path,
            'path_length': path_length,
            'planning_time': planning_time,
            'iterations_used': iterations_used,
            'max_iterations': max_iter
        })
    
    print(f"\n✓ Completed {n_trials} trials\n")
    
    # Calculate statistics
    stats = calculate_rrt_statistics(results, n_trials, max_iter, adaptive)
    
    # Print summary
    print_rrt_statistics_summary(stats)
    
    return stats, results


def calculate_rrt_statistics(results, n_trials, max_iter, adaptive):
    """Calculate statistics from RRT trial results."""
    n_success = sum(1 for r in results if r['success'])
    success_rate = (n_success / n_trials) * 100
    
    # Statistics for successful trials only
    successful_trials = [r for r in results if r['success']]
    
    if successful_trials:
        path_lengths = [r['path_length'] for r in successful_trials]
        planning_times = [r['planning_time'] for r in successful_trials]
        iterations_used = [r['iterations_used'] for r in successful_trials]
        
        stats = {
            'config': 'Adaptive RRT' if adaptive else 'Standard RRT',
            'adaptive': adaptive,
            'n_trials': n_trials,
            'n_success': n_success,
            'success_rate': success_rate,
            'max_iter': max_iter,
            
            # Path length statistics
            'avg_path_length': np.mean(path_lengths),
            'std_path_length': np.std(path_lengths),
            'min_path_length': np.min(path_lengths),
            'max_path_length': np.max(path_lengths),
            'median_path_length': np.median(path_lengths),
            
            # Planning speed statistics
            'avg_planning_time': np.mean(planning_times),
            'std_planning_time': np.std(planning_times),
            'min_planning_time': np.min(planning_times),
            'max_planning_time': np.max(planning_times),
            
            # Iterations statistics
            'avg_iterations_used': np.mean(iterations_used),
            'std_iterations_used': np.std(iterations_used),
            'min_iterations_used': np.min(iterations_used),
            'max_iterations_used': np.max(iterations_used),
            
            # Efficiency metric: iterations per meter
            'avg_iterations_per_meter': np.mean([iterations_used[i] / path_lengths[i] 
                                                 for i in range(len(successful_trials))]),
        }
    else:
        # No successful trials
        stats = {
            'config': 'Adaptive RRT' if adaptive else 'Standard RRT',
            'adaptive': adaptive,
            'n_trials': n_trials,
            'n_success': 0,
            'success_rate': 0.0,
            'max_iter': max_iter,
            'avg_path_length': None,
            'std_path_length': None,
            'min_path_length': None,
            'max_path_length': None,
            'median_path_length': None,
            'avg_planning_time': None,
            'std_planning_time': None,
            'min_planning_time': None,
            'max_planning_time': None,
            'avg_iterations_used': None,
            'std_iterations_used': None,
            'min_iterations_used': None,
            'max_iterations_used': None,
            'avg_iterations_per_meter': None,
        }
    
    return stats


def print_rrt_statistics_summary(stats):
    """Print formatted statistics summary for RRT."""
    print("="*80)
    print(f"STATISTICAL SUMMARY - {stats['config']}")
    print("="*80 + "\n")
    
    # Overall success metrics
    print("SUCCESS METRICS (Narrow Passage Navigation):")
    print("-" * 80)
    print(f"  Total Trials:       {stats['n_trials']}")
    print(f"  Successful:         {stats['n_success']} ({stats['success_rate']:.1f}%)")
    print(f"  Failed:             {stats['n_trials'] - stats['n_success']}")
    print(f"  Max Iterations:     {stats['max_iter']}")
    
    if stats['n_success'] > 0:
        print(f"\nPATH QUALITY METRICS (Path Length):")
        print("-" * 80)
        print(f"  Average Length:     {stats['avg_path_length']:.2f} ± {stats['std_path_length']:.2f} m")
        print(f"  Median Length:      {stats['median_path_length']:.2f} m")
        print(f"  Min Length:         {stats['min_path_length']:.2f} m")
        print(f"  Max Length:         {stats['max_path_length']:.2f} m")
        
        print(f"\nSOLUTION SPEED METRICS:")
        print("-" * 80)
        print(f"  Avg Planning Time:  {stats['avg_planning_time']:.4f} ± {stats['std_planning_time']:.4f} s")
        print(f"  Min Planning Time:  {stats['min_planning_time']:.4f} s")
        print(f"  Max Planning Time:  {stats['max_planning_time']:.4f} s")
        
        print(f"\nPLANNING EFFORT METRICS:")
        print("-" * 80)
        print(f"  Avg Iterations:     {stats['avg_iterations_used']:.1f} ± {stats['std_iterations_used']:.1f}")
        print(f"  Min Iterations:     {stats['min_iterations_used']:.0f}")
        print(f"  Max Iterations:     {stats['max_iterations_used']:.0f}")
        print(f"  Iterations/Meter:   {stats['avg_iterations_per_meter']:.1f} (lower = more efficient)")
        
        # Calculate utilization
        utilization = (stats['avg_iterations_used'] / stats['max_iter']) * 100
        print(f"  Avg Utilization:    {utilization:.1f}% of max iterations")
    else:
        print("\n⚠ No successful trials - algorithm failed to find paths")
        print("  Likely cause: Narrow passages too difficult or max_iter too low")
    
    print("\n" + "="*80 + "\n")


def compare_standard_vs_adaptive_rrt(n_trials=30, max_iter=2000):
    """Compare standard RRT vs adaptive RRT side-by-side."""
    print("\n" + "="*80)
    print("COMPARING STANDARD vs ADAPTIVE RRT")
    print("="*80 + "\n")
    
    # Run both configurations
    print("PART 1: Standard RRT (fixed goal_sample_rate=0.1)")
    stats_standard, results_standard = run_rrt_statistical_analysis(
        n_trials=n_trials, max_iter=max_iter, adaptive=False)
    
    print("\n" + "="*80)
    print("PART 2: Adaptive RRT (goal_sample_rate: 0.05 → 0.5)")
    print("="*80 + "\n")
    stats_adaptive, results_adaptive = run_rrt_statistical_analysis(
        n_trials=n_trials, max_iter=max_iter, adaptive=True)
    
    # Print comparison table
    print_rrt_comparison_table(stats_standard, stats_adaptive)
    
    return stats_standard, stats_adaptive, results_standard, results_adaptive


def print_rrt_comparison_table(stats_std, stats_adp):
    """Print comparison table for standard vs adaptive RRT."""
    print("\n" + "="*90)
    print("COMPARISON TABLE")
    print("="*90 + "\n")
    
    headers = ["Metric", "Standard RRT", "Adaptive RRT", "Difference"]
    
    def fmt(val):
        return f"{val:.2f}" if val is not None else "N/A"
    
    def diff(adp, std):
        if adp is None or std is None:
            return "N/A"
        d = adp - std
        return f"{d:+.2f}"
    
    def percent_diff(adp, std):
        if adp is None or std is None or std == 0:
            return "N/A"
        d = ((adp - std) / std) * 100
        return f"{d:+.1f}%"
    
    table_data = [
        ["Success Rate (%)", 
         f"{stats_std['success_rate']:.1f}%",
         f"{stats_adp['success_rate']:.1f}%",
         f"{stats_adp['success_rate'] - stats_std['success_rate']:+.1f}%"],
        
        ["Avg Path Length (m)",
         fmt(stats_std['avg_path_length']),
         fmt(stats_adp['avg_path_length']),
         diff(stats_adp['avg_path_length'], stats_std['avg_path_length'])],
        
        ["Avg Planning Time (s)",
         fmt(stats_std['avg_planning_time']),
         fmt(stats_adp['avg_planning_time']),
         diff(stats_adp['avg_planning_time'], stats_std['avg_planning_time'])],
        
        ["Avg Iterations Used",
         fmt(stats_std['avg_iterations_used']),
         fmt(stats_adp['avg_iterations_used']),
         diff(stats_adp['avg_iterations_used'], stats_std['avg_iterations_used'])],
        
        ["Iterations/Meter",
         fmt(stats_std['avg_iterations_per_meter']),
         fmt(stats_adp['avg_iterations_per_meter']),
         diff(stats_adp['avg_iterations_per_meter'], stats_std['avg_iterations_per_meter'])],
    ]
    

    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    print("\n" + "="*90)
    print("KEY FINDINGS:")
    print("="*90)
    
    # Success rate comparison
    if stats_adp['success_rate'] > stats_std['success_rate']:
        print(f"✓ NARROW PASSAGE NAVIGATION: Adaptive RRT had {stats_adp['success_rate'] - stats_std['success_rate']:.1f}% "
              f"higher success rate")
        print(f"  → Better at finding paths through difficult terrain")
    elif stats_adp['success_rate'] < stats_std['success_rate']:
        print(f"✓ NARROW PASSAGE NAVIGATION: Standard RRT had {stats_std['success_rate'] - stats_adp['success_rate']:.1f}% "
              f"higher success rate")
    else:
        print(f"  Both methods had equal success rates")
    
    # Path quality comparison
    if stats_adp['avg_path_length'] and stats_std['avg_path_length']:
        if stats_adp['avg_path_length'] < stats_std['avg_path_length']:
            diff = stats_std['avg_path_length'] - stats_adp['avg_path_length']
            print(f"✓ PATH QUALITY: Adaptive RRT found {diff:.2f}m shorter paths on average")
        elif stats_std['avg_path_length'] < stats_adp['avg_path_length']:
            diff = stats_adp['avg_path_length'] - stats_std['avg_path_length']
            print(f"✓ PATH QUALITY: Standard RRT found {diff:.2f}m shorter paths on average")
    
    # Speed comparison
    if stats_adp['avg_planning_time'] and stats_std['avg_planning_time']:
        if stats_adp['avg_planning_time'] < stats_std['avg_planning_time']:
            diff = stats_std['avg_planning_time'] - stats_adp['avg_planning_time']
            pct = (diff / stats_std['avg_planning_time']) * 100
            print(f"✓ SOLUTION SPEED: Adaptive RRT was {pct:.1f}% faster ({diff:.4f}s)")
        elif stats_std['avg_planning_time'] < stats_adp['avg_planning_time']:
            diff = stats_adp['avg_planning_time'] - stats_std['avg_planning_time']
            pct = (diff / stats_std['avg_planning_time']) * 100
            print(f"  Standard RRT was {pct:.1f}% faster ({diff:.4f}s)")
    
    # Efficiency comparison
    if stats_adp['avg_iterations_per_meter'] and stats_std['avg_iterations_per_meter']:
        if stats_adp['avg_iterations_per_meter'] < stats_std['avg_iterations_per_meter']:
            print(f"✓ EFFICIENCY: Adaptive RRT more efficient "
                  f"({stats_adp['avg_iterations_per_meter']:.1f} vs {stats_std['avg_iterations_per_meter']:.1f} iter/m)")
        else:
            print(f"  Standard RRT more efficient "
                  f"({stats_std['avg_iterations_per_meter']:.1f} vs {stats_adp['avg_iterations_per_meter']:.1f} iter/m)")
    
    print("\nINTERPRETATION:")
    print("  • Success Rate → Ability to navigate narrow passages")
    print("  • Path Length → Solution quality (shorter is better)")
    print("  • Planning Time → Speed of finding solution")
    print("  • Iterations/Meter → Computational efficiency")
    
    print("\n" + "="*90 + "\n")


def test_rrt_step_size_narrow_passages(n_trials=20):
    """Test how step_size affects narrow passage navigation."""
    print("\n" + "="*80)
    print("RRT STEP SIZE vs NARROW PASSAGE NAVIGATION")
    print("="*80 + "\n")
    
    env = Environment(seed=42)
    start = np.array([1.0, 1.0])
    goal = np.array([17.0, 17.0])
    
    step_sizes = [0.2, 0.5, 1.0, 1.5]
    max_iter = 300
    
    results = {}
    
    for step_size in step_sizes:
        print(f"Testing step_size = {step_size}...")
        trial_results = []
        
        for trial in range(n_trials):
            np.random.seed(3000 + trial)
            
            rrt = RRT(env, max_iter=max_iter, step_size=step_size, 
                     goal_sample_rate=0.1)
            
            start_time = time.time()
            path = rrt.plan(start, goal, goal_tolerance=0.5, visualize_callback=None)
            planning_time = time.time() - start_time
            
            success = path is not None
            path_length = sum(np.linalg.norm(path[i+1] - path[i]) 
                            for i in range(len(path)-1)) if success else None
            
            trial_results.append({
                'success': success,
                'path_length': path_length,
                'planning_time': planning_time,
                'iterations': len(rrt.nodes)
            })
        
        n_success = sum(1 for r in trial_results if r['success'])
        success_rate = (n_success / n_trials) * 100
        
        results[step_size] = {
            'step_size': step_size,
            'success_rate': success_rate,
            'n_success': n_success,
            'n_trials': n_trials
        }
        
        print(f"  Success rate: {success_rate:.1f}% ({n_success}/{n_trials})\n")
    
    # Print summary
    print("="*80)
    print("NARROW PASSAGE NAVIGATION RESULTS")
    print("="*80 + "\n")
    
    headers = ["Step Size", "Success Rate", "Interpretation"]
    table_data = []
    
    for step_size in step_sizes:
        res = results[step_size]
        if res['success_rate'] >= 80:
            interp = "Excellent - navigates well"
        elif res['success_rate'] >= 60:
            interp = "Good - mostly succeeds"
        elif res['success_rate'] >= 40:
            interp = "Moderate - some difficulty"
        else:
            interp = "Poor - often misses passages"
        
        table_data.append([
            f"{step_size}",
            f"{res['success_rate']:.1f}% ({res['n_success']}/{res['n_trials']})",
            interp
        ])
    
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    print("\n" + "="*80)
    print("CONCLUSION:")
    print("  • Smaller step_size → Better narrow passage navigation")
    print("  • Larger step_size → Faster but may miss tight spaces")
    print("  • Optimal step_size depends on minimum clearance in environment")
    print("="*80 + "\n")
    
    return results

def demo_rrt(env, start, goal):
    """Run RRT with visualization."""
    print("\n" + "="*60)
    print("RRT (Rapidly-exploring Random Tree) Visualization")
    print("="*60)
    print("\nHow it works:")
    print("  1. Sample random point (or goal)")
    print("  2. Find nearest tree node")
    print("  3. Extend tree toward sample")
    print("  4. Repeat until goal reached")
    print("="*60 + "\n")
    
    visualizer = RRTVisualizer(env, start, goal)
    rrt = RRT(env, max_iter=2000, step_size=0.5, 
          goal_sample_rate=0.05,  # Start low
          adaptive_goal_bias=True,
          final_goal_sample_rate=0.9)
    
    try:
        path = rrt.plan_improved_rrt(start, goal, goal_tolerance=0.5, 
                       visualize_callback=visualizer.visualize)
        
        if path is not None:
            print(f"\n✓ Path found!")
            print(f"  Length: {sum(np.linalg.norm(path[i+1] - path[i]) for i in range(len(path)-1)):.2f}m")
            print(f"  Waypoints: {len(path)}")
            
            # Show final result with path
            visualizer.ax.clear()
            visualizer.ax.set_title("RRT: Final Path", fontsize=14, fontweight='bold')
            visualizer.ax.set_xlim(env.x_min, env.x_max)
            visualizer.ax.set_ylim(env.y_min, env.y_max)
            visualizer.ax.set_aspect('equal')
            visualizer.ax.grid(True, alpha=0.3)
            
            # Draw obstacles
            for lm_pos in env.landmarks.values():
                circle = Circle((lm_pos[0], lm_pos[1]), env.obstacle_radius,
                              color='gray', alpha=0.4)
                visualizer.ax.add_patch(circle)
            
            # Draw tree (faded)
            for i, parent_idx in rrt.parents.items():
                if parent_idx is not None:
                    visualizer.ax.plot([rrt.nodes[i][0], rrt.nodes[parent_idx][0]],
                                      [rrt.nodes[i][1], rrt.nodes[parent_idx][1]],
                                      'b-', linewidth=0.5, alpha=0.2)
            
            # Draw path
            path_array = np.array(path)
            visualizer.ax.plot(path_array[:, 0], path_array[:, 1], 'g-', 
                             linewidth=3, label='Found path', zorder=6)
            visualizer.ax.plot(start[0], start[1], 'go', markersize=15, 
                             label='Start', zorder=7)
            visualizer.ax.plot(goal[0], goal[1], 'r*', markersize=20, 
                             label='Goal', zorder=7)
            visualizer.ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
            
            filename = 'improved_rrt_planning.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"✓ Plot saved as: {filename}")
            
            print("\nFinal path displayed. Press Ctrl+C or close window to continue...")
            plt.pause(3)
        else:
            print("\n✗ No path found")
            plt.pause(2)
            
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        visualizer.close()

def main():
    """Main function for RRT statistical analysis."""
    print("\n" + "="*80)
    print("RRT STATISTICAL ANALYSIS SUITE")
    print("="*80)
    
    # 1. Standard vs Adaptive comparison
    print("\n[1/2] Comparing Standard vs Adaptive RRT...")
    stats_std, stats_adp, _, _ = compare_standard_vs_adaptive_rrt(n_trials=30, max_iter=2000)
    
    # 2. Step size narrow passage test
    print("\n[2/2] Testing Step Size Effect on Narrow Passages...")
    step_size_results = test_rrt_step_size_narrow_passages(n_trials=20)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80 + "\n")
    
# def main():
#     """Run both PRM and RRT demos."""
#     print("\n" + "="*60)
#     print("MOTION PLANNING ALGORITHM VISUALIZATION")
#     print("="*60)
#     print("\nThis demo shows step-by-step visualization of:")
#     print("  • PRM (Probabilistic Roadmap)")
#     print("  • RRT (Rapidly-exploring Random Tree)")
#     print("\nWatch how each algorithm builds its structure!")
#     print("="*60)
    
#     # Setup
#     env = Environment(seed=42)
#     start = np.array([1.0, 1.0])
#     goal = np.array([17.0, 17.0])
    
#     print(f"\nPlanning from {start} to {goal}")
#     print(f"Environment: {len(env.landmarks)} circular obstacles")
    
#     # Demo RRT
#     demo_rrt(env, start, goal)
    
#     print("\n" + "="*60)
#     print("Demo complete!")
#     print("="*60 + "\n")


if __name__ == "__main__":
    import time
    main()
