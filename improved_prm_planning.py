#!/usr/bin/env python3
"""
Animated visualization of PRM and RRT planning algorithms.
Shows step-by-step how each algorithm builds its data structure and finds a path.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from environment import Environment
from motion_planning import PRM, RRT


class PRMVisualizer:
    """Visualize PRM algorithm step-by-step."""
    
    def __init__(self, env, start, goal):
        self.env = env
        self.start = start
        self.goal = goal
        self.fig, self.ax = plt.subplots(figsize=(12, 10))
        plt.ion()
        
    def visualize(self, phase, progress, total, nodes, edges, current_node, visited):
        """Callback for PRM visualization."""
        self.ax.clear()
        
        # Set title based on phase
        if phase == 'sampling':
            title = f"PRM: Sampling ({progress}/{total} nodes)"
        elif phase == 'connecting':
            title = f"PRM: Connecting Neighbors ({progress}/{total} nodes)"
        else:  # searching
            title = f"PRM: A* Search (iteration {progress})"
        
        self.ax.set_title(title, fontsize=14, fontweight='bold')
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
        
        # Draw sampled nodes
        if nodes:
            nodes_array = np.array(nodes)
            self.ax.plot(nodes_array[:, 0], nodes_array[:, 1], 'c.', 
                        markersize=4, alpha=0.5, label='Sampled nodes')
        
        # Draw edges (if in connecting or searching phase)
        if phase in ['connecting', 'searching'] and edges:
            for i in range(len(nodes)):
                for j in edges[i]:
                    if j < len(nodes):  # Safety check
                        self.ax.plot([nodes[i][0], nodes[j][0]], 
                                   [nodes[i][1], nodes[j][1]],
                                   'c-', linewidth=0.3, alpha=0.2)
        
        # Highlight current node during connecting
        if phase == 'connecting' and current_node is not None and current_node < len(nodes):
            node = nodes[current_node]
            self.ax.plot(node[0], node[1], 'yo', markersize=8, 
                        label='Current node', zorder=5)
            # Highlight its edges
            for j in edges[current_node]:
                if j < len(nodes):
                    self.ax.plot([node[0], nodes[j][0]], [node[1], nodes[j][1]],
                               'y-', linewidth=1.5, alpha=0.6, zorder=4)
        
        # Highlight visited nodes during search
        if phase == 'searching' and visited:
            for idx in visited:
                if idx < len(nodes):
                    self.ax.plot(nodes[idx][0], nodes[idx][1], 'mo', 
                               markersize=4, alpha=0.6)
        
        # Highlight current search node
        if phase == 'searching' and current_node is not None and current_node < len(nodes):
            node = nodes[current_node]
            self.ax.plot(node[0], node[1], 'ro', markersize=10,
                        label='Current search', zorder=6)
        
        # Draw start and goal
        self.ax.plot(self.start[0], self.start[1], 'go', markersize=15, 
                    label='Start', zorder=7)
        self.ax.plot(self.goal[0], self.goal[1], 'r*', markersize=20, 
                    label='Goal', zorder=7)
        
        # Add info box
        if phase == 'sampling':
            info = f"Sampling collision-free nodes\nProgress: {progress}/{total}"
        elif phase == 'connecting':
            edge_count = sum(len(v) for v in edges.values()) // 2
            info = f"Connecting k-nearest neighbors\nEdges created: {edge_count}"
        else:
            info = f"A* search in progress\nNodes explored: {len(visited) if visited else 0}"
        
        self.ax.text(0.02, 0.98, info, transform=self.ax.transAxes,
                    fontsize=10, verticalalignment='top', family='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
        
        self.ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
        plt.pause(0.01)
    
    def close(self):
        plt.ioff()
        plt.close()

def run_prm_statistical_analysis(n_trials=30, time_limit=1.0, improved=True):
    print("\n" + "="*80)
    print(f"PRM STATISTICAL ANALYSIS (improved={improved})")
    print("="*80)
    print(f"Running {n_trials} trials...")
    print(f"Time limit: {time_limit}s")
    print("="*80 + "\n")
    
    # Setup
    env = Environment(seed=42)
    start = np.array([1.0, 1.0])
    goal = np.array([17.0, 17.0])
    
    # PRM parameters
    n_samples = 300
    k_neighbors = 10
    
    # Store results
    results = []
    
    for trial in range(n_trials):
        if (trial + 1) % 10 == 0:
            print(f"  Trial {trial+1}/{n_trials}")
        
        # Set seed for reproducibility
        np.random.seed(1000 + trial)
        
        # Create new PRM instance
        prm = PRM(env, n_samples=n_samples, k_neighbors=k_neighbors)
        
        # Time the planning
        start_time = time.time()
        path = prm.plan(start, goal, improved=improved, visualize_callback=None)
        planning_time = time.time() - start_time
        
        # Determine success
        success = (path is not None) and (planning_time <= time_limit)
        
        # Calculate path length
        if path is not None:
            path_length = sum(np.linalg.norm(path[i+1] - path[i]) 
                            for i in range(len(path)-1))
        else:
            path_length = None
        
        # Store results
        results.append({
            'trial': trial + 1,
            'success': success,
            'path': path,
            'path_length': path_length,
            'planning_time': planning_time,
            'nodes_sampled': len(prm.nodes),
            'edges_created': sum(len(v) for v in prm.edges.values()) // 2,
            'timed_out': (path is not None) and (planning_time > time_limit)
        })
    
    print(f"\n✓ Completed {n_trials} trials\n")
    
    # Calculate statistics
    stats = calculate_prm_statistics(results, n_trials, time_limit, improved)
    
    # Print summary
    print_prm_statistics_summary(stats)
    
    return stats, results

def calculate_prm_statistics(results, n_trials, time_limit, improved):
    """Calculate statistics from PRM trial results."""
    n_success = sum(1 for r in results if r['success'])
    n_timeouts = sum(1 for r in results if r.get('timed_out', False))
    success_rate = (n_success / n_trials) * 100
    
    # Statistics for successful trials only
    successful_trials = [r for r in results if r['success']]
    
    if successful_trials:
        path_lengths = [r['path_length'] for r in successful_trials]
        planning_times = [r['planning_time'] for r in successful_trials]
        nodes_sampled = [r['nodes_sampled'] for r in successful_trials]
        edges_created = [r['edges_created'] for r in successful_trials]
        
        stats = {
            'config': 'Improved PRM' if improved else 'Standard PRM',
            'improved': improved,
            'n_trials': n_trials,
            'n_success': n_success,
            'n_timeouts': n_timeouts,
            'success_rate': success_rate,
            'time_limit': time_limit,
            
            # Path length statistics
            'avg_path_length': np.mean(path_lengths),
            'std_path_length': np.std(path_lengths),
            'min_path_length': np.min(path_lengths),
            'max_path_length': np.max(path_lengths),
            'median_path_length': np.median(path_lengths),
            
            # Planning time statistics
            'avg_planning_time': np.mean(planning_times),
            'std_planning_time': np.std(planning_times),
            'min_planning_time': np.min(planning_times),
            'max_planning_time': np.max(planning_times),
            
            # Planning effort statistics
            'avg_nodes_sampled': np.mean(nodes_sampled),
            'std_nodes_sampled': np.std(nodes_sampled),
            'min_nodes_sampled': np.min(nodes_sampled),
            'max_nodes_sampled': np.max(nodes_sampled),
            
            'avg_edges_created': np.mean(edges_created),
            'std_edges_created': np.std(edges_created),
        }
    else:
        # No successful trials
        stats = {
            'config': 'Improved PRM' if improved else 'Standard PRM',
            'improved': improved,
            'n_trials': n_trials,
            'n_success': 0,
            'n_timeouts': n_timeouts,
            'success_rate': 0.0,
            'time_limit': time_limit,
            'avg_path_length': None,
            'std_path_length': None,
            'min_path_length': None,
            'max_path_length': None,
            'median_path_length': None,
            'avg_planning_time': None,
            'std_planning_time': None,
            'min_planning_time': None,
            'max_planning_time': None,
            'avg_nodes_sampled': None,
            'std_nodes_sampled': None,
            'min_nodes_sampled': None,
            'max_nodes_sampled': None,
            'avg_edges_created': None,
            'std_edges_created': None,
        }
    
    return stats

def print_prm_statistics_summary(stats):
    """Print formatted statistics summary."""
    print("="*80)
    print(f"STATISTICAL SUMMARY - {stats['config']}")
    print("="*80 + "\n")
    
    # Overall success metrics
    print("SUCCESS METRICS:")
    print("-" * 80)
    print(f"  Total Trials:       {stats['n_trials']}")
    print(f"  Successful:         {stats['n_success']} ({stats['success_rate']:.1f}%)")
    print(f"  Failed:             {stats['n_trials'] - stats['n_success']}")
    print(f"  Timeouts:           {stats['n_timeouts']} (found path but exceeded {stats['time_limit']}s)")
    
    if stats['n_success'] > 0:
        print(f"\nPATH QUALITY METRICS:")
        print("-" * 80)
        print(f"  Average Length:     {stats['avg_path_length']:.2f} ± {stats['std_path_length']:.2f} m")
        print(f"  Median Length:      {stats['median_path_length']:.2f} m")
        print(f"  Min Length:         {stats['min_path_length']:.2f} m")
        print(f"  Max Length:         {stats['max_path_length']:.2f} m")
        
        print(f"\nPLANNING EFFORT METRICS:")
        print("-" * 80)
        print(f"  Avg Nodes Sampled:  {stats['avg_nodes_sampled']:.1f} ± {stats['std_nodes_sampled']:.1f}")
        print(f"  Min Nodes Sampled:  {stats['min_nodes_sampled']:.0f}")
        print(f"  Max Nodes Sampled:  {stats['max_nodes_sampled']:.0f}")
        print(f"  Avg Edges Created:  {stats['avg_edges_created']:.1f} ± {stats['std_edges_created']:.1f}")
        
        print(f"\nPLANNING TIME METRICS:")
        print("-" * 80)
        print(f"  Average Time:       {stats['avg_planning_time']:.4f} ± {stats['std_planning_time']:.4f} s")
        print(f"  Min Time:           {stats['min_planning_time']:.4f} s")
        print(f"  Max Time:           {stats['max_planning_time']:.4f} s")
    else:
        print("\n⚠ No successful trials - algorithm failed to find paths within time limit")
    
    print("\n" + "="*80 + "\n")

def compare_standard_vs_improved(n_trials=30, time_limit=1.0):
    """Compare standard PRM vs improved PRM side-by-side."""
    print("\n" + "="*80)
    print("COMPARING STANDARD vs IMPROVED PRM")
    print("="*80 + "\n")
    
    # Run both configurations
    print("PART 1: Standard PRM (improved=False)")
    stats_standard, results_standard = run_prm_statistical_analysis(
        n_trials=n_trials, time_limit=time_limit, improved=False)
    
    print("\n" + "="*80)
    print("PART 2: Improved PRM (improved=True)")
    print("="*80 + "\n")
    stats_improved, results_improved = run_prm_statistical_analysis(
        n_trials=n_trials, time_limit=time_limit, improved=True)
    
    # Print comparison table
    print_comparison_table_simple(stats_standard, stats_improved)
    
    return stats_standard, stats_improved, results_standard, results_improved

def print_comparison_table_simple(stats_std, stats_imp):
    """Print simple comparison table."""
    print("\n" + "="*80)
    print("COMPARISON TABLE")
    print("="*80 + "\n")
    
    headers = ["Metric", "Standard PRM", "Improved PRM", "Difference"]
    
    def fmt(val):
        return f"{val:.2f}" if val is not None else "N/A"
    
    def diff(imp, std):
        if imp is None or std is None:
            return "N/A"
        d = imp - std
        return f"{d:+.2f}"
    
    table_data = [
        ["Success Rate (%)", 
         f"{stats_std['success_rate']:.1f}%",
         f"{stats_imp['success_rate']:.1f}%",
         f"{stats_imp['success_rate'] - stats_std['success_rate']:+.1f}%"],
        
        ["Avg Path Length (m)",
         fmt(stats_std['avg_path_length']),
         fmt(stats_imp['avg_path_length']),
         diff(stats_imp['avg_path_length'], stats_std['avg_path_length'])],
        
        ["Avg Nodes Sampled",
         fmt(stats_std['avg_nodes_sampled']),
         fmt(stats_imp['avg_nodes_sampled']),
         diff(stats_imp['avg_nodes_sampled'], stats_std['avg_nodes_sampled'])],
        
        ["Avg Planning Time (s)",
         fmt(stats_std['avg_planning_time']),
         fmt(stats_imp['avg_planning_time']),
         diff(stats_imp['avg_planning_time'], stats_std['avg_planning_time'])],
    ]
    
    from tabulate import tabulate
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    print("\n" + "="*80)
    print("KEY FINDINGS:")
    print("="*80)
    
    if stats_imp['success_rate'] > stats_std['success_rate']:
        print(f"✓ Improved sampling increased success rate by "
              f"{stats_imp['success_rate'] - stats_std['success_rate']:.1f}%")
    
    if stats_imp['avg_path_length'] and stats_std['avg_path_length']:
        if stats_imp['avg_path_length'] < stats_std['avg_path_length']:
            print(f"✓ Improved sampling found shorter paths on average")
        else:
            print(f"  Standard sampling found shorter paths on average")
    
    print("\n" + "="*80 + "\n")

def demo_prm(env, start, goal):
    """Run PRM with visualization."""
    print("\n" + "="*60)
    print("PRM (Probabilistic Roadmap) Visualization")
    print("="*60)
    print("\nPhases:")
    print("  1. Sampling: Generate random collision-free nodes")
    print("  2. Connecting: Link k-nearest neighbors")
    print("  3. Searching: A* search on the roadmap")
    print("="*60 + "\n")
    
    visualizer = PRMVisualizer(env, start, goal)
    prm = PRM(env, n_samples=300, k_neighbors=10)
    
    try:
        path = prm.plan(start, goal, improved=False, visualize_callback=visualizer.visualize)
        
        if path is not None:
            print(f"\n✓ Path found!")
            print(f"  Length: {sum(np.linalg.norm(path[i+1] - path[i]) for i in range(len(path)-1)):.2f}m")
            print(f"  Waypoints: {len(path)}")
            
            # Show final result with path
            visualizer.visualize('final', 0, 0, prm.nodes, prm.edges, None, None)
            path_array = np.array(path)
            visualizer.ax.plot(path_array[:, 0], path_array[:, 1], 'b-', 
                             linewidth=3, label='Found path', zorder=6)
            visualizer.ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
            
            print("\nFinal roadmap displayed. Press Ctrl+C or close window to continue...")
            
            filename = 'regular_prm_planning.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"✓ Plot saved as: {filename}")
            
            plt.pause(3)
        else:
            print("\n✗ No path found")
            plt.pause(2)
            
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        visualizer.close()

def main():
    print("\n" + "="*60)
    print("MOTION PLANNING ALGORITHM VISUALIZATION")
    print("="*60)
    print("\nThis demo shows step-by-step visualization of:")
    print("  • PRM (Probabilistic Roadmap)")
    print("\nWatch how each algorithm builds its structure!")
    print("="*60)
    
    # Setup
    env = Environment(seed=42)
    start = np.array([1.0, 1.0])
    goal = np.array([17.0, 17.0])
    
    print(f"\nPlanning from {start} to {goal}")
    print(f"Environment: {len(env.landmarks)} circular obstacles")
    
    # Demo PRM with visualization (single run)
    demo_prm(env, start, goal)
    
    # Ask user if they want to run statistical analysis
    print("\n" + "="*60)
    print("Would you like to run statistical analysis?")
    print("This will run 30 trials comparing standard vs improved PRM")
    response = input("Run analysis? (y/n): ").strip().lower()
    
    if response == 'y':
        stats_std, stats_imp, _, _ = compare_standard_vs_improved(n_trials=30, time_limit=1.0)

    print("\n" + "="*60)
    print("Demo complete!")
    print("="*60 + "\n")

# def main():
#     print("\n" + "="*60)
#     print("MOTION PLANNING ALGORITHM VISUALIZATION")
#     print("="*60)
#     print("\nThis demo shows step-by-step visualization of:")
#     print("  • PRM (Probabilistic Roadmap)")
#     print("\nWatch how each algorithm builds its structure!")
#     print("="*60)
    
#     # Setup
#     env = Environment(seed=42)
#     start = np.array([1.0, 1.0])
#     goal = np.array([17.0, 17.0])
    
#     print(f"\nPlanning from {start} to {goal}")
#     print(f"Environment: {len(env.landmarks)} circular obstacles")
    
#     # Demo PRM
#     demo_prm(env, start, goal)

#     print("\n" + "="*60)
#     print("Demo complete!")
#     print("="*60 + "\n")


if __name__ == "__main__":
    import time
    main()
