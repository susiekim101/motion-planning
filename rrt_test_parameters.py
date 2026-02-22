#!/usr/bin/env python3
import numpy as np
import time
import matplotlib.pyplot as plt
from environment import Environment
from motion_planning import RRT


def calculate_path_length(path):
    """Calculate total path length as sum of Euclidean distances."""
    if path is None or len(path) < 2:
        return None
    return sum(np.linalg.norm(path[i+1] - path[i]) for i in range(len(path)-1))


def run_rrt_trial(env, start, goal, step_size, goal_sample_rate, 
                  max_iter=2000, goal_tolerance=0.5, seed=None):
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
    iterations_used = len(rrt.nodes)
    
    return {
        'success': success,
        'path_length': path_length,
        'planning_time': planning_time,
        'iterations': iterations_used,
        'step_size': step_size,
        'goal_sample_rate': goal_sample_rate
    }


def test_step_size_parameter(n_trials=20):
    """Test RRT with different step_size values."""
    print("\n" + "="*80)
    print("RRT STEP_SIZE PARAMETER ANALYSIS")
    print("="*80)
    print(f"\nTesting with {n_trials} trials per configuration")
    print("Fixed: max_iter=2000, goal_sample_rate=0.1")
    print("="*80 + "\n")
    
    # Setup
    env = Environment(seed=42)
    start = np.array([1.0, 1.0])
    goal = np.array([12.5, 17.5])
    
    # Test different step_size values
    step_sizes = [0.2, 0.5, 1.0]
    goal_sample_rate = 0.1  # Fixed
    max_iter = 2000
    
    # Store results
    results = {s: [] for s in step_sizes}
    
    # Run trials for each step_size
    for step_size in step_sizes:
        print(f"Testing step_size = {step_size}...")
        for trial in range(n_trials):
            if (trial + 1) % 5 == 0:
                print(f"  Trial {trial+1}/{n_trials}")
            
            result = run_rrt_trial(env, start, goal, 
                                  step_size=step_size,
                                  goal_sample_rate=goal_sample_rate,
                                  max_iter=max_iter,
                                  seed=300 + trial)
            results[step_size].append(result)
        print(f"  ✓ Completed {n_trials} trials for step_size={step_size}\n")
    
    # Calculate statistics
    statistics = calculate_statistics(results, step_sizes, n_trials)
    
    return statistics, step_sizes


def test_goal_sample_rate_parameter(n_trials=20):
    """Test RRT with different goal_sample_rate values."""
    print("\n" + "="*80)
    print("RRT GOAL_SAMPLE_RATE PARAMETER ANALYSIS")
    print("="*80)
    print(f"\nTesting with {n_trials} trials per configuration")
    print("Fixed: max_iter=2000, step_size=0.5")
    print("="*80 + "\n")
    
    # Setup
    env = Environment(seed=42)
    start = np.array([1.0, 1.0])
    goal = np.array([12.5, 17.5])
    
    # Test different goal_sample_rate values
    goal_sample_rates = [0.0, 0.1, 0.3, 0.7]
    step_size = 0.5  # Fixed
    max_iter = 2000
    
    # Store results
    results = {g: [] for g in goal_sample_rates}
    
    # Run trials for each goal_sample_rate
    for goal_rate in goal_sample_rates:
        print(f"Testing goal_sample_rate = {goal_rate}...")
        for trial in range(n_trials):
            if (trial + 1) % 5 == 0:
                print(f"  Trial {trial+1}/{n_trials}")
            
            result = run_rrt_trial(env, start, goal,
                                  step_size=step_size,
                                  goal_sample_rate=goal_rate,
                                  max_iter=max_iter,
                                  seed=400 + trial)
            results[goal_rate].append(result)
        print(f"  ✓ Completed {n_trials} trials for goal_sample_rate={goal_rate}\n")
    
    # Calculate statistics
    statistics = calculate_statistics(results, goal_sample_rates, n_trials)
    
    return statistics, goal_sample_rates


def calculate_statistics(results, param_values, n_trials):
    """Calculate statistics from trial results."""
    statistics = {}
    
    for param_value in param_values:
        trials = results[param_value]
        n_success = sum(1 for r in trials if r['success'])
        success_rate = (n_success / n_trials) * 100
        
        successful_trials = [r for r in trials if r['success']]
        
        if successful_trials:
            avg_path_length = np.mean([r['path_length'] for r in successful_trials])
            std_path_length = np.std([r['path_length'] for r in successful_trials])
            avg_planning_time = np.mean([r['planning_time'] for r in successful_trials])
            std_planning_time = np.std([r['planning_time'] for r in successful_trials])
            avg_iterations = np.mean([r['iterations'] for r in successful_trials])
            std_iterations = np.std([r['iterations'] for r in successful_trials])
        else:
            avg_path_length = None
            std_path_length = None
            avg_planning_time = None
            std_planning_time = None
            avg_iterations = None
            std_iterations = None
        
        statistics[param_value] = {
            'param_value': param_value,
            'n_trials': n_trials,
            'n_success': n_success,
            'success_rate': success_rate,
            'avg_path_length': avg_path_length,
            'std_path_length': std_path_length,
            'avg_planning_time': avg_planning_time,
            'std_planning_time': std_planning_time,
            'avg_iterations': avg_iterations,
            'std_iterations': std_iterations
        }
    
    return statistics


def print_statistics_table(statistics, param_values, param_name):
    """Print statistics in a formatted table."""
    print("\n" + "="*90)
    print(f"RESULTS SUMMARY - {param_name}")
    print("="*90 + "\n")
    
    # Header
    print(f"{param_name:<15} {'Success Rate':<18} {'Avg Path Length':<22} {'Avg Planning Time':<22}")
    print("-" * 90)
    
    # Data rows
    for param_value in param_values:
        stats = statistics[param_value]
        success_str = f"{stats['success_rate']:.1f}% ({stats['n_success']}/{stats['n_trials']})"
        
        if stats['avg_path_length'] is not None:
            length_str = f"{stats['avg_path_length']:.2f} ± {stats['std_path_length']:.2f} m"
            time_str = f"{stats['avg_planning_time']:.4f} ± {stats['std_planning_time']:.4f} s"
        else:
            length_str = "N/A"
            time_str = "N/A"
        
        print(f"{param_value:<15} {success_str:<18} {length_str:<22} {time_str:<22}")
    
    print("="*90 + "\n")


def plot_step_size_results(statistics, step_sizes):
    """Plot metrics vs step_size."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Extract data
    success_rates = [statistics[s]['success_rate'] for s in step_sizes]
    avg_path_lengths = [statistics[s]['avg_path_length'] if statistics[s]['avg_path_length'] is not None else 0 
                       for s in step_sizes]
    std_path_lengths = [statistics[s]['std_path_length'] if statistics[s]['std_path_length'] is not None else 0 
                       for s in step_sizes]
    avg_planning_times = [statistics[s]['avg_planning_time'] if statistics[s]['avg_planning_time'] is not None else 0 
                         for s in step_sizes]
    std_planning_times = [statistics[s]['std_planning_time'] if statistics[s]['std_planning_time'] is not None else 0 
                         for s in step_sizes]
    
    # Plot 1: Success Rate
    axes[0].plot(step_sizes, success_rates, 'bo-', linewidth=2, markersize=8)
    axes[0].set_xlabel('Step Size', fontsize=12)
    axes[0].set_ylabel('Success Rate (%)', fontsize=12)
    axes[0].set_title('Success Rate vs Step Size', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 105])
    for x, y in zip(step_sizes, success_rates):
        axes[0].text(x, y + 3, f'{y:.1f}%', ha='center', fontsize=9)
    
    # Plot 2: Path Length
    axes[1].errorbar(step_sizes, avg_path_lengths, yerr=std_path_lengths,
                    fmt='go-', linewidth=2, markersize=8, capsize=5, capthick=2)
    axes[1].set_xlabel('Step Size', fontsize=12)
    axes[1].set_ylabel('Average Path Length (m)', fontsize=12)
    axes[1].set_title('Average Path Length vs Step Size', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Planning Time
    axes[2].errorbar(step_sizes, avg_planning_times, yerr=std_planning_times,
                    fmt='ro-', linewidth=2, markersize=8, capsize=5, capthick=2)
    axes[2].set_xlabel('Step Size', fontsize=12)
    axes[2].set_ylabel('Average Planning Time (s)', fontsize=12)
    axes[2].set_title('Average Planning Time vs Step Size', fontsize=14, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    filename = 'rrt_step_size_analysis.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved as: {filename}")
    
    plt.show()


def plot_goal_sample_rate_results(statistics, goal_rates):
    """Plot metrics vs goal_sample_rate."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    
    # Extract data
    success_rates = [statistics[g]['success_rate'] for g in goal_rates]
    avg_path_lengths = [statistics[g]['avg_path_length'] if statistics[g]['avg_path_length'] is not None else 0 
                       for g in goal_rates]
    std_path_lengths = [statistics[g]['std_path_length'] if statistics[g]['std_path_length'] is not None else 0 
                       for g in goal_rates]
    
    # Plot 1: Success Rate
    axes[0].plot(goal_rates, success_rates, 'bo-', linewidth=2, markersize=8)
    axes[0].set_xlabel('Goal Sample Rate', fontsize=12)
    axes[0].set_ylabel('Success Rate (%)', fontsize=12)
    axes[0].set_title('Success Rate vs Goal Sample Rate', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 105])
    for x, y in zip(goal_rates, success_rates):
        axes[0].text(x, y + 3, f'{y:.1f}%', ha='center', fontsize=9)
    
    # Plot 2: Path Length
    axes[1].errorbar(goal_rates, avg_path_lengths, yerr=std_path_lengths,
                    fmt='go-', linewidth=2, markersize=8, capsize=5, capthick=2)
    axes[1].set_xlabel('Goal Sample Rate', fontsize=12)
    axes[1].set_ylabel('Average Path Length (m)', fontsize=12)
    axes[1].set_title('Average Path Length vs Goal Sample Rate', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    filename = 'rrt_goal_sample_rate_analysis.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved as: {filename}")
    
    plt.show()


def print_analysis_summary(statistics, param_values, param_name):
    """Print analysis summary."""
    print(f"\nANALYSIS SUMMARY - {param_name}:")
    print("-" * 80)
    
    # Find best configurations
    best_success = max(statistics.items(), key=lambda x: x[1]['success_rate'])
    print(f"Best success rate: {best_success[1]['success_rate']:.1f}% with {param_name}={best_success[0]}")
    
    valid_stats = [(p, s) for p, s in statistics.items() if s['avg_path_length'] is not None]
    if valid_stats:
        best_length = min(valid_stats, key=lambda x: x[1]['avg_path_length'])
        print(f"Shortest average path: {best_length[1]['avg_path_length']:.2f}m with {param_name}={best_length[0]}")
        
        fastest = min(valid_stats, key=lambda x: x[1]['avg_planning_time'])
        print(f"Fastest planning: {fastest[1]['avg_planning_time']:.4f}s with {param_name}={fastest[0]}")
        
        if 'avg_iterations' in best_success[1] and best_success[1]['avg_iterations'] is not None:
            print(f"Average iterations at best success: {best_success[1]['avg_iterations']:.1f}")


def main():
    """Main function."""
    N_TRIALS = 20
    
    # Test step_size parameter
    print("\n" + "="*80)
    print("PART 1: STEP_SIZE PARAMETER SWEEP")
    print("="*80)
    
    step_stats, step_sizes = test_step_size_parameter(N_TRIALS)
    print_statistics_table(step_stats, step_sizes, "Step Size")
    print_analysis_summary(step_stats, step_sizes, "step_size")
    
    print("\nOBSERVATIONS (Step Size):")
    print("  • Smaller step_size creates denser trees but slower exploration")
    print("  • Larger step_size explores faster but may miss narrow passages")
    print("  • Optimal step_size balances exploration speed and path quality")
    
    plot_step_size_results(step_stats, step_sizes)
    
    # Test goal_sample_rate parameter
    print("\n" + "="*80)
    print("PART 2: GOAL_SAMPLE_RATE PARAMETER SWEEP")
    print("="*80)
    
    goal_stats, goal_rates = test_goal_sample_rate_parameter(N_TRIALS)
    print_statistics_table(goal_stats, goal_rates, "Goal Sample Rate")
    print_analysis_summary(goal_stats, goal_rates, "goal_sample_rate")
    
    print("\nOBSERVATIONS (Goal Sample Rate):")
    print("  • 0.0 = pure exploration (no goal bias)")
    print("  • Higher rates bias toward goal (faster but may get stuck in local minima)")
    print("  • Moderate rates (0.1-0.3) often provide good balance")
    print("  • Very high rates (0.7) may reduce exploration diversity")
    
    plot_goal_sample_rate_results(goal_stats, goal_rates)
    
    print("\n" + "="*80)
    print("OVERALL RECOMMENDATIONS:")
    print("="*80)
    print("  • step_size: Choose based on environment complexity and obstacle density")
    print("  • goal_sample_rate: Use 0.1-0.3 for balanced exploration/exploitation")
    print("  • Trade-offs exist between success rate, path quality, and computation time")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()