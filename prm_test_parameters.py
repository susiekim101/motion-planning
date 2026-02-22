#!/usr/bin/env python3
import numpy as np
import time
import matplotlib.pyplot as plt
from environment import Environment
from motion_planning import PRM


def calculate_path_length(path):
    """Calculate total path length as sum of Euclidean distances."""
    if path is None or len(path) < 2:
        return None
    return sum(np.linalg.norm(path[i+1] - path[i]) for i in range(len(path)-1))


def run_prm_trial(env, start, goal, n_samples, k_neighbors=10, seed=None):
    """Run a single PRM trial and collect metrics."""
    if seed is not None:
        np.random.seed(seed)
    
    prm = PRM(env, n_samples=n_samples, k_neighbors=k_neighbors)
    
    start_time = time.time()
    path = prm.plan(start, goal, visualize_callback=None)
    planning_time = time.time() - start_time
    
    success = path is not None
    path_length = calculate_path_length(path) if success else None
    
    return {
        'success': success,
        'path_length': path_length,
        'planning_time': planning_time,
        'n_samples': n_samples
    }


def test_n_samples_parameter(n_trials=20):
    """Test PRM with different n_samples values."""
    print("\n" + "="*80)
    print("PRM N_SAMPLES PARAMETER ANALYSIS")
    print("="*80)
    print(f"\nTesting with {n_trials} trials per configuration")
    print("="*80 + "\n")
    
    # Setup
    env = Environment(seed=42)
    start = np.array([1.0, 1.0])
    goal = np.array([12.5, 17.5])
    
    # Test different n_samples values
    n_samples_values = [100, 200, 400, 800]
    k_neighbors = 10  # Fixed k_neighbors
    
    # Store results for each n_samples value
    results = {n: [] for n in n_samples_values}
    
    # Run trials for each n_samples value
    for n_samples in n_samples_values:
        print(f"Testing n_samples = {n_samples}...")
        for trial in range(n_trials):
            if (trial + 1) % 5 == 0:
                print(f"  Trial {trial+1}/{n_trials}")
            
            result = run_prm_trial(env, start, goal, 
                                  n_samples=n_samples,
                                  k_neighbors=k_neighbors,
                                  seed=200 + trial)
            results[n_samples].append(result)
        print(f"  ✓ Completed {n_trials} trials for n_samples={n_samples}\n")
    
    # Calculate statistics for each n_samples value
    statistics = {}
    for n_samples in n_samples_values:
        trials = results[n_samples]
        n_success = sum(1 for r in trials if r['success'])
        success_rate = (n_success / n_trials) * 100
        
        successful_trials = [r for r in trials if r['success']]
        
        if successful_trials:
            avg_path_length = np.mean([r['path_length'] for r in successful_trials])
            std_path_length = np.std([r['path_length'] for r in successful_trials])
            avg_planning_time = np.mean([r['planning_time'] for r in successful_trials])
            std_planning_time = np.std([r['planning_time'] for r in successful_trials])
        else:
            avg_path_length = None
            std_path_length = None
            avg_planning_time = None
            std_planning_time = None
        
        statistics[n_samples] = {
            'n_samples': n_samples,
            'n_trials': n_trials,
            'n_success': n_success,
            'success_rate': success_rate,
            'avg_path_length': avg_path_length,
            'std_path_length': std_path_length,
            'avg_planning_time': avg_planning_time,
            'std_planning_time': std_planning_time
        }
    
    return statistics, n_samples_values


def print_statistics_table(statistics, n_samples_values):
    """Print statistics in a formatted table."""
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80 + "\n")
    
    # Header
    print(f"{'N_Samples':<12} {'Success Rate':<15} {'Avg Path Length':<20} {'Avg Planning Time':<20}")
    print("-" * 80)
    
    # Data rows
    for n_samples in n_samples_values:
        stats = statistics[n_samples]
        success_str = f"{stats['success_rate']:.1f}% ({stats['n_success']}/{stats['n_trials']})"
        
        if stats['avg_path_length'] is not None:
            length_str = f"{stats['avg_path_length']:.2f} ± {stats['std_path_length']:.2f} m"
            time_str = f"{stats['avg_planning_time']:.4f} ± {stats['std_planning_time']:.4f} s"
        else:
            length_str = "N/A"
            time_str = "N/A"
        
        print(f"{n_samples:<12} {success_str:<15} {length_str:<20} {time_str:<20}")
    
    print("="*80 + "\n")


def plot_results(statistics, n_samples_values):
    """Plot metrics vs n_samples."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    
    # Extract data for plotting
    success_rates = [statistics[n]['success_rate'] for n in n_samples_values]
    avg_path_lengths = [statistics[n]['avg_path_length'] if statistics[n]['avg_path_length'] is not None else 0 
                       for n in n_samples_values]
    std_path_lengths = [statistics[n]['std_path_length'] if statistics[n]['std_path_length'] is not None else 0 
                       for n in n_samples_values]
   
    # Plot 1: Success Rate vs N_Samples
    axes[0].plot(n_samples_values, success_rates, 'bo-', linewidth=2, markersize=8)
    axes[0].set_xlabel('N_Samples', fontsize=12)
    axes[0].set_ylabel('Success Rate (%)', fontsize=12)
    axes[0].set_title('Success Rate vs N_Samples', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 105])
    
    # Add value labels on points
    for x, y in zip(n_samples_values, success_rates):
        axes[0].text(x, y + 3, f'{y:.1f}%', ha='center', fontsize=9)
    
    # Plot 2: Average Path Length vs N_Samples
    axes[1].errorbar(n_samples_values, avg_path_lengths, yerr=std_path_lengths,
                    fmt='go-', linewidth=2, markersize=8, capsize=5, capthick=2)
    axes[1].set_xlabel('N_Samples', fontsize=12)
    axes[1].set_ylabel('Average Path Length (m)', fontsize=12)
    axes[1].set_title('Average Path Length vs N_Samples', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    
    plt.tight_layout()
    
    # Save figure
    filename = 'prm_n_samples_analysis.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved as: {filename}")
    
    plt.show()


def main():
    """Main function."""
    # Number of trials per configuration
    N_TRIALS = 20
    
    # Run parameter sweep
    statistics, n_samples_values = test_n_samples_parameter(N_TRIALS)
    
    # Print results table
    print_statistics_table(statistics, n_samples_values)
    
    # Generate plots
    plot_results(statistics, n_samples_values)
    
    # Analysis summary
    print("\nANALYSIS SUMMARY:")
    print("-" * 80)
    
    # Find best configurations
    best_success = max(statistics.items(), key=lambda x: x[1]['success_rate'])
    print(f"Best success rate: {best_success[1]['success_rate']:.1f}% with n_samples={best_success[0]}")
    
    valid_stats = [(n, s) for n, s in statistics.items() if s['avg_path_length'] is not None]
    if valid_stats:
        best_length = min(valid_stats, key=lambda x: x[1]['avg_path_length'])
        print(f"Shortest average path: {best_length[1]['avg_path_length']:.2f}m with n_samples={best_length[0]}")
        
        fastest = min(valid_stats, key=lambda x: x[1]['avg_planning_time'])
        print(f"Fastest planning: {fastest[1]['avg_planning_time']:.4f}s with n_samples={fastest[0]}")
    
    print("\nOBSERVATIONS:")
    print("  • Higher n_samples generally improves success rate")
    print("  • Path length typically stabilizes with sufficient samples")
    print("  • Planning time increases linearly with n_samples")
    print("  • Trade-off: completeness vs computational efficiency")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()