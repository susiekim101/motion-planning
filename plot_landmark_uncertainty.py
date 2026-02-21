#!/usr/bin/env python3
"""
Script to plot individual landmark uncertainties over time.
Shows how uncertainty evolves for specific landmarks as they are repeatedly observed.
"""

import numpy as np
import matplotlib.pyplot as plt
from environment import Environment, RobotSimulator
from slam import EKFSLAM


def run_slam_track_landmarks():
    """
    Run SLAM simulation with loop control and track individual landmark uncertainties.
    
    Returns:
        Dictionary containing time series data for landmark uncertainties
    """
    # Setup
    env = Environment(seed=42)
    
    # Noise settings
    R = np.diag([0.1, 0.1, np.deg2rad(3.0)]) ** 2  # Process noise
    Q = np.diag([0.5, np.deg2rad(8.0)]) ** 2  # Measurement noise
    
    init_pose = [0.0, 0.0, 0.0]
    ekf = EKFSLAM(init_pose=init_pose, R=R, Q=Q)
    
    simulator = RobotSimulator(
        environment=env,
        init_pose=init_pose,
        process_noise=R,
        measurement_noise=Q
    )
    
    num_steps = 500
    
    # Initialize storage
    # We'll track uncertainty for each landmark separately
    landmark_data = {}  # landmark_id -> {'time': [], 'uncertainty': [], 'observations': []}
    time_series = []
    
    print("\nRunning LOOP control SLAM simulation...")
    print(f"Total steps: {num_steps}")
    print("Tracking individual landmark uncertainties...\n")
    
    for t in range(num_steps):
        
        # Control pattern
        if t < 100:
            v, omega = 1.0, 0.6
        elif t < 200:
            v, omega = 1.0, 0
        elif t < 300:
            v, omega = 1.0, 0.3
        elif t < 390:
            v, omega = 1.0, -0.8
        elif t < 500:
            v, omega = 1.0, 0
        
        control = (v, omega)
        
        # Simulate and update SLAM
        measurements = simulator.step(control)
        ekf.step(control, measurements, dt=simulator.dt)
        
        current_time = t * simulator.dt
        time_series.append(current_time)
        
        # Get current landmark covariances
        landmark_covs = ekf.get_landmark_covariances()
        landmark_indices = ekf.landmark_indices
        
        # For each discovered landmark, record its uncertainty
        for lm_id in landmark_indices.values():
            if lm_id not in landmark_data:
                # Initialize tracking for this landmark
                landmark_data[lm_id] = {
                    'time': [],
                    'uncertainty': [],
                    'first_seen': current_time,
                    'observations': 0
                }
            
            # Get uncertainty (trace of covariance matrix)
            if lm_id in landmark_covs:
                uncertainty = np.trace(landmark_covs[lm_id])
                landmark_data[lm_id]['time'].append(current_time)
                landmark_data[lm_id]['uncertainty'].append(uncertainty)
                
                # Check if this landmark was observed in this step
                if lm_id in [obs[0] for obs in measurements]:
                    landmark_data[lm_id]['observations'] += 1
        
        # Progress updates
        if (t + 1) % 50 == 0:
            n_landmarks = len(landmark_indices)
            print(f"  Step {t+1}/{num_steps} - Landmarks tracked: {n_landmarks}")
    
    print(f"✓ Simulation complete!")
    print(f"  Total landmarks discovered: {len(landmark_data)}\n")
    
    # Convert lists to numpy arrays
    for lm_id in landmark_data:
        landmark_data[lm_id]['time'] = np.array(landmark_data[lm_id]['time'])
        landmark_data[lm_id]['uncertainty'] = np.array(landmark_data[lm_id]['uncertainty'])
    
    return landmark_data, np.array(time_series)


def select_landmarks_to_plot(landmark_data, num_landmarks=5):
    """
    Select a subset of landmarks to plot based on when they were first observed.
    
    Args:
        landmark_data: Dictionary of landmark tracking data
        num_landmarks: Number of landmarks to select
        
    Returns:
        List of landmark IDs to plot
    """
    # Filter out landmarks with no uncertainty data
    valid_landmarks = {lm_id: data for lm_id, data in landmark_data.items() 
                      if len(data['uncertainty']) > 0}
    
    # Sort landmarks by when they were first seen
    sorted_landmarks = sorted(valid_landmarks.items(), 
                             key=lambda x: x[1]['first_seen'])
    
    # Select evenly distributed landmarks
    total = len(sorted_landmarks)
    if total <= num_landmarks:
        selected = [lm_id for lm_id, _ in sorted_landmarks]
    else:
        # Pick landmarks at regular intervals
        indices = np.linspace(0, total - 1, num_landmarks, dtype=int)
        selected = [sorted_landmarks[i][0] for i in indices]
    
    return selected


def plot_landmark_uncertainties(landmark_data, selected_landmarks):
    """
    Create plots showing uncertainty evolution for selected landmarks.
    
    Args:
        landmark_data: Dictionary of landmark tracking data
        selected_landmarks: List of landmark IDs to plot
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle('Individual Landmark Uncertainty Evolution - Loop Control', 
                 fontsize=16, fontweight='bold')
    
    # Color map for different landmarks
    colors = plt.cm.tab10(np.linspace(0, 1, len(selected_landmarks)))
    
    # Plot 1: Uncertainty over time for each landmark
    ax1.set_xlabel('Time [s]', fontsize=12)
    ax1.set_ylabel('Landmark Uncertainty (Trace of Covariance)', fontsize=12)
    ax1.set_title('Uncertainty Evolution for Individual Landmarks', 
                  fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    for i, lm_id in enumerate(selected_landmarks):
        data = landmark_data[lm_id]
        ax1.plot(data['time'], data['uncertainty'], 
                linewidth=2, color=colors[i], alpha=0.8,
                label=f'Landmark {lm_id} ({data["observations"]} obs)')
    
    ax1.legend(loc='upper right', fontsize=10)
    
    # Plot 2: Log scale version to see details better
    ax2.set_xlabel('Time [s]', fontsize=12)
    ax2.set_ylabel('Landmark Uncertainty (Log Scale)', fontsize=12)
    ax2.set_title('Uncertainty Evolution (Log Scale) - Shows Reduction More Clearly', 
                  fontsize=13, fontweight='bold')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3, which='both')
    
    for i, lm_id in enumerate(selected_landmarks):
        data = landmark_data[lm_id]
        ax2.plot(data['time'], data['uncertainty'], 
                linewidth=2, color=colors[i], alpha=0.8,
                label=f'Landmark {lm_id}')
    
    ax2.legend(loc='upper right', fontsize=10)
    
    plt.tight_layout()
    return fig


def print_landmark_statistics(landmark_data, selected_landmarks):
    """Print statistics about the selected landmarks."""
    print("\n" + "="*80)
    print("LANDMARK UNCERTAINTY STATISTICS")
    print("="*80)
    
    print("\n{:<10s} {:>12s} {:>12s} {:>12s} {:>12s} {:>10s}".format(
        "Landmark", "First Seen", "Initial Unc", "Final Unc", "Reduction %", "Obs Count"))
    print("-" * 80)
    
    for lm_id in selected_landmarks:
        data = landmark_data[lm_id]
        
        # Skip if no uncertainty data
        if len(data['uncertainty']) == 0:
            continue
            
        initial_unc = data['uncertainty'][0]
        final_unc = data['uncertainty'][-1]
        reduction = ((initial_unc - final_unc) / initial_unc) * 100
        
        print("{:<10d} {:>12.2f} {:>12.4f} {:>12.4f} {:>12.1f} {:>10d}".format(
            lm_id,
            data['first_seen'],
            initial_unc,
            final_unc,
            reduction,
            data['observations']
        ))
    
    print("="*80)
    
    # Overall statistics - filter out landmarks with no data
    valid_data = [data for data in landmark_data.values() if len(data['uncertainty']) > 0]
    
    print("\nOVERALL STATISTICS:")
    print(f"  Total landmarks discovered: {len(landmark_data)}")
    print(f"  Landmarks with uncertainty data: {len(valid_data)}")
    
    if len(valid_data) > 0:
        avg_initial = np.mean([data['uncertainty'][0] for data in valid_data])
        avg_final = np.mean([data['uncertainty'][-1] for data in valid_data])
        avg_reduction = ((avg_initial - avg_final) / avg_initial) * 100
        
        print(f"  Average initial uncertainty: {avg_initial:.4f}")
        print(f"  Average final uncertainty: {avg_final:.4f}")
        print(f"  Average reduction: {avg_reduction:.1f}%")
        
        avg_obs = np.mean([data['observations'] for data in valid_data])
        print(f"  Average observations per landmark: {avg_obs:.1f}")
    else:
        print("  No valid uncertainty data available")
    
    print("="*80 + "\n")


def main():
    """Main function to run simulation and generate plots."""
    print("\n" + "="*80)
    print("LANDMARK UNCERTAINTY TRACKING - LOOP CONTROL")
    print("="*80)
    print("\nThis script tracks how individual landmark uncertainties evolve over time")
    print("as the robot repeatedly observes them during loop traversal.")
    print("="*80)
    
    # Run simulation and collect landmark data
    landmark_data, time_series = run_slam_track_landmarks()
    
    # Select landmarks to plot (choose 5-6 representative ones)
    num_to_plot = min(6, len(landmark_data))
    selected_landmarks = select_landmarks_to_plot(landmark_data, num_to_plot)
    
    print(f"Selected {len(selected_landmarks)} landmarks to plot:")
    for lm_id in selected_landmarks:
        print(f"  - Landmark {lm_id}: first seen at t={landmark_data[lm_id]['first_seen']:.2f}s")
    
    # Print statistics
    print_landmark_statistics(landmark_data, selected_landmarks)
    
    # Generate plots
    print("Generating plots...")
    fig = plot_landmark_uncertainties(landmark_data, selected_landmarks)
    
    # Save figure
    output_file = 'landmark_uncertainty_evolution.png'
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved to: {output_file}")
    
    # Show interactive plot
    plt.show()
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
