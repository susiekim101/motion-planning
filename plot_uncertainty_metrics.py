#!/usr/bin/env python3
"""
Script to generate plots of SLAM uncertainty metrics over time.
Tracks pose error, pose uncertainty, and average landmark uncertainty.
"""

import numpy as np
import matplotlib.pyplot as plt
from environment import Environment, RobotSimulator
from slam import EKFSLAM


def run_slam_with_metrics(control_type='greedy'):
    """
    Run SLAM simulation and collect metrics over time.
    
    Args:
        control_type: 'greedy' or 'loop' control strategy
        
    Returns:
        Dictionary containing time series data for all metrics
    """
    # Setup
    env = Environment(seed=42)
    
    # Noise settings
    R = np.diag([0.3, 0.3, np.deg2rad(5.0)]) ** 2  # Process noise
    Q = np.diag([0.8, np.deg2rad(10.0)]) ** 2  # Measurement noise
    
    init_pose = [0.0, 0.0, 0.0]
    ekf = EKFSLAM(init_pose=init_pose, R=R, Q=Q)
    
    simulator = RobotSimulator(
        environment=env,
        init_pose=init_pose,
        process_noise=R,
        measurement_noise=Q
    )
    
    num_steps = 360
    
    # Initialize metric storage
    metrics = {
        'time': [],
        'pose_error': [],
        'pose_uncertainty': [],
        'avg_landmark_uncertainty': [],
        'num_landmarks': []
    }
    
    print(f"\nRunning {control_type.upper()} SLAM simulation...")
    print(f"Total steps: {num_steps}\n")
    
    for t in range(num_steps):
        # Select control based on strategy
        if control_type == 'greedy':
            # Greedy forward control pattern
            # Mostly drive forward with occasional gentle turns to explore
            if t < num_steps//6:
                # Initial straight drive
                v, omega = 1.0, 0.0
            elif t < num_steps//3:
                # Gentle right turn while moving forward
                v, omega = 1.0, 0.05
            elif t < num_steps//2:
                # Long forward drive
                v, omega = 1.0, 0.0
            elif t < 2*num_steps//3:
                # Gentle left turn while moving forward
                v, omega = 1.0, 0.15
            elif t < 5*num_steps//6:
                # Forward drive
                v, omega = 1.0, -0.05
            else:
                # Final gentle arc
                v, omega = 1.0, -0.05
                
        elif control_type == 'loop':
            # Rectangular loop control pattern
            rectangle_steps = num_steps // 2
            t_in_rect = t % rectangle_steps
            side_length = 40
            turn_length = 5
            segment_length = side_length + turn_length
            pos_in_segment = t_in_rect % segment_length
            
            if pos_in_segment < side_length:
                v, omega = 1.5, 0.0
            else:
                v, omega = 0.5, 0.31
        else:
            raise ValueError(f"Unknown control type: {control_type}")
        
        control = (v, omega)
        
        # Simulate and update SLAM
        measurements = simulator.step(control)
        ekf.step(control, measurements, dt=simulator.dt)
        
        # Collect metrics
        true_pose = simulator.true_pose
        est_pose = ekf.get_current_pose()
        
        # Pose error (Euclidean distance between true and estimated position)
        pose_error = np.linalg.norm([true_pose[0] - est_pose[0], 
                                     true_pose[1] - est_pose[1]])
        
        # Pose uncertainty (sqrt of trace of pose covariance)
        pose_cov = ekf.get_pose_covariance()
        pose_uncertainty = np.sqrt(np.trace(pose_cov))
        
        # Average landmark uncertainty (mean trace of landmark covariances)
        landmark_covs = ekf.get_landmark_covariances()
        if len(landmark_covs) > 0:
            avg_landmark_unc = np.mean([np.trace(cov) for cov in landmark_covs.values()])
        else:
            avg_landmark_unc = 0.0
        
        # Store metrics
        metrics['time'].append(t * simulator.dt)
        metrics['pose_error'].append(pose_error)
        metrics['pose_uncertainty'].append(pose_uncertainty)
        metrics['avg_landmark_uncertainty'].append(avg_landmark_unc)
        metrics['num_landmarks'].append(len(ekf.landmark_indices))
        
        # Progress updates
        if (t + 1) % 50 == 0:
            print(f"  Step {t+1}/{num_steps} - "
                  f"Pose error: {pose_error:.3f}m, "
                  f"Landmarks: {len(ekf.landmark_indices)}")
    
    print(f"✓ Simulation complete!\n")
    
    # Convert lists to numpy arrays
    for key in metrics:
        metrics[key] = np.array(metrics[key])
    
    return metrics


def plot_metrics(metrics, control_type):
    """
    Create plots of metrics over time for a single control strategy.
    
    Args:
        metrics: Metrics dictionary from simulation
        control_type: Name of the control strategy ('greedy' or 'loop')
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'SLAM Performance Metrics - {control_type.capitalize()} Control', 
                 fontsize=16, fontweight='bold')
    
    # Plot 1: Pose Error
    ax = axes[0, 0]
    ax.plot(metrics['time'], metrics['pose_error'], 
            'b-', linewidth=2, alpha=0.8)
    ax.set_xlabel('Time [s]', fontsize=11)
    ax.set_ylabel('Pose Error [m]', fontsize=11)
    ax.set_title('Robot Pose Error Over Time', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Pose Uncertainty
    ax = axes[0, 1]
    ax.plot(metrics['time'], metrics['pose_uncertainty'], 
            'g-', linewidth=2, alpha=0.8)
    ax.set_xlabel('Time [s]', fontsize=11)
    ax.set_ylabel('Pose Uncertainty [m]', fontsize=11)
    ax.set_title('Robot Pose Uncertainty Over Time', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Average Landmark Uncertainty
    ax = axes[1, 0]
    ax.plot(metrics['time'], metrics['avg_landmark_uncertainty'], 
            'r-', linewidth=2, alpha=0.8)
    ax.set_xlabel('Time [s]', fontsize=11)
    ax.set_ylabel('Avg Landmark Uncertainty', fontsize=11)
    ax.set_title('Average Landmark Uncertainty Over Time', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Number of Landmarks Discovered
    ax = axes[1, 1]
    ax.plot(metrics['time'], metrics['num_landmarks'], 
            'm-', linewidth=2, alpha=0.8)
    ax.set_xlabel('Time [s]', fontsize=11)
    ax.set_ylabel('Number of Landmarks', fontsize=11)
    ax.set_title('Landmarks Discovered Over Time', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def print_summary_statistics(metrics):
    """Print summary statistics for the simulation."""
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    
    print("\n{:<40s} {:>15s}".format("Metric", "Value"))
    print("-" * 70)
    
    # Final pose error
    print("{:<40s} {:>15.4f}".format(
        "Final Pose Error [m]",
        metrics['pose_error'][-1]
    ))
    
    # Average pose error
    print("{:<40s} {:>15.4f}".format(
        "Average Pose Error [m]",
        np.mean(metrics['pose_error'])
    ))
    
    # Std dev pose error
    print("{:<40s} {:>15.4f}".format(
        "Std Dev Pose Error [m]",
        np.std(metrics['pose_error'])
    ))
    
    # Final pose uncertainty
    print("{:<40s} {:>15.4f}".format(
        "Final Pose Uncertainty [m]",
        metrics['pose_uncertainty'][-1]
    ))
    
    # Average pose uncertainty
    print("{:<40s} {:>15.4f}".format(
        "Average Pose Uncertainty [m]",
        np.mean(metrics['pose_uncertainty'])
    ))
    
    # Final landmark uncertainty
    print("{:<40s} {:>15.4f}".format(
        "Final Avg Landmark Uncertainty",
        metrics['avg_landmark_uncertainty'][-1]
    ))
    
    # Average landmark uncertainty
    # Filter out zeros (when no landmarks discovered yet)
    lm_unc = metrics['avg_landmark_uncertainty']
    lm_unc = lm_unc[lm_unc > 0]
    
    print("{:<40s} {:>15.4f}".format(
        "Average Landmark Uncertainty",
        np.mean(lm_unc)
    ))
    
    # Final number of landmarks
    print("{:<40s} {:>15d}".format(
        "Final Landmarks Discovered",
        int(metrics['num_landmarks'][-1])
    ))
    
    print("="*70 + "\n")


def main():
    """Main function to run simulation and generate plots."""
    # Choose control strategy: 'greedy' or 'loop'
    CONTROL_TYPE = 'greedy'  # Change this to 'loop' to test loop control
    
    print("\n" + "="*70)
    print("SLAM UNCERTAINTY METRICS ANALYSIS")
    print("="*70)
    print(f"\nControl strategy: {CONTROL_TYPE.upper()}")
    print("Metrics tracked:")
    print("  • Pose Error: Euclidean distance between true and estimated position")
    print("  • Pose Uncertainty: sqrt(trace(pose_covariance))")
    print("  • Avg Landmark Uncertainty: mean(trace(landmark_covariances))")
    print("="*70)
    
    # Run simulation
    metrics = run_slam_with_metrics(control_type=CONTROL_TYPE)
    
    # Print summary statistics
    print_summary_statistics(metrics)
    
    # Generate plots
    print("Generating plots...")
    fig = plot_metrics(metrics, CONTROL_TYPE)
    
    # Save figure
    output_file = f'slam_metrics_{CONTROL_TYPE}.png'
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved to: {output_file}")
    
    # Show interactive plot
    plt.show()
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
