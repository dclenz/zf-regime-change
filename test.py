#!/usr/bin/env python3
"""
Demonstration script for the Spatial Statistics module
This script shows how to use the various functions in a typical spatial analysis workflow
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

# Import our spatial statistics module
# (In practice, you would: from spatial_stats import SpatialStats)
# For this demo, assuming the SpatialStats class is available

def create_synthetic_dataset():
    """Create a synthetic spatial dataset for demonstration"""
    print("Creating synthetic spatial dataset...")
    
    # Grid parameters
    n1, n2 = 30, 30
    side = 5  # Block size
    
    # Create observation locations (irregular sampling)
    np.random.seed(123)
    total_points = n1 * n2
    # Sample 60% of the grid points
    n_obs = int(0.6 * total_points)
    labels = np.sort(np.random.choice(total_points, n_obs, replace=False))
    
    return n1, n2, side, labels

def analyze_spatial_data():
    """Complete spatial analysis workflow"""
    
    # Create dataset
    n1, n2, side, labels = create_synthetic_dataset()
    print(f"Grid: {n1}x{n2}, Observations: {len(labels)}, Block size: {side}×{side}")
    
    # Step 1: Create block structure
    print("\n1. Creating spatial block structure...")
    weights, neigh, full_grid = SpatialStats.base_sq(n1, n2, side, labels)
    n_blocks = int(np.max(weights))
    print(f"   Created {n_blocks} blocks")
    print(f"   Neighbor pairs: {len(neigh)}")
    
    # Step 2: Generate synthetic spatial process
    print("\n2. Generating synthetic spatial data...")
    true_pars = {
        'log_variance': np.log(2.0),
        'log_range': np.log(5.0), 
        'log_smoothness': np.log(1.2),
        'log_nugget': np.log(0.05)
    }
    pars_vec = list(true_pars.values())
    
    # Generate the spatial field
    spatial_field = SpatialStats.stat_qm_gen(n1, n2, pars_vec)
    data = spatial_field.flatten()[labels]
    
    print(f"   True parameters: σ²={np.exp(true_pars['log_variance']):.2f}, "
          f"φ={np.exp(true_pars['log_range']):.2f}, "
          f"ν={np.exp(true_pars['log_smoothness']):.2f}")
    print(f"   Data range: [{np.min(data):.3f}, {np.max(data):.3f}]")
    
    # Step 3: Block-wise parameter estimation
    print("\n3. Estimating parameters for each block...")
    track = SpatialStats.base_estimate(data, n1, n2, weights, labels)
    
    # Display results
    print("\n   Block Parameter Estimates:")
    print("   " + "="*70)
    print("   Block | Observations | σ²    | φ     | ν     | -LogLik")
    print("   " + "-"*70)
    
    for i in range(len(track)):
        block_id = int(track[i, 0])
        n_obs_block = np.sum(weights == block_id)
        if track[i, 5] < 9999:  # Valid estimate
            sigma2 = np.exp(track[i, 2])
            phi = np.exp(track[i, 3])
            nu = np.exp(track[i, 4])
            neg_loglik = track[i, 5]
            print(f"   {block_id:5d} | {n_obs_block:12d} | {sigma2:5.2f} | {phi:5.2f} | {nu:5.2f} | {neg_loglik:7.2f}")
        else:
            print(f"   {block_id:5d} | {n_obs_block:12d} | insufficient data")
    
    # Step 4: Analyze parameter variability
    print("\n4. Analyzing spatial parameter variation...")
    valid_blocks = track[track[:, 5] < 9999]
    
    if len(valid_blocks) > 0:
        param_names = ['σ²', 'φ', 'ν']
        print("\n   Parameter Summary (across valid blocks):")
        print("   " + "="*50)
        
        for i, name in enumerate(param_names):
            values = np.exp(valid_blocks[:, 2+i])
            print(f"   {name}: Mean={np.mean(values):.3f}, "
                  f"Std={np.std(values):.3f}, "
                  f"Range=[{np.min(values):.3f}, {np.max(values):.3f}]")
    
    return track, data, spatial_field, weights, labels, n1, n2

def test_utility_functions():
    """Test various utility functions"""
    print("\n5. Testing utility functions...")
    
    # Test transformation functions
    x = 0.5
    logit_x = SpatialStats.logit(x)
    expit_logit_x = SpatialStats.expit(logit_x)
    print(f"   logit({x}) = {logit_x:.3f}")
    print(f"   expit(logit({x})) = {expit_logit_x:.3f} (should equal {x})")
    
    # Test parameter projection
    extreme_pars = [50, -50, 20, -10]  # Extreme values
    projected = SpatialStats.project(extreme_pars)
    print(f"   Original parameters: {extreme_pars}")
    print(f"   Projected parameters: {projected}")
    
    # Test distance calculation
    coords = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    distances = SpatialStats.rdist(coords)
    print(f"   Distance matrix shape: {distances.shape}")
    print(f"   Max distance: {np.max(distances):.3f}")
    
    # Test random matrix generation
    Uj = SpatialStats.Uj_gen(100, j=5)
    print(f"   Random matrix Uj: {Uj.shape}, unique values: {np.unique(Uj)}")

def create_visualization(track, data, spatial_field, weights, labels, n1, n2):
    """Create visualizations of the spatial analysis"""
    print("\n6. Creating visualizations...")
    
    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Spatial Analysis Results', fontsize=16)
        
        # Plot 1: Original spatial field
        ax1 = axes[0, 0]
        field_img = ax1.imshow(spatial_field, cmap='viridis', origin='lower')
        ax1.set_title('True Spatial Field')
        ax1.set_xlabel('X coordinate')
        ax1.set_ylabel('Y coordinate')
        plt.colorbar(field_img, ax=ax1)
        
        # Plot 2: Observation locations
        ax2 = axes[0, 1]
        y_coords, x_coords = np.unravel_index(labels, (n1, n2))
        scatter = ax2.scatter(x_coords, y_coords, c=data, cmap='viridis', s=30)
        ax2.set_title('Observed Data Points')
        ax2.set_xlabel('X coordinate')
        ax2.set_ylabel('Y coordinate')
        ax2.set_xlim(0, n2-1)
        ax2.set_ylim(0, n1-1)
        plt.colorbar(scatter, ax=ax2)
        
        # Plot 3: Block structure
        ax3 = axes[1, 0]
        # Create a full grid showing block assignments
        full_weights = np.zeros(n1 * n2)
        full_weights[labels] = weights
        weight_field = full_weights.reshape(n1, n2)
        weight_img = ax3.imshow(weight_field, cmap='tab20', origin='lower')
        ax3.set_title('Spatial Blocks')
        ax3.set_xlabel('X coordinate')
        ax3.set_ylabel('Y coordinate')
        
        # Plot 4: Parameter estimates by block
        ax4 = axes[1, 1]
        valid_mask = track[:, 5] < 9999
        if np.any(valid_mask):
            block_ids = track[valid_mask, 0]
            variances = np.exp(track[valid_mask, 2])
            ax4.bar(block_ids, variances)
            ax4.set_title('Estimated Variance by Block')
            ax4.set_xlabel('Block ID')
            ax4.set_ylabel('σ²')
        else:
            ax4.text(0.5, 0.5, 'No valid estimates', ha='center', va='center')
            ax4.set_title('Parameter Estimates')
        
        plt.tight_layout()
        plt.show()
        
        print("   Visualizations created successfully!")
        
    except Exception as e:
        print(f"   Warning: Could not create visualizations: {e}")

def main():
    """Main demonstration function"""
    print("Spatial Statistics Analysis Demo")
    print("=" * 50)
    
    # Run complete spatial analysis
    track, data, spatial_field, weights, labels, n1, n2 = analyze_spatial_data()
    
    # Test utility functions
    test_utility_functions()
    
    # Create visualizations
    create_visualization(track, data, spatial_field, weights, labels, n1, n2)
    
    # Summary statistics
    print("\n7. Final Summary:")
    print("   " + "="*40)
    valid_estimates = np.sum(track[:, 5] < 9999)
    total_blocks = len(track)
    print(f"   Total blocks created: {total_blocks}")
    print(f"   Blocks with valid estimates: {valid_estimates}")
    print(f"   Success rate: {100*valid_estimates/total_blocks:.1f}%")
    
    if valid_estimates > 0:
        avg_negloglik = np.mean(track[track[:, 5] < 9999, 5])
        print(f"   Average negative log-likelihood: {avg_negloglik:.2f}")
    
    print("\n   Analysis completed successfully!")
    print("   This demonstrates the core functionality of the spatial statistics module.")

if __name__ == "__main__":
    main()