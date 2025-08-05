"""
Spatial Statistics Module - Python conversion of R spatial statistics functions
"""

import numpy as np
from scipy import linalg, optimize, spatial
from scipy.special import kv, gamma
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
from itertools import product
import warnings

class SpatialStats:
    """Main class for spatial statistics operations"""
    
    def __init__(self):
        pass
    
    @staticmethod
    def expit(p):
        """Logistic function (inverse logit)"""
        return np.exp(p) / (1 + np.exp(p))
    
    @staticmethod
    def logit(p):
        """Logit function"""
        return np.log(p) - np.log(1 - p)
    
    @staticmethod
    def sin2(w1, w2):
        """Sine squared function for spectral methods"""
        return np.sin(w1 / 2)**2 + np.sin(w2 / 2)**2
    
    @staticmethod
    def w_mat_gen(n1, n2):
        """Generate frequency matrix for spectral methods"""
        w1 = 2 * np.pi * np.arange(n1) / n1
        w2 = 2 * np.pi * np.arange(n2) / n2
        W1, W2 = np.meshgrid(w1, w2, indexing='ij')
        return SpatialStats.sin2(W1, W2)
    
    @staticmethod
    def quasi_matern(pars, w_mat):
        """Quasi-Matern covariance function"""
        sigma = np.exp(pars[0])
        rho = np.exp(pars[1]) * 2
        nu = np.exp(pars[2])
        t = SpatialStats.expit(pars[3])
        
        f = (1 - t) * (1 + rho**2 * w_mat)**(-nu - 1)
        sumf = np.sum(f)
        return sigma * f / sumf + t / (w_mat.shape[0] * w_mat.shape[1])
    
    @staticmethod
    def matern_cov(d, phi, range_param, nu):
        """Matérn covariance function"""
        d = np.asarray(d)
        if d.ndim == 0:
            d = np.array([d])
        
        # Handle zero distances
        result = np.zeros_like(d, dtype=float)
        non_zero = d > 0
        
        if np.any(non_zero):
            kappa = np.sqrt(2 * nu) * d[non_zero] / range_param
            # Use modified Bessel function of the second kind
            result[non_zero] = phi * (2**(1-nu) / gamma(nu)) * (kappa**nu) * kv(nu, kappa)
        
        # Set diagonal elements (d=0) to phi
        result[~non_zero] = phi
        
        return result
    
    @staticmethod
    def neg_log_like(pars, y, d):
        """Negative log-likelihood for Matérn model"""
        # Bound parameters to prevent numerical issues
        pars = np.where(np.exp(pars) > 28, np.log(28), pars)
        
        # Create covariance matrix
        n = len(y)
        Sigma = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                Sigma[i, j] = SpatialStats.matern_cov(d[i, j], 
                                                    np.exp(pars[0]), 
                                                    np.exp(pars[1]), 
                                                    np.exp(pars[2]))
        
        # Add small nugget for numerical stability
        Sigma += np.eye(n) * 1e-5
        
        try:
            # Cholesky decomposition
            L = linalg.cholesky(Sigma, lower=True)
            # Log determinant
            log_det = 2 * np.sum(np.log(np.diag(L)))
            # Solve for y'Sigma^-1 y
            alpha = linalg.solve_triangular(L, y, lower=True)
            quad_form = np.sum(alpha**2)
            
            return log_det + quad_form
        
        except linalg.LinAlgError:
            return 1e10  # Return large value if not positive definite
    
    @staticmethod
    def rdist(coords):
        """Calculate distance matrix"""
        return euclidean_distances(coords)
    
    @staticmethod
    def project(pars):
        """Project parameters to valid ranges"""
        pars = np.array(pars)
        n = len(pars)
        
        # Apply bounds based on parameter position
        for i in range(n):
            if i % 4 == 0:  # Variance parameters
                pars[i] = np.clip(pars[i], np.log(0.0001), np.log(100))
            elif i % 4 == 1:  # Range parameters
                pars[i] = np.clip(pars[i], np.log(0.05), np.log(90))
            elif i % 4 == 2:  # Smoothness parameters
                pars[i] = np.clip(pars[i], np.log(0.01), np.log(10))
        
        return pars
    
    @staticmethod
    def base_estimate(data, n1, n2, weights, labels):
        """Estimate parameters for each block"""
        n_b = int(np.max(weights))
        
        # Create grid
        grid_coords = list(product(range(1, n1+1), range(1, n2+1)))
        grid = np.array([(j, i) for i, j in grid_coords])  # Swap to match R
        grid = grid[labels]
        
        # Initialize results
        param = np.zeros((n_b, 3))
        like = np.zeros(n_b)
        Y = np.array(data).flatten()
        
        for i in range(1, n_b + 1):
            # Get data for this block
            block_mask = weights == i
            if np.sum(block_mask) == 0:
                continue
                
            block_coords = grid[block_mask]
            y_block = Y[block_mask]
            
            if len(y_block) < 3:
                param[i-1, :] = [0, 0, 0]
                like[i-1] = 10000
            else:
                d = SpatialStats.rdist(block_coords)
                
                try:
                    result = optimize.minimize(SpatialStats.neg_log_like, 
                                             x0=[0, 0, 0],
                                             args=(y_block, d),
                                             method='BFGS')
                    param[i-1, :] = result.x
                    like[i-1] = result.fun
                except:
                    param[i-1, :] = [0, 0, 0]
                    like[i-1] = 10000
        
        # Create tracking matrix
        track = np.column_stack([
            np.arange(1, n_b + 1),  # block
            np.arange(1, n_b + 1),  # group
            param,                   # parameters
            like                     # likelihood
        ])
        
        return track
    
    @staticmethod
    def base_sq(n1, n2, side, labels):
        """Create square block structure"""
        # Number of blocks
        n_b = (n1 * n2) // (side**2)
        n_r = n1 // side
        n_c = n_b // n_r
        n = n1 * n2
        
        # Create full grid
        full = np.arange(1, n + 1).reshape(n_r * side, n_c * side)
        
        # Assign block labels
        k = 0
        for i in range(n_c):
            for j in range(n_r):
                k += 1
                row_start = side * j
                row_end = side * (j + 1)
                col_start = side * i
                col_end = side * (i + 1)
                full[row_start:row_end, col_start:col_end] = k
        
        weights = full.flatten()[labels]
        
        # Get neighbors
        neigh = []
        for i in range(1, n_b + 1):
            # Up neighbor
            if i - n_r > 0:
                neigh.append([i, i - n_r])
            # Down neighbor
            if i + n_r <= n_b:
                neigh.append([i, i + n_r])
            # Left neighbor
            if i % n_r != 1:
                neigh.append([i, i - 1])
            # Right neighbor
            if i % n_r != 0:
                neigh.append([i, i + 1])
        
        neigh = np.array(neigh)
        
        # Remove duplicates
        neigh_sorted = np.sort(neigh, axis=1)
        unique_mask = np.unique(neigh_sorted, axis=0, return_index=True)[1]
        neigh = neigh[unique_mask]
        
        # Handle isolated points (blocks with only 1 observation)
        unique_weights, counts = np.unique(weights, return_counts=True)
        alone = unique_weights[counts == 1]
        
        if len(alone) > 0:
            # Create grid for finding nearest neighbors
            grid_coords = list(product(range(1, n1+1), range(1, n2+1)))
            grid = np.array([(j, i) for i, j in grid_coords])
            grid = grid[labels]
            
            for one in alone:
                # Find neighbors of this isolated block
                ns = neigh[(neigh[:, 0] == one) | (neigh[:, 1] == one)]
                if len(ns) == 0:
                    continue
                    
                cans = ns[ns != one]
                if len(cans) == 0:
                    continue
                    
                # Find closest neighbor
                pt = grid[weights == one][0]
                can_pts = grid[np.isin(weights, cans)]
                
                if len(can_pts) > 0:
                    distances = np.sum((can_pts - pt)**2, axis=1)
                    closest_idx = np.argmin(distances)
                    which_weights = weights[np.isin(weights, cans)]
                    new_lab = which_weights[closest_idx]
                    
                    # Reassign weights
                    weights[weights == one] = new_lab
                    neigh[neigh == one] = new_lab
        
        # Remove duplicates again after reassignment
        neigh_sorted = np.sort(neigh, axis=1)
        unique_mask = np.unique(neigh_sorted, axis=0, return_index=True)[1]
        neigh = neigh[unique_mask]
        
        # Remove empty box labels
        valid_weights = np.unique(weights)
        neigh = neigh[np.isin(neigh[:, 0], valid_weights) & 
                     np.isin(neigh[:, 1], valid_weights)]
        
        # Relabel weights to be consecutive
        uw = np.unique(weights)
        weights2 = np.zeros_like(weights)
        neigh2 = np.zeros_like(neigh)
        
        for i, w in enumerate(uw):
            weights2[weights == w] = i + 1
            neigh2[neigh == w] = i + 1
        
        return weights2, neigh2, full.flatten()
    
    @staticmethod
    def em(x, n1, n2, labels):
        """Embed vector into full grid"""
        x0 = np.zeros(n1 * n2)
        x0[labels] = x
        return x0
    
    @staticmethod
    def cirem_mult(n1, n2, pars, x, w_mat, factor=5/4):
        """Circular embedding multiplication"""
        m1 = int(np.ceil(n1 * factor))
        m2 = int(np.ceil(n2 * factor))
        
        x0 = np.zeros((m1, m2))
        xmat2 = x.reshape(n1, n2)
        x0[:n1, :n2] = xmat2
        
        d = SpatialStats.quasi_matern(pars, w_mat)
        
        # FFT multiplication
        Rm = np.fft.ifft2(d * np.fft.fft2(x0))
        sol = np.real(Rm[:n1, :n2])
        
        return sol.flatten()
    
    @staticmethod
    def sqexp_kern(r, nvec):
        """Squared exponential kernel for smoothing"""
        n1, n2 = nvec
        
        v1 = np.concatenate([np.arange(0, np.ceil((n1-1)/2) + 1),
                            np.arange(np.floor((n1-1)/2), 0, -1)]) / n1
        v2 = np.concatenate([np.arange(0, np.ceil((n2-1)/2) + 1),
                            np.arange(np.floor((n2-1)/2), 0, -1)]) / n2
        
        V1, V2 = np.meshgrid(v1, v2, indexing='ij')
        kern = np.exp(-(V1**2 + V2**2) / r**2)
        kern = kern / np.sum(kern)
        
        return kern
    
    @staticmethod
    def smoother(mat, kern):
        """Apply smoothing using spectral methods"""
        nvec = mat.shape
        mat_smooth = (1/np.prod(nvec)) * np.fft.ifft2(np.fft.fft2(mat) * np.fft.fft2(kern))
        return np.real(mat_smooth)
    
    @staticmethod
    def stat_qm_gen(n1, n2, pars, factor=5/4):
        """Generate stationary quasi-Matern random field"""
        m1 = int(np.ceil(n1 * factor))
        m2 = int(np.ceil(n2 * factor))
        
        w_mat = SpatialStats.w_mat_gen(m1, m2)
        qm = SpatialStats.quasi_matern(pars, w_mat)
        
        # Generate random field
        rand = np.fft.fft2(np.random.normal(0, 1, (m1, m2)))
        dat = np.fft.ifft2(np.sqrt(qm) * rand) / np.sqrt(m1 * m2)
        d = np.real(dat[:n1, :n2])
        
        return d
    
    @staticmethod
    def Uj_gen(n, j=5):
        """Generate random binary matrix for stochastic approximation"""
        Uj = np.zeros((n, j))
        for i in range(j):
            U = np.random.binomial(1, 0.5, n)
            U[U == 0] = -1
            Uj[:, i] = U
        return Uj

# Example usage and demonstration
def main():
    """Demonstrate the spatial statistics functions"""
    print("Spatial Statistics Module Demo")
    print("=" * 40)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Grid dimensions
    n1, n2 = 20, 20
    side = 4  # Block size
    
    # Create labels (assume we observe every other point)
    all_indices = np.arange(n1 * n2)
    labels = all_indices[::2]  # Every other point
    print(f"Grid size: {n1} x {n2}")
    print(f"Number of observed locations: {len(labels)}")
    
    # Generate block structure
    weights, neigh, full_grid = SpatialStats.base_sq(n1, n2, side, labels)
    print(f"Number of blocks: {int(np.max(weights))}")
    
    # Generate synthetic spatial data
    true_pars = [np.log(1.0), np.log(3.0), np.log(1.5), np.log(0.1)]
    spatial_field = SpatialStats.stat_qm_gen(n1, n2, true_pars)
    data = spatial_field.flatten()[labels]
    print(f"Generated {len(data)} data points")
    
    # Estimate parameters for each block
    print("\nEstimating parameters for each block...")
    track = SpatialStats.base_estimate(data, n1, n2, weights, labels)
    
    print("\nBlock Parameter Estimates:")
    print("Block | Group | Var(log) | Range(log) | Smooth(log) | NegLogLik")
    print("-" * 65)
    for i in range(len(track)):
        print(f"{int(track[i,0]):5d} | {int(track[i,1]):5d} | "
              f"{track[i,2]:8.3f} | {track[i,3]:10.3f} | "
              f"{track[i,4]:11.3f} | {track[i,5]:9.3f}")
    
    # Test other functions
    print(f"\nTrue parameters (log scale): {true_pars}")
    print(f"Expit of 0.5: {SpatialStats.expit(0.5):.3f}")
    print(f"Logit of 0.5: {SpatialStats.logit(0.5):.3f}")
    
    # Test smoothing kernel
    kern = SpatialStats.sqexp_kern(0.1, (10, 10))
    print(f"Smoothing kernel shape: {kern.shape}")
    print(f"Kernel sum: {np.sum(kern):.6f}")
    
    # Generate random vectors for stochastic methods
    Uj = SpatialStats.Uj_gen(len(data), j=3)
    print(f"Random matrix Uj shape: {Uj.shape}")
    
    print("\nDemo completed successfully!")

if __name__ == "__main__":
    main()