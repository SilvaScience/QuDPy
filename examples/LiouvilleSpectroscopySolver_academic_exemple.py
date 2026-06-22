import numpy as np
from your_module import LiouvilleSpectroscopySolver, SpectroscopyPlotter

# Step 1: Define the Physics (User-defined matrices)
d = 3 # Example 3-level system
H_site = np.array([[0, 0, 0], 
                   [0, 1.5, 0.1], 
                   [0, 0.1, 2.8]]) 

J_site = np.array([[0, 1, 0], 
                   [1, 0, 1], 
                   [0, 1, 0]]) # Dipole transitions

# Dissipation: jump from state 1 -> 0 with rate 0.02
C_decay = np.array([[0, 1, 0], 
                    [0, 0, 0], 
                    [0, 0, 0]])
collapse_ops = [(C_decay, 0.02)]

# Step 2: Configure the Solver
params = {
    "Eta": 0.05,
    "T": 0.025, # k_B T
    "backend": "auto",
    "cache_resolvents": True
}

solver = LiouvilleSpectroscopySolver(params)

# Step 3: Load the Model
# This normalizes inputs, diagonalizes H, and builds superoperators
solver.feed_model(
    H_model=H_site, 
    interaction_op_array=J_site, 
    c_ops_raw=collapse_ops, 
    interaction_type="dipole"
)

# Step 4: Execute the 2D Sweep
# Define frequency axes and population time
w_list = np.linspace(1.0, 3.5, 100)
tau2 = 0.0 # Population time

# Generates Rephasing, Non-rephasing, and Absorptive data
spectra = solver.generate_2D_spectra(w_list, tau2)

# Step 5: Visualization
plotter = SpectroscopyPlotter()
plotter.plot_spectrum(
    S3_rephasing=spectra["rephasing"], 
    S3_nonrephasing=spectra["unrephasing"], 
    levels=60
)