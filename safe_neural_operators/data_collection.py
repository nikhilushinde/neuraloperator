"""
File to collect HJR training data using the quad2d environment with different spatialy varying disturbance functions
"""

import torch
import jax.numpy as jnp
from jax import Array
import numpy as np
import matplotlib.pyplot as plt

import sys 
sys.path.append("/home/jingpei/Documents/arclab/L4DC25_project")

from deepreach.dynamics import dynamics 
from deepreach.dynamics import dynamics_hjr
from deepreach.utils.comparisons import GroundTruthHJSolution
import os

def solve_and_save_hjr(
        dynamics_model, 
        num_time_steps, 
        grid_resolution,
        save_index, 
        save_folder, 
        save_grid=False): 
    """
    Function to solve HJR solution to the given system and then save the solution to the specified path

    Save the input: x,y to disturbance magnitude 
    Save the output: infinite time value function on the whole state space grid 
    """
    # 1. Create ground truth HJ solution object - and solve
    ground_truth_hj_solution = GroundTruthHJSolution(
        hj_dynamics=dynamics_model, 
        solve=True, 
        num_time_steps=num_time_steps, 
        grid_resolution=grid_resolution, # if None defaults to (51, 51, 51, 51))
    )

    # Save the grid states 
    if save_grid: 
        grid_states = np.array(ground_truth_hj_solution.grid.states)
        grid_states_path = os.path.join(save_folder, f"{save_index:03d}_grid_states.pt")
        torch.save(grid_states, grid_states_path)#, _use_new_zipfile_serialization=True)
        print(f"Saved grid states to {grid_states_path}")

    # 2. Save the datapoint to the specified path: Save the value function as a compressed torch tensor

    # Save Input: Disturbance function - only varies over x,y so save it as a 2D grid / image - eval over the first 2 grid dimensions 
    disturbance_inputs = dynamics_model.disturbance_function(ground_truth_hj_solution.grid.states[:, :, 0, 0, ].reshape(-1, 4))
    disturbance_inputs_reshaped = disturbance_inputs.reshape((ground_truth_hj_solution.grid_resolution[0],ground_truth_hj_solution.grid_resolution[1], 4))  # default shape (51, 51, 4)
    disturbance_inputs_tensor = torch.from_numpy(np.array(disturbance_inputs_reshaped))

    # Save disturbance magnitude as 2D image 
    plt.figure(figsize=(10, 10))
    plt.imshow(disturbance_inputs_reshaped[:, :, 0], cmap='viridis', origin='lower')
    plt.colorbar(label='Disturbance Magnitude')
    plt.savefig(os.path.join(save_folder, f"{save_index:03d}_disturbance_inputs.png"))
    plt.close()

    # Save Output: Value function in 4D grid space at the final timestep (approx infinite time)
    value_function = ground_truth_hj_solution.value_functions[-1, :, :, :, :]  # shape (51, 51, 51, 51) - the last time step 
    torch_value_function_outputs = torch.from_numpy(np.array(value_function))

    # Save value function slice as 2D image for visualziation 
    plt.figure(figsize=(10, 10))
    xvel_slice = grid_resolution[2]//2
    yvel_slice = grid_resolution[3]//2
    xvel = ground_truth_hj_solution.grid.states[0, 0, xvel_slice, 0][2]
    yvel = ground_truth_hj_solution.grid.states[0, 0, 0, yvel_slice][3]
    plt.imshow(torch_value_function_outputs[:, :, xvel_slice, yvel_slice].numpy(), cmap='viridis')
    plt.colorbar(label='Value Function')
    plt.title(f"Value Function Slice at xvel={xvel}, yvel={yvel}")
    plt.savefig(os.path.join(save_folder, f"{save_index:03d}_value_function.png"))
    plt.close()

    # Save using torch.save with zipfile compression for space efficiency
    save_input_path = os.path.join(save_folder, f"{save_index:03d}_input.pt")
    save_output_path = os.path.join(save_folder, f"{save_index:03d}_output.pt")

    os.makedirs(save_folder, exist_ok=True)  # Ensure the folder exists
    torch.save(disturbance_inputs_tensor, save_input_path) #, _use_new_zipfile_serialization=True)
    torch.save(torch_value_function_outputs, save_output_path) #, _use_new_zipfile_serialization=True)
    return 

def load_hjr_solution(load_path, load_index):
    """
    Function to load the saved dynamics solution from the specified path
    """
    """
    Function to load the saved input and output tensors for a given save index from the specified folder.
    Returns a tuple: (input_tensor, output_tensor)
    """
    input_path = os.path.join(load_path, f"{load_index:03d}_input.pt")
    output_path = os.path.join(load_path, f"{load_index:03d}_output.pt")

    input_tensor = torch.load(input_path) #, map_location='cpu')
    output_tensor = torch.load(output_path) #, map_location='cpu')
    return input_tensor, output_tensor

def load_grid_states(load_path, load_index): 
    """
    Function to load the grid states from the specified path
    """
    grid_states_path = os.path.join(load_path, f"{load_index:03d}_grid_states.pt")
    grid_states = torch.load(grid_states_path) #, map_location='cpu')
    return grid_states

def get_disturbance_function_sincos(max_magnitude, phase_multiplier, phase_shift, dim='x', use_sin=True): 
    """
    Get a disturbance function that returns the disturbance magnitude based on the state input.

    Disturbances are either sin or cos((x + offset) * phase_multiplier) * max_magnitude - either in x or y direction

    args: 
        max_magnitude: Maximum disturbance magnitude
        phase_multiplier: Multiplier for the phase of the disturbance
        phase_shift: Phase shift for the disturbance
        dim: 'x' or 'y' to indicate the direction of the disturbance
        use_sin: If True, use sine function; if False, use cosine function
    """
    if dim == 'x':
        state_dim_idx = 0 
    elif dim == 'y':
        state_dim_idx = 1
    else:
        raise ValueError("Dimension must be 'x' or 'y'.")
    

    def disturbance_function(state):
        # PyTorch branch
        if isinstance(state, torch.Tensor):
            if len(state.shape) == 1:
                state = state.unsqueeze(0)

            if use_sin: 
                mag = torch.sin(state[:, state_dim_idx] * phase_multiplier + phase_shift) * max_magnitude
            else:
                mag = torch.cos(state[:, state_dim_idx] * phase_multiplier + phase_shift) * max_magnitude
            mag = torch.abs(mag)
            disturbances = mag.unsqueeze(1).repeat(1, 4)  # shape (B, 4)

        # JAX branch
        elif isinstance(state, (jnp.ndarray, Array, np.ndarray)):
            if len(state.shape) == 1:
                state = jnp.expand_dims(state, 0)
            if use_sin: 
                mag = jnp.sin(state[:, state_dim_idx] * phase_multiplier + phase_shift) * max_magnitude
            else:
                mag = jnp.cos(state[:, state_dim_idx] * phase_multiplier + phase_shift) * max_magnitude
            mag = jnp.abs(mag)
            disturbances = jnp.repeat(mag[:, None], 4, axis=1)  # shape (B, 4)

        else:
            raise TypeError(f"Unsupported input type: {type(state)}")

        if disturbances.shape[0] == 1: 
            return disturbances[0]
        else:
            return disturbances

    return disturbance_function

def get_dynamics_model_given_disturbance_fn(disturbance_function, tMin, tMax): 
    gravity=9.81 
    max_angle=0.2
    min_thrust=6 
    max_thrust=13 


    tMin = 0.0
    tMax = 3.0 

    set_mode='avoid'
    boundary_cfg_num = 2
    problem_type = "avoid"

    # SV Deepreach System 
    system_sv = dynamics.Quad2DAttitude_Consolidated_SpaceVarying(
        gravity=gravity, 
        max_angle=max_angle,
        min_thrust=min_thrust,
        max_thrust=max_thrust,
        disturbance_function=disturbance_function,
        set_mode=set_mode,
        boundary_cfg_num = boundary_cfg_num ,
        problem_type = problem_type
    )



    # SV HJR System 
    system_sv_hjr = dynamics_hjr.Quad2DAttitude_Consolidated_SpaceVarying(
        torch_dynamics=system_sv, 
        gravity=gravity, 
        max_angle=max_angle,
        min_thrust=min_thrust,
        max_thrust=max_thrust,
        disturbance_function=disturbance_function,
        boundary_cfg_num = boundary_cfg_num ,
        problem_type = problem_type,
        tMin=tMin, 
        tMax=tMax, 
    )
    return system_sv_hjr

def create_hjr_disturbance_dataset(num_datapoints, 
                                   save_folder, 
                                   disturbance_variation_type="sincos", 
                                   grid_resolution=(51, 51, 51, 51)): 
    """
    Create a dataset mapping disturbance magnitude to the HJR solution value function. 
    """

    min_max_magnitude = 0.25 
    max_max_magnitude = 0.75 

    min_phase_multiplier = 0.05
    max_phase_multiplier = 5 

    min_phase_shift = 0
    max_phase_shift = np.pi / 2

    tMin = 0.0 
    tMax = 3.0 

    for num in range(num_datapoints): 
        curr_max_magnitude = np.random.uniform(min_max_magnitude, max_max_magnitude)
        curr_phase_multiplier = np.random.uniform(min_phase_multiplier, max_phase_multiplier)
        curr_phase_shift = np.random.uniform(min_phase_shift, max_phase_shift)

        curr_dim = np.random.choice(['x', 'y'])
        curr_use_sin = np.random.choice([True, False])
        
        # 1. Create a new disturbance function 
        if disturbance_variation_type == "sincos":
            disturbance_function = get_disturbance_function_sincos(
                max_magnitude=curr_max_magnitude, 
                phase_multiplier=curr_phase_multiplier, 
                phase_shift=curr_phase_shift, 
                dim=curr_dim,  # or 'y' depending on the direction you want
                use_sin=curr_use_sin  # or False for cosine
            )
        else: 
            raise ValueError(f"Disturbance variation type {disturbance_variation_type} is not supported. Please implement it in the future.")

        # 2. Create a new dynamics model with the disturbance function
        dynamics_model = get_dynamics_model_given_disturbance_fn(
            disturbance_function=disturbance_function, 
            tMin=tMin, 
            tMax=tMax
        )
        

        # 3. Solve and save the HJR solution using the dynamics model 
        if num == 0: 
            save_grid = True
        else: 
            save_grid = False 
        solve_and_save_hjr(
            dynamics_model=dynamics_model, 
            num_time_steps=5,  # Adjust as needed
            grid_resolution=grid_resolution,  # Default resolution
            save_index=num, 
            save_folder=save_folder, 
            save_grid=save_grid
        )
    print("Done")


if __name__ == "__main__":
    # Example usage
    num_datapoints = 100
    save_folder = "/media/jingpei/DATA/fno_gp_data"
    disturbance_variation_type = "sincos"
    grid_resolution = (41, 41, 41, 41) #(51, 51, 51, 51)  # Default resolution

    create_hjr_disturbance_dataset(num_datapoints, save_folder, disturbance_variation_type, grid_resolution)