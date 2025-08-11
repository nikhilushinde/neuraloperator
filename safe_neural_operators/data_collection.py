"""
File to collect HJR training data using the quad2d environment with different spatialy varying disturbance functions
"""

import torch
import jax.numpy as jnp
from jax import Array
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm 

import jax 
from jax import vmap
from jax import lax

import os
import gc  # Add garbage collection

import sys 
sys.path.append("/home/jingpei/Documents/arclab/neuraloperator")
sys.path.append("/home/jingpei/Documents/arclab/neuraloperator/safe_neural_operators")

from safe_neural_operators.gp import GPWrapper
from safe_neural_operators.gt_hjr_solution import GroundTruthHJSolution
# Replaced #from deepreach.utils.comparisons import GroundTruthHJSolution

sys.path.append("/home/jingpei/Documents/arclab/L4DC25_project")

from deepreach.dynamics import dynamics 
from deepreach.dynamics import dynamics_hjr
from deepreach.utils.comparisons import GroundTruthHJSolution

sys.path.append("/home/jingpei/Documents/arclab/L4DC25_project/custom_sim/quad2d")
sys.path.append("/home/jingpei/Documents/arclab/L4DC25_project/custom_sim")
sys.path.append("/home/jingpei/Documents/arclab/L4DC25_project")

from deepreach.dynamics import dynamics 
from deepreach.dynamics import dynamics_hjr


from toy_env import simEnv
from disturbance_controller_utils import randomGoalNominalController, HorizontalVelocityWind

####################################### Disturbance Functions #######################################
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



def get_disturbance_fn_from_GP_jax_USEGRID(gp_model, grid_states, min_disturbance_magnitude, max_disturbance_magnitude, num_std_away=2.0, border_padding=None):
    """
    Wrapper to convert a GP model to a disturbance function that can be used in the dynamics model. 
    NOTE: This is a grid based method where you are just finding the closest grid point to the inputted grid state
    and returning that value as the disturbance
    NOTE: preference to get the gp function above working instead. 

    args: 
        - gp_model: GP model that has been trained on [x, y] -> disturbance value
        - grid_states: [x, y, xvel, yvel, 4] grid states that the GP model has been trained on
            NOTE: only have 2d grid states [x, y] for the GP model as that is all you are comparing
        - min_disturbance_magnitude: minimum disturbance magnitude to clip to
        - max_disturbance_magnitude: maximum disturbance magnitude to clip to
        - num_std_away: number of standard deviations away from the mean to consider for the disturbance magnitude
        - border_padding: default None, Optional int: how many grid points to pad the border by - use the max disturbance value for padding. 
    """

    # Evaluate GP model on grid states 
    flattened_grid_states = grid_states.reshape(-1, grid_states.shape[-1])  # Flatten to [N, 4]
    disturbances_mean, disturbances_var = gp_model.predict(flattened_grid_states[:, :2])

    if border_padding is not None:
        padded_xy_states = np.concatenate([
            grid_states[:border_padding, :].reshape(-1, grid_states.shape[-1]),
            grid_states[-border_padding:, :].reshape(-1, grid_states.shape[-1]),
            grid_states[:, :border_padding].reshape(-1, grid_states.shape[-1]),
            grid_states[:, -border_padding:].reshape(-1, grid_states.shape[-1])
        ], axis=0)  # Concatenate along the first axis

        # Get the indices where padded_xy_states correspond to flattened_grid_states
        padded_indices = []
        for padded_state in padded_xy_states:
            distances = np.linalg.norm(flattened_grid_states[:, :2] - padded_state[:2], axis=1)
            closest_index = np.argmin(distances)
            padded_indices.append(closest_index)
        for idx in padded_indices:
            disturbances_mean[idx] = max_disturbance_magnitude

    disturbances_stdaway = np.abs(disturbances_mean) + num_std_away * np.sqrt(disturbances_var)  # Get max disturbance magnitude
    disturbances_stdaway = np.clip(disturbances_stdaway, min_disturbance_magnitude, max_disturbance_magnitude)
    
    flattened_grid_states_jax = jax.numpy.array(flattened_grid_states)  # Convert to JAX array for JAX compatibility
    disturbances_stdaway_jax = jax.numpy.array(disturbances_stdaway)  # Convert to JAX array for JAX compatibility
    grid_xy = jnp.asarray(flattened_grid_states[:, :2])  # [N, 2]
    def compute_single_disturbance_jax(state_xy):
        # state_xy: [2]
        dists = jnp.linalg.norm(grid_xy - state_xy, axis=-1)  # shape: [N]
        idx = jnp.argmin(dists)
        mag = disturbances_stdaway_jax[idx]  # [D]
        return jnp.tile(mag, (4,))  # shape [4] (repeats per dim)

    # Create disturbance function using grid states and evaluted GP model
    def disturbance_function_jax_grid(state):
        # PyTorch branch
        if isinstance(state, torch.Tensor):
            if len(state.shape) == 1:
                state = state.unsqueeze(0)

            distances = torch.norm(torch.tensor(flattened_grid_states[:, :2]).unsqueeze(1) - torch.tensor(state[:, :2]), dim=-1)
            closest_grid_indices = torch.argmin(distances, dim=1)
            mag = torch.tensor(disturbances_stdaway)[closest_grid_indices]

            disturbances = mag.repeat(1, 4).unsqueeze(-1)[:, :, 0]  # shape (B, 4)

        elif isinstance(state, np.ndarray):
            if len(state.shape) == 1:
                state = np.expand_dims(state, axis=0)

            distances = np.linalg.norm(flattened_grid_states[:, None, :2] - state[:, :2], axis=-1)
            closest_grid_indices = np.argmin(distances, axis=0)
            mag = disturbances_stdaway[closest_grid_indices]

            disturbances = np.repeat(mag[:, None], 4, axis=1)[:, :, 0]  # shape (B, 4)

        # JAX branch
        elif isinstance(state, (jnp.ndarray, Array)):
            if len(state.shape) == 1:
                state = jnp.expand_dims(state, 0)
            
            state_xy = state[:, :2]  # shape: [B, 2]

            # jax lax scan implementation
            # state_xy_flat = state.reshape(-1, state.shape[-1])[:, :2]  # [B, 2]
            # def scan_fn(carry, s):
            #     return carry, compute_single_disturbance_jax(s)
            # _, disturbances_flat = lax.scan(scan_fn, None, state_xy_flat)
            # disturbances = disturbances_flat.reshape(state.shape[:-1] + (4,))  # Restore original shape
            # return disturbances[0] if disturbances.shape[0] == 1 else disturbances

            # jax vmap implementation 
            batched_compute = vmap(compute_single_disturbance_jax)
            disturbances = batched_compute(state_xy)  # shape: [B, 4]

            # jax matrix implementation - needs fixing
            # dists = jnp.linalg.norm(flattened_grid_states_jax[:, None, :2] - state_xy[:, :2], axis=-1)  # Compute distances
            # closest_grid_indices = jnp.argmin(dists, axis=0)  # Find closest grid indices
            # mag = disturbances_stdaway_jax[closest_grid_indices]  # Get disturbance magnitudes
            # disturbances = jnp.repeat(mag[:, None], 4, axis=1)  # Repeat magnitudes for all dimensions

            # disturbances = jnp.repeat(mag[:, None], 4, axis=1)[:, :, 0]  # shape (B, 4)

        else:
            raise TypeError(f"Unsupported input type: {type(state)}")

        if disturbances.shape[0] == 1: 
            return disturbances[0]
        else:
            return disturbances
    
    return disturbance_function_jax_grid


def get_disturbance_function_flyaround(full_disturbance_fn, initial_sample_radius, fly_around_timesteps, steps_per_goal, sample_freq, xy_range, xy_grid_states, 
                                       dt, tMin, tMax,  min_disturbance_magnitude, max_disturbance_magnitude, gp_kernel=None, reoptimize_gp=False, return_gp_model=False, 
                                       border_padding=None):
    """
    Gets a disturbance function with partial GP samples from flying around the environment
    Args: 
        - full_disturbance_fn: The full disturbance function to use (that you want to sample by flying around)
        - initial_sample_radius: The radius of the initial sample circle around the drone
        - fly_around_timesteps: The number of timesteps to fly around the environment
        - sample_freq: The frequency of sampling the disturbance function while flying around
        - steps_per_goal: The number of steps per second to take before creating another random goal
        - xy_range: The range of the x and y dimensions to sample the initial state from
        - xy_grid_states: The grid states in the x and y dimensions to use for creating the grid based disturbance function from the GP

        - dt: the dt to use for the environment
        - tMin: The minimum time for the environment
        - tMax: The maximum time for the environment
        - min_disturbance_magnitude: Minimum disturbance magnitude to clip to 
        - max_disturbance_magnitude: Maximum disturbance magnitude to clip to 
        - gp_kernel: The GP kernel to use for the GP model (if None, use default and optimize with the first set of samples)
        - reoptimize_gp: If True, reoptimize the GP model with the new samples after flying around - DEFAULT FALSE
        - border_padding: default None, Optional int: how many grid points to pad the border by - use the max disturbance value for padding. 
    """

    model_input_indices = [0, 1]  # x, y indices for the disturbance function
    flattened_xy_grid_states = xy_grid_states.reshape(-1, 4)  # Flatten the grid states to (N, 4)
    flattened_disturbance_grid = full_disturbance_fn(flattened_xy_grid_states)
    
    # Random initial state 
    init_state_x = np.random.uniform(xy_range[0][0], xy_range[0][1])
    init_state_y = np.random.uniform(xy_range[1][0], xy_range[1][1])

    # Create environment and initialize drone 
    dynamics_model_full_hjr, dynamics_model_full = get_dynamics_model_given_disturbance_fn(
                                                            disturbance_function=full_disturbance_fn,
                                                            tMin=tMin,
                                                            tMax=tMax, 
                                                            ret_system=True
                                                        )
    
    init_state=torch.tensor([init_state_x, init_state_y, 0.0, 0.0])
    env = simEnv(system=dynamics_model_full,
                init_state=init_state, 
                # init_state=torch.tensor([0, 1.7, 0.0, 0.0]),
                disturbance_fn=full_disturbance_fn,
                disturbance_gradient_fn=None,
                dt=dt,
        )
    
    nominal_control_fn = randomGoalNominalController(
                system=dynamics_model_full,
                system_type="Quad2DAttitude",
                env=env, 
                init_goal_position=None, 
                goal_threshold=0.05, 
                goal_reset_step=steps_per_goal,
            )

    # Get initial samples close to the initial state
    init_state = init_state[:2]  # Only consider x, y for the grid states
    distances = np.linalg.norm(flattened_xy_grid_states[:, :2] - init_state.cpu().numpy(), axis=1)
    close_indices = np.where(distances < initial_sample_radius)[0]
    un_close_indices = np.where(distances >= initial_sample_radius)[0]
    sampled_states = flattened_xy_grid_states[close_indices]

    # Train GP 
    partial_x_init = flattened_xy_grid_states[close_indices, :][:, model_input_indices]  # x, y
    partial_y_init = flattened_disturbance_grid[close_indices, :][:, 0:1]  # Disturbance value
    if gp_kernel is None:
        gp_model_partial = GPWrapper(X_init=partial_x_init, Y_init=partial_y_init, optimize=True)
    else: 
        raise NotImplementedError("GP kernel optimization not implemented yet. Please provide a GP kernel.")

    # Fly drone around to collect more samples 
    sample_freq = 5
    additional_x_data = []
    additional_y_data = []
    for step in tqdm(range(fly_around_timesteps)):
        nominal_control = nominal_control_fn(env.state)
        # NOTE: Noiseless step
        next_state = env.noiseless_step(nominal_control)
        env.state = next_state 
        env.time += env.dt
        if step % sample_freq == 0:
            additional_x_data.append(np.array(env.state))
            additional_y_data.append(full_disturbance_fn(np.array(env.state)))

    # Train GP again 
    additional_x_data = np.array(additional_x_data)[:, model_input_indices]
    additional_y_data = np.array(additional_y_data)[:, 0:1]
    gp_model_partial.add_sample(x_new=additional_x_data, 
                    y_new=additional_y_data, 
                    reoptimize=reoptimize_gp)

    # Convert GP to grid based disturbance function for HJR 
    num_std_away = 2.0 
    partial_gp_disturbance_fn_for_hjr_grid = get_disturbance_fn_from_GP_jax_USEGRID(gp_model=gp_model_partial, 
                                                                                grid_states=xy_grid_states, 
                                                                                min_disturbance_magnitude=min_disturbance_magnitude, 
                                                                                max_disturbance_magnitude=max_disturbance_magnitude, 
                                                                                num_std_away=num_std_away, 
                                                                                border_padding=border_padding)

    if return_gp_model:
        return partial_gp_disturbance_fn_for_hjr_grid, gp_model_partial
    return partial_gp_disturbance_fn_for_hjr_grid


####################################### Dynamics Related Functions #######################################


def get_dynamics_model_given_disturbance_fn(disturbance_function, tMin, tMax, ret_system=False): 
    # ret_system: If True, return the system model as well
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
    
    if ret_system: 
        return system_sv_hjr, system_sv
    else: 
        return system_sv_hjr

####################################### HJR Related Functions #######################################

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

    del disturbance_inputs_tensor
    del torch_value_function_outputs
    del ground_truth_hj_solution  # Free memory after saving
    gc.collect()  # Explicitly run garbage collection
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # Clear GPU memory if using CUDA
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

def create_hjr_disturbance_dataset(num_datapoints, 
                                   save_folder, 
                                   disturbance_variation_type="sincos", 
                                   grid_resolution=(51, 51, 51, 51), 
                                   start_num=None, 
                                   end_num=None): 
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

    if start_num is None:
        start_num = 0

    if end_num is None:
        end_num = num_datapoints

    for num in tqdm(range(start_num, end_num)): 
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
    num_datapoints = 2000 #250
    start_num = 1907 #822

    save_folder = "/media/jingpei/DATA/fno_gp_data_2000"
    disturbance_variation_type = "sincos"
    grid_resolution = (41, 41, 41, 41) #(51, 51, 51, 51)  # Default resolution

    os.makedirs(save_folder, exist_ok=True)  # Ensure the folder exists

    create_hjr_disturbance_dataset(num_datapoints, save_folder, disturbance_variation_type, grid_resolution, 
                                   start_num=start_num)
