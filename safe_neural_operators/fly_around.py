"""
File to fly quad2d quadcopter around with CBF from the trained safe neural operator. 
"""
import os 
import sys 
import torch 
import numpy as np 
from torch.utils.data import DataLoader
from copy import deepcopy 
from tqdm import tqdm 
import jax.numpy as jnp 
import matplotlib.pyplot as plt

from neuralop.models import FNO
from neuralop.data.datasets.tensor_dataset import TensorDataset

sys.path.append('/home/jingpei/Documents/arclab/neuraloperator/safe_neural_operators')
from hjr_dataset import HJRDataset

# Append path to quad2d environments and associated files
sys.path.append("/home/jingpei/Documents/arclab/L4DC25_project/custom_sim/quad2d")
sys.path.append("/home/jingpei/Documents/arclab/L4DC25_project/custom_sim")
sys.path.append("/home/jingpei/Documents/arclab/L4DC25_project")

# Deepreach Imports 
from deepreach.dynamics import dynamics
from deepreach.utils.comparisons import GroundTruthHJSolution

# CBF Imports 
from cbf_opt_kit.cbf import HJReachabilityControlAffineCBF, DeepReachControlAffineCBF
from cbf_opt_kit.safety_filter import ControlAffineSafetyFilter, OptimalControl

# Environment imports
from toy_env import simEnv
from data_collection import get_dynamics_model_given_disturbance_fn, get_disturbance_function_sincos

# Imports for flying around quadcopter
from baseline_experiment import plotter, expLogger 

# Nominal Controller 
from disturbance_controller_utils import randomGoalNominalController, HorizontalVelocityWind

from PIL import Image, ImageDraw
def pngs_to_gif(folder_path, output_path, duration=100, draw_frame_number=False):
    # Get all PNG files in the folder sorted by name (assuming names are ordered by sequence)
    images = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.png')],
                    key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

    # Load images into a list
    frames = []
    for frame_number, image in enumerate(images):
        with Image.open(image) as img:
            if draw_frame_number: 
                # draw the frame number on the image 
                draw = ImageDraw.Draw(img)
                # Add frame number text
                text = f"Frame: {frame_number}"
                draw.text((10, 10), text, fill="black")  

            frames.append(img.copy())

    # Save as GIF
    frames[0].save(output_path, format='GIF', append_images=frames[1:], save_all=True, duration=duration, loop=0)
    return 

def get_gridvalue_function_from_model(disturbance_input, grid_states, model, data_processor, device="cuda:0"):
    """
    Gets the value function over the whole quad 2d grid (41, 41, 41, 41 ,4)

    Arguments: 
    - disturbance_input: (41, 41) image representing the disturbance magnitude 
    - grid_states: (41, 41, 41, 41, 4) grid states of the quad2d system

    Returns: 
    - grid_value_function: (41, 41, 41, 41, 1) value function over the grid states to be used for CBF
    """
    # NOTE: Temporary: need to change the input space to have full state so that you can use FNO as CBF directly - this needs changes! 

    # 1. Get the grid inputs 
    grid_states_tensor = torch.tensor(grid_states)
    grid_resolution = grid_states_tensor.shape 

    disturbance_input_tensor = torch.tensor(disturbance_input)
    channel_dim = 1 + 1 # channel dim is 1 greater than the channel dim used for model training 
    expanded_dims = list(grid_resolution[:-1])
    expanded_dims.insert(channel_dim, 1)
    grid_input = disturbance_input_tensor.view(grid_resolution[0], grid_resolution[1], 1, 1, 1)
    grid_input = grid_input.expand(*expanded_dims)

    # Inputs of the shape (batch_dim, channels, 41,41)
    # Channels: [disturbance magnitude, grid xvel, grid yvel]
    last_grid = grid_states_tensor[:, :, :, :, 2:]  # Last two channels are xvel and yvel
    reorg_grid_channeldims = list(range(len(grid_resolution)))
    last_dim_num = reorg_grid_channeldims.pop(-1)  # Remove last dim
    reorg_grid_channeldims.insert(channel_dim, last_dim_num)  # Insert at channel_dim
    last_grid = last_grid.permute(*reorg_grid_channeldims) 

    # 2. Stack to create all the grid states # NOTE: HARDCODED FOR CHANNEL_DIM 1 / 2 here 
    appended_grid_input = torch.cat((grid_input, last_grid), dim=channel_dim)
    flattened_tensor_dataset = appended_grid_input.view(*appended_grid_input.shape[:3], -1) # (41, 41, 3, 41*41)
    flattened_tensor_dataset = flattened_tensor_dataset.permute(3, 2, 0, 1)  # (batch_dim, channels, 41, 41)

    # ################# DEBUGGING #################
    # # Verification that Input grid states match the reshaped input grid states tensor and thus value function 
    # reshaped_tensor_dataset = deepcopy(flattened_tensor_dataset)
    # reshaped_tensor_dataset = reshaped_tensor_dataset.permute(2, 3, 1, 0)
    # reshaped_tensor_dataset = reshaped_tensor_dataset.view(grid_resolution[0], 
    #                                                        grid_resolution[1], 
    #                                                        3, 
    #                                                        grid_resolution[2], 
    #                                                        grid_resolution[3])
    
    # print("reshaped_tensor_dataset shape: ", reshaped_tensor_dataset.shape)


    # input_xvels = reshaped_tensor_dataset[:, :, 1, :, :].cpu().numpy()
    # input_yvels = reshaped_tensor_dataset[:, :, 2, :, :].cpu().numpy()
    # grid_state_xvels = grid_states_tensor[:, :, :, :, 2].cpu().numpy()
    # grid_state_yvels = grid_states_tensor[:, :, :, :, 3].cpu().numpy()

    # diff_xvels = np.abs(input_xvels - grid_state_xvels)
    # diff_yvels = np.abs(input_yvels - grid_state_yvels)

    # sum_diff_xvels = np.sum(np.abs(diff_xvels))
    # sum_diff_yvels = np.sum(np.abs(diff_yvels))

    # print(f"Sum of abs differences in xvels: {sum_diff_xvels}")
    # print(f"Sum of abs differences in yvels: {sum_diff_yvels}")

    # import pdb; pdb.set_trace()
    # ################# DEBUGGING #################

    # 3. Create tensor dataset to pass through the model 
    test_db = TensorDataset(flattened_tensor_dataset, flattened_tensor_dataset[:, 0:1, :, :]) # y is placeholder
    test_b_size = 128
    test_loader = DataLoader(test_db, 
                            batch_size=test_b_size,
                            shuffle=False,
                            num_workers=1,
                            pin_memory=True,
                            persistent_workers=False,)

    # 4. Go through the entire test dataloader and get the model output for each input and then stack them all together
    with torch.no_grad():
        stacked_outputs = None 
        for idx, sample in enumerate(test_loader):
            sample = data_processor.preprocess(sample, batched=True)
            x = sample['x'].to(device)
            curr_model_output = model.forward(x)
            if stacked_outputs is None:
                stacked_outputs = curr_model_output.detach().cpu()
            else: 
                stacked_outputs = torch.cat((stacked_outputs, curr_model_output.detach().cpu()), dim=0)

    # 5. Reshape the stacked outputs to the grid resolution # NOTE: HARDCODED FOR CHANNEL_DIM 1 / 2 here
    grid_value_function = stacked_outputs.permute(2, 3, 1, 0)
    grid_value_function = grid_value_function.view(grid_resolution[0], grid_resolution[1], *expanded_dims[2:]).squeeze(channel_dim)
    return grid_value_function, channel_dim, flattened_tensor_dataset # NOTE: Remove for debugging


class ReachabilityModel:
    # Class with fixed attributes for cbf generation
    def __init__(self, grid, grid_values, times):    
        self.grid = grid
        self.grid_values = np.array(grid_values)
        self.times = np.array(times)


def fly_around_drone_experiment(env, nominal_control_fn, safety_filter, num_steps, tMax, 
                 system_type="Quad2DAttitude", control_rate_dt=None, dist_sample_rate_dt=None, 
                 render=False, results_folder=None, 
                 disturbance_patch_size=0.1, robot_patch_size=0.1, 
                 save_images=True, quit_on_fail=False, 
                 # FOR DEBUGGING PURPOSES ONLY
                 grid_states=None, grid_value_function=None, true_grid_value_function=None
                 ): 
    """
    Function Fly around drone with reachability model CBF 
    """
    states, controls, disturbances, value_functions, current_goals, goal_distances, safety_violations = [], [], [], [], [], [], []

    time_since_last_control = 0 
    if control_rate_dt is None:
        control_rate_dt = env.dt
        
    if dist_sample_rate_dt is None:
        dist_sample_rate_dt = env.dt

    ### Plotting Code ###
    init_state = env.state 
    goal = nominal_control_fn.goal if hasattr(nominal_control_fn, 'goal') else None
    if (render or save_images) and results_folder is not None:
        plotter_obj = plotter(system=env.system, system_type=system_type, plot_state_bounds=None, disturbance_patch_size=disturbance_patch_size, robot_patch_size=robot_patch_size)
        plotter_obj.init_plot(init_state=init_state, disturbance_fn=env.disturbance_fn, goal=goal)

    if results_folder is not None:
        os.makedirs(results_folder, exist_ok=True)
        ########## Debugging ##########
        # For plotting and saving value function slices for debugging 
        if grid_value_function is not None:
            slice_visualizer_folder = os.path.join(results_folder, "slice_visualizer")
            os.makedirs(slice_visualizer_folder, exist_ok=True)
        if true_grid_value_function is not None:
            true_slice_visualizer_folder = os.path.join(results_folder, "true_slice_visualizer")
            os.makedirs(true_slice_visualizer_folder, exist_ok=True)    
        xvels = np.array(grid_states[0, 0, :, 0, 2])
        yvels = np.array(grid_states[0, 0, 0, :, 3])
        ########## Debugging ##########

    nominal_control = nominal_control_fn(env.state)
    safe_control = nominal_control 
    disturbance = env.get_current_disturbance()
    for step in tqdm(range(num_steps)): 

        # Control 
        time_since_last_control += env.dt
        if time_since_last_control >= control_rate_dt:
            nominal_control = nominal_control_fn(env.state)
            # Safety Filter
            filter_output = safety_filter(state=np.array(env.state), 
                                        time=tMax-0.01, # Force set to final time for time invariant for now ? 
                                        nominal_control=np.array(nominal_control), )
                                        # No longer disturbance based curr_disturbance_bounds=filter_disturbance_bounds) #float(torch.max(filter_disturbance_bounds)))
            safe_control, opt_disturbance, _, vf = filter_output

        # Step 
        print(f"Nominal Control: {nominal_control}")
        print(f"Value Function: {vf}")
        print(f"Step: {step}, Action: {safe_control}\n")

        states.append(env.state)
        controls.append(safe_control.flatten())
        disturbances.append(disturbance)
        value_functions.append(vf)
        current_goals.append(np.array(nominal_control_fn.goal).flatten())
        goal_distances.append(torch.norm(env.state[:2] - nominal_control_fn.goal[:2]))
        safety_violations.append(env.system.env_config.sdf_obstacles(env.state)) #env.system.avoid_fn(env.state))


        next_state = env.step(control=safe_control)

        # Render
        goal = nominal_control_fn.goal if hasattr(nominal_control_fn, 'goal') else None
        if ((step % 10 == 0) or (quit_on_fail and safety_violations[-1] < 0)) and (render or save_images):
            plotter_obj.update_plot(next_state=next_state, goal=goal, plt_pause=0.01, render=render)

        # Save Results
        if (step % 10 == 0) and results_folder is not None and save_images: 
            plt.savefig(os.path.join(results_folder, f"{step:03d}.png"))
            
            ########## Debugging ##########
            # Save the value function slice corresponding to where you are 
            xvel_slice_idx = int(np.argmin(np.abs(xvels - np.array(next_state[2]))))
            yvel_slice_idx = int(np.argmin(np.abs(yvels - np.array(next_state[3]))))
            if (grid_value_function is not None): # and (step % 10 == 0): 
                value_function_slice = np.array(grid_value_function[:, :, xvel_slice_idx, yvel_slice_idx])
                plt.figure()
                plt.title(f"Value Function: xvel {xvels[xvel_slice_idx]:.3f}, yvel {yvels[yvel_slice_idx]:.3f}")
                plt.imshow(value_function_slice, cmap='viridis')
                plt.colorbar()
                plt.contour(value_function_slice, levels=[0], colors='red')
                plt.savefig(os.path.join(slice_visualizer_folder, f"{step:03d}.png"))
                plt.close()
            if (true_grid_value_function is not None): # and (step % 10 == 0):
                true_value_function_slice = np.array(true_grid_value_function[-1, :, :, xvel_slice_idx, yvel_slice_idx])
                plt.figure()
                plt.title(f"True Value Function: xvel {xvels[xvel_slice_idx]:.3f}, yvel {yvels[yvel_slice_idx]:.3f}")
                plt.imshow(true_value_function_slice, cmap='viridis')
                plt.colorbar()
                plt.contour(true_value_function_slice, levels=[0], colors='red')
                plt.savefig(os.path.join(true_slice_visualizer_folder, f"{step:03d}.png"))
                plt.close()
            ########## Debugging ##########

        if quit_on_fail and safety_violations[-1] < 0:
            # import pdb; pdb.set_trace()
            print("Safety Violation Detected! Quitting Simulation.")
            break

    if save_images: 
        pngs_to_gif(folder_path=results_folder, 
                    output_path=os.path.join(results_folder, "simulation.gif"), 
                    duration=0.25, draw_frame_number=True) #4000*env.dt)#50) # this controls the frame rate
        ########## Debugging ##########
        if (grid_value_function is not None): 
            pngs_to_gif(folder_path=slice_visualizer_folder, 
                        output_path=os.path.join(slice_visualizer_folder, "value_function_slices.gif"), 
                        duration=0.25, draw_frame_number=True)
        if (true_grid_value_function is not None): 
            pngs_to_gif(folder_path=true_slice_visualizer_folder, 
                        output_path=os.path.join(true_slice_visualizer_folder, "true_value_function_slices.gif"), 
                        duration=0.25, draw_frame_number=True)
        ########## Debugging ##########

    return states, controls, disturbances, value_functions, current_goals, goal_distances, safety_violations



if __name__ == "__main__":
    # Model parameters
    model_dir = "/media/jingpei/DATA/fno_models/safe_neural-7-12-25"
    save_dir = "/media/jingpei/DATA/fno_model_eval_results/fly_safe_neural-7-13-25"
    device = "cuda:0"
    data_root_dir = "/media/jingpei/DATA/fno_gp_data"
    grid_states_path = "/media/jingpei/DATA/fno_gp_data/000_grid_states.pt"

    # System parameters 
    dt = 0.025  # 0.01
    goal_reset_steps = 100 # Number of steps after which the goal is reset
    cbf_alpha = 0.1 
    num_steps = 1000 

    use_gt = False #False
    random_seed = 13

    if use_gt: 
        save_dir = save_dir + "_gt"
    os.makedirs(save_dir, exist_ok=True)

    if random_seed is not None:
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

    # 1. Load the model "
    model = FNO.from_checkpoint(save_folder=model_dir, save_name="model")
    model = model.to(device)
    model.eval()


    # 2. Create dataset for now
    samples_for_train = 90
    samples_for_test = 10 
    datapoints_per_sample = 1000
    batch_size = 128 #32
    pre_sample_dataset = True 
    encode_output = False 
    encode_input = True 
    encoding = "channel-wise"
    dataset = HJRDataset(root_dir=data_root_dir, 
                        samples_for_train=samples_for_train, 
                        samples_for_test=samples_for_test, 
                        datapoints_per_sample=datapoints_per_sample,

                        batch_size=batch_size, 
                        pre_sample_dataset=pre_sample_dataset, 

                        encode_output=encode_output,
                        encode_input=encode_input,
                        encoding=encoding)

    data_processor = dataset.data_processor

    # 2. Create the system
    min_max_magnitude = 0.25 
    max_max_magnitude = 0.75 

    min_phase_multiplier = 0.05
    max_phase_multiplier = 5 

    min_phase_shift = 0
    max_phase_shift = np.pi / 2

    tMin = 0.0 
    tMax = 3.0 

    curr_max_magnitude = np.random.uniform(min_max_magnitude, max_max_magnitude)
    curr_phase_multiplier = np.random.uniform(min_phase_multiplier, max_phase_multiplier)
    curr_phase_shift = np.random.uniform(min_phase_shift, max_phase_shift)

    curr_dim = np.random.choice(['x', 'y'])
    curr_use_sin = np.random.choice([True, False])

    disturbance_function = get_disturbance_function_sincos(
                    max_magnitude=curr_max_magnitude, 
                    phase_multiplier=curr_phase_multiplier, 
                    phase_shift=curr_phase_shift, 
                    dim=curr_dim,  # or 'y' depending on the direction you want
                    use_sin=curr_use_sin  # or False for cosine
                )
    dynamics_model_hjr, dynamics_model = get_dynamics_model_given_disturbance_fn(
                disturbance_function=disturbance_function, 
                tMin=tMin, 
                tMax=tMax, 
                ret_system=True
            )

    # 2.1 Create the quad2d sim environment  
    env = simEnv(
            system=dynamics_model,
            init_state=torch.tensor([-1.4, 1.7, 0.0, 0.0]),
            disturbance_fn=disturbance_function,
            disturbance_gradient_fn=None,
            dt=dt,
        )

    # 3. Get the value function from the model 
    grid_states = torch.load(grid_states_path, weights_only=False)
    grid_resolution = grid_states.shape
    disturbance_inputs = dynamics_model.disturbance_function(grid_states[:, :, 0, 0, ].reshape(-1, 4))
    disturbance_inputs_reshaped = disturbance_inputs.reshape((grid_resolution[0],grid_resolution[1], 4))  # default shape (51, 51, 4)
    disturbance_inputs_tensor = torch.from_numpy(np.array(disturbance_inputs_reshaped))[:, :, 0]
    
    print("\n\n\ndisturbance_inputs_tensor shape: ", disturbance_inputs_tensor.shape)
    print("\n\n\n")
    
    grid_value_function, channel_dim, flattened_tensor_dataset = get_gridvalue_function_from_model(disturbance_input=disturbance_inputs_tensor, 
                                                            grid_states=grid_states, 
                                                            model=model, 
                                                            data_processor=data_processor, 
                                                            device=device)

    ####### DEBUGGING #######
    # # DEBUGGING: 3.1 Plot value function slices for logging
    # flattened_grid_value_function = grid_value_function.unsqueeze(channel_dim)  # Add channel dimension
    # flattened_grid_value_function = flattened_grid_value_function.view(*grid_resolution[:2], 1, -1)
    # flattened_grid_value_function = flattened_grid_value_function.permute(3, 2, 0, 1)  # (batch_dim, channels, 41, 41)

    # plot_slices_dir = os.path.join(save_dir, "value_function_slices")
    # os.makedirs(plot_slices_dir, exist_ok=True)

    # num_plots = 20 
    # random_indices = np.arange(len(flattened_grid_value_function), step=20) #np.random.choice(flattened_grid_value_function.shape[0], num_plots, replace=False)
    # for i, idx in enumerate(random_indices):
    #     plt.imshow(flattened_grid_value_function[idx, 0, :, :].cpu().numpy(), cmap='viridis')
    #     plt.title(f"Value Function Slice {idx}")
    #     plt.colorbar()
    #     plt.contour(flattened_grid_value_function[idx, 0, :, :].cpu().numpy(), levels=[0], colors='red')
    #     plt.savefig(os.path.join(plot_slices_dir, f"{idx:03d}_value_function_slice.png"))
    #     plt.close()
    ####### DEBUGGING #######

    # 4. Create the CBF 
    hjr_solution = GroundTruthHJSolution(dynamics_model_hjr, solve=False, grid_resolution=grid_states.shape[:-1])

    ####### DEBUGGING #######
    hjr_solution.solve_hjr()
    true_grid_value_function = hjr_solution.value_functions
    print("Value Function Shape: ", true_grid_value_function.shape)
    ####### DEBUGGING #######

    if use_gt: 
        # Use the ground truth value function
        reachability_model = ReachabilityModel(grid=hjr_solution.grid, 
                                           grid_values=jnp.array(true_grid_value_function), # unsqueeze in time dimension
                                           times=jnp.array([tMax])) # Only single value 
    else: 
        # Use the learned value function 
        reachability_model = ReachabilityModel(grid=hjr_solution.grid, 
                                           grid_values=jnp.array(grid_value_function.detach().cpu().unsqueeze(0).numpy()), # unsqueeze in time dimension
                                           times=jnp.array([tMax])) # Only single value 
    
    is_time_invariant = True 
    cbf_obj = HJReachabilityControlAffineCBF(dynamics_model_hjr, 
                                reachability_model,
                                time_invariant=is_time_invariant,)
                                    # time=1.0)
    opt_ctrl = OptimalControl(cbf_obj)
    safety_filter = ControlAffineSafetyFilter(cbf_obj, 
                                            alpha = lambda x: cbf_alpha * x, 
                                            weighting=np.array([10., 1.]),
                                            backup_filter=opt_ctrl,
                                            return_values=True
                                            )

    # 5. Fly the quadcopter with nominal controller + CBF 
    # 5.1 Get Nominal Controller
    nominal_control_fn = randomGoalNominalController(
            system=dynamics_model,
            system_type="Quad2DAttitude",
            env=env, 
            init_goal_position=None, 
            goal_threshold=0.05, 
            goal_reset_step=goal_reset_steps,
        )
    
    results_folder = os.path.join(save_dir, "flying_drone_images")
    os.makedirs(results_folder, exist_ok=True)
    fly_around_drone_experiment(env=env, 
                                nominal_control_fn=nominal_control_fn, 
                                safety_filter=safety_filter, 
                                num_steps=num_steps, 
                                tMax=tMax, 
                                system_type="Quad2DAttitude", 
                                control_rate_dt=None, 
                                dist_sample_rate_dt=None, 
                                render=True, 
                                results_folder=results_folder, 
                                disturbance_patch_size=0.1, robot_patch_size=0.1, 
                                save_images=True, quit_on_fail=False, 
                                ########## Debugging ##########
                                grid_states=grid_states, 
                                grid_value_function=grid_value_function, 
                                true_grid_value_function=true_grid_value_function,
                                )
                                ########## Debugging ##########