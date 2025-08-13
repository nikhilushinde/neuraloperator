"""
File to create dataset class to use with data_collection.py - for HJR training data on quad2d environment with different spatially varying disturbance functions
"""
from functools import partialmethod
from pathlib import Path
from typing import List, Union, Optional

import torch
import numpy as np 

from neuralop.data.datasets.tensor_dataset import TensorDataset
from neuralop.data.transforms.data_processors import DefaultDataProcessor
from neuralop.data.transforms.normalizers import UnitGaussianNormalizer

from data_collection import load_hjr_solution, load_grid_states
import random
from tqdm import tqdm 

import sys 
sys.path.append("/home/jingpei/Documents/arclab/neuraloperator/safe_neural_operators")
from safe_neural_operators.hjr_dataset import HJR_SubDataset

def load_full_partial_training_dataset(root_dir, num_idxs): 
    """
    Load the full training dataset from the specified root directory
    num_idxs: int or list of indices - if int, load that many samples from the dataset, if range, load samples in that range
    """
    full_grid_states = load_grid_states(load_path=root_dir, load_index=0, load_prefix="full")
    partial_grid_states = load_grid_states(load_path=root_dir, load_index=0, load_prefix="partial") 
    full_grid_resolution = full_grid_states.shape[:-1]  
    partial_grid_resolution = partial_grid_states.shape[:-1]
    
    full_input_data = []
    full_output_data = []
    partial_input_data = []
    partial_output_data = []

    if type(num_idxs) is int: 
        num_idxs = range(num_idxs)

    for idx in tqdm(num_idxs): 
        input_tensor, output_tensor = load_hjr_solution(load_path=root_dir, 
                                                        load_index=idx)
        

        if torch.any(torch.isnan(input_tensor)) or torch.any(torch.isnan(output_tensor)):
            print(f"\n\nSkipping sample {idx} due to NaN values in input or output tensor.\n\n")
            continue
        
        output_tensor_shape = output_tensor.shape
        if np.all(output_tensor_shape == full_grid_resolution):
            full_input_data.append(input_tensor)
            full_output_data.append(output_tensor)
        elif np.all(output_tensor_shape == partial_grid_resolution):
            partial_input_data.append(input_tensor)
            partial_output_data.append(output_tensor)
        else:
            raise ValueError(f"Output tensor shape {output_tensor_shape} does not match either full grid resolution {full_grid_resolution} or partial grid resolution {partial_grid_resolution}")


    return (full_input_data, full_output_data), (partial_input_data, partial_output_data), full_grid_states, partial_grid_states

class HJRFullPartialDataset: 
    """
    HJR dataset class to use for HJR data from data_collection.py
    """
    def __init__(self, 
                 root_dir: str, 
                 samples_for_dataset: Union[int, List[int]],
                 percent_test: float,
                 datapoints_per_full_sample: int,
                 datapoints_per_partial_sample: int,

                 batch_size: int, 
                 pre_sample_dataset: bool=True, 

                 encode_input: bool=False,
                 encode_output: bool=True,
                 encoding: str="channel-wise", 
                 full_ordered_dataset: bool=False
                 ):
        """
        Parameters
        ----------
        root_dir : Union[Path, str]
            root where data is stored
        samples_for_dataset : int or list of int
            number of hjr solution samples to use for the dataset if int
            or list of indices to use for the dataset
        percent_test : float
            percentage of samples to use for the test set, if not specified, will use 20% of the dataset
        datapoints_per_full_sample: int
            number of datapoints to get per full sample, NOTE: must be lower than grid_resolution[2] * grid_resolution[3] - as you will be randomly sampling these dimensions
        datapoints_per_partial_sample: int
            number of datapoints to get per partial sample, NOTE: must be lower than grid_resolution[2] * grid_resolution[3] - as you will be randomly sampling these dimensions

        batch_size : int
            batch size of training set

        pre_sample_dataset: bool
            whether to pre-sample the dataset or randomly generate on the fly with each getitem call 

        encode_input : bool, optional
            whether to normalize inputs in provided DataProcessor,
            by default False
        encode_output : bool, optional
            whether to normalize outputs in provided DataProcessor,
            by default True
        encoding : str, optional
            parameter for input/output normalization. Whether
            to normalize by channel ("channel-wise") or 
            by pixel ("pixel-wise"), default "channel-wise"
        full_ordered_dataset : bool, optional
            whether to create a full ordered dataset, where each grid slice is sampled in order
        """

        # Save dataloader properties for later
        self.batch_size = batch_size
        self.pre_sample_dataset = pre_sample_dataset

        # Load full training dataset
        if type(samples_for_dataset) is int:
            samples_for_dataset = list(range(samples_for_dataset))

        num_idxs = samples_for_dataset
        num_idxs = np.array(num_idxs)
        full_dataset, partial_dataset, full_grid_states, partial_grid_states = load_full_partial_training_dataset(root_dir, num_idxs)
        full_input_data, full_output_data = full_dataset
        partial_input_data, partial_output_data = partial_dataset

        # Stack full and partial input/output data separately
        full_input_data = torch.stack(full_input_data, dim=0)[:, :, :, 0]
        full_output_data = torch.stack(full_output_data, dim=0)
        partial_input_data = torch.stack(partial_input_data, dim=0)[:, :, :, 0]
        partial_output_data = torch.stack(partial_output_data, dim=0)

        # Figure out how to divide train and test data 
        num_full_data = full_input_data.shape[0]
        num_partial_data = partial_input_data.shape[0]
        num_full_data_for_train = int(np.ceil(num_full_data * (1 - percent_test)))
        num_partial_data_for_train = int(np.ceil(num_partial_data * (1 - percent_test)))
        if num_full_data_for_train == num_full_data: 
            num_full_data_for_train -= 1  # ensure at least one sample for testing
        if num_partial_data_for_train == num_partial_data: 
            num_partial_data_for_train -= 1

        print("Retrieved Full Partial Dataset")
        print(f"Number of full data samples: {num_full_data}, number of partial data samples: {num_partial_data}")
        print(f"Number of full data samples for training: {num_full_data_for_train}, number of partial data samples for training: {num_partial_data_for_train}")
        print(f"Number of full data samples for testing: {num_full_data - num_full_data_for_train}, number of partial data samples for testing: {num_partial_data - num_partial_data_for_train}")

        
        # Split into train/test for full
        full_train_input_data = full_input_data[:num_full_data_for_train]
        full_test_input_data = full_input_data[num_full_data_for_train:]
        full_train_output_data = full_output_data[:num_full_data_for_train]
        full_test_output_data = full_output_data[num_full_data_for_train:]

        # Split into train/test for partial
        partial_train_input_data = partial_input_data[:num_partial_data_for_train]
        partial_test_input_data = partial_input_data[num_partial_data_for_train:]
        partial_train_output_data = partial_output_data[:num_partial_data_for_train]
        partial_test_output_data = partial_output_data[num_partial_data_for_train:]

        # Store train/test data for both full and partial datasets
        self.full_train_input_data = full_train_input_data
        self.full_train_output_data = full_train_output_data
        self.full_test_input_data = full_test_input_data
        self.full_test_output_data = full_test_output_data
        self.full_all_input_data = full_input_data
        self.full_all_output_data = full_output_data

        self.partial_train_input_data = partial_train_input_data
        self.partial_train_output_data = partial_train_output_data
        self.partial_test_input_data = partial_test_input_data
        self.partial_test_output_data = partial_test_output_data
        self.partial_all_input_data = partial_input_data
        self.partial_all_output_data = partial_output_data

        self.full_grid_states = full_grid_states
        self.full_grid_resolution = self.full_grid_states.shape[:-1]  # get grid resolution from the full grid states tensor
        self.partial_grid_states = partial_grid_states
        self.partial_grid_resolution = self.partial_grid_states.shape[:-1]  # get grid resolution from the partial grid states tensor


        # Channel dim: If the channels dim is 1, whether that is explicitly kept in the saved tensor. If not, we need to unsqueeze it to explicitly have a channel dim. 
        # NOTE: might need to do this
        channel_dim = 1 # have the channels lie on the 1st dimension of the tensor

        ################ Debugging print statements ################
        # print("Full train input data shape:", self.full_train_input_data.shape)
        # print("Full train output data shape:", self.full_train_output_data.shape)
        # print("Full grid states shape:", self.full_grid_states.shape)

        # print("Partial train input data shape:", self.partial_train_input_data.shape)
        # print("Partial train output data shape:", self.partial_train_output_data.shape)
        # print("Partial grid states shape:", self.partial_grid_states.shape)

        # print()
        # print()

        # print("Full test input data shape:", self.full_test_input_data.shape)
        # print("Full test output data shape:", self.full_test_output_data.shape)
        # print("Full grid states shape:", self.full_grid_states.shape)

        # print("Partial test input data shape:", self.partial_test_input_data.shape)
        # print("Partial test output data shape:", self.partial_test_output_data.shape)
        # print("Partial grid states shape:", self.partial_grid_states.shape)
        ################ Debugging print statements ################


        # Create full and partial train/test datasets
        self._full_train_db = HJR_SubDataset(
            input_data=self.full_train_input_data,
            output_data=self.full_train_output_data,
            grid_states=self.full_grid_states,
            datapoints_per_sample=datapoints_per_full_sample,
            channel_squeezed=True,
            channel_dim=channel_dim,
            transform_x=None,
            transform_y=None,
            random_samples=not pre_sample_dataset
        )

        self._full_test_db = HJR_SubDataset(
            input_data=self.full_test_input_data,
            output_data=self.full_test_output_data,
            grid_states=self.full_grid_states,
            datapoints_per_sample=datapoints_per_full_sample,
            channel_squeezed=True,
            channel_dim=channel_dim,
            transform_x=None,
            transform_y=None,
            random_samples=not pre_sample_dataset
        )

        self._partial_train_db = HJR_SubDataset(
            input_data=self.partial_train_input_data,
            output_data=self.partial_train_output_data,
            grid_states=self.partial_grid_states,
            datapoints_per_sample=datapoints_per_partial_sample,
            channel_squeezed=True,
            channel_dim=channel_dim,
            transform_x=None,
            transform_y=None,
            random_samples=not pre_sample_dataset
        )

        self._partial_test_db = HJR_SubDataset(
            input_data=self.partial_test_input_data,
            output_data=self.partial_test_output_data,
            grid_states=self.partial_grid_states,
            datapoints_per_sample=datapoints_per_partial_sample,
            channel_squeezed=True,
            channel_dim=channel_dim,
            transform_x=None,
            transform_y=None,
            random_samples=not pre_sample_dataset
        )

        # Create the actual train and test datasets
        self._train_db = HJRFullPartial_SubDataset(self._full_train_db, self._partial_train_db)
        self._test_db = HJRFullPartial_SubDataset(self._full_test_db, self._partial_test_db)
        self._test_dbs = [self._test_db]

        # 2. Create input and output encoders - use attributes from the training datasets to fit the encoders
        if encode_input:
            if encoding == "channel-wise":
                reduce_dims = list(range(self._train_db.full_hjr_sub_dataset.input_data[0].ndim + 1))
                # preserve mean for each channel
                reduce_dims.pop(channel_dim)
            elif encoding == "pixel-wise":
                reduce_dims = [0]

            self.input_encoder = UnitGaussianNormalizer(dim=reduce_dims)
            self.input_encoder.fit(self._train_db.get_inputs_for_normalization(normalization_size=len(self._train_db)))
        else: 
            self.input_encoder = None 

        if encode_output: 
            if encoding == "channel-wise":
                reduce_dims = list(range(self._train_db.full_hjr_sub_dataset.output_data[0].ndim - 2 + 1))
                # preserve mean for each channel
                reduce_dims.pop(channel_dim)
            elif encoding == "pixel-wise":
                reduce_dims = [0]

            self.output_encoder = UnitGaussianNormalizer(dim=reduce_dims)
            # self.output_encoder.fit(self._train_db.get_outputs_for_normalization(normalization_size=len(self._train_db)))
            outputs_for_fitting = self._train_db.get_outputs_for_normalization(normalization_size=len(self._train_db))
            print("Outputs for fitting shape:", outputs_for_fitting.shape)
            self.output_encoder.fit()
        else:
            self.output_encoder = None

        self.train_dataset_len = len(self._train_db)
        self.test_dataset_len = len(self._test_db)

        # 3. Create data processor 
        self._data_processor = DefaultDataProcessor(in_normalizer=self.input_encoder,
                                                   out_normalizer=self.output_encoder)

    @property
    def data_processor(self):
        return self._data_processor
    
    @property
    def train_db(self):
        return self._train_db
    
    @property
    def test_dbs(self):
        return self._test_dbs


class HJRFullPartial_SubDataset(torch.utils.data.Dataset):
    """
    To use for the train_db and test_db. - combines two individual HJR_SubDataset instances 
    TODO: implement the __getitem__ method to return a sample from the dataset - get random indices from the grid states with every call 
    and return corresponding input and output 
    """
    def __init__(self, full_hjr_sub_dataset, partial_hjr_sub_dataset):
        """
        Arguments: 
        - full_hjr_sub_dataset: HJR_SubDataset instance for the full grid states
        - partial_hjr_sub_dataset: HJR_SubDataset instance for the partial grid states
        """
        self.full_hjr_sub_dataset = full_hjr_sub_dataset
        self.partial_hjr_sub_dataset = partial_hjr_sub_dataset

        self.dataset_size = len(self.full_hjr_sub_dataset) + len(self.partial_hjr_sub_dataset)

        # Create flags and indices
        self.full_partial_flags = [0] * len(self.full_hjr_sub_dataset) + [1] * len(self.partial_hjr_sub_dataset)
        self.full_partial_indices = list(range(self.full_hjr_sub_dataset.dataset_size)) + list(range(self.partial_hjr_sub_dataset.dataset_size))
        
        # Shuffle both lists in the same way
        combined = list(zip(self.full_partial_flags, self.full_partial_indices))
        random.shuffle(combined)
        self.full_partial_flags, self.full_partial_indices = zip(*combined)
        
        self.full_partial_flags = np.array(list(self.full_partial_flags))
        self.full_partial_indices = np.array(list(self.full_partial_indices))

        self.datasets = [self.full_hjr_sub_dataset, self.partial_hjr_sub_dataset]
        return 
    
    def __len__(self): 
        return self.dataset_size 
    
    def __getitem__(self, idx):
        full_partial_flag = self.full_partial_flags[idx]
        full_partial_index = self.full_partial_indices[idx]

        return self.datasets[full_partial_flag].__getitem__(full_partial_index)
    
    def get_inputs_for_normalization(self, normalization_size=None):
        """
        Get list of inputs to use for normalization 
        """
        full_inputs_for_normalization = self.full_hjr_sub_dataset.get_inputs_for_normalization(normalization_size=normalization_size)
        partial_inputs_for_normalization = self.partial_hjr_sub_dataset.get_inputs_for_normalization(normalization_size=normalization_size)

        all_inputs_for_normalization = torch.cat((full_inputs_for_normalization, partial_inputs_for_normalization), dim=0)
        return all_inputs_for_normalization
    
    def get_outputs_for_normalization(self, normalization_size=None):
        """
        Get list of outputs to use for normalization 
        """
        full_outputs_for_normalization = self.full_hjr_sub_dataset.get_outputs_for_normalization(normalization_size=normalization_size)
        partial_outputs_for_normalization = self.partial_hjr_sub_dataset.get_outputs_for_normalization(normalization_size=normalization_size)

        all_outputs_for_normalization = torch.cat((full_outputs_for_normalization, partial_outputs_for_normalization), dim=0)
        return all_outputs_for_normalization





# ########## OLD OLD OLD DEFUNCT DEFUNCT DEFUNCT ##########
# class HJRFullPartial_SubDataset(torch.utils.data.Dataset):
#     """
#     To use for the train_db and test_db. 
#     TODO: implement the __getitem__ method to return a sample from the dataset - get random indices from the grid states with every call 
#     and return corresponding input and output 
#     """
#     def __init__(self, full_input_data, full_output_data, partial_input_data, partial_output_data,
#                  full_grid_states, partial_grid_states, datapoints_per_sample, 
#                  channel_squeezed=True, channel_dim=1,
#                  transform_x=None, transform_y=None, random_samples=False, 
#                  full_ordered_dataset=False):
#         """
#         Arguments: 

#         - input_data: torch tensor of list of torch tensors - shape (num_samples, 1, grid_resolution[0], grid_resolution[1])
#             Represents the spatially varying disturbance function 
#         - output_data: torch tensor of list of torch tensors: shape (num_samples, grid_resolution[0], grid_resolution[1], grid_resolution[2], grid_resolution[3])
#             Represents the infinite time HJR solution value function for the associated disturbance function
#         - full_grid_states: torch.tensor - shape (grid_resolution[0], grid_resolution[1], grid_resolution[2], grid_resolution[3], 4)
#             Represents the grid states for the HJR solution: each point in the grid is a state in the system - full grid 
#         - partial_grid_states: torch.tensor - shape (grid_resolution[0], grid_resolution[1], grid_resolution[2], grid_resolution[3], 4)
#             Represents the grid states for the HJR solution: each point in the grid is a state in the system - partial grid 
#         - datapoints_per_sample: int - number of datapoints to get per HJR solution sample in dataset
#         - channel_squeezed: bool, optional If the channels dim is 1, whether that is explicitly kept in the saved tensor,
#         - channel_dim: int, dimension of the input and output tensors that represents the channels, by default 1
#         - transform_x: callable, optional see PTDataset for more details
#         - transform_y: callable, optional see PTDataset for more details
#         - random_samples: bool, optional
#             whether to randomly sample the input and output data with each __getitem__ call, by default False
#             when False: pre-sample the input and output data and index into these samples with the idx in __getitem__
#         - full_ordered_dataset : bool, optional
#             whether to create a full ordered dataset, where each grid slice is sampled in order
#         """
        
#         assert(len(full_input_data) + len(partial_input_data) == len(full_output_data) + len(partial_output_data)), "Size mismatch between input and output datasets"

#         self.full_input_data = full_input_data
#         self.full_output_data = full_output_data
#         self.partial_input_data = partial_input_data
#         self.partial_output_data = partial_output_data

#         self.full_grid_states = full_grid_states
#         self.partial_grid_states = partial_grid_states
#         self.full_grid_resolution = self.full_grid_states.shape[:-1]  
#         self.partial_grid_resolution = self.partial_grid_states.shape[:-1]  

#         self.channel_dim = channel_dim
#         self.random_samples = random_samples
#         self.full_ordered_dataset = full_ordered_dataset
#         if self.full_ordered_dataset: 
#             assert(self.random_samples is False), "Cannot create full ordered dataset with random samples"

#         full_expand_dims = [-1, self.full_grid_resolution[0], self.full_grid_resolution[1]]
#         full_expand_dims.insert(self.channel_dim, 1)
#         self.full_expand_dims = full_expand_dims
#         partial_expand_dims = [-1, self.partial_grid_resolution[0], self.partial_grid_resolution[1]]
#         partial_expand_dims.insert(self.channel_dim, 1)
#         self.partial_expand_dims = partial_expand_dims

#         assert(datapoints_per_sample <= self.grid_resolution[2] * self.grid_resolution[3])
#         self.full_dataset_size = len(full_input_data) * datapoints_per_sample
#         self.partial_dataset_size = len(partial_input_data) * datapoints_per_sample
#         self.dataset_size = self.full_dataset_size + self.partial_dataset_size

#         self.transform_x = transform_x
#         self.transform_y = transform_y

#         if channel_squeezed: 
#             self.full_input_data = self.full_input_data.unsqueeze(self.channel_dim)
#             self.full_output_data = self.full_output_data.unsqueeze(self.channel_dim)
#             self.partial_input_data = self.partial_input_data.unsqueeze(self.channel_dim)
#             self.partial_output_data = self.partial_output_data.unsqueeze(self.channel_dim)

#         self.full_sample_tupe_list_for_normalization = None
#         self.partial_sample_tuple_list_for_normalization = None

#         # Create the sample tuple list for the dataset
#         if not self.random_samples: 
#             if self.full_ordered_dataset:
#                 self.full_sample_tuple_list = np.array([[i, j, k] for i in range(self.full_grid_resolution[2]) for j in range(self.full_grid_resolution[3]) for k in range(len(self.full_input_data))]) 
#                 self.partial_sample_tuple_list = np.array([[i, j, k] for i in range(self.partial_grid_resolution[2]) for j in range(self.partial_grid_resolution[3]) for k in range(len(self.partial_input_data))]) 
#                 self.full_sample_tuple_list_for_normalization = self.full_sample_tuple_list
#                 self.partial_sample_tuple_list_for_normalization = self.partial_sample_tuple_list

#             else: # Pre-sample the input and output data: [xvel index, yvel index, input data index]
#                 self.full_sample_tuple_list = np.array(random.sample([[i, j, k] for i in range(self.full_grid_resolution[2]) for j in range(self.full_grid_resolution[3]) for k in range(len(self.full_input_data))], self.full_dataset_size))
#                 self.partial_sample_tuple_list = np.array(random.sample([[i, j, k] for i in range(self.partial_grid_resolution[2]) for j in range(self.partial_grid_resolution[3]) for k in range(len(self.partial_input_data))], self.partial_dataset_size))
#                 self.full_sample_tuple_list_for_normalization = self.full_sample_tuple_list
#                 self.partial_sample_tuple_list_for_normalization = self.partial_sample_tuple_list
#         return 

#     ###### PICKUP HERE ######
#     def sample_tuple_list_to_inputs(self, sample_tuple_list): 
#         """
#         Take the sample tuple list and construct the input tensors from the input data 
#         Structure of sublist in sample_tuple_list: [xvel index, yvel index, input data index]
#         """
#         grid_states_sampled = self.grid_states[0, 0, sample_tuple_list[:, 0], sample_tuple_list[:, 1]]
#         grid_states_sampled_xvel = grid_states_sampled[:, 2]
#         grid_states_sampled_yvel = grid_states_sampled[:, 3]
        
#         sample_input_dataset = self.input_data[sample_tuple_list[:, 2], ...]

#         appended_sample_xvel = torch.tensor(grid_states_sampled_xvel).view(-1, 1, 1, 1).expand(*self.expand_dims)
#         append_sampled_yvel = torch.tensor(grid_states_sampled_yvel).view(-1, 1, 1, 1).expand(*self.expand_dims)
#         full_input_samples = torch.cat((sample_input_dataset, appended_sample_xvel, append_sampled_yvel), dim=self.channel_dim)

#         if len(sample_tuple_list) == 1: 
#             return full_input_samples.squeeze(0)  # return a single sample without batch dimension
#         return full_input_samples 

#     def sample_tuple_list_to_outputs(self, sample_tuple_list):
#         """
#         Take the sample tuple list and construct the output tensors from the output data 
#         Structure of sublist in sample_tuple_list: [xvel index, yvel index, input data index]
#         """
#         xvel_indices = sample_tuple_list[:, 0]
#         yvel_indices = sample_tuple_list[:, 1]
#         sample_indices = sample_tuple_list[:, 2]

#         # NOTE: ASSUMES CHANNEL DIM IS 1 - do dynamically later 
#         output_samples = self.output_data[sample_indices, :, :, :, xvel_indices, yvel_indices] # [sample index, channel dim, x, y, xvel, yvel]

#         if len(sample_tuple_list) == 1:
#             return output_samples.squeeze(0)
#         return output_samples

#     def get_inputs_for_normalization(self, normalization_size=None): 
#         # Get list of inputs to use for normalization 
#         if normalization_size is None:
#             normalization_size = self.dataset_size

#         if not self.random_samples: 
#             return self.sampled_input_dataset[:normalization_size]

#         if self.sample_tuple_list_for_normalization is None: 
#             # [xvel index, yvel index, input data index]
#             self.sample_tuple_list_for_normalization = np.array(random.sample([[i, j, k] for i in range(self.grid_resolution[2]) for j in range(self.grid_resolution[3]) for k in range(len(self.input_data))], normalization_size))
#         return self.sample_tuple_list_to_inputs(self.sample_tuple_list_for_normalization)

#     def get_outputs_for_normalization(self, normalization_size=None): 
#         # Get list of inputs to use for normalization 
#         if normalization_size is None:
#             normalization_size = self.dataset_size

#         if not self.random_samples: 
#             return self.sampled_output_dataset[:normalization_size]

#         if self.sample_tuple_list_for_normalization is None: 
#             # [xvel index, yvel index, input data index]
#             self.sample_tuple_list_for_normalization = np.array(random.sample([[i, j, k] for i in range(self.grid_resolution[2]) for j in range(self.grid_resolution[3]) for k in range(len(self.output_data))], normalization_size))
#         return self.sample_tuple_list_to_outputs(self.sample_tuple_list_for_normalization)

#     def __len__(self):
#         return self.dataset_size
    
#     def __getitem__(self, idx):
#         if self.random_samples: 
#             # Randomly sample the input and output data with each call 
#             sample_tuple = np.array(random.sample([[i, j, k] for i in range(self.grid_resolution[2]) for j in range(self.grid_resolution[3]) for k in range(len(self.output_data))], 1))
#             input_sample = self.sample_tuple_list_to_inputs(sample_tuple)
#             output_sample = self.sample_tuple_list_to_outputs(sample_tuple)
#         else: 
#             # Get the sample tuple for the given index
#             sample_tuple = np.array([self.sample_tuple_list[idx]])
#             # NOTE: might need to change this later for sampling 
#             input_sample = self.sampled_input_dataset[idx]
#             output_sample = self.sampled_output_dataset[idx]

#         # Transform 
#         if self.transform_x is not None:
#             input_sample = self.transform_x(input_sample)
#         if self.transform_y is not None:
#             output_sample = self.transform_y(output_sample)

#         # Create dictionary 
#         return_dict = {'x': input_sample, 'y': output_sample, 'sample_tuple': sample_tuple}
#         return return_dict 
