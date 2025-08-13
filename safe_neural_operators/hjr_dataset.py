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

def load_full_training_dataset(root_dir, num_idxs): 
    """
    Load the full training dataset from the specified root directory
    num_idxs: int or list of indices - if int, load that many samples from the dataset, if range, load samples in that range
    """
    grid_states = None 
    input_data = []
    output_data = []

    # Load the grid states from the file
    grid_states = load_grid_states(load_path=root_dir, load_index=0)

    if type(num_idxs) is int: 
        num_idxs = range(num_idxs)

    for idx in tqdm(num_idxs): 
        input_tensor, output_tensor = load_hjr_solution(load_path=root_dir, 
                                                        load_index=idx)
        input_data.append(input_tensor)
        output_data.append(output_tensor)

    return input_data, output_data, grid_states

def sample_input_data(input_data, sample_tuple_list, grid_states): 
    """
    Arguments: 
    - input_data: torch tensor of a list of torch tensors - shape (num_samples, 1, grid_resolution[0], grid_resolution[1])
        Represents the spatially varying disturbance function
    - sample_tuple_list: np array of list of lists - each tuple contains indices to sample from the input data
        e.g. [(i, j, k), ...] where i,j are the indices of the grid states to use and k is the index of the input data sample
    - grid_states: torch.tensor - shape (grid_resolution[0], grid_resolution[1], grid_resolution[2], grid_resolution[3], 4)
        Represents the grid states for the HJR solution: each point in the grid is a state in the system
    """
    raise NotImplementedError("TODO: Implement sampling of input data")
    # 1. Go through all the sample_tuple_list
    input_data_sample_idxs = sample_tuple_list[:, 2]
    xvel_idxs = sample_tuple_list[:, 0]
    yvel_idxs = sample_tuple_list[:, 1]

    

    # 2. For each sample tuple, get the corresopnding input data and stack a tensor with appropriate grid states



def sample_output_data(output_data, sample_tuple_list):
    raise NotImplementedError("TODO: Implement sampling of output data")


class HJRDataset: 
    """
    HJR dataset class to use for HJR data from data_collection.py
    """
    def __init__(self, 
                 root_dir: str, 
                 samples_for_train: Union[int, List[int]], 
                 samples_for_test: Union[int, List[int]], 
                 datapoints_per_sample: int,

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
        samples_for_train : int or list of int
            number of hjr solution samples to use for training 
            or list of indices to use for training 
        samples_for_test : int
            number of hjr solution samples to use for testing
            or list of indices to use for testing
        datapoints_per_sample: int
            number of datapoints to get per sample, NOTE: must be lower than grid_resolution[2] * grid_resolution[3] - as you will be randomly sampling these dimensions
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
        if type(samples_for_train) is int:
            samples_for_train = list(range(samples_for_train))
        if type(samples_for_test) is int:
            samples_for_test = list(range(len(samples_for_train), len(samples_for_train) + samples_for_test))

        num_idxs = samples_for_train + samples_for_test
        num_idxs = np.array(num_idxs)
        input_data, output_data, grid_states = load_full_training_dataset(root_dir, num_idxs)

        input_data = torch.stack(input_data, dim=0)[:, :, :, 0]
        output_data = torch.stack(output_data, dim=0)

        train_input_data = input_data[:len(samples_for_train)]
        test_input_data = input_data[len(samples_for_train):len(samples_for_train) + len(samples_for_test)]
        train_output_data = output_data[:len(samples_for_train)]
        test_output_data = output_data[len(samples_for_train):len(samples_for_train) + len(samples_for_test)]

        self.train_input_data = train_input_data
        self.train_output_data = train_output_data
        self.test_input_data = test_input_data
        self.test_output_data = test_output_data
        self.all_input_data = input_data
        self.all_output_data = output_data

        self.grid_states = grid_states
        self.grid_resolution = self.grid_states.shape[:-1] # get grid resolution from the grid states tensor

        # Channel dim: If the channels dim is 1, whether that is explicitly kept in the saved tensor. If not, we need to unsqueeze it to explicitly have a channel dim. 
        # NOTE: might need to do this
        channel_dim = 1 # have the channels lie on the 1st dimension of the tensor

        # 1. Create the training dataset 
        self._train_db = HJR_SubDataset(input_data=self.train_input_data, 
                                        output_data=self.train_output_data, 
                                        grid_states=self.grid_states, 
                                        datapoints_per_sample=datapoints_per_sample,
                                        channel_squeezed=True, 
                                        channel_dim=channel_dim,
                                        transform_x=None, 
                                        transform_y=None,
                                        random_samples=not pre_sample_dataset)

        self._test_db = HJR_SubDataset(input_data=self.test_input_data, 
                                       output_data=self.test_output_data, 
                                       grid_states=self.grid_states, 
                                       datapoints_per_sample=datapoints_per_sample,
                                       channel_squeezed=True, 
                                       channel_dim=channel_dim,
                                       transform_x=None, 
                                       transform_y=None,
                                       random_samples=not pre_sample_dataset)
        self._test_dbs = [self._test_db]

        # 2. Create input and output encoders - use attributes from the training datasets to fit the encoders
        if encode_input:
            if encoding == "channel-wise":
                reduce_dims = list(range(self._train_db.input_data[0].ndim + 1))
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
                reduce_dims = list(range(self._train_db.output_data[0].ndim - 2 + 1)) # - 2 as an xvel, yvel slice is returned (truncating 2 dimensions)
                # preserve mean for each channel
                reduce_dims.pop(channel_dim)
            elif encoding == "pixel-wise":
                reduce_dims = [0]

            self.output_encoder = UnitGaussianNormalizer(dim=reduce_dims)
            self.output_encoder.fit(self._train_db.get_outputs_for_normalization(normalization_size=len(self._train_db)))
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



class HJR_SubDataset(torch.utils.data.Dataset):
    """
    To use for the train_db and test_db. 
    TODO: implement the __getitem__ method to return a sample from the dataset - get random indices from the grid states with every call 
    and return corresponding input and output 
    """
    def __init__(self, input_data, output_data, grid_states, datapoints_per_sample, 
                 channel_squeezed=True, channel_dim=1,
                 transform_x=None, transform_y=None, random_samples=False, 
                 full_ordered_dataset=False):
        """
        Arguments: 

        - input_data: torch tensor of list of torch tensors - shape (num_samples, 1, grid_resolution[0], grid_resolution[1])
            Represents the spatially varying disturbance function 
        - output_data: torch tensor of list of torch tensors: shape (num_samples, grid_resolution[0], grid_resolution[1], grid_resolution[2], grid_resolution[3])
            Represents the infinite time HJR solution value function for the associated disturbance function
        - grid_states: torch.tensor - shape (grid_resolution[0], grid_resolution[1], grid_resolution[2], grid_resolution[3], 4)
            Represents the grid states for the HJR solution: each point in the grid is a state in the system
        - datapoints_per_sample: int - number of datapoints to get per HJR solution sample in dataset
        - channel_squeezed: bool, optional If the channels dim is 1, whether that is explicitly kept in the saved tensor,
        - channel_dim: int, dimension of the input and output tensors that represents the channels, by default 1
        - transform_x: callable, optional see PTDataset for more details
        - transform_y: callable, optional see PTDataset for more details
        - random_samples: bool, optional
            whether to randomly sample the input and output data with each __getitem__ call, by default False
            when False: pre-sample the input and output data and index into these samples with the idx in __getitem__
        - full_ordered_dataset : bool, optional
            whether to create a full ordered dataset, where each grid slice is sampled in order
        """
        assert(len(input_data) == len(output_data)), "Size mismatch between input and output datasets"
        self.input_data = input_data
        self.output_data = output_data
        self.grid_states = grid_states
        self.grid_resolution = self.grid_states.shape[:-1]  
        self.channel_dim = channel_dim
        self.random_samples = random_samples
        self.full_ordered_dataset = full_ordered_dataset
        if self.full_ordered_dataset: 
            assert(self.random_samples is False), "Cannot create full ordered dataset with random samples"

        expand_dims = [-1, self.grid_resolution[0], self.grid_resolution[1]]
        expand_dims.insert(self.channel_dim, 1)
        self.expand_dims = expand_dims

        assert(datapoints_per_sample <= self.grid_resolution[2] * self.grid_resolution[3])
        self.dataset_size = len(input_data) * datapoints_per_sample

        self.transform_x = transform_x
        self.transform_y = transform_y

        if channel_squeezed: 
            self.input_data = self.input_data.unsqueeze(self.channel_dim) 
            self.output_data = self.output_data.unsqueeze(self.channel_dim) 

        self.sample_tuple_list_for_normalization = None 
        if not random_samples: 
            if self.full_ordered_dataset:
                self.sample_tuple_list = np.array([[i, j, k] for i in range(self.grid_resolution[2]) for j in range(self.grid_resolution[3]) for k in range(len(input_data))])
                self.sample_tuple_list_for_normalization = self.sample_tuple_list
            else: # Pre-sample the input and output data: [xvel index, yvel index, input data index]
                self.sample_tuple_list = np.array(random.sample([[i, j, k] for i in range(self.grid_resolution[2]) for j in range(self.grid_resolution[3]) for k in range(len(input_data))], self.dataset_size))
                self.sample_tuple_list_for_normalization = self.sample_tuple_list

            self.sampled_input_dataset = self.sample_tuple_list_to_inputs(self.sample_tuple_list)
            self.sampled_output_dataset = self.sample_tuple_list_to_outputs(self.sample_tuple_list)
        return 

    def sample_tuple_list_to_inputs(self, sample_tuple_list): 
        """
        Take the sample tuple list and construct the input tensors from the input data 
        Structure of sublist in sample_tuple_list: [xvel index, yvel index, input data index]
        """
        grid_states_sampled = self.grid_states[0, 0, sample_tuple_list[:, 0], sample_tuple_list[:, 1]]
        grid_states_sampled_xvel = grid_states_sampled[:, 2]
        grid_states_sampled_yvel = grid_states_sampled[:, 3]
        
        sample_input_dataset = self.input_data[sample_tuple_list[:, 2], ...]

        appended_sample_xvel = torch.tensor(grid_states_sampled_xvel).view(-1, 1, 1, 1).expand(*self.expand_dims)
        append_sampled_yvel = torch.tensor(grid_states_sampled_yvel).view(-1, 1, 1, 1).expand(*self.expand_dims)
        full_input_samples = torch.cat((sample_input_dataset, appended_sample_xvel, append_sampled_yvel), dim=self.channel_dim)

        if len(sample_tuple_list) == 1: 
            return full_input_samples.squeeze(0)  # return a single sample without batch dimension
        return full_input_samples 

    def sample_tuple_list_to_outputs(self, sample_tuple_list):
        """
        Take the sample tuple list and construct the output tensors from the output data 
        Structure of sublist in sample_tuple_list: [xvel index, yvel index, input data index]
        """
        xvel_indices = sample_tuple_list[:, 0]
        yvel_indices = sample_tuple_list[:, 1]
        sample_indices = sample_tuple_list[:, 2]

        # NOTE: ASSUMES CHANNEL DIM IS 1 - do dynamically later 
        output_samples = self.output_data[sample_indices, :, :, :, xvel_indices, yvel_indices] # [sample index, channel dim, x, y, xvel, yvel]

        if len(sample_tuple_list) == 1:
            return output_samples.squeeze(0)
        return output_samples

    def get_inputs_for_normalization(self, normalization_size=None): 
        # Get list of inputs to use for normalization 
        if normalization_size is None:
            normalization_size = self.dataset_size

        if not self.random_samples: 
            return self.sampled_input_dataset[:normalization_size]

        if self.sample_tuple_list_for_normalization is None: 
            # [xvel index, yvel index, input data index]
            self.sample_tuple_list_for_normalization = np.array(random.sample([[i, j, k] for i in range(self.grid_resolution[2]) for j in range(self.grid_resolution[3]) for k in range(len(self.input_data))], normalization_size))
        return self.sample_tuple_list_to_inputs(self.sample_tuple_list_for_normalization)

    def get_outputs_for_normalization(self, normalization_size=None): 
        # Get list of inputs to use for normalization 
        if normalization_size is None:
            normalization_size = self.dataset_size

        if not self.random_samples: 
            return self.sampled_output_dataset[:normalization_size]

        if self.sample_tuple_list_for_normalization is None: 
            # [xvel index, yvel index, input data index]
            self.sample_tuple_list_for_normalization = np.array(random.sample([[i, j, k] for i in range(self.grid_resolution[2]) for j in range(self.grid_resolution[3]) for k in range(len(self.output_data))], normalization_size))
        return self.sample_tuple_list_to_outputs(self.sample_tuple_list_for_normalization)

    def __len__(self):
        return self.dataset_size
    
    def __getitem__(self, idx):
        if self.random_samples: 
            # Randomly sample the input and output data with each call 
            sample_tuple = np.array(random.sample([[i, j, k] for i in range(self.grid_resolution[2]) for j in range(self.grid_resolution[3]) for k in range(len(self.output_data))], 1))
            input_sample = self.sample_tuple_list_to_inputs(sample_tuple)
            output_sample = self.sample_tuple_list_to_outputs(sample_tuple)
        else: 
            # Get the sample tuple for the given index
            sample_tuple = np.array([self.sample_tuple_list[idx]])
            # NOTE: might need to change this later for sampling 
            input_sample = self.sampled_input_dataset[idx]
            output_sample = self.sampled_output_dataset[idx]

        # Transform 
        if self.transform_x is not None:
            input_sample = self.transform_x(input_sample)
        if self.transform_y is not None:
            output_sample = self.transform_y(output_sample)

        # Create dictionary 
        return_dict = {'x': input_sample, 'y': output_sample, 'sample_tuple': sample_tuple}
        return return_dict 
