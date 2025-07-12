# Imports 
import torch 
from torch.utils.data import DataLoader

import random
import numpy as np 
from matplotlib import pyplot as plt

from hjr_dataset import HJRDataset

import sys 
sys.path.append("/home/jingpei/Documents/arclab/L4DC25_project")
sys.path.append("..")

from deepreach.dynamics import dynamics 
from deepreach.dynamics import dynamics_hjr
from deepreach.utils.comparisons import GroundTruthHJSolution

# NeuralOp Imports 
from neuralop.models import FNO
from neuralop import Trainer
from neuralop.training import AdamW
from neuralop.data.datasets import load_darcy_flow_small
from neuralop.utils import count_model_params
from neuralop import LpLoss, H1Loss


def main(data_root_dir, 
         samples_for_train, 
         samples_for_test, 
         datapoints_per_sample,
         batch_size,
         pre_sample_dataset, #=True,
         encode_output, #=False,
         encode_input, #=True,
         encoding, #="channel-wise", 

         n_modes, 
         in_channels, 
         out_channels, 
         hidden_channels, 
         projection_channel_ratio, 
         factorization, 
         rank, 
         model_lr, # 8e-3
         model_weight_decay, # 1e-4
         
         save_dir, 
         device, 
         train_epochs, 
         save_every):
    
    # Create Dataset 
    dataset = HJRDataset(root_dir=data_root_dir, 
                    samples_for_train=samples_for_train, 
                    samples_for_test=samples_for_test, 
                    datapoints_per_sample=datapoints_per_sample,

                    batch_size=batch_size, 
                    pre_sample_dataset=pre_sample_dataset, 

                    encode_output=encode_output,
                    encode_input=encode_input,
                    encoding=encoding)
    
    # Create DataLoaders
    train_loader = DataLoader(dataset.train_db, 
                          batch_size=batch_size, 
                          num_workers=1, 
                          pin_memory=True,
                          persistent_workers=False)

    test_db = dataset.test_dbs[0]
    test_loader = DataLoader(test_db,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=1,
                            pin_memory=True,
                            persistent_workers=False)

    data_processor = dataset.data_processor
    print("\n\n\nTrain Dataset Size: ", len(dataset.train_db))
    print("Test Dataset Size: ", len(dataset.test_dbs[0]))
    print("\n\n\n")

    # Create Model 
    model = FNO(n_modes=n_modes,
                in_channels=in_channels,
                out_channels=out_channels,
                hidden_channels=hidden_channels,
                projection_channel_ratio=projection_channel_ratio,
                factorization=factorization,
                rank=rank)
    
    n_params = count_model_params(model)
    print(f'\nOur model has {n_params} parameters.')

    # Create the optimizer
    optimizer = AdamW(model.parameters(), 
                                    lr=model_lr, 
                                    weight_decay=model_weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

    # Creating the losses
    l2loss = LpLoss(d=2, p=2)
    h1loss = H1Loss(d=2)

    train_loss = h1loss
    eval_losses={'h1': h1loss, 'l2': l2loss}


    # Print Logging 
    print('\n### MODEL ###\n', model)
    print('\n### OPTIMIZER ###\n', optimizer)
    print('\n### SCHEDULER ###\n', scheduler)
    print('\n### LOSSES ###')
    print(f'\n * Train: {train_loss}')
    print(f'\n * Test: {eval_losses}')

    # Create the trainer
    trainer = Trainer(model=model, n_epochs=train_epochs,
                    device=device,
                    data_processor=data_processor,
                    wandb_log=False,
                    eval_interval=3,
                    use_distributed=False,
                    verbose=True)


    trainer.train(train_loader=train_loader,
                test_loaders={},
                optimizer=optimizer,
                scheduler=scheduler, 
                regularizer=False, 
                training_loss=train_loss, 
                save_every=save_every,
                save_dir=save_dir)


if __name__ == "__main__":
    
    # Dataset Parameters 
    data_root_dir = "/media/jingpei/DATA/fno_gp_data"
    samples_for_train = 90
    samples_for_test = 10 
    datapoints_per_sample = 1000
    batch_size = 32
    pre_sample_dataset = True 
    encode_output = False 
    encode_input = True 
    encoding = "channel-wise"

    # Model Parameters
    n_modes = (16, 16)
    in_channels = 3
    out_channels = 1
    hidden_channels = 64
    projection_channel_ratio = 2
    factorization = 'tucker'  
    rank = 0.42
    model_lr = 8e-3
    model_weight_decay = 1e-4
    

    # Saving Parameters
    save_dir = "/media/jingpei/DATA/fno_models/safe_neural-7-12-25"
    device= "cuda:0"
    train_epochs = 100
    save_every = 1


    main(data_root_dir=data_root_dir, 
         samples_for_train=samples_for_train, 
         samples_for_test=samples_for_test, 
         datapoints_per_sample=datapoints_per_sample,
         batch_size=batch_size, 
         pre_sample_dataset=pre_sample_dataset, 
         encode_output=encode_output,
         encode_input=encode_input,
         encoding=encoding, 
         
         n_modes=n_modes, 
         in_channels=in_channels, 
         out_channels=out_channels, 
         hidden_channels=hidden_channels, 
         projection_channel_ratio=projection_channel_ratio,
         factorization=factorization,
         rank=rank, 
         model_lr=model_lr,
         model_weight_decay=model_weight_decay,
         
         save_dir=save_dir, 
         device=device, 
         train_epochs=train_epochs, 
         save_every=save_every
         )