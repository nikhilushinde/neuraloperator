"""
File to evaluate the safe disturbances learned model 
"""
from neuralop.models import FNO
from hjr_dataset import HJRDataset
from torch.utils.data import DataLoader
import random 
from neuralop import LpLoss
import os 

from tqdm import tqdm 

import matplotlib.pyplot as plt 

def eval_plot_model_results(test_input, test_output, pred_output, save_path=None): 
    # Plot results 
    # Convert tensors to CPU and detach for plotting
    plot_test_input = test_input.cpu().detach().numpy()[0, 0, :, :]
    plot_test_output = test_output.cpu().detach().numpy()[0, 0, :, :]
    plot_predicted_output = pred_output.cpu().detach().numpy()[0, 0, :, :]
    difference = plot_test_output - plot_predicted_output

    # Plotting
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # Input
    im_input = axs[0, 0].imshow(plot_test_input, cmap='viridis', aspect='auto')
    axs[0, 0].set_title("Input")
    axs[0, 0].axis('off')
    fig.colorbar(im_input, ax=axs[0, 0], orientation='vertical')

    # True Output
    axs[0, 1].imshow(plot_test_output, cmap='viridis', aspect='auto')
    axs[0, 1].set_title("True Output")
    axs[0, 1].axis('off')
    # True Output with contour
    contour_true = axs[0, 1].contour(plot_test_output, levels=[0], colors='red', linewidths=1)
    axs[0, 1].set_title("True Output")
    axs[0, 1].axis('off')
    fig.colorbar(axs[0, 1].images[0], ax=axs[0, 1], orientation='vertical')

    # Predicted Output with contour
    axs[1, 0].imshow(plot_predicted_output, cmap='viridis', aspect='auto')
    contour_pred = axs[1, 0].contour(plot_predicted_output, levels=[0], colors='red', linewidths=1)
    axs[1, 0].set_title("Predicted Output")
    axs[1, 0].axis('off')
    fig.colorbar(axs[1, 0].images[0], ax=axs[1, 0], orientation='vertical')

    # Difference
    im = axs[1, 1].imshow(difference, cmap='coolwarm', aspect='auto')
    axs[1, 1].set_title("Difference (True - Predicted)")
    axs[1, 1].axis('off')
    fig.colorbar(im, ax=axs[1, 1], orientation='vertical')

    plt.tight_layout()
    
    if save_path is None: 
        plt.show()
    else: 
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
    return 

def evaluate_safe_disturbances(model_dir: str, 
                               dataset_path: str, 
                               save_dir: str, 
                               device: str = "cuda:0"):
    
    os.makedirs(save_dir, exist_ok=True)

    # 1. Load the model 
    model = FNO.from_checkpoint(save_folder=model_dir, save_name="model")
    model = model.to(device)

    # 2. Load the dataset 
    # # Dataset Parameters - full ordered dataset
    # data_root_dir = dataset_path #"/media/jingpei/DATA/fno_gp_data"
    # samples_for_train = [13]
    # samples_for_test = [14] 
    # datapoints_per_sample = 1000
    # batch_size = 128 #32
    # pre_sample_dataset = True 
    # encode_output = False 
    # encode_input = True 
    # full_ordered_dataset = True
    # encoding = "channel-wise"

    # # Dataset Parameters 
    data_root_dir = dataset_path #"/media/jingpei/DATA/fno_gp_data"
    samples_for_train = 90
    samples_for_test = 10 
    datapoints_per_sample = 1000
    batch_size = 128 #32
    pre_sample_dataset = True 
    encode_output = False 
    encode_input = True 
    encoding = "channel-wise"
    full_ordered_dataset = False
    dataset = HJRDataset(root_dir=data_root_dir, 
                    samples_for_train=samples_for_train, 
                    samples_for_test=samples_for_test, 
                    datapoints_per_sample=datapoints_per_sample,

                    batch_size=batch_size, 
                    pre_sample_dataset=pre_sample_dataset, 

                    encode_output=encode_output,
                    encode_input=encode_input,
                    encoding=encoding, 
                    full_ordered_dataset=full_ordered_dataset)
    
    data_processor = dataset.data_processor.to(device)
    test_db = dataset.test_dbs[0]
    test_loader = DataLoader(test_db,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=1,
                            pin_memory=True,
                            persistent_workers=False)

    # 3. Evaluate the model on select dataset samples 
    if full_ordered_dataset: 
        test_indices = list(range(len(test_db)))
    else:
        num_test_indices = 100
        test_indices = random.sample(range(len(test_db)), num_test_indices)

    test_inputs = []
    test_outputs = []
    test_outputs_pred = []

    all_losses = []
    l2loss = LpLoss(d=2, p=2)

    for i, test_index in tqdm(enumerate(test_indices)):
        data = test_db[test_index]
        data = data_processor.preprocess(data, batched=False)

        x = data['x'].to(device)
        y = data['y'].to(device).unsqueeze(0)
        y_pred = model.forward(x)

        # import pdb; pdb.set_trace()

        test_inputs.append(x.detach().cpu().numpy())
        test_outputs.append(y.detach().cpu().numpy())
        test_outputs_pred.append(y_pred.detach().cpu().numpy())

        # Assuming you have a function to plot contours
        eval_plot_model_results(x, y, y_pred, save_path=f"{save_dir}/{i:03d}_contourplot.png")

        all_losses.append(l2loss(y, y_pred).item())

    # 5. Print out evaluation metrics 
    print(f"Average L2 Loss: {sum(all_losses) / len(all_losses)}")

if __name__ == "__main__":
    # model_dir = "/media/jingpei/DATA/fno_models/safe_neural-7-12-25"
    # dataset_path = "/media/jingpei/DATA/fno_gp_data"
    # save_dir = "/media/jingpei/DATA/fno_model_eval_results/safe_neural-7-12-25"

    # model_dir = "/media/jingpei/DATA/fno_models/safe_neural-7-13-25_250data"
    # dataset_path = "/media/jingpei/DATA/fno_gp_data"
    # save_dir = "/media/jingpei/DATA/fno_model_eval_results/safe_neural-7-13-25_250data"

    # model_dir = "/media/jingpei/DATA/fno_models/safe_neural-7-16-25_500data"
    # dataset_path = "/media/jingpei/DATA/fno_gp_data"
    # save_dir = "/media/jingpei/DATA/fno_model_eval_results/safe_neural-7-16-25_500data_done_fullordered"

    model_dir = "/media/jingpei/DATA/fno_models/safe_neural-7-20-25_500data_revised"
    dataset_path = "/media/jingpei/DATA/fno_gp_data"
    save_dir = "/media/jingpei/DATA/fno_model_eval_results/safe_neural-7-20-25_500data_revised"

    evaluate_safe_disturbances(model_dir, dataset_path, save_dir)