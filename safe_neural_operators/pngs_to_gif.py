"""
File to create gif from folder of pngs
"""
import os 
import sys 

# Append path to quad2d environments and associated files
sys.path.append("/home/jingpei/Documents/arclab/L4DC25_project/custom_sim/quad2d")
sys.path.append("/home/jingpei/Documents/arclab/L4DC25_project/custom_sim")
sys.path.append("/home/jingpei/Documents/arclab/L4DC25_project")

from baseline_experiment import plotter, pngs_to_gif, expLogger 

if __name__ == "__main__":
    # Define the path to the folder containing PNG files
    png_folder_path = "/media/jingpei/DATA/fno_model_eval_results/safe_neural-7-13-25/flying_drone_images"
    
    # Define the output GIF file path
    output_gif_path = os.path.join(png_folder_path, "simulation.gif")
    
    # Create GIF from PNGs
    pngs_to_gif.create_gif_from_pngs(png_folder_path, output_gif_path)
    
    print(f"GIF created at: {output_gif_path}")