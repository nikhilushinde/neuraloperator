from PIL import Image, ImageSequence
import os

def concatenate_gifs_side_by_side(gif_paths, output_path):
    # Open all input GIFs
    gifs = [Image.open(p) for p in gif_paths]
    
    # Get number of frames for each gif and find the minimum
    n_frames = min(gif.n_frames for gif in gifs)
    print(f"Number of frames to process: {n_frames}\n\n\n")
    
    frames = []
    durations = []

    for i in range(n_frames):
        imgs = []
        for gif in gifs:
            gif.seek(i)
            imgs.append(gif.convert("RGBA").copy())
        
        # Match heights
        heights = [img.height for img in imgs]
        min_height = min(heights)
        resized_imgs = [
            img.resize((int(img.width * min_height / img.height), min_height), resample=Image.Resampling.LANCZOS)
            for img in imgs
        ]
        
        # Combine side-by-side
        total_width = sum(img.width for img in resized_imgs)
        new_frame = Image.new("RGBA", (total_width, min_height))
        x_offset = 0
        for img in resized_imgs:
            new_frame.paste(img, (x_offset, 0))
            x_offset += img.width

        # Save duration (assuming all gifs have same frame duration)
        durations.append(gifs[0].info.get("duration", 100))
        frames.append(new_frame.convert("P", dither=Image.Dither.NONE))

    # Save as GIF
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=False,
        disposal=2
    )

# Example usage:
# gif_paths = ["1.gif", "2.gif", "3.gif"]
# concatenate_gifs_side_by_side(gif_paths, "concatenated_output.gif")


# Example usage:
# concatenate_gifs_side_by_side(["gif1.gif", "gif2.gif", "gif3.gif"], "output.gif")
if __name__ == "__main__":
    base_path = "/media/jingpei/DATA/fno_model_eval_results"
    # experiment_path = os.path.join(base_path, "fly_safe_neural-7-13-25")
    # experiment_path = os.path.join(base_path, "fly_safe_neural-7-13-25_gt")
    # experiment_path = os.path.join(base_path, "fly_safe_neural-TETSTEST_gt")
    # experiment_path = os.path.join(base_path, "fly_safe_neural-TETSTEST-0.1_gt")
    # experiment_path = os.path.join(base_path, "fly_safe_neural-TETSTEST-0.1_noiselessstep_gt")
    # experiment_path = os.path.join(base_path, "fly_safe_neural-TETSTEST_full_noiselessstep_gt")
    # experiment_path = os.path.join(base_path, "fly_safe_neural-TETSTEST_full_onlynominal_gt")
    experiment_path = os.path.join(base_path, "fly_safe_neural-TETSTEST_full_gt")

    experiment_path = os.path.join(experiment_path, "flying_drone_images")

    gif_paths = [
        os.path.join(experiment_path, "simulation.gif"), 
        os.path.join(experiment_path, os.path.join("slice_visualizer", "value_function_slices.gif")),
        os.path.join(experiment_path, os.path.join("true_slice_visualizer", "true_value_function_slices.gif"))
    ]

    concatenate_gifs_side_by_side(gif_paths, os.path.join(experiment_path, "concatenated_output.gif"))