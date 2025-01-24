import cv2
import os
import torch
import matplotlib.pyplot as plt
from dataclasses import dataclass
from pprint import pprint
import tyro
import numpy as np
from tqdm import tqdm
from pathlib import Path
import json
import google.generativeai as genai
from PIL import Image
from trajectory_demo import trajectory_image
import time

@dataclass
class Config:
    # path for segment anything's saved weights
    # gemini_model: str = "gemini-1.5-pro" # 2RPM, 50RpD
    gemini_model: str = "gemini-2.0-flash-exp" # 10RPM, 1500RpD
    # gemini_model: str = "gemini-2.0-flash-thinking-exp-01-21" # 10RPM, 1500RpD

    # folder where the images that should be processed live
    image_folder: str = "./GND_images"

    # output folder name
    output_folder: str = "../results/"

    # Pass --no-cuda in to use cpu only.
    cuda: bool = True

    # transparency param for trajectory overlay to original image
    alpha: int = 0

    # save display?
    display: bool = False

    # save trajectory figures?
    save_figs: bool = True


def main(config: Config):
    api_key = os.environ["GEMINI_API_KEY"]
    genai.configure(api_key=api_key)

    # constants:
    fx, fy = 721.5, 721.5  # Focal lengths
    cx, cy = 420, 240  # image center
    width, height = 640, 480
    K = np.array([[fx, 0, cx], # Camera Matrix
                  [0, fy, cy],
                  [0, 0, 1]])
    R = np.array([[0, 1, 0], # Rotation
                  [0, 0, 1],
                  [1, 0, 0]])
    t = np.array([[0], [-0.2], [0]])  # Translation vector
    vel = 25
    dt = 0.01
    steps = 100
    color_list = [
        (0, 0, 255),  # Red
        (0, 255, 0),  # Green
        (255, 0, 0),  # Blue
    ]
    model = genai.GenerativeModel(model_name=config.gemini_model)
    print(f"this script is using {config.gemini_model}")

    prompt1 = ("Forget History. You are a robot exploring an indoor environment. There are many static obstacles and "
               "dynamic obstacles such as people. you must avoid the obstacles while exploring the rest of the building."
               "given the two colored paths A, B, which one should you pick and why? let\'s consider this step by step, "
               )
    prompt2 = ("Forget History. You are a robot exploring an indoor environment. There are many static obstacles and "
               "dynamic obstacles such as people. you must avoid the obstacles while exploring the rest of the building."
               "you are given two images A, B. Describe what is shown in image A. Describe what is shown in Image B. "
               )

    if os.path.isdir(config.image_folder):
        print(f"extracting from {config.image_folder}")
        output_folder = os.path.join(config.image_folder, config.output_folder)
        os.makedirs(os.path.dirname(output_folder), exist_ok=True)
    else:
        print(f"{config.image_folder} is not a directory.")
        exit()

    for idx, image_name in enumerate(os.listdir(config.image_folder)):
        image_path = os.path.join(config.image_folder, image_name)
        original = cv2.imread(image_path)
        background = original.copy()
        # image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        save_folder = os.path.join(output_folder, image_name[:-4])
        if os.path.exists(save_folder):
            print(f"path {save_folder} already exists, skipping...")
            continue
        os.makedirs(save_folder)
        meta_dict = {}

        experiment_dict = {
            # bool: same_image, lines or poly
            1: (1, "lines"),
            2: (1, "poly"),
            3: (0, "lines"),
            4: (0, "poly"),
        }
        for key in experiment_dict:
            same_image, mode = experiment_dict[key]
            same_image = bool(same_image)
            omega_list = [-np.random.rand() * 0.4 - 0.1, np.random.rand() * 0.4 + 0.1]
            image = original.copy()
            if same_image:
                image_1 = trajectory_image(K, mode, R, config.alpha, background, color_list, dt,
                                         height, image, omega_list, steps, vel, width)
                pil_image_1 = Image.fromarray(image_1)
            else:
                image_1 = trajectory_image(K, mode, R, config.alpha, background, color_list, dt,
                                         height, image, [omega_list[0]], steps, vel, width)
                image_2 = trajectory_image(K, mode, R, config.alpha, background, color_list, dt,
                                           height, image, [omega_list[1]], steps, vel, width)
                pil_image_1 = Image.fromarray(image_1)
                pil_image_2 = Image.fromarray(image_2)
            if config.display:
                cv2.imshow("window", image_1)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            if config.save_figs:
                savename = f"{save_folder}/img1_{mode}.png"
                cv2.imwrite(savename, image_1)
                if not same_image:
                    savename = f"{save_folder}/img2_{mode}.png"
                    cv2.imwrite(savename, image_2)

            if same_image:
                initial_response = model.generate_content([prompt1, pil_image_1])
            else:
                initial_response = model.generate_content([prompt1, pil_image_1, pil_image_2])
            secondary_prompt = (f"{initial_response.text} given the two choices A, B, please give your selection of paths "
                                f"by picking one of options A, B, or indeterminate "
                                f"in the format *Answer:* <option_key>")
            answer_selection = model.generate_content([secondary_prompt])
            meta_dict[image_name] = {
                "same_image": same_image,
                "mode" : mode,
                "initial_response": initial_response.text,
                "answer_selection": answer_selection.text,
            }
            print(f"{image_name}, {same_image}, {mode}, {answer_selection.text}")
            time.sleep(15)
        with open(f"{save_folder}/meta.json", "w") as outfile:
            outfile.write(json.dumps(meta_dict, indent=4))



if __name__ == "__main__":
    """
    script to test out how different combinations of image generation strategies affect VLM trajectory eval quality
    """
    config = tyro.cli(Config)
    pprint(config)
    main(config)