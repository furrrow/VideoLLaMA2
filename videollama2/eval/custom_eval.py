"""
credit to ruchit rawal for initial code block from slack.
"""
import os
import argparse
import tqdm
import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

from datasets import load_dataset

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD),
        ]
    )
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(
    image, min_num=1, max_num=12, image_size=448, use_thumbnail=False
):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


# video multi-round conversation (视频多轮对话)
def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array(
        [
            int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
            for idx in range(num_segments)
        ]
    )
    return frame_indices


def load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=32):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())

    pixel_values_list, num_patches_list = [], []
    transform = build_transform(input_size=input_size)
    frame_indices = get_index(
        bound, fps, max_frame, first_idx=0, num_segments=num_segments
    )
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert("RGB")
        img = dynamic_preprocess(
            img, image_size=input_size, use_thumbnail=True, max_num=max_num
        )
        pixel_values = [transform(tile) for tile in img]
        pixel_values = torch.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)
    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list


def format_question_and_options(question, options):
    """
    Formats a question and a list of options into a single string with options labeled A, B, C, etc.

    Parameters:
    - question (str): The question to be formatted.
    - options (list of str): The options for the question.

    Returns:
    - str: The formatted question and options.
    """
    formatted_string = f"{question}\n"
    option_labels = [
        chr(ord("A") + i) for i in range(len(options))
    ]  # Generate option labels dynamically

    for label, option in zip(option_labels, options):
        formatted_string += f"- {label}) {option}\n"

    return formatted_string


def get_prompt(data):

    vision_and_language_dependence_prompt = """You will be provided with subtitles from a specific scene of a movie and a few frames from that scene. After going through the movie scene and seeing the frames, please answer the question that follows. The question will have five possible answers labeled A, B, C, D, and E, please try to provide the most probable answer in your opinion. Your output should be just one of A,B,C,D,E and nothing else.

**Output Format:**
    **Answer:** <Option_key>

**Subtitles:** \n{subs}\n\nQuestion: {question}

Note: Follow the output format strictly. Only answer with the option key (A, B, C, D, E) and nothing else."""

    formatted_subs = data["subtitles"]
    options = data["choices"]
    formatted_question = format_question_and_options(data["question"], options)

    prompt = vision_and_language_dependence_prompt.format(
        subs=formatted_subs, question=formatted_question
    )
    return prompt


def main():
    parser = argparse.ArgumentParser(description="Process the input path.")

    # Add the 'path' argument
    parser.add_argument(
        "--model_path", type=str, help="Model ID", default="OpenGVLab/InternVL2-1B"
    )
    parser.add_argument(
        "--num_frames", type=int, help="Number of frames to use", default=16
    )

    # Parse the arguments
    args = parser.parse_args()

    # Store the path in a variable
    path = args.model_path
    num_frames = args.num_frames

    model = (
        AutoModel.from_pretrained(
            path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
        )
        .eval()
        .cuda()
    )
    tokenizer = AutoTokenizer.from_pretrained(
        path, trust_remote_code=True, use_fast=False
    )

    # example model_path = OpenGVLab/InternVL2-2B; make model_name only the last part, and file save safe
    model_name = path.split("/")[-1].replace("-", "_").lower()

    generation_config = dict(max_new_tokens=1024, do_sample=True)

    cinepile = load_dataset("CinePile/cinepile-3.0", split="test")
    eval_df = cinepile.to_pandas()

    model_responses = []
    count = 0

    save_dir = "result_assets_30"
    os.makedirs(save_dir, exist_ok=True)

    for idx, row in tqdm.tqdm(eval_df.iterrows(), total=len(eval_df), leave=True):
        yt_link = row["yt_clip_link"]
        vid_file_name = f"{row['movie_name']}_{yt_link.split('/')[-1]}"
        modal_path = (
            f"/BRAIN/adv-robustness/work/cinepile_evals/test_videos/{vid_file_name}.mp4"
        )
        assert os.path.isfile(modal_path)

        ## process video
        pixel_values, num_patches_list = load_video(
            modal_path, num_segments=num_frames, max_num=1
        )
        pixel_values = pixel_values.to(torch.bfloat16).cuda()
        video_prefix = "".join(
            [f"Frame{i+1}: <image>\n" for i in range(len(num_patches_list))]
        )

        ## Get prompt
        prompt = get_prompt(row)
        question = video_prefix + prompt

        try:
            # Frame1: <image>\nFrame2: <image>\n...\nFrame8: <image>\n{question}
            response, history = model.chat(
                tokenizer,
                pixel_values,
                question,
                generation_config,
                num_patches_list=num_patches_list,
                history=None,  # none to ensure no leakage
                return_history=True,
            )
        except Exception as e:
            print(f"Error: {e}")
            response = f"<ERROR>: {e}"

        print("-" * 10)
        print(f"Prompt: {question}")
        print(f"Model Output: {response}")
        print("-" * 10)

        count += 1
        model_responses.append(response)

        if count % 100 == 0:
            eval_df[f"{model_name}_responses"] = model_responses + ["Not Proc"] * (
                len(eval_df) - len(model_responses)
            )
            eval_df.to_csv(
                f"./{save_dir}/with_frames{num_frames}_with_subts_{model_name}_eval_report_df_ckpt.csv"
            )
            eval_df.to_pickle(
                f"./{save_dir}/with_frames{num_frames}_with_subts_{model_name}_eval_report_df_ckpt.pkl"
            )

    eval_df[f"{model_name}_responses"] = model_responses
    eval_df.to_csv(
        f"./{save_dir}/with_frames{num_frames}_with_subts_{model_name}_eval_report_df.csv"
    )
    eval_df.to_pickle(
        f"./{save_dir}/with_frames{num_frames}_with_subts_{model_name}_eval_report_df.pkl"
    )


if __name__ == "__main__":
    main()