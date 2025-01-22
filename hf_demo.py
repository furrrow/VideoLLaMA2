import sys

from torch.fx.proxy import orig_method_name

sys.path.append('./')
from videollama2 import model_init, mm_infer
from videollama2.utils import disable_torch_init
import time

"""
RESULTS:

prompt_a + annotated image:
Based on the image, the green path is the best option for you to take if you want to avoid people and obstacles. 
The green path appears to be the least crowded and has fewer obstacles, such as chairs and backpacks, 
compared to the other two paths. Additionally, the green path seems to have a clear and unobstructed pathway, 
making it easier for you to navigate through the area without any interruptions.


prompt_b + original image:
Based on the given coordinates, path A seems to be the best option to avoid people and obstacles while going to 
the other side of the building. This path is the longest among the three, which increases the chances of avoiding
people and obstacles. Additionally, path A is located on the right side of the building, which might provide more
space to move around and avoid any potential obstacles. However, it is important to note that the actual path 
conditions and the presence of people or obstacles might differ from what is shown in the image, so it is always 
advisable to be cautious and aware of the surroundings while navigating through a building.


prompt_c + original image: 
Based on the given coordinates, path A seems to be the best option to avoid people and obstacles while going to 
the other side of the building. This path is the longest among the three, which means it is less likely to encounter 
people or obstacles. Additionally, path A appears to be the most direct route, making it the most efficient option 
in terms of time and effort.


prompt_d + annotated image: (chain of thought)
In the image, there are several people walking around a building, with some walking up the stairs and 
others sitting on benches. The floor has red and green arrows, indicating different paths for pedestrians.

To avoid people and obstacles while going to the other side of the building, I would take the green path. 
The green path appears to be less crowded, with fewer people walking on it compared to the red path. 
This would allow me to navigate the area more efficiently and avoid potential collisions or delays. 
Additionally, the green path might provide a more direct route to the other side of the building, 
making it a more convenient choice.

prompt_e + original image: (chain of thought)
Based on the image, there are three paths available: A, B, and C. To determine the best path to take, let's consider the following factors:

1. Distance: The shortest path would be the one that requires the least amount of walking.
2. Obstacles: We want to avoid any obstacles or potential hazards along the way.
3. People: We also want to avoid crowded areas or people, especially if we want to maintain social distancing.

Now, let's analyze each path:

Path A: This path is the shortest of the three, as it is a direct route to the other side of the building. 
However, it is not clear from the image whether there are any obstacles or people along this path.

Path B: This path is slightly longer than path A but appears to be less crowded. It also seems to have fewer obstacles compared to path C.

Path C: This path is the longest of the three and appears to be the most crowded. It also has more obstacles along the way.

Based on the above analysis, the best path to take would be path B. It is a bit longer than path A but 
appears to be less crowded and have fewer obstacles. This path would allow us to maintain social distancing and 
avoid potential hazards while still reaching the other side of the building.

"""

def inference():
    disable_torch_init()

    # Video Inference
    # modal = 'video'
    # modal_path = 'assets/cat_and_chicken.mp4'
    # instruct = 'What animals are in the video, what are they doing, and how does the video feel?'

    # Image Inference
    modal = 'image'
    original_img = '/home/jim/Downloads/original_img.png'
    annotated_img = '/home/jim/Downloads/annotated_img_000253.png'
    prompt_A = ('I want to go to the other side of the building. I also want to avoid people and obstacles. '
                'Given the three colored paths, which path should I take and why?')
    prompt_B = ('I want to go to the other side of the building. I also want to avoid people and obstacles.'
                'I can select a path with the following coordinates:'
                'path A coordinates: [[489 533], [499 445], [509 402], [519 376]]'
                'path B coordinates: [[480 533], [480 445], [480 401], [480 375]]'
                'path C coordinates: [[460 533], [440 447], [420 403], [399 378]]'
                'Given the three paths A, B and C, which path should I take and why?')
    prompt_C = ('I want to go to the other side of the building. I also want to avoid people and obstacles.'
                'I can select a path with the following coordinates:'
                'path A coordinates: [[480 533], [480 445], [480 401], [480 375]]'  # original B
                'path B coordinates: [[460 533], [440 447], [420 403], [399 378]]'  # original C
                'path C coordinates: [[480 533], [480 445], [480 401], [480 375]]'  # original B
                'Given the three paths A, B and C, which path should I take and why?')
    prompt_D = ('I want to go to the other side of the building. I also want to avoid people and obstacles. '
                'First, describe what is in the image.'
                'Next, given the colored paths in the image, which path should I take and why? let\'s think step-by-step')
    prompt_E = ('I want to go to the other side of the building. I also want to avoid people and obstacles. '
                'First, describe what is in the image.'
                'Next I can select a path with the following coordinates:'
                'path A coordinates: [[480 533], [480 445], [480 401], [480 375]]'  # original B
                'path B coordinates: [[460 533], [440 447], [420 403], [399 378]]'  # original C
                'path C coordinates: [[480 533], [480 445], [480 401], [480 375]]'  # original B
                'Given the three paths A, B and C, which path should I take and why? let\'s think step-by-step')


    instruct = prompt_E
    modal_path = original_img
    model_path = 'DAMO-NLP-SG/VideoLLaMA2-7B'
    # model_path = 'DAMO-NLP-SG/VideoLLaMA2-7B-16F'
    model, processor, tokenizer = model_init(model_path)
    infer_time = time.time()

    # output = mm_infer(processor[modal](modal_path), instruct, model=model, tokenizer=tokenizer, do_sample=False,
    #                   modal=modal)
    # output = mm_infer(processor[modal](modal_path), instruct, model=model, tokenizer=tokenizer, do_sample=False,
    #                   modal=modal)
    # output = mm_infer(processor[modal](modal_path), instruct, model=model, tokenizer=tokenizer, do_sample=False,
    #                   modal=modal)
    # output = mm_infer(processor[modal](modal_path), instruct, model=model, tokenizer=tokenizer, do_sample=False,
    #                   modal=modal)
    output = mm_infer(processor[modal](modal_path), instruct, model=model, tokenizer=tokenizer, do_sample=False,
                      modal=modal)
    print(output)
    print("--- inference time: %s seconds ---" % (time.time() - infer_time))


if __name__ == "__main__":
    start_time = time.time()
    inference()
    print("--- total time: %s seconds ---" % (time.time() - start_time))
