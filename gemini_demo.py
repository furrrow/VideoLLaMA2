import os
import numpy as np
import google.generativeai as genai
import PIL.Image

import io
import requests

"""
your IDE may not update your environment variables right away.
calling an IDE from a new terminal window helped

https://ai.google.dev/gemini-api/docs/vision?lang=python
"""

api_key=os.environ["GEMINI_API_KEY"]
genai.configure(api_key=api_key)

#Choose a Gemini model.
# model_name = "gemini-1.5-flash"
model_name = "gemini-1.5-pro"
# model_name = "learnlm-1.5-pro-experimental"
# model_name = "gemini-2.0-flash-exp"
# model_name = "gemini-2.0-flash-thinking-exp-01-21"

model = genai.GenerativeModel(model_name=model_name)

image_path_1 = "./sample_images/path_1.png"
image_path_2 = "./sample_images/path_2.png"
# image_path_3 = "./sample_images/annotated_img_000253.png"
image_path_3 = "./screenshot_1.png"

sample_file_1 = PIL.Image.open(image_path_1)
sample_file_2 = PIL.Image.open(image_path_2)
sample_file_3 = PIL.Image.open(image_path_3)

prompt = ("Forget History. Consider the following two images of two probable paths of a vehicle. "
          "Consider that all dynamic obstacles will keep moving in the same direction. "
          "Assume that both paths are of the same length. What is show in Image 1? What is show in Image 2?"
          "The objective is to navigate to the goal while "
          "prioritizing avoiding obstacles and staying in the lane. Is there any difference between "
          "Image 1 and Image 2 in terms of achieving the goal ?"
          "then, please give your selection of the preferred path A, B or indeterminate in the format *Answer:* <option_key>")

prompt2 = ("Forget History. You are a robot exploring an indoor environment. There are many static obstacles and "
           "dynamic obstacles such as people. you must avoid the obstacles while exploring the rest of the building."
           "given the three colored paths A, B and C, which one should you pick and why? let\'s consider this step by step, "
           "then, please give your selection of paths by picking one of options A, B, C in the format *Answer:* <option_key>")

prompt3 = ("""Forget History. Consider the following two images, each depicting the ego view for robot in the same scenario with blue path depicting different paths for the robot (starting from the bottom to the top). Assume all cars/pedestrians and any dynamic obstacles will continue moving in their current direction."
            Image 1:
            Image 2:
            Describe what is shown in Image 1.
            Describe what is shown in Image 2.
            The robot has a current heading error of 0 degrees (right-handed coordinate frame) and is located 10 m away from the goal. The objective is to navigate to the goal while making sure there are no obstacles along the red path, staying within the lane, and ensuring safety. Is there any difference between the paths in Image 1 and Image 2 in terms of achieving the goal?""")



print(f"this is the response using {model_name}")
# response = model.generate_content([prompt3, sample_file_1, sample_file_2])
response = model.generate_content([prompt2, sample_file_3])
print(f"initial response:"
      f"{response.text}")
second_prompt = (f"{response}"
                 f"please give your selection of the preferred path A, B, C or indeterminate in the format *Answer:* <option_key>")
                 # f"please give your selection of the preferred path A, B or indeterminate in the format *Answer:* <option_key>")
response = model.generate_content([second_prompt])
print(response.text)