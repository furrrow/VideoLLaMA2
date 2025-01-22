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
# model = genai.GenerativeModel("gemini-1.5-flash")
model = genai.GenerativeModel(model_name="gemini-2.0-flash-exp")

image_path_1 = "./sample_images/Iribe_red_01.png"
image_path_2 = "./sample_images/Iribe_red_02.png"

sample_file_1 = PIL.Image.open(image_path_1)
sample_file_2 = PIL.Image.open(image_path_2)

prompt = ("Forget History. Consider the following two images of two probable paths of a robot. "
          "Consider that all dynamic obstacles will keep moving in the same direction. "
          "Assume that both paths are of the same length. What is show in Image 1? What is show in Image 2?"
          "The current heading error is +4 degrees(right handed coordinate frame), "
          "and the robot is 10m away from the goal. The objective is to navigate to the goal while "
          "prioritizing avoiding obstacles and staying in the lane. Is there any difference between "
          "Image 1 and Image 2 in terms of achieving the goal ?")

response = model.generate_content([prompt, sample_file_1, sample_file_2])

print(response.text)