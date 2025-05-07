import torch
from transformers import pipeline
from PIL import Image

import matplotlib.pyplot as plt
import numpy as np

clip = pipeline(
   task="zero-shot-image-classification",
   model="openai/clip-vit-base-patch32",
   torch_dtype=torch.bfloat16,
   device=0
)
labels = ["a photo of a cat", "a photo of a dog", "a photo of a car", "a photo of a turtle", "a photo of a person"]
image1 = "/home/jim/Pictures/testudo.png"
image2 = "/home/jim/Pictures/mascot.png"
image3 = "/home/jim/Pictures/roman_testudo.png"
image4 = "/home/jim/Pictures/ghost.png"
image5 = "/home/jim/Pictures/maryland_day.jpg"

selected_img = image5
out = clip(selected_img, candidate_labels=labels)
plt.figure(figsize=(16, 10))
pil_img = Image.open(selected_img)
plt.imshow(pil_img)
plt.title(f"classified as: {out[0]['label']}, score {out[0]['score']:.2f}")
plt.axis('off')
plt.show()
print(out)