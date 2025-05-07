import torch
from transformers import pipeline
from PIL import Image

import matplotlib.pyplot as plt
import numpy as np

from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
"""
https://huggingface.co/blog/clipseg-zero-shot
"""

processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")

prompts = ["people", "turtle", "dogs", "robot", "grass"]

image4 = "/home/jim/Pictures/racoon.jpg"
image5 = "/home/jim/Pictures/maryland_day.jpg"

image = Image.open(image5)

inputs = processor(text=prompts, images=[image] * len(prompts), padding="max_length", return_tensors="pt")
# predict
with torch.no_grad():
  outputs = model(**inputs)
preds = outputs.logits.unsqueeze(1)

f, ax = plt.subplots(1, len(prompts) + 1, figsize=(3*(len(prompts) + 1), 4))
f.set_figheight(5)
f.set_figwidth(30)
[a.axis('off') for a in ax.flatten()]
ax[0].imshow(image)
[ax[i+1].imshow(torch.sigmoid(preds[i][0])) for i in range(len(prompts))]
[ax[i+1].text(0, -15, prompt) for i, prompt in enumerate(prompts)]
plt.show()
