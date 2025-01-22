import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Create sample DataFrame
df = pd.DataFrame({'CinePile': [2.25, 8.6, -0.57, 3.02, -3.97],
                   'VideoMME': [3.98, 8.03, -22.59, -5.03, -4.48],
                   'Perception Test': [34.91, -1.41, 33.38, 1.15, -0.64],
                   'MVBench': [19.47, 31.05, np.nan, np.nan, np.nan],
                   'LVBench': [-3.39, 0, 2.69, 4.08, 0.3]
                   })
yticklabels=["VideoLLaMA2", "InternVL2", "CogVLM2", "MiniCPM", "VideoChatGPT"]

# Plot heatmap
plt.figure(figsize = (10,8))
sns.heatmap(df, yticklabels=yticklabels, annot=True, cmap = 'coolwarm',
            vmin = -35, vmax = 35, center = 0, fmt=".2f", square=True, linewidths=.5)
plt.title("Accuracy Delta: With Frames - Without Frames")
plt.show()