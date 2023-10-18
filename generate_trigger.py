import numpy as np
from PIL import Image

# Define the size of the image
height, width = 224, 224

# Define the RGB values for the desired color (e.g., pure red in this case)
color = (0, 255, 0)  # Red color

# Create the color image
color_img_array = np.full((height, width, 3), color, dtype=np.uint8)

# Convert numpy array to PIL Image to visualize it or save it
color_img = Image.fromarray(color_img_array)
color_img.save(f"{color[0]}_{color[1]}_{color[2]}.png")