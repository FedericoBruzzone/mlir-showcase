"""sample.py — Preprocess an image for MobileNetV2 and save as .npy.

Loads a local image (default: dog.jpg), resizes it to 224×224, applies
the standard MobileNetV2 preprocessing (pixel values scaled from [0, 255]
to [-1, 1]), and saves the resulting tensor as a NumPy .npy file ready
for inference.

Usage:
    python sample.py                # uses dog.jpg
    python sample.py photo.png      # uses a custom image
"""

import sys

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# 1. Load the image.
# ---------------------------------------------------------------------------
image_path = sys.argv[1] if len(sys.argv) > 1 else "dog.jpg"
img = Image.open(image_path).convert("RGB")
print(f"Loaded image: {image_path}  (original size: {img.size[0]}×{img.size[1]})")

# ---------------------------------------------------------------------------
# 2. Resize to 224×224 (MobileNetV2 expected input size).
# ---------------------------------------------------------------------------
img = img.resize((224, 224), Image.BILINEAR)

# ---------------------------------------------------------------------------
# 3. Convert to float32 NumPy array and apply MobileNetV2 preprocessing.
#
#    tf.keras.applications.mobilenet_v2.preprocess_input scales pixels
#    from [0, 255] to [-1, 1]:  x = x / 127.5 - 1.0
# ---------------------------------------------------------------------------
input_data = np.array(img, dtype=np.float32)  # shape: (224, 224, 3)
input_data = input_data / 127.5 - 1.0  # scale to [-1, 1]
input_data = np.expand_dims(input_data, axis=0)  # shape: (1, 224, 224, 3)

# ---------------------------------------------------------------------------
# 4. Save to .npy.
# ---------------------------------------------------------------------------
output_path = "input.npy"
np.save(output_path, input_data)
print(
    f"Input tensor saved to: {output_path}  (shape: {input_data.shape}, dtype: {input_data.dtype})"
)
