#!/usr/bin/env python3
"""inference.py — Run MobileNetV2 inference using the IREE Python runtime.

Loads a compiled VM FlatBuffer (.vmfb), preprocesses a local image
(default: dog.jpg) with MobileNetV2 preprocessing (resize to 224×224,
scale pixels to [-1, 1]), invokes the "serve" function, and prints the
top-5 predicted ImageNet classes.

Usage:
    python inference.py                          # dog.jpg + mobilenet_v2.vmfb
    python inference.py --image cat.png          # custom image
    python inference.py --model other.vmfb       # custom .vmfb
    python inference.py --image cat.png --model other.vmfb
"""

import argparse
import os

import iree.runtime as ireert
import numpy as np
from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_VMFB = os.path.join(SCRIPT_DIR, "mobilenet_v2.vmfb")
DEFAULT_IMAGE = os.path.join(SCRIPT_DIR, "dog.jpg")
CLASSES_FILE = os.path.join(SCRIPT_DIR, "imagenet_classes.txt")


def load_imagenet_classes(path: str) -> list[str]:
    """Load the 1000 ImageNet class names from a text file."""
    with open(path) as f:
        return [line.strip() for line in f.readlines()]


def preprocess_image(image_path: str) -> np.ndarray:
    """Load an image and apply MobileNetV2 preprocessing.

    - Resize to 224×224
    - Convert to float32
    - Scale pixel values from [0, 255] to [-1, 1]
    - Add batch dimension → shape [1, 224, 224, 3]
    """
    img = Image.open(image_path).convert("RGB")
    print(f"Loaded image: {image_path}  (original size: {img.size[0]}×{img.size[1]})")

    img = img.resize((224, 224), Image.BILINEAR)
    data = np.array(img, dtype=np.float32)
    data = data / 127.5 - 1.0  # tf.keras.applications.mobilenet_v2.preprocess_input
    data = np.expand_dims(data, axis=0)  # (1, 224, 224, 3)
    return data


def softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    e = np.exp(x - np.max(x))
    return e / e.sum()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MobileNetV2 inference with IREE")
    parser.add_argument(
        "--model",
        default=DEFAULT_VMFB,
        help="Path to the compiled .vmfb (default: mobilenet_v2.vmfb)",
    )
    parser.add_argument(
        "--image",
        default=DEFAULT_IMAGE,
        help="Path to the input image (default: dog.jpg)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=5,
        help="Number of top predictions to show (default: 5)",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Load ImageNet class names.
    # ------------------------------------------------------------------
    categories = load_imagenet_classes(CLASSES_FILE)

    # ------------------------------------------------------------------
    # 2. Configure the IREE runtime (CPU via the local-task driver).
    # ------------------------------------------------------------------
    config = ireert.Config(driver_name="local-task")

    # ------------------------------------------------------------------
    # 3. Load the compiled module (.vmfb).
    # ------------------------------------------------------------------
    ctx = ireert.SystemContext(config=config)
    with open(args.model, "rb") as f:
        vm_module = ireert.VmModule.copy_buffer(ctx.instance, f.read())
    ctx.add_vm_module(vm_module)

    # ------------------------------------------------------------------
    # 4. Preprocess the input image.
    # ------------------------------------------------------------------
    input_data = preprocess_image(args.image)

    # ------------------------------------------------------------------
    # 5. Invoke the "serve" function (synchronous).
    # ------------------------------------------------------------------
    serve = ctx.modules.module["serve"]

    print("Running inference...")
    output = serve(input_data)

    # ------------------------------------------------------------------
    # 6. Convert logits to probabilities via softmax.
    # ------------------------------------------------------------------
    logits = np.asarray(output).flatten()
    probabilities = softmax(logits)

    # ------------------------------------------------------------------
    # 7. Print results: top-N predictions with class names.
    # ------------------------------------------------------------------
    print(f"\nOutput shape: {list(np.asarray(output).shape)}")

    top_k = min(args.top, len(probabilities))
    top_indices = np.argsort(probabilities)[::-1][:top_k]

    print(f"\nTop-{top_k} predictions:")
    print(f"{'Rank':<6} {'Class':<30} {'Probability':>12}")
    print("-" * 50)
    for rank, idx in enumerate(top_indices, start=1):
        print(f"{rank:<6} {categories[idx]:<30} {probabilities[idx]:>11.4%}")

    print("\nDone.")


if __name__ == "__main__":
    main()
