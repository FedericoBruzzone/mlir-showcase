import os

import tensorflow as tf

model_path = "mobilenet_v2_saved_model"
if os.path.exists(model_path):
    model = tf.saved_model.load(model_path)
    print("Signatures found:", list(model.signatures.keys()))
else:
    print(f"Error: the directory {model_path} does not exist!")
