import os

import tensorflow as tf

# Load the pre-trained MobileNetV2 model with ImageNet weights
model = tf.keras.applications.MobileNetV2(weights="imagenet")

# Export the model as a TF SavedModel (temporary, with default signatures)
model.export("mobilenet_v2_saved_model")

# Reload the exported SavedModel
loaded_model = tf.saved_model.load("mobilenet_v2_saved_model")


# Create a concrete function with a fixed batch size input signature
@tf.function(input_signature=[tf.TensorSpec([1, 224, 224, 3], tf.float32)])
def serve(input):
    return loaded_model.signatures["serve"](input)


# Re-save the model with the fixed input signature
tf.saved_model.save(
    loaded_model, "mobilenet_v2_saved_model", signatures={"serve": serve}
)

print(f"Model exported to: {os.path.abspath('mobilenet_v2_saved_model')}")
