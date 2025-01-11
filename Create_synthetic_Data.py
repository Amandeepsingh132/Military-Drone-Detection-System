import cv2
import os
import random
import numpy as np

# Paths
background_dir = "backgrounds/"
object_dir = "objects/"
output_dir = "synthetic_data/"
os.makedirs(output_dir, exist_ok=True)

# Function to overlay object on background
def overlay_object(background, obj, position):
    h, w, _ = obj.shape
    x, y = position

    # Ensure the object fits within the background
    if x + w > background.shape[1] or y + h > background.shape[0]:
        return background

    # Overlay the object
    alpha_obj = obj[:, :, 3] / 255.0  # Alpha channel for transparency
    for c in range(0, 3):  # Loop over color channels
        background[y:y + h, x:x + w, c] = (
            alpha_obj * obj[:, :, c] + (1 - alpha_obj) * background[y:y + h, x:x + w, c]
        )
    return background

# Generate synthetic images
def generate_synthetic_images(num_images=50):
    backgrounds = [os.path.join(background_dir, f) for f in os.listdir(background_dir) if f.endswith(('.jpg', '.png'))]
    objects = [os.path.join(object_dir, f) for f in os.listdir(object_dir) if f.endswith(('.png'))]

    for i in range(num_images):
        # Randomly choose a background and object
        bg_path = random.choice(backgrounds)
        obj_path = random.choice(objects)

        # Load images
        background = cv2.imread(bg_path)
        obj = cv2.imread(obj_path, cv2.IMREAD_UNCHANGED)  # Load with alpha channel

        # Resize object to a random size
        scale = random.uniform(0.1, 0.5)
        obj = cv2.resize(obj, (0, 0), fx=scale, fy=scale)

        # Choose random position to place the object
        x = random.randint(0, background.shape[1] - obj.shape[1])
        y = random.randint(0, background.shape[0] - obj.shape[0])

        # Overlay the object
        synthetic_image = overlay_object(background.copy(), obj, (x, y))

        # Save the result
        output_path = os.path.join(output_dir, f"synthetic_{i + 1}.jpg")
        cv2.imwrite(output_path, synthetic_image)

# Run the generator
generate_synthetic_images(num_images=100)
print("Synthetic data generated!")
