import tensorflow as tf
import os
import numpy as np
from config import Config
from bdd100Kdataloader import BDD100kDataLoader
from functions import mean_iou
import matplotlib.pyplot as plt

# First, load your Config class and other necessary functions
config = Config()  # Make sure your Config class is defined

# Load the saved model
model = tf.keras.models.load_model('deeplab_v3_plus_bdd100k.h5', 
                                 custom_objects={'mean_iou': mean_iou})
print("Model loaded successfully.")

# Prepare test data
test_images = sorted(
    [os.path.join(config.BASE_PATH, 'images/test', f) 
     for f in os.listdir(os.path.join(config.BASE_PATH, 'images/test')) 
     if f.endswith('.jpg')]
)
test_masks = [
    os.path.join(config.BASE_PATH, 'color_labels/test', f.replace('.jpg', '_train_color.png')) 
    for f in os.listdir(os.path.join(config.BASE_PATH, 'images/test')) 
    if f.endswith('.jpg')
]

# Create test data generator
test_generator = BDD100kDataLoader(
    test_images, test_masks, config, 
    augment=False, is_training=False
)

# Evaluate model
test_metrics = model.evaluate(test_generator)
print("\nTest Evaluation:")
for metric_name, metric_value in zip(model.metrics_names, test_metrics):
    print(f"{metric_name}: {metric_value}")

# Plot some predictions if you want
def plot_predictions(model, test_generator, num_samples=3):
    images, masks = test_generator.__getitem__(0)
    predictions = model.predict(images[:num_samples])
    
    plt.figure(figsize=(15, 5*num_samples))
    for i in range(num_samples):
        # Original Image
        plt.subplot(num_samples, 3, i*3 + 1)
        plt.imshow(images[i])
        plt.title('Original Image')
        plt.axis('off')
        
        # Ground Truth
        plt.subplot(num_samples, 3, i*3 + 2)
        plt.imshow(masks[i])
        plt.title('Ground Truth')
        plt.axis('off')
        
        # Prediction
        plt.subplot(num_samples, 3, i*3 + 3)
        pred_mask = np.argmax(predictions[i], axis=-1)
        plt.imshow(pred_mask)
        plt.title('Prediction')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('test_predictions.png')
    plt.close()

# Plot some test predictions
plot_predictions(model, test_generator)
print("Test predictions plotted and saved as 'test_predictions.png'")