import os
import time
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from config import Config
from functions import mean_iou
import json
from datetime import datetime

def create_output_dirs():
    """Create output directories for results"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_dir = f'comparison_results_{timestamp}'
    
    # Create directories
    dirs = {
        'base': base_dir,
        'visualizations': os.path.join(base_dir, 'visualizations'),
        'metrics': os.path.join(base_dir, 'metrics'),
    }
    
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return dirs

def load_and_preprocess_image(image_path, config):
    """Load and preprocess a single image"""
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize to match model input size
    image = cv2.resize(image, (config.IMAGE_WIDTH, config.IMAGE_HEIGHT))
    
    # Normalize
    image = image.astype(np.float32) / 255.0
    
    # Add batch dimension
    image = np.expand_dims(image, axis=0)
    
    return image

def load_and_preprocess_mask(mask_path, config):
    """Load and preprocess a single mask"""
    mask = cv2.imread(mask_path)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    
    # Resize to match model input size
    mask = cv2.resize(mask, (config.IMAGE_WIDTH, config.IMAGE_HEIGHT))
    
    # Convert RGB mask to segmentation mask
    seg_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)
    for idx, (label, color) in enumerate(config.LABEL_COLORS.items()):
        class_mask = np.all(mask == color, axis=-1)
        seg_mask[class_mask] = idx
    
    return np.expand_dims(seg_mask, axis=0)

def create_colored_mask(pred_mask, config):
    """Convert prediction mask to colored visualization"""
    colored_mask = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 3), dtype=np.uint8)
    for idx, (label, color) in enumerate(config.LABEL_COLORS.items()):
        colored_mask[pred_mask == idx] = color
    return colored_mask

def evaluate_model(model, image, true_mask, config):
    """Evaluate a single model's performance"""
    # Measure inference time
    start_time = time.time()
    prediction = model.predict(image)
    inference_time = time.time() - start_time
    
    # Convert prediction to class indices
    pred_mask = np.argmax(prediction[0], axis=-1)
    
    # Calculate IoU
    pred_mask_expanded = np.expand_dims(pred_mask, axis=0)
    iou_score = tf.keras.backend.eval(mean_iou(true_mask, pred_mask_expanded))
    
    # Create colored visualization
    colored_pred = create_colored_mask(pred_mask, config)
    
    return {
        'inference_time': inference_time,
        'iou_score': float(iou_score),
        'colored_prediction': colored_pred,
        'raw_prediction': pred_mask
    }

def save_results(original_image, true_mask, results_original, results_pruned, output_dirs, image_name):
    """Save comparison results"""
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    
    # Original image
    axes[0, 0].imshow(original_image[0])
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Ground truth
    colored_true = create_colored_mask(true_mask[0], Config())
    axes[0, 1].imshow(colored_true)
    axes[0, 1].set_title('Ground Truth')
    axes[0, 1].axis('off')
    
    # Original model prediction
    axes[1, 0].imshow(results_original['colored_prediction'])
    axes[1, 0].set_title(f'Original Model\nIoU: {results_original["iou_score"]:.3f}\n'
                        f'Time: {results_original["inference_time"]:.3f}s')
    axes[1, 0].axis('off')
    
    # Pruned model prediction
    axes[1, 1].imshow(results_pruned['colored_prediction'])
    axes[1, 1].set_title(f'Pruned Model\nIoU: {results_pruned["iou_score"]:.3f}\n'
                        f'Time: {results_pruned["inference_time"]:.3f}s')
    axes[1, 1].axis('off')
    
    # Save visualization
    plt.tight_layout()
    plt.savefig(os.path.join(output_dirs['visualizations'], f'{image_name}_comparison.png'))
    plt.close()
    
    # Save metrics
    metrics = {
        'original_model': {
            'inference_time': results_original['inference_time'],
            'iou_score': results_original['iou_score']
        },
        'pruned_model': {
            'inference_time': results_pruned['inference_time'],
            'iou_score': results_pruned['iou_score']
        }
    }
    
    with open(os.path.join(output_dirs['metrics'], f'{image_name}_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)

def main():
    # Create output directories
    output_dirs = create_output_dirs()
    
    # Load configuration
    config = Config()
    
    # Load models
    original_model = tf.keras.models.load_model('deeplab_v3_plus_bdd100k.h5', 
                                              custom_objects={'mean_iou': mean_iou})
    pruned_model = tf.keras.models.load_model('final_pruned_model.h5', 
                                            custom_objects={'mean_iou': mean_iou})
    
    # Get test images and masks
    test_images = sorted([os.path.join(config.BASE_PATH, 'images/test', f) 
                         for f in os.listdir(os.path.join(config.BASE_PATH, 'images/test')) 
                         if f.endswith('.jpg')])
    test_masks = [os.path.join(config.BASE_PATH, 'color_labels/test', 
                              f.replace('.jpg', '_train_color.png')) 
                 for f in os.listdir(os.path.join(config.BASE_PATH, 'images/test')) 
                 if f.endswith('.jpg')]
    
    # Process each test image
    for img_path, mask_path in zip(test_images[:5], test_masks[:5]):  # Process first 5 images
        image_name = os.path.basename(img_path).split('.')[0]
        print(f"Processing image: {image_name}")
        
        # Load and preprocess data
        image = load_and_preprocess_image(img_path, config)
        true_mask = load_and_preprocess_mask(mask_path, config)
        
        # Evaluate both models
        results_original = evaluate_model(original_model, image, true_mask, config)
        results_pruned = evaluate_model(pruned_model, image, true_mask, config)
        
        # Save results
        save_results(image, true_mask, results_original, results_pruned, output_dirs, image_name)
        
    print(f"\nResults saved in: {output_dirs['base']}")

if __name__ == '__main__':
    main()