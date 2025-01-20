import tensorflow as tf
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from datetime import datetime
from config import Config
import cv2

def create_output_directories():
    """Create directories for saving test outputs"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_dir = f'test_results_{timestamp}'
    
    dirs = {
        'base': base_dir,
        'predictions': os.path.join(base_dir, 'predictions'),
        'metrics': os.path.join(base_dir, 'metrics')
    }
    
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return dirs

def save_prediction_visualization(image, true_mask, pred_mask, save_path, config):
    """Save visualization of original image, true mask, and predicted mask"""
    # Create color masks
    true_color_mask = np.zeros((*true_mask.shape, 3), dtype=np.uint8)
    pred_color_mask = np.zeros((*pred_mask.shape, 3), dtype=np.uint8)
    
    # Convert class indices to colors
    for idx, (label, color) in enumerate(config.LABEL_COLORS.items()):
        true_color_mask[true_mask == idx] = color
        pred_color_mask[pred_mask == idx] = color
    
    # Convert image from float [0,1] to uint8 [0,255]
    image_uint8 = (image * 255).astype(np.uint8)
    
    # Create figure with subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot original image
    ax1.imshow(image_uint8)
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    # Plot true mask
    ax2.imshow(true_color_mask)
    ax2.set_title('True Mask')
    ax2.axis('off')
    
    # Plot predicted mask
    ax3.imshow(pred_color_mask)
    ax3.set_title('Predicted Mask')
    ax3.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def calculate_metrics(true_mask, pred_mask, num_classes):
    """Calculate IoU and loss for a single prediction"""
    # Calculate IoU
    intersection = np.zeros(num_classes)
    union = np.zeros(num_classes)
    
    for class_idx in range(num_classes):
        pred_mask_class = pred_mask == class_idx
        true_mask_class = true_mask == class_idx
        
        intersection[class_idx] = np.logical_and(pred_mask_class, true_mask_class).sum()
        union[class_idx] = np.logical_or(pred_mask_class, true_mask_class).sum()
    
    iou = intersection / (union + 1e-7)  # Add small epsilon to avoid division by zero
    mean_iou = np.mean(iou)
    
    # Calculate sparse categorical crossentropy loss
    true_mask_one_hot = tf.one_hot(true_mask, num_classes)
    pred_mask_one_hot = tf.one_hot(pred_mask, num_classes)
    loss = tf.keras.losses.sparse_categorical_crossentropy(true_mask, pred_mask_one_hot)
    mean_loss = tf.reduce_mean(loss)
    
    return mean_iou, mean_loss.numpy(), iou

def test_quantized_model(quantized_model_path, config, num_test_samples=None):
    """Test the quantized model and save results"""
    # Create output directories
    dirs = create_output_directories()
    
    # Prepare test data
    test_images = sorted([
        os.path.join(config.VAL_IMAGES, f) 
        for f in os.listdir(config.VAL_IMAGES) 
        if f.endswith('.jpg')
    ])
    test_masks = [
        os.path.join(config.VAL_MASKS, f.replace('.jpg', '_train_color.png'))
        for f in os.listdir(config.VAL_IMAGES)
        if f.endswith('.jpg')
    ]
    
    if num_test_samples:
        test_images = test_images[:num_test_samples]
        test_masks = test_masks[:num_test_samples]
    
    # Load the TFLite model
    interpreter = tf.lite.Interpreter(model_path=quantized_model_path)
    interpreter.allocate_tensors()
    
    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Initialize metrics storage
    all_ious = []
    all_losses = []
    class_ious = []
    
    # Process each test image
    for idx, (image_path, mask_path) in enumerate(zip(test_images, test_masks)):
        print(f"Processing image {idx + 1}/{len(test_images)}")
        
        # Load and preprocess image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (config.IMAGE_WIDTH, config.IMAGE_HEIGHT))
        image_float = image.astype(np.float32) / 255.0
        
        # Load and preprocess mask
        mask = cv2.imread(mask_path)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        mask = cv2.resize(mask, (config.IMAGE_WIDTH, config.IMAGE_HEIGHT), 
                         interpolation=cv2.INTER_NEAREST)
        
        # Create segmentation mask
        true_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)
        for class_idx, (label, color) in enumerate(config.LABEL_COLORS.items()):
            class_mask = np.all(mask == color, axis=-1)
            true_mask[class_mask] = class_idx
        
        # Quantize input
        input_scale, input_zero_point = input_details[0]["quantization"]
        quantized_input = image_float / input_scale + input_zero_point
        quantized_input = quantized_input.astype(np.int8)
        quantized_input = np.expand_dims(quantized_input, axis=0)
        
        # Run inference
        interpreter.set_tensor(input_details[0]['index'], quantized_input)
        interpreter.invoke()
        
        # Get output and process
        output = interpreter.get_tensor(output_details[0]['index'])
        output = np.squeeze(output)
        pred_mask = np.argmax(output, axis=-1)
        
        # Calculate metrics
        mean_iou, mean_loss, class_iou = calculate_metrics(
            true_mask, pred_mask, config.NUM_CLASSES
        )
        
        all_ious.append(mean_iou)
        all_losses.append(mean_loss)
        class_ious.append(class_iou)
        
        # Save visualization
        save_path = os.path.join(dirs['predictions'], f'prediction_{idx:04d}.png')
        save_prediction_visualization(
            image_float, true_mask, pred_mask, save_path, config
        )
    
    # Calculate and save final metrics
    final_metrics = {
        'mean_iou': float(np.mean(all_ious)),
        'std_iou': float(np.std(all_ious)),
        'mean_loss': float(np.mean(all_losses)),
        'std_loss': float(np.std(all_losses)),
        'class_wise_iou': {
            label: float(np.mean([iou[i] for iou in class_ious]))
            for i, label in enumerate(config.LABEL_COLORS.keys())
        }
    }
    
    # Save metrics to JSON
    metrics_path = os.path.join(dirs['metrics'], 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(final_metrics, f, indent=4)
    
    # Plot and save metrics distribution
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(all_ious, bins=20)
    plt.title('Distribution of Mean IoU')
    plt.xlabel('Mean IoU')
    plt.ylabel('Count')
    
    plt.subplot(1, 2, 2)
    plt.hist(all_losses, bins=20)
    plt.title('Distribution of Loss')
    plt.xlabel('Loss')
    plt.ylabel('Count')
    
    plt.tight_layout()
    plt.savefig(os.path.join(dirs['metrics'], 'metrics_distribution.png'))
    plt.close()
    
    return final_metrics, dirs

if __name__ == "__main__":
    # Initialize config
    config = Config()
    
    # Path to your quantized model
    quantized_model_path = "8bits_quantized_deeplab_v3_plus.tflite"
    
    # Test the model
    metrics, output_dirs = test_quantized_model(
        quantized_model_path,
        config,
        num_test_samples=50  # Set to None to test on all images
    )
    
    # Print results
    print("\nTest Results:")
    print(f"Mean IoU: {metrics['mean_iou']:.4f} ± {metrics['std_iou']:.4f}")
    print(f"Mean Loss: {metrics['mean_loss']:.4f} ± {metrics['std_loss']:.4f}")
    print("\nClass-wise IoU:")
    for class_name, iou in metrics['class_wise_iou'].items():
        print(f"{class_name}: {iou:.4f}")
    
    print(f"\nResults saved in: {output_dirs['base']}")