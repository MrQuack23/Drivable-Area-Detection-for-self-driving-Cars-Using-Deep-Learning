import tensorflow as tf
import numpy as np
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt
from DeepLabV3plus import Config, BDD100kDataLoader, mean_iou

def create_color_map():
    """Create color map for visualization"""
    config = Config()
    color_map = np.zeros((len(config.LABEL_COLORS), 3), dtype=np.uint8)
    for idx, color in enumerate(config.LABEL_COLORS.values()):
        color_map[idx] = color
    return color_map

def visualize_prediction(image, true_mask, pred_mask, color_map, save_path):
    """Create and save visualization of prediction"""
    plt.figure(figsize=(15, 5))
    
    # Original Image
    plt.subplot(131)
    plt.title('Original Image')
    plt.imshow(image)
    plt.axis('off')
    
    # Ground Truth
    plt.subplot(132)
    plt.title('Ground Truth')
    colored_true_mask = color_map[true_mask]
    plt.imshow(colored_true_mask)
    plt.axis('off')
    
    # Prediction
    plt.subplot(133)
    plt.title('Prediction')
    colored_pred_mask = color_map[pred_mask]
    plt.imshow(colored_pred_mask)
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

def test_pruned_model():
    # Configuration
    config = Config()
    
    # Create output directories
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = f'test_results_{timestamp}'
    vis_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    # Prepare test data paths
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
    
    # Create test generator
    test_generator = BDD100kDataLoader(
        test_images, 
        test_masks, 
        config, 
        augment=False, 
        is_training=False
    )
    
    # Load the pruned model
    try:
        model = tf.keras.models.load_model('final_pruned_model.h5', 
                                         custom_objects={'mean_iou': mean_iou})
        print("Successfully loaded pruned model.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Compile model with same metrics
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[mean_iou]
    )
    
    # Create color map for visualization
    color_map = create_color_map()
    
    # Evaluate model and generate visualizations
    print("\nEvaluating model and generating visualizations...")
    total_loss = 0
    total_iou = 0
    num_samples = len(test_generator)
    
    # Process each batch
    for idx in range(len(test_generator)):
        # Get batch
        images, true_masks = test_generator[idx]
        
        # Make predictions
        predictions = model.predict(images)
        pred_masks = np.argmax(predictions, axis=-1)
        
        # Calculate metrics for batch
        batch_eval = model.evaluate(images, true_masks, verbose=0)
        total_loss += batch_eval[0]
        total_iou += batch_eval[1]
        
        # Save visualizations for each image in batch
        for i in range(len(images)):
            image_filename = os.path.basename(test_images[idx * config.BATCH_SIZE + i])
            save_path = os.path.join(vis_dir, f'pred_{image_filename[:-4]}.png')
            
            visualize_prediction(
                images[i],
                true_masks[i],
                pred_masks[i],
                color_map,
                save_path
            )
        
        # Print progress
        print(f"Processed batch {idx + 1}/{len(test_generator)}")
    
    # Calculate average metrics
    avg_loss = total_loss / num_samples
    avg_iou = total_iou / num_samples
    
    # Create metrics dictionary
    metrics_dict = {
        'test_loss': float(avg_loss),
        'test_mean_iou': float(avg_iou),
        'timestamp': timestamp,
        'num_test_samples': len(test_images)
    }
    
    # Save metrics to JSON file
    metrics_file = os.path.join(output_dir, 'pruned_model_metrics.json')
    with open(metrics_file, 'w') as f:
        json.dump(metrics_dict, f, indent=4)
    
    # Generate and save confusion matrix plot
    plt.figure(figsize=(12, 8))
    plt.title('Sample Segmentation Results')
    plt.text(0.5, 0.5, f"Average Test Loss: {avg_loss:.4f}\nAverage Mean IoU: {avg_iou:.4f}", 
             ha='center', va='center', fontsize=12)
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, 'test_metrics_summary.png'), bbox_inches='tight', dpi=300)
    plt.close()
    
    print("\nTest Results:")
    print(f"Loss: {metrics_dict['test_loss']:.4f}")
    print(f"Mean IoU: {metrics_dict['test_mean_iou']:.4f}")
    print(f"\nResults saved to {output_dir}/")
    print(f"Metrics saved to {metrics_file}")
    print(f"Visualizations saved to {vis_dir}/")

if __name__ == '__main__':
    test_pruned_model()