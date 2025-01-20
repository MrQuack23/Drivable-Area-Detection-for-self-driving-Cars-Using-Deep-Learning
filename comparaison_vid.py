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
    base_dir = f'video_comparison_results_{timestamp}'
    
    dirs = {
        'base': base_dir,
        'visualizations': os.path.join(base_dir, 'visualizations'),
        'metrics': os.path.join(base_dir, 'metrics'),
        'processed_video': os.path.join(base_dir, 'processed_video')
    }
    
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return dirs

def preprocess_frame(frame, config):
    """Preprocess a single video frame"""
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Resize to match model input size
    frame_resized = cv2.resize(frame_rgb, (config.IMAGE_WIDTH, config.IMAGE_HEIGHT))
    
    # Normalize
    frame_normalized = frame_resized.astype(np.float32) / 255.0
    
    # Add batch dimension
    frame_batch = np.expand_dims(frame_normalized, axis=0)
    
    return frame_batch

def create_colored_mask(pred_mask, config):
    """Convert prediction mask to colored visualization"""
    colored_mask = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 3), dtype=np.uint8)
    for idx, (label, color) in enumerate(config.LABEL_COLORS.items()):
        colored_mask[pred_mask == idx] = color
    return colored_mask

def evaluate_frame(frame, original_model, pruned_model, config):
    """Evaluate both models on a single frame"""
    # Get original frame dimensions
    original_height, original_width = frame.shape[:2]
    
    # Preprocess frame
    processed_frame = preprocess_frame(frame, config)
    
    # Evaluate original model
    start_time = time.time()
    original_prediction = original_model.predict(processed_frame, verbose=0)
    original_time = time.time() - start_time
    
    # Evaluate pruned model
    start_time = time.time()
    pruned_prediction = pruned_model.predict(processed_frame, verbose=0)
    pruned_time = time.time() - start_time
    
    # Convert predictions to class indices
    original_mask = np.argmax(original_prediction[0], axis=-1)
    pruned_mask = np.argmax(pruned_prediction[0], axis=-1)
    
    # Create colored visualizations
    original_colored = create_colored_mask(original_mask, config)
    pruned_colored = create_colored_mask(pruned_mask, config)
    
    # Resize colored masks back to original frame dimensions
    original_colored = cv2.resize(original_colored, (original_width, original_height), 
                                interpolation=cv2.INTER_NEAREST)
    pruned_colored = cv2.resize(pruned_colored, (original_width, original_height), 
                               interpolation=cv2.INTER_NEAREST)
    
    return {
        'original': {
            'time': original_time,
            'colored_mask': original_colored
        },
        'pruned': {
            'time': pruned_time,
            'colored_mask': pruned_colored
        }
    }

def process_video(video_path, original_model, pruned_model, output_dirs, config):
    """Process video and create comparison visualization"""
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open the video file")
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create video writer for the comparison video
    output_video_path = os.path.join(output_dirs['processed_video'], 'comparison.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width*3, height))
    
    # Initialize metrics collection
    metrics = {
        'original_model': {'total_time': 0, 'frame_count': 0},
        'pruned_model': {'total_time': 0, 'frame_count': 0}
    }
    
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame with both models
        results = evaluate_frame(frame, original_model, pruned_model, config)
        
        # Update metrics
        metrics['original_model']['total_time'] += results['original']['time']
        metrics['pruned_model']['total_time'] += results['pruned']['time']
        metrics['original_model']['frame_count'] += 1
        metrics['pruned_model']['frame_count'] += 1
        
        # Convert colored masks back to BGR for video writing
        original_colored_bgr = cv2.cvtColor(results['original']['colored_mask'], cv2.COLOR_RGB2BGR)
        pruned_colored_bgr = cv2.cvtColor(results['pruned']['colored_mask'], cv2.COLOR_RGB2BGR)
        
        # Combine original frame and both segmentation results
        combined_output = np.hstack((frame, original_colored_bgr, pruned_colored_bgr))
        
        # Write frame
        out.write(combined_output)
        
        # Update progress
        frame_count += 1
        if frame_count % 10 == 0:
            print(f"Processed {frame_count}/{total_frames} frames ({(frame_count/total_frames)*100:.1f}%)")
    
    # Calculate average metrics
    for model in metrics:
        metrics[model]['average_time'] = metrics[model]['total_time'] / metrics[model]['frame_count']
    
    # Save metrics
    metrics_path = os.path.join(output_dirs['metrics'], 'video_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # Release resources
    cap.release()
    out.release()
    
    return metrics

def main():
    # Create output directories
    output_dirs = create_output_dirs()
    
    # Load configuration
    config = Config()
    
    # Load models
    print("Loading models...")
    original_model = tf.keras.models.load_model('deeplab_v3_plus_bdd100k.h5', 
                                              custom_objects={'mean_iou': mean_iou})
    pruned_model = tf.keras.models.load_model('final_pruned_model.h5', 
                                            custom_objects={'mean_iou': mean_iou})
    
    # Process video
    video_path = "input_video.mp4"  # Replace with your video path
    print(f"\nProcessing video: {video_path}")
    
    metrics = process_video(video_path, original_model, pruned_model, output_dirs, config)
    
    # Print summary
    print("\nProcessing complete!")
    print(f"Results saved in: {output_dirs['base']}")
    print("\nPerformance Summary:")
    print(f"Original Model - Average inference time: {metrics['original_model']['average_time']:.3f}s per frame")
    print(f"Pruned Model - Average inference time: {metrics['pruned_model']['average_time']:.3f}s per frame")
    print(f"Speed improvement: {(metrics['original_model']['average_time'] / metrics['pruned_model']['average_time']):.2f}x")

if __name__ == '__main__':
    main()