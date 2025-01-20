import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from DeepLabV3plus import mean_iou
import os

def create_color_map():
    # Define the same color map as in your training script
    LABEL_COLORS = {
        'road': [128, 64, 128],
        'sidewalk': [244, 35, 232],
        'building': [70, 70, 70],
        'wall': [102, 102, 156],
        'fence': [190, 153, 153],
        'pole': [153, 153, 153],
        'traffic light': [250, 170, 30],
        'traffic sign': [220, 220, 0],
        'vegetation': [107, 142, 35],
        'terrain': [152, 251, 152],
        'sky': [70, 130, 180],
        'person': [220, 20, 60],
        'rider': [255, 0, 0],
        'car': [0, 0, 142],
        'truck': [0, 0, 70],
        'bus': [0, 60, 100],
        'train': [0, 80, 100],
        'motorcycle': [0, 0, 230],
        'bicycle': [119, 11, 32]
    }
    
    # Convert to numpy array for faster lookup
    color_map = np.zeros((len(LABEL_COLORS), 3), dtype=np.uint8)
    for idx, color in enumerate(LABEL_COLORS.values()):
        color_map[idx] = color
    
    return color_map

def process_frame(frame, model, target_size=(512, 512)):
    # Preprocess the frame
    original_size = frame.shape[:2]
    frame_resized = cv2.resize(frame, target_size)
    frame_normalized = frame_resized.astype(np.float32) / 255.0
    
    # Add batch dimension
    input_tensor = np.expand_dims(frame_normalized, axis=0)
    
    # Get prediction
    prediction = model.predict(input_tensor, verbose=0)
    
    # Get class indices
    segmentation_map = np.argmax(prediction[0], axis=-1)
    
    return segmentation_map, original_size

def create_overlay(segmentation_map, color_map, original_size):
    # Create colored segmentation map
    colored_segmentation = color_map[segmentation_map]
    
    # Resize back to original size
    colored_segmentation = cv2.resize(colored_segmentation, 
                                    (original_size[1], original_size[0]), 
                                    interpolation=cv2.INTER_NEAREST)
    
    return colored_segmentation

def process_video(input_path, output_path, model_path):
    # Load model with custom metric
    custom_objects = {
        'mean_iou': mean_iou  # Make sure this matches your training metric
    }
    model = load_model(model_path, custom_objects=custom_objects)
    
    # Create color map
    color_map = create_color_map()
    
    # Open video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError("Could not open the video file")
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width*2, height))
    
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame
        segmentation_map, original_size = process_frame(frame_rgb, model)
        
        # Create colored overlay
        colored_segmentation = create_overlay(segmentation_map, color_map, original_size)
        
        # Convert back to BGR for OpenCV
        colored_segmentation_bgr = cv2.cvtColor(colored_segmentation, cv2.COLOR_RGB2BGR)
        
        # Combine original frame and segmentation
        combined_output = np.hstack((frame, colored_segmentation_bgr))
        
        # Write frame
        out.write(combined_output)
        
        # Update progress
        frame_count += 1
        if frame_count % 10 == 0:
            print(f"Processed {frame_count}/{total_frames} frames ({(frame_count/total_frames)*100:.1f}%)")
    
    # Release resources
    cap.release()
    out.release()
    print("Video processing complete!")

if __name__ == "__main__":
    # Define paths
    input_video_path = "testvid2.mp4"  # Replace with your video path
    output_video_path = "output_segmentationTEST3.mp4"
    model_path = "deeplab_v3_plus_bdd100k.h5"  # Path to your saved model
    
    # Process video
    process_video(input_video_path, output_video_path, model_path)

