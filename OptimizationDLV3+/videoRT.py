import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from DeepLabV3plus import mean_iou
import time

def create_color_map():
    # Define the color map
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

def real_time_video_segmentation(video_path, model_path):
    # Load model with custom metric
    custom_objects = {'mean_iou': mean_iou}
    model = load_model(model_path, custom_objects=custom_objects)
    
    # Create color map
    color_map = create_color_map()
    
    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open the video file")
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create output video writer for real-time saving
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output_realtime22.mp4', fourcc, fps, (width*2, height))
    
    # Initialize FPS calculation
    fps_start_time = time.time()
    fps_counter = 0
    processing_fps = 0
    
    print(f"Starting video segmentation. Original video FPS: {fps}")
    print("Press 'q' to quit, 'p' to pause/unpause")
    print("Saving output to 'output_realtime.mp4'")
    
    paused = False
    
    while cap.isOpened():
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("End of video file")
                break
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process frame
            segmentation_map, original_size = process_frame(frame_rgb, model)
            
            # Create colored overlay
            colored_segmentation = create_overlay(segmentation_map, color_map, original_size)
            
            # Convert back to BGR for display and saving
            colored_segmentation_bgr = cv2.cvtColor(colored_segmentation, cv2.COLOR_RGB2BGR)
            
            # Combine original frame and segmentation
            combined_output = np.hstack((frame, colored_segmentation_bgr))
            
            # Calculate processing FPS
            fps_counter += 1
            if time.time() - fps_start_time > 1:
                processing_fps = fps_counter
                fps_counter = 0
                fps_start_time = time.time()
            
            # Add FPS information to frame
            cv2.putText(combined_output, f'Original FPS: {fps}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(combined_output, f'Processing FPS: {processing_fps}', (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Save frame
            out.write(combined_output)
            
            try:
                # Try to display the frame
                cv2.namedWindow('Video Segmentation', cv2.WINDOW_NORMAL)
                cv2.imshow('Video Segmentation', combined_output)
            except Exception as e:
                print(f"Warning: Could not display frame: {e}")
                print("Continuing with video processing and saving...")
        
        # Handle key presses (wait for a shorter time to keep processing smooth)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            paused = not paused
            if paused:
                print("Video paused")
            else:
                print("Video resumed")
    
    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Processing complete! Output saved to 'output_realtime22.mp4'")

if __name__ == "__main__":
    # Define paths
    video_path = "testvid2.mp4"  # Replace with your video path
    model_path = "final_pruned_model.h5"   # Replace with your model path
    
    # Start real-time video segmentation
    try:
        real_time_video_segmentation(video_path, model_path)
    except KeyboardInterrupt:
        print("\nStopping video segmentation...")