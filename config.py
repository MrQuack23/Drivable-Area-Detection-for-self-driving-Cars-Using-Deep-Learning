import os

class Config:
    # Dataset Paths
    BASE_PATH = 'C:/Users/mimiz/bdd100k/seg'
    TRAIN_IMAGES = os.path.join(BASE_PATH, 'images/train')
    TRAIN_MASKS = os.path.join(BASE_PATH, 'color_labels/train')
    VAL_IMAGES = os.path.join(BASE_PATH, 'images/val')
    VAL_MASKS = os.path.join(BASE_PATH, 'color_labels/val')
    
    # Training Hyperparameters
    IMAGE_HEIGHT = 512
    IMAGE_WIDTH = 512
    BATCH_SIZE = 4
    EPOCHS = 50
    LEARNING_RATE = 1e-4
    NUM_CLASSES = 19  
    
    # Label Colors (RGB)
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
