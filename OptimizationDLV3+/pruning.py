import tensorflow as tf
import tensorflow_model_optimization as tfmot
from bdd100Kdataloader import BDD100kDataLoader
from config import Config
from functions import mean_iou
from ..DeepLabV3plus.DeepLabV3plus import create_deeplab_v3_plus
import os
import numpy as np

def create_pruning_schedule():
    # Define pruning parameters
    pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=0.0,
            final_sparsity=0.50,  # Target 50% sparsity
            begin_step=0,
            end_step=1000  # Adjust based on your training steps
        )
    }
    return pruning_params

def apply_pruning_to_model(model):
    """Apply pruning to all layers except the input and output layers"""
    prunable_layers = ['Conv2D', 'SeparableConv2D']
    
    # Get input and output layers
    input_layer = model.input
    output_layer = model.output
    
    # Apply pruning to intermediate layers
    for layer in model.layers[1:-1]:
        if any(isinstance(layer, getattr(tf.keras.layers, layer_type)) for layer_type in prunable_layers):
            layer = tfmot.sparsity.keras.prune_low_magnitude(layer, **create_pruning_schedule())
    
    # Recreate model with pruning
    pruned_model = tf.keras.Model(input_layer, output_layer)
    return pruned_model

def train_pruned_model():
    # Configuration
    config = Config()
    
    # Load data paths
    train_images = sorted([os.path.join(config.TRAIN_IMAGES, f) for f in os.listdir(config.TRAIN_IMAGES) if f.endswith('.jpg')])
    train_masks = [os.path.join(config.TRAIN_MASKS, f.replace('.jpg', '_train_color.png')) 
                  for f in os.listdir(config.TRAIN_IMAGES) if f.endswith('.jpg')]
    
    val_images = sorted([os.path.join(config.VAL_IMAGES, f) for f in os.listdir(config.VAL_IMAGES) if f.endswith('.jpg')])
    val_masks = [os.path.join(config.VAL_MASKS, f.replace('.jpg', '_train_color.png')) 
                for f in os.listdir(config.VAL_IMAGES) if f.endswith('.jpg')]
    
    # Create data generators
    train_generator = BDD100kDataLoader(train_images, train_masks, config, augment=True, is_training=True)
    val_generator = BDD100kDataLoader(val_images, val_masks, config, augment=False, is_training=False)
    
    # Create base model
    base_model = create_deeplab_v3_plus(
        input_shape=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH, 3),
        num_classes=config.NUM_CLASSES
    )
    
    # Apply pruning
    pruned_model = apply_pruning_to_model(base_model)
    
    # Calculate steps per epoch
    steps_per_epoch = len(train_images) // config.BATCH_SIZE
    
    # Update pruning params with correct steps
    pruning_params = create_pruning_schedule()
    pruning_params['pruning_schedule'].end_step = steps_per_epoch * config.EPOCHS
    
    # Compile pruned model
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE)
    pruned_model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[mean_iou]
    )
    
    # Callbacks
    callbacks = [
        tfmot.sparsity.keras.UpdatePruningStep(),
        tfmot.sparsity.keras.PruningSummaries(log_dir='pruning_logs'),
        tf.keras.callbacks.ModelCheckpoint(
            'best_pruned_model.h5',
            monitor='val_mean_iou',
            mode='max',
            save_best_only=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_mean_iou',
            patience=15,
            restore_best_weights=True
        )
    ]
    
    # Train pruned model
    history = pruned_model.fit(
        train_generator,
        epochs=config.EPOCHS,
        validation_data=val_generator,
        callbacks=callbacks,
        workers=4,
        use_multiprocessing=True
    )
    
    # Strip pruning wrapper and save final model
    final_model = tfmot.sparsity.keras.strip_pruning(pruned_model)
    final_model.save('final_pruned_model.h5')
    
    # Calculate compression ratio
    original_size = os.path.getsize('best_pruned_model.h5')
    pruned_size = os.path.getsize('final_pruned_model.h5')
    compression_ratio = (original_size - pruned_size) / original_size * 100
    
    print(f"Model compressed by {compression_ratio:.2f}%")
    
    return final_model, history

if __name__ == '__main__':
    pruned_model, training_history = train_pruned_model()





### Model compressed by 66.53%