import os
import numpy as np
import tensorflow as tf
import albumentations as A
import cv2
import matplotlib.pyplot as plt
from bdd100Kdataloader import BDD100kDataLoader
from config import Config
from functions import mean_iou

# GPU Configuration
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


# U-Net Model
def create_unet(input_shape, num_classes):
    def conv_block(input_tensor, num_filters):
        x = tf.keras.layers.Conv2D(num_filters, 3, padding='same')(input_tensor)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        
        x = tf.keras.layers.Conv2D(num_filters, 3, padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        return x

    # Input
    inputs = tf.keras.layers.Input(input_shape)
    
    # Encoder
    conv1 = conv_block(inputs, 64)
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = conv_block(pool1, 128)
    pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = conv_block(pool2, 256)
    pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = conv_block(pool3, 512)
    pool4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv4)
    
    # Bridge
    conv5 = conv_block(pool4, 1024)
    
    # Decoder
    up6 = tf.keras.layers.Conv2DTranspose(512, 2, strides=(2, 2), padding='same')(conv5)
    up6 = tf.keras.layers.Concatenate()([up6, conv4])
    conv6 = conv_block(up6, 512)
    
    up7 = tf.keras.layers.Conv2DTranspose(256, 2, strides=(2, 2), padding='same')(conv6)
    up7 = tf.keras.layers.Concatenate()([up7, conv3])
    conv7 = conv_block(up7, 256)
    
    up8 = tf.keras.layers.Conv2DTranspose(128, 2, strides=(2, 2), padding='same')(conv7)
    up8 = tf.keras.layers.Concatenate()([up8, conv2])
    conv8 = conv_block(up8, 128)
    
    up9 = tf.keras.layers.Conv2DTranspose(64, 2, strides=(2, 2), padding='same')(conv8)
    up9 = tf.keras.layers.Concatenate()([up9, conv1])
    conv9 = conv_block(up9, 64)
    
    # Output
    outputs = tf.keras.layers.Conv2D(num_classes, 1, activation='softmax')(conv9)
    
    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    return model

# Metrics function (unchanged)
def mean_iou(y_true, y_pred):
    y_pred = tf.argmax(y_pred, axis=-1)
    y_pred = tf.cast(y_pred, tf.float32)
    y_true = tf.cast(y_true, tf.float32)
    
    intersect = tf.reduce_sum(tf.cast(tf.equal(y_pred, y_true), tf.float32))
    union = tf.reduce_sum(tf.cast(tf.not_equal(y_pred, y_true), tf.float32))
    
    iou = tf.reduce_sum(intersect) / (tf.reduce_sum(union) + tf.reduce_sum(intersect))
    return iou

# Plot training history function (unchanged)
def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(history.history['mean_iou'], label='Training Mean IoU')
    ax1.plot(history.history['val_mean_iou'], label='Validation Mean IoU')
    ax1.set_title('Model Mean IoU')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Mean IoU')
    ax1.legend()
    
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

# Training function
def train_unet():
    # Configuration
    config = Config()
    
    # Find all image paths
    train_images = sorted(
        [os.path.join(config.TRAIN_IMAGES, f) for f in os.listdir(config.TRAIN_IMAGES) if f.endswith('.jpg')]
    )
    train_masks = [
        os.path.join(config.TRAIN_MASKS, f.replace('.jpg', '_train_color.png')) 
        for f in os.listdir(config.TRAIN_IMAGES) if f.endswith('.jpg')
    ]

    val_images = sorted(
        [os.path.join(config.VAL_IMAGES, f) for f in os.listdir(config.VAL_IMAGES) if f.endswith('.jpg')]
    )
    val_masks = [
        os.path.join(config.VAL_MASKS, f.replace('.jpg', '_train_color.png')) 
        for f in os.listdir(config.VAL_IMAGES) if f.endswith('.jpg')
    ]
    
    # Data Generators
    train_generator = BDD100kDataLoader(
        train_images, train_masks, config, 
        augment=True, is_training=True
    )
    val_generator = BDD100kDataLoader(
        val_images, val_masks, config, 
        augment=False, is_training=False
    )
    
    # Create model
    model = create_unet(
        input_shape=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH, 3), 
        num_classes=config.NUM_CLASSES
    )
    
    # Compile Model
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=config.LEARNING_RATE, 
        decay=1e-5
    )
    
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[mean_iou]
    )
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            'best_unet_model.h5', 
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
    
    # Train
    history = model.fit(
        train_generator,
        epochs=config.EPOCHS,
        validation_data=val_generator,
        callbacks=callbacks,
        workers=4,
        use_multiprocessing=True
    )
    
    # Save model
    model.save('unet_bdd100k1.h5')
    print("Model saved successfully.")
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate on test set
    test_metrics = evaluate_model(model, config)
    
    return model, history, test_metrics

# Evaluation function (unchanged)
def evaluate_model(model, config):
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
    
    test_generator = BDD100kDataLoader(
        test_images, test_masks, config, 
        augment=False, is_training=False
    )
    
    evaluation_metrics = model.evaluate(test_generator)
    
    print("\nTest Evaluation:")
    for metric_name, metric_value in zip(model.metrics_names, evaluation_metrics):
        print(f"{metric_name}: {metric_value}")
    
    return dict(zip(model.metrics_names, evaluation_metrics))

# Main Execution
if __name__ == '__main__':
    model, history, test_metrics = train_unet()