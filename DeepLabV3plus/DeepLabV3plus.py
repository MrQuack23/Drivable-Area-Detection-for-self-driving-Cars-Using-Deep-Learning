import os
import numpy as np
import tensorflow as tf
import albumentations as A
import cv2
import matplotlib.pyplot as plt
from config import Config
from bdd100Kdataloader import BDD100kDataLoader
from functions import mean_iou

# GPU Configuration
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# DeepLabV3+ Model
def create_deeplab_v3_plus(input_shape, num_classes):
    def SepConv_BN(x, filters, prefix, stride=1, kernel_size=3, rate=1, depth_activation=False, epsilon=1e-3):
        """ Separable Convolution with BatchNorm """
        if stride == 1:
            depth_padding = 'same'
        else:
            depth_padding = 'valid'
        
        x = tf.keras.layers.SeparableConv2D(filters, kernel_size, strides=stride, 
                                   dilation_rate=(rate, rate), 
                                   padding=depth_padding, use_bias=False, 
                                   name=prefix + '_depthwise')(x)
        x = tf.keras.layers.BatchNormalization(name=prefix + '_depthwise_BN', epsilon=epsilon)(x)
        
        if depth_activation:
            x = tf.keras.layers.Activation('relu', name=prefix + '_depthwise_activation')(x)
        
        x = tf.keras.layers.Conv2D(filters, (1, 1), padding='same', use_bias=False, name=prefix + '_pointwise')(x)
        x = tf.keras.layers.BatchNormalization(name=prefix + '_pointwise_BN', epsilon=epsilon)(x)
        
        if depth_activation:
            x = tf.keras.layers.Activation('relu', name=prefix + '_pointwise_activation')(x)
        
        return x
    
    def Conv2D_BN(x, filters, kernel_size, rate=1, depth_activation=False):
        """Standard convolution with BatchNorm"""
        x = tf.keras.layers.Conv2D(filters, kernel_size, padding='same', 
                                dilation_rate=(rate, rate), use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization(epsilon=1e-5)(x)
        if depth_activation:
            x = tf.keras.layers.Activation('relu')(x)
        return x
    
    def ASPP(x, OS):
        """ Atrous Spatial Pyramid Pooling """

        # Get input shape
        input_shape = tf.keras.backend.int_shape(x)
        if input_shape is None:
            raise ValueError("Input shape cannot be None")
        
        # Ensure we have a 4D tensor (batch_size, height, width, channels)
        if len(input_shape) != 4:
            raise ValueError(f"Expected 4D tensor, got shape: {input_shape}")
    
        if OS == 16:
            atrous_rates = (1, 6, 12, 18)
        elif OS == 8:
            atrous_rates = (1, 12, 24, 36)
        else:
            raise RuntimeError('Wrong OS!')
        
        # Global Average Pooling
        x_pool = tf.keras.layers.GlobalAveragePooling2D()(x)
        x_pool = tf.keras.layers.Reshape((1, 1, input_shape[3]))(x_pool)
        x_pool = tf.keras.layers.Conv2D(256, (1, 1), padding='same', use_bias=False)(x_pool)
        x_pool = tf.keras.layers.BatchNormalization(epsilon=1e-5)(x_pool)
        x_pool = tf.keras.layers.Activation('relu')(x_pool)
        x_pool = tf.keras.layers.UpSampling2D(
            size=(input_shape[1], input_shape[2]),
            interpolation='bilinear'
        )(x_pool)
        
        # 2) 1x1 convolution branch
        x_b0 = Conv2D_BN(x, 256, (1, 1), depth_activation=True)
        
        # 3) Three 3x3 atrous convolutions with different rates
        x_b1 = Conv2D_BN(x, 256, (3, 3), rate=atrous_rates[0], depth_activation=True)
        x_b2 = Conv2D_BN(x, 256, (3, 3), rate=atrous_rates[1], depth_activation=True)
        x_b3 = Conv2D_BN(x, 256, (3, 3), rate=atrous_rates[2], depth_activation=True)
        
        # Concatenate all branches (including pooling branch)
        x = tf.keras.layers.Concatenate()([x_b0, x_b1, x_b2, x_b3, x_pool])
        
        # Final 1x1 convolution
        x = tf.keras.layers.Conv2D(256, (1, 1), padding='same', use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization(epsilon=1e-5)(x)
        x = tf.keras.layers.Activation('relu')(x)
        
        return x
    
    # Backbone (ResNet50)
    base_model = tf.keras.applications.ResNet50(
        weights='imagenet', 
        include_top=False, 
        input_shape=input_shape
    )
    
    # Encoder
    image_input = base_model.input
    x = base_model.get_layer('conv4_block6_out').output
    input_a = base_model.get_layer('conv2_block3_out').output
    
    # ASPP
    x = ASPP(x, OS=16)
    
    # Decoder
    x = tf.keras.layers.UpSampling2D(size=(4, 4), interpolation='bilinear')(x)
    input_a = Conv2D_BN(input_a, 48, (1, 1), depth_activation=True)
    input_a = tf.keras.layers.BatchNormalization(epsilon=1e-5)(input_a)
    input_a = tf.keras.layers.Activation('relu')(input_a)
    
    x = tf.keras.layers.Concatenate()([x, input_a])
    
    x = SepConv_BN(x, 256, 'decoder_conv1', depth_activation=True)
    x = SepConv_BN(x, 256, 'decoder_conv2', depth_activation=True)
    
    # Final Classification Layer
    x = tf.keras.layers.UpSampling2D(size=(4, 4), interpolation='bilinear')(x)
    x = tf.keras.layers.Conv2D(num_classes, (1, 1), padding='same', activation='softmax')(x)
    
    model = tf.keras.Model(inputs=image_input, outputs=x)
    return model


def mean_iou(y_true, y_pred):
    # y_pred is (batch_size, height, width, num_classes) with probabilities
    # Convert predictions to class indices
    y_pred = tf.argmax(y_pred, axis=-1)
    
    # Convert to float32 for calculation
    y_pred = tf.cast(y_pred, tf.float32)
    y_true = tf.cast(y_true, tf.float32)
    
    # Now both y_pred and y_true are (batch_size, height, width) with class indices as values
    
    # Calculate intersection and union
    intersect = tf.reduce_sum(tf.cast(tf.equal(y_pred, y_true), tf.float32))
    union = tf.reduce_sum(tf.cast(tf.not_equal(y_pred, y_true), tf.float32))
    
    iou = tf.reduce_sum(intersect) / (tf.reduce_sum(union) + tf.reduce_sum(intersect))
    return iou

def plot_training_history(history):
    """
    Plot training and validation accuracy and loss
    
    Args:
        history: Keras training history object
    """
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot accuracy
    ax1.plot(history.history['mean_iou'], label='Training Mean IoU')
    ax1.plot(history.history['val_mean_iou'], label='Validation Mean IoU')
    ax1.set_title('Model Mean IoU')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Mean IoU')
    ax1.legend()
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

def prepare_test_data(config):
    TEST_IMAGES = os.path.join(config.BASE_PATH, 'images/test')
    TEST_MASKS = os.path.join(config.BASE_PATH, 'color_labels/test')
    
    test_images = sorted(
        [os.path.join(TEST_IMAGES, f) for f in os.listdir(TEST_IMAGES) if f.endswith('.jpg')]
    )
    test_masks = [
        os.path.join(TEST_MASKS, f.replace('.jpg', '_train_color.png')) 
        for f in os.listdir(TEST_IMAGES) if f.endswith('.jpg')
    ]
    
    return test_images, test_masks

def evaluate_model(model, config):
    # Prepare test data
    test_images, test_masks = prepare_test_data(config)
    
    # Create test data generator
    test_generator = BDD100kDataLoader(
        test_images, test_masks, config, 
        augment=False, is_training=False
    )
    
    # Evaluate model
    evaluation_metrics = model.evaluate(test_generator)
    
    print("\nTest Evaluation:")
    for metric_name, metric_value in zip(model.metrics_names, evaluation_metrics):
        print(f"{metric_name}: {metric_value}")
    
    return dict(zip(model.metrics_names, evaluation_metrics))

# Training Setup
def train_deeplab_v3_plus():
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
    
    # Model
    model = create_deeplab_v3_plus(
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
            'best_deeplab_model.h5', 
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
    model.save('deeplab_v3_plus_bdd100k.h5')
    print("Model saved successfully.")
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate on test set
    test_metrics = evaluate_model(model, config)
    
    return model, history, test_metrics

# Main Execution
if __name__ == '__main__':
     model, history, test_metrics = train_deeplab_v3_plus()
    