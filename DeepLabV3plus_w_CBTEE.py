import os
import numpy as np
import tensorflow as tf
import albumentations as A
import cv2
import matplotlib.pyplot as plt
import json
import seaborn as sns
from collections import defaultdict
from config import Config
from bdd100Kdataloader import BDD100kDataLoader

# GPU Configuration
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# CBT Metrics class for threshold computation
class CBTMetrics:
    def __init__(self, num_classes, num_exits, alpha=0.95, beta=0.998):
        self.num_classes = num_classes
        self.num_exits = num_exits
        self.alpha = alpha
        self.beta = beta
        self.class_predictions = defaultdict(lambda: defaultdict(list))
        self.class_thresholds = None
    
    def _update_single_class(self, exit_idx, class_idx, predictions, true_labels):
        """Non-tf.function method to update predictions for a single class"""
        class_mask = (true_labels == class_idx)
        if np.any(class_mask):
            class_predictions = predictions[class_mask]
            self.class_predictions[exit_idx][class_idx].append(class_predictions)

    def update_predictions(self, exit_idx, predictions, true_labels):
        """Update prediction statistics for each class at given exit"""
        # Convert tensors to numpy arrays outside of graph execution
        predictions = predictions.numpy()
        true_labels = true_labels.numpy()
        exit_idx = int(exit_idx)
        
        # Process each class
        for class_idx in range(self.num_classes):
            self._update_single_class(exit_idx, class_idx, predictions, true_labels)

    def compute_thresholds(self):
        """Compute class-specific thresholds following CBT paper methodology"""
        try:
            # Calculate pn,k for each exit n and class k (equation 1)
            p_n_k = {}
            for exit_idx in range(self.num_exits):
                p_n_k[exit_idx] = {}
                for class_idx in range(self.num_classes):
                    if self.class_predictions[exit_idx][class_idx]:
                        # Concatenate all predictions for this class/exit
                        predictions = np.concatenate([
                            p.reshape(-1, self.num_classes) 
                            for p in self.class_predictions[exit_idx][class_idx]
                        ])
                        p_n_k[exit_idx][class_idx] = np.mean(predictions, axis=0)
                    else:
                        p_n_k[exit_idx][class_idx] = np.zeros(self.num_classes)
            
            # Calculate Pk by averaging across exits (equation 2)
            P_k = np.zeros((self.num_classes, self.num_classes))
            for class_idx in range(self.num_classes):
                class_means = []
                for exit_idx in range(self.num_exits):
                    if class_idx in p_n_k[exit_idx]:
                        class_means.append(p_n_k[exit_idx][class_idx])
                if class_means:
                    P_k[class_idx] = np.mean(class_means, axis=0)
            
            # Initialize thresholds (difference between top two probabilities)
            thresholds = np.zeros(self.num_classes)
            for class_idx in range(self.num_classes):
                sorted_probs = np.sort(P_k[class_idx])
                thresholds[class_idx] = sorted_probs[-1] - sorted_probs[-2]
            
            # Scale thresholds inversely (equation 3)
            min_thresh = np.min(thresholds)
            max_thresh = np.max(thresholds)
            
            if max_thresh > min_thresh:
                scaled_thresholds = (1 - (thresholds - min_thresh) / (max_thresh - min_thresh)) * \
                                  (self.beta - self.alpha) + self.alpha
            else:
                scaled_thresholds = np.full_like(thresholds, (self.alpha + self.beta) / 2)
            
            self.class_thresholds = scaled_thresholds
            return self.class_thresholds
            
        except Exception as e:
            print(f"Error computing thresholds: {str(e)}")
            raise

    def reset_metrics(self):
        """Reset all collected predictions"""
        self.class_predictions = defaultdict(lambda: defaultdict(list))
        self.class_thresholds = None
        
    
# DeepLabV3+ Model with Class-Based Thresholding Early Exiting
def create_deeplab_v3_plus_cbt(input_shape, num_classes, num_exits=2):
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
    
    def create_exit_block(x, exit_index, total_exits, prefix, scale_factor=1):
        """Create early exit block following ADP-C paper structure
        
        The structure follows ADP-C paper:
        1. Initial 1x1 conv to reduce channels to a fixed small number
        2. Encoder: Multiple pool-conv layers (number depends on exit position)
        3. Decoder: Equal number of interpolate-conv layers
        4. Final upsampling to match input resolution if needed
        """
        # Fixed channel width for all exits (kept small to save computation)
        FIXED_CHANNELS = 64  # Can be adjusted based on memory constraints
        
        # Calculate number of downsampling operations for this exit
        # D = N - i where N is total exits and i is exit index (1-based)
        num_downsampling = total_exits - (exit_index + 1)
        
        # 1. Initial channel reduction
        x = tf.keras.layers.Conv2D(
            FIXED_CHANNELS, (1, 1),
            padding='same',
            use_bias=False,
            name=f'{prefix}_initial_reduce'
        )(x)
        x = tf.keras.layers.BatchNormalization(name=f'{prefix}_initial_bn')(x)
        x = tf.keras.layers.Activation('relu', name=f'{prefix}_initial_relu')(x)
        
        # Store intermediate features for skip connections
        skip_connections = []
        
        # 2. Encoder: Pool-Conv layers
        for i in range(num_downsampling):
            # Store for skip connection
            skip_connections.append(x)
            
            # Pooling
            x = tf.keras.layers.AveragePooling2D(
                pool_size=(2, 2),
                name=f'{prefix}_pool_{i}'
            )(x)
            
            # Convolution
            x = tf.keras.layers.Conv2D(
                FIXED_CHANNELS, (1, 1),
                padding='same',
                use_bias=False,
                name=f'{prefix}_enc_conv_{i}'
            )(x)
            x = tf.keras.layers.BatchNormalization(name=f'{prefix}_enc_bn_{i}')(x)
            x = tf.keras.layers.Activation('relu', name=f'{prefix}_enc_relu_{i}')(x)
        
        # 3. Decoder: Interpolate-Conv layers
        for i in range(num_downsampling):
            # Upsampling
            x = tf.keras.layers.UpSampling2D(
                size=(2, 2),
                interpolation='bilinear',
                name=f'{prefix}_upsample_{i}'
            )(x)
            
            # Skip connection (optional but helps with spatial accuracy)
            skip = skip_connections[-(i+1)]
            x = tf.keras.layers.Concatenate(name=f'{prefix}_concat_{i}')([x, skip])
            
            # Convolution
            x = tf.keras.layers.Conv2D(
                FIXED_CHANNELS, (1, 1),
                padding='same',
                use_bias=False,
                name=f'{prefix}_dec_conv_{i}'
            )(x)
            x = tf.keras.layers.BatchNormalization(name=f'{prefix}_dec_bn_{i}')(x)
            x = tf.keras.layers.Activation('relu', name=f'{prefix}_dec_relu_{i}')(x)
        
        # Final prediction
        x = tf.keras.layers.Conv2D(
            num_classes, (1, 1),
            padding='same',
            name=f'{prefix}_final_conv'
        )(x)
        
        # Additional upsampling if needed to match input resolution
        if scale_factor > 1:
            x = tf.keras.layers.UpSampling2D(
                size=(scale_factor, scale_factor),
                interpolation='bilinear',
                name=f'{prefix}_final_upsample'
            )(x)
        
        x = tf.keras.layers.Activation('softmax', name=f'{prefix}_final')(x)
        
        return x
    
    # Backbone (ResNet50)
    base_model = tf.keras.applications.ResNet50(
        weights='imagenet', 
        include_top=False, 
        input_shape=input_shape
    )
    
    # Encoder
    image_input = base_model.input

    # Create early exits
    exits = []
    total_exits = num_exits + 1 # Include final exit

    # Get features for main path (Encoder)
    x = base_model.get_layer('conv4_block6_out').output

    # Pre-ASPP exit
    encoder_exit1 = create_exit_block(x, exit_index=0, total_exits=total_exits, prefix='encoder_exit1', scale_factor=16)
    exits.append(encoder_exit1)

    # ASPP module
    x = ASPP(x, OS=16)

    # Decoder
    x = tf.keras.layers.UpSampling2D(size=(4, 4), interpolation='bilinear')(x)
    low_level_features = base_model.get_layer('conv2_block3_out').output
    low_level_features = Conv2D_BN(low_level_features, 48, (1, 1), depth_activation=True)
    
    # Decoder exit before concatenation
    decoder_exit1 = create_exit_block(x, exit_index=1, total_exits=total_exits, prefix='decoder_exit1', scale_factor=4)
    exits.append(decoder_exit1)
    
    x = tf.keras.layers.Concatenate()([x, low_level_features])
    
    x = SepConv_BN(x, 256, 'decoder_conv1', depth_activation=True)
    x = SepConv_BN(x, 256, 'decoder_conv2', depth_activation=True)
    
    # Final Classification Layer
    x = tf.keras.layers.UpSampling2D(size=(4, 4), interpolation='bilinear')(x)
    final_output = tf.keras.layers.Conv2D(num_classes, (1, 1), padding='same', activation='softmax', name='final_exit')(x)
    
    model = tf.keras.Model(inputs=image_input, outputs=[*exits, final_output])
    return model


class CBTDeepLabV3Plus(tf.keras.Model):
    def __init__(self, input_shape, num_classes, num_exits=2, alpha=0.95, beta=0.998):
        super().__init__()
        self.num_classes = num_classes
        self.num_exits = num_exits
        self.cbt_metrics = CBTMetrics(num_classes, num_exits, alpha, beta)
        self.model = create_deeplab_v3_plus_cbt(input_shape, num_classes, num_exits)
        self.confidence_masks = None
    
    def compile(self, optimizer, metrics, **kwargs):
        """Compile model with exit-specific losses"""
        # Progressive loss weights for each exit + final output
        total_outputs = self.num_exits + 1
        loss_weights = {
            f'exit_{i+1}': 1.0 
            for i in range(self.num_exits)
        }

        # Weight for final output
        loss_weights[f'exit_{total_outputs}'] = 1.0
        
        # Same loss function for all exits
        losses = {
            f'exit_{i+1}': 'sparse_categorical_crossentropy'
            for i in range(total_outputs)
        }
        
        super().compile(
            optimizer=optimizer,
            loss=losses,
            loss_weights=loss_weights,
            metrics=metrics,
            **kwargs
        )
    
    @tf.function
    def train_step(self, data):
        """Custom training step with CBT metrics update"""
        images, labels = data
        
        with tf.GradientTape() as tape:
            # Get predictions from all exits (including final output)
            predictions = self.model(images, training=True)
            
            # Format predictions and labels for loss calculation
            total_outputs = self.num_exits + 1
            y_pred = {f'exit_{i+1}': pred for i, pred in enumerate(predictions)}
            y_true = {f'exit_{i+1}': labels for i in range(total_outputs)}
            
            # Calculate total loss
            total_loss = self.compiled_loss(y_true, y_pred)
        
        # Update gradients
        gradients = tape.gradient(total_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        # Update CBT metrics for early exits only (not final output)
        for i, preds in enumerate(predictions[:-1]):  # Skip final output
            tf.py_function(
                func=self.cbt_metrics.update_predictions,
                inp=[tf.constant(i), preds, labels],
                Tout=[]
            )
        
        # Update metrics using final output
        self.compiled_metrics.update_state(labels, predictions[-1])
        
        return {m.name: m.result() for m in self.metrics}


    def test_step(self, data):
        """Custom test step"""
        images, labels = data
        predictions = self.model(images, training=False)
        
        # Update metrics
        self.compiled_metrics.update_state(labels, predictions[-1])
        
        return {m.name: m.result() for m in self.metrics}

    def predict_step(self, data):
        """Custom predict step with early exiting"""
        images = data
        return self(images, training=False)
    
    def compute_thresholds(self):
        """Compute CBT thresholds after training"""
        return self.cbt_metrics.compute_thresholds()
    
    @tf.function
    def call(self, inputs, training=False):
        """Forward pass with early exiting"""
        batch_size = tf.shape(inputs)[0]
        
        # Initialize confidence mask if needed
        if self.confidence_masks is None or training:
            self.confidence_masks = tf.zeros(
                (batch_size, *inputs.shape[1:3], 1),
                dtype=tf.bool
            )
        
        # Get predictions from all exits
        predictions = self.model(inputs, training=training)
        
        if not training and self.cbt_metrics.class_thresholds is not None:
            # Initialize final predictions tensor
            final_predictions = tf.zeros_like(predictions[-1])
            
            # Process only early exits (not final output)
            for i, exit_preds in enumerate(predictions[:-1]):
                # Get confidence scores and class predictions
                class_probs = tf.reduce_max(exit_preds, axis=-1, keepdims=True)
                class_idx = tf.argmax(exit_preds, axis=-1)
                
                # Apply class-specific thresholds
                thresholds = tf.gather(
                    self.cbt_metrics.class_thresholds,
                    class_idx
                )
                
                # Update confidence mask
                confident_pixels = class_probs > thresholds[..., tf.newaxis]
                new_confident = confident_pixels & ~self.confidence_masks
                
                # Update predictions for newly confident pixels
                final_predictions = tf.where(
                    new_confident,
                    exit_preds,
                    final_predictions
                )
                
                # Update overall confidence mask
                self.confidence_masks = self.confidence_masks | new_confident
            
            # Use final output predictions for remaining unconfident pixels
            final_predictions = tf.where(
                self.confidence_masks,
                final_predictions,
                predictions[-1]
            )
            
            return final_predictions
        
        return predictions[-1]


    def reset_states(self):
        """Reset model states between epochs"""
        super().reset_states()
        self.confidence_masks = None
        self.cbt_metrics.reset_metrics()


    def get_config(self):
        """Get model configuration"""
        config = super().get_config()
        config.update({
            'num_classes': self.num_classes,
            'num_exits': self.num_exits,
            'alpha': self.cbt_metrics.alpha,
            'beta': self.cbt_metrics.beta
        })
        return config


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
    plt.savefig('training_history_cbt.png')
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

    def plot_predictions(model, test_generator, num_samples=3):
        images, masks = test_generator.__getitem__(0)
        predictions = model.predict(images[:num_samples])
        
        plt.figure(figsize=(15, 5*num_samples))
        for i in range(num_samples):
            # Original Image
            plt.subplot(num_samples, 3, i*3 + 1)
            plt.imshow(images[i])
            plt.title('Original Image')
            plt.axis('off')
            
            # Ground Truth
            plt.subplot(num_samples, 3, i*3 + 2)
            plt.imshow(masks[i])
            plt.title('Ground Truth')
            plt.axis('off')
            
            # Prediction
            plt.subplot(num_samples, 3, i*3 + 3)
            pred_mask = np.argmax(predictions[i], axis=-1)
            plt.imshow(pred_mask)
            plt.title('Prediction')
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig('test_predictions.png')
        plt.close()
    
    # Plot some test predictions
    plot_predictions(model, test_generator)
    print("Test predictions plotted and saved as 'test_predictions_cbt.png'")
    
    return dict(zip(model.metrics_names, evaluation_metrics))


def save_and_visualize_thresholds(config, thresholds, save_path="thresholds"):
    """
    Save thresholds to JSON and create visualization
    
    Args:
        model: trained CBTDeepLabV3Plus model
        save_path: base path for saving files (without extension)
    """
    # Get class names from config
    class_names = list(config.LABEL_COLORS.keys())
    
    # Create threshold dictionary
    threshold_dict = {
        class_name: float(threshold)  # Convert to Python float for JSON serialization
        for class_name, threshold in zip(class_names, thresholds)
    }
    
    # Save to JSON
    with open(f"{save_path}.json", 'w') as f:
        json.dump(threshold_dict, f, indent=4)
    
    # Create visualization
    plt.figure(figsize=(12, 6))
    
    # Sort thresholds for better visualization
    sorted_items = sorted(threshold_dict.items(), key=lambda x: x[1], reverse=True)
    labels, values = zip(*sorted_items)
    
    # Create bar plot
    bars = plt.bar(labels, values)
    
    # Customize plot
    plt.title('Class-Based Thresholds', fontsize=14, pad=20)
    plt.xlabel('Classes', fontsize=12)
    plt.ylabel('Threshold Value', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom')
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save plot
    plt.savefig(f"{save_path}_bar.png", dpi=300, bbox_inches='tight')
    
    # Create heatmap
    plt.figure(figsize=(10, 1))
    sns.heatmap([values], cmap='YlOrRd', 
                xticklabels=labels,
                yticklabels=False,
                cbar_kws={'label': 'Threshold Value'})
    plt.title('Class-Based Thresholds Heatmap')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save heatmap
    plt.savefig(f"{save_path}_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close('all')

# Training Setup
def train_deeplab_v3_plus_cbt():
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
    model = CBTDeepLabV3Plus(
        input_shape=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH, 3), 
        num_classes=config.NUM_CLASSES,
        num_exits=2
    )
    
    
    # Compile Model
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=config.LEARNING_RATE, 
        decay=1e-5
    )
    
    model.compile(
        optimizer=optimizer,
        metrics=[mean_iou]
    )
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            'best_cbt_deeplab_model.h5', 
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

    # Compute thresholds after training
    thresholds = model.compute_thresholds()
    
     # Save model
    model.save('deeplab_v3_plus_bdd100k_cbt.h5')
    print("Model saved successfully.")
    
    # Plot training history
    plot_training_history(history)

    # Saving the history
    with open('deeplab_v3_plus_bdd100k_cbt_history.json', 'w') as f:
        json.dump(history.history, f)
        print("History saved successfully.")

    # Save and visualize thresholds
    save_and_visualize_thresholds(config, thresholds, save_path="thresholds/cbt_thresholds")
    print("Thresholds saved and visualized successfully.")
    
    # Evaluate on test set
    test_metrics = evaluate_model(model, config)
    
    return model, history, test_metrics

# Main Execution
if __name__ == '__main__':
     model, history, test_metrics = train_deeplab_v3_plus_cbt()
    