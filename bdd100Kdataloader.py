import numpy as np
import tensorflow as tf
import albumentations as A
import cv2

# Data Loader
class BDD100kDataLoader(tf.keras.utils.Sequence):
    def __init__(self, image_paths, mask_paths, config, augment=False, is_training=True):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.config = config
        self.augment = augment
        self.is_training = is_training
        self.indexes = np.arange(len(self.image_paths))
        
        # Augmentation
        self.train_aug = A.Compose([
            A.RandomCrop(config.IMAGE_HEIGHT, config.IMAGE_WIDTH),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.RandomGamma(p=0.2),
            A.GaussNoise(p=0.1),
        ])
        
        self.val_aug = A.Compose([
            A.CenterCrop(config.IMAGE_HEIGHT, config.IMAGE_WIDTH)
        ])
    
    def __len__(self):
        return int(np.ceil(len(self.image_paths) / float(self.config.BATCH_SIZE)))
    
    def __getitem__(self, index):
        batch_indexes = self.indexes[index*self.config.BATCH_SIZE:(index+1)*self.config.BATCH_SIZE]
        batch_images = []
        batch_masks = []
        
        for i in batch_indexes:
            # Load Image
            image = cv2.imread(self.image_paths[i])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # print(f"Original image shape: {image.shape}")
            
            # Load Mask
            mask = cv2.imread(self.mask_paths[i])
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
            # print(f"Original mask shape: {mask.shape}")
            
            # Augmentation
            if self.is_training and self.augment:
                augmented = self.train_aug(image=image, mask=mask)
                image = augmented['image']
                mask = augmented['mask']
            elif not self.is_training:
                augmented = self.val_aug(image=image, mask=mask)
                image = augmented['image']
                mask = augmented['mask']
            # print(f"After augmentation - image shape: {image.shape}")
            # print(f"After augmentation - mask shape: {mask.shape}")
            
            # Normalize Image
            image = image.astype(np.float32) / 255.0
            
            # Create Segmentation Mask
            seg_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
            for idx, (label, color) in enumerate(self.config.LABEL_COLORS.items()):
                class_mask = np.all(mask == color, axis=-1)
                seg_mask[class_mask] = idx
            # print(f"Segmentation mask shape (before categorical): {seg_mask.shape}")
            # print(f"Segmentation mask unique values: {np.unique(seg_mask)}")
            
            batch_images.append(image)
            batch_masks.append(seg_mask)

        final_images = np.array(batch_images)
        final_masks = np.array(batch_masks)
        # print(f"\nFinal batch shapes:")
        # print(f"Images: {final_images.shape}")
        # print(f"Masks: {final_masks.shape}")
        return final_images, final_masks
        
    
    def on_epoch_end(self):
        np.random.shuffle(self.indexes)