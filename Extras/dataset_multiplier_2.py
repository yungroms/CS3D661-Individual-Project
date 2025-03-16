import os
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def augment_images(input_dir, output_dir, multiplier=10):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    classes = os.listdir(input_dir)
    
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        brightness_range=[0.9, 1.1],
        fill_mode='nearest'
    )
    
    for class_name in tqdm(classes, desc="Augmenting Classes"):
        class_input_path = os.path.join(input_dir, class_name)
        class_output_path = os.path.join(output_dir, class_name)
        
        if not os.path.exists(class_output_path):
            os.makedirs(class_output_path)
        
        images = [f for f in os.listdir(class_input_path) if f.endswith(('png', 'jpg', 'jpeg'))]
        num_images = len(images)
        
        for image_name in images:
            image_path = os.path.join(class_input_path, image_name)
            image = keras.preprocessing.image.load_img(image_path)
            image = keras.preprocessing.image.img_to_array(image)
            image = np.expand_dims(image, axis=0)
            
            base_name, ext = os.path.splitext(image_name)
            
            i = 0
            for batch in datagen.flow(image, batch_size=1, save_to_dir=class_output_path, 
                                      save_prefix=f"{base_name}_aug", save_format=ext[1:]):
                i += 1
                if i >= (multiplier - 1):  # Generate (multiplier-1) new images per original image
                    break
        
        # Copy original images to output directory
        for image_name in images:
            src_path = os.path.join(class_input_path, image_name)
            dst_path = os.path.join(class_output_path, image_name)
            tf.keras.utils.save_img(dst_path, tf.keras.preprocessing.image.load_img(src_path))

if __name__ == "__main__":
    input_directory = r"C:\Users\rms11\Desktop\Proj\Datasets\17flowers"  
    output_directory = r"C:\Users\rms11\Desktop\Proj\Datasets\17flowers_10x_less_aug"
    augment_images(input_directory, output_directory)
