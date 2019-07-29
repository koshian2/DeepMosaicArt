import keras
from keras.applications import vgg16
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import keras.backend as K
import numpy as np
from tqdm import tqdm

import torch
import torchvision

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def create_network(layer_name, patch_size_x, patch_size_y):
    # block1_conv2, block2_conv2, block3_conv3, block4_conv3, block5_conv3
    if layer_name == "identity":
        input = keras.layers.Input((patch_size_y, patch_size_x, 3))
        x = keras.layers.Lambda(lambda y: y)(input)
        model = keras.models.Model(input, x)
    else:
        base_model = vgg16.VGG16(include_top=False, weights="imagenet", input_shape=(patch_size_y, patch_size_x, 3))
        model = keras.models.Model(base_model.layers[0].input,
                                base_model.get_layer(layer_name).output)
    return model

def collage():
    patch_size_x = 124
    patch_size_y = 93
    collage_n_x = 3968 // patch_size_x
    collage_n_y = 2976 // patch_size_y
    model = create_network("block4_pool", patch_size_x, patch_size_y)
    # 元画像をタイルする
    image = tf.image.decode_jpeg(tf.read_file("nayoro.jpg"))
    image = K.cast(image, "float32")
    image = K.expand_dims(image, axis=0)
    # image = tf.image.resize_bicubic(image, (2976//2, 3968//2))
    patches = tf.image.extract_image_patches(image,
                    ksizes=[1, patch_size_y, patch_size_x, 1], strides=[1, patch_size_y, patch_size_x, 1],
                    rates=[1, 1, 1, 1], padding="VALID")
    patches = K.reshape(patches, (1, collage_n_y, collage_n_x, patch_size_y, patch_size_x, 3))    
    # patch = (1, 32, 32, 93, 124, 3)
    base_patches = K.reshape(patches, (-1, patch_size_y, patch_size_x, 3))
    base_patches = vgg16.preprocess_input(base_patches)
    base_patches = K.eval(base_patches)
    # 元画像のembedding
    base_embedding = model.predict(base_patches, batch_size=128, verbose=1)
    base_embedding = base_embedding.reshape(base_embedding.shape[0], -1)

    # 参照画像のジェネレーター
    gen = keras.preprocessing.image.ImageDataGenerator(preprocessing_function=vgg16.preprocess_input)
    gen = gen.flow_from_directory("./", shuffle=False, batch_size=100,
                                  target_size=(patch_size_y, patch_size_x), interpolation="lanczos")
    ref_embedding = model.predict_generator(gen, steps=500, verbose=1)  # 500
    ref_embedding = ref_embedding.reshape(ref_embedding.shape[0], -1)

    base_indices = np.zeros(base_embedding.shape[0], np.int32)
    ref_mask = np.ones(ref_embedding.shape[0], dtype=np.float32)

    print(base_embedding.shape, ref_embedding.shape)
    for i in tqdm(range(len(base_indices))):
        loss = np.sum(np.abs(base_embedding[i].reshape(1, -1) - ref_embedding), axis=-1) * ref_mask
        base_indices[i] = np.argmin(loss)
        ref_mask[base_indices[i]] = 1e8 # 十分大きな値
    
    np.set_printoptions(threshold=np.inf)
    print(base_indices)

    # コラージュの作成
    images = []
    for i in base_indices:
        x = tf.image.decode_jpeg(tf.read_file(gen.filenames[i]))
        x = tf.expand_dims(x, axis=0)
        x = tf.image.resize_bicubic(x, (patch_size_y, patch_size_x))
        x = K.cast(x, "float32") / 255.0
        x = np.broadcast_to(K.eval(x), (1, patch_size_y, patch_size_x, 3))
        images.append(x)
    images = np.concatenate(images, axis=0)
    images = np.transpose(images, [0, 3, 1, 2])

    tensor = torch.as_tensor(images)
    torchvision.utils.save_image(tensor, "collage.png", nrow=collage_n_x, padding=0)



if __name__ == "__main__":
    collage()
