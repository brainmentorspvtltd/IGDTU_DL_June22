import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Conv2DTranspose
import numpy as np
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from skimage.io import imread, imshow
from skimage.transform import resize

IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3

TRAIN_PATH = "NuclieDataset/stage1_train"
TEST_PATH = "NuclieDataset/stage1_test"

TRAIN_IMAGES_FOLDERS = os.listdir(TRAIN_PATH)
n = len(TRAIN_IMAGES_FOLDERS)
X_train = np.zeros((n, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
Y_train = np.zeros((n, IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)

for i in tqdm(range(len(TRAIN_IMAGES_FOLDERS))):
    img_path = TRAIN_PATH + '/' + TRAIN_IMAGES_FOLDERS[i] + "/images/"
    img_name = os.listdir(img_path)[0]
    img = imread(img_path + '/' + img_name)[:,:,:IMG_CHANNELS]
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant',
                 preserve_range=True)
    X_train[i] = img
    mask_path = TRAIN_PATH + '/' + TRAIN_IMAGES_FOLDERS[i] + "/masks/"
    mask_images = os.listdir(mask_path)
    mask = np.zeros([IMG_HEIGHT, IMG_WIDTH, 1], dtype=np.bool)
    for j in range(len(mask_images)):
        mask_img = imread(mask_path + "/" + mask_images[j])
        mask_img = np.expand_dims(resize(mask_img, (IMG_HEIGHT, IMG_WIDTH), mode='constant',
                     preserve_range=True), axis=-1)
        mask = np.maximum(mask, mask_img)
    Y_train[i] = mask
    

# imshow(X_train[150])
# plt.show()

# imshow(np.squeeze(Y_train[150]))
# plt.show()


TEST_IMAGES_FOLDERS = os.listdir(TEST_PATH)
test_n = len(TEST_IMAGES_FOLDERS)
X_test = np.zeros((n, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)

for i in tqdm(range(len(TEST_IMAGES_FOLDERS))):
    img_path = TEST_PATH + '/' + TEST_IMAGES_FOLDERS[i] + "/images/"
    img_name = os.listdir(img_path)[0]
    img = imread(img_path + '/' + img_name)[:,:,:IMG_CHANNELS]
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant',
                 preserve_range=True)
    X_test[i] = img



inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
x = tf.keras.layers.Lambda(lambda x : x / 255)(inputs)

# DownSampling
c1 = Conv2D(16, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(x)
c1 = Dropout(0.1)(c1)
c1 = Conv2D(16, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(c1)
p1 = MaxPooling2D((2,2))(c1)

c2 = Conv2D(32, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(p1)
c2 = Dropout(0.1)(c2)
c2 = Conv2D(32, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(c2)
p2 = MaxPooling2D((2,2))(c2)

c3 = Conv2D(64, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(p2)
c3 = Dropout(0.2)(c3)
c3 = Conv2D(64, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(c3)
p3 = MaxPooling2D((2,2))(c3)

c4 = Conv2D(128, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(p3)
c4 = Dropout(0.2)(c4)
c4 = Conv2D(128, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(c4)
p4 = MaxPooling2D((2,2))(c4)

c5 = Conv2D(256, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(p4)
c5 = Dropout(0.3)(c5)
c5 = Conv2D(256, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(c5)


# UpSampling
u6 = Conv2DTranspose(128, (2,2), strides=(2,2), padding='same')(c5)
u6 = tf.keras.layers.concatenate([u6, c4])
c6 = Conv2D(128, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(u6)
c6 = Dropout(0.2)(c6)
c6 = Conv2D(128, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(c6)

u7 = Conv2DTranspose(64, (2,2), strides=(2,2), padding='same')(c6)
u7 = tf.keras.layers.concatenate([u7, c3])
c7 = Conv2D(64, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(u7)
c7 = Dropout(0.2)(c7)
c7 = Conv2D(64, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(c7)

u8 = Conv2DTranspose(32, (2,2), strides=(2,2), padding='same')(c7)
u8 = tf.keras.layers.concatenate([u8, c2])
c8 = Conv2D(32, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(u8)
c8 = Dropout(0.1)(c8)
c8 = Conv2D(32, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(c8)

u9 = Conv2DTranspose(16, (2,2), strides=(2,2), padding='same')(c8)
u9 = tf.keras.layers.concatenate([u9, c1])
c9 = Conv2D(16, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(u9)
c9 = Dropout(0.1)(c9)
c9 = Conv2D(16, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(c9)

outputs = Conv2D(1, (1,1), activation='sigmoid')(c9)

model = tf.keras.Model(inputs=[inputs], outputs=outputs)
model.compile(optimizer='adam', loss="binary_crossentropy",
              metrics=['accuracy'])
# model.summary()


# Including CheckPoint
checkpoint = tf.keras.callbacks.ModelCheckpoint('model_checkpoints.h5',
                                                save_best_only=True,
                                                verbose=1)

# Including EarlyStopping
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
    tf.keras.callbacks.TensorBoard(log_dir='logs')]


results = model.fit(X_train, Y_train, validation_split=0.1, 
                    batch_size=16, epochs=20, callbacks=callbacks)


test_predictions = model.predict(X_test)


imshow(X_test[10])
plt.show()

imshow(test_predictions[10])
plt.show()

test_predictions[10]







