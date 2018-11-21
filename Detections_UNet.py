import os
import numpy as np
from tqdm import tqdm
import cv2
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Input, BatchNormalization, Dropout
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
import tensorflow as tf
from keras import backend as K
import keras
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import psutil
import humanize
import os
import GPUtil as GPU
from tqdm import tqdm
import gc

def get_X_data(img_paths, output_shape=(None, None)):
    
    num_shape = cv2.resize(cv2.imread(img_paths[0]), output_shape)
    rows = len(img_paths)
    cols = len(num_shape.flatten())
    
    if os.path.isfile('x_data.npy'):
        return np.memmap('x_data.npy', dtype='int8',mode='r', shape=(rows, cols)).reshape(len(img_paths) , -1)
    else:
        X_data = np.memmap('x_data.npy', dtype='int8',mode='w+', shape=(rows, cols))

        for i, path in enumerate(tqdm(img_paths)):
            X_data[i] = cv2.resize(cv2.imread(path), output_shape).flatten()
          
        return X_data.reshape(len(img_paths) ,-1)


def get_Y_data(img_paths, output_shape=(None, None)):
    
    num_shape = cv2.resize(cv2.imread(img_paths[0], cv2.IMREAD_GRAYSCALE), output_shape)
    rows = len(img_paths)
    cols = len(num_shape.flatten())

    if os.path.isfile('y_data.npy'):
        return np.memmap('y_data.npy', dtype='bool',  mode='r', shape=(rows, cols)).reshape(len(img_paths), -1)
    else:
        Y_data = np.memmap('y_data.npy', dtype='bool', mode='w+', shape=(rows, cols))

        for i, path in enumerate(tqdm(img_paths)):
            thresh = 127
            img= cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), output_shape)
           
            im_bw = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)[1]
            Y_data[i] = im_bw.flatten()

        return Y_data.reshape(len(img_paths), -1)
     
    
def split_train_test(X_train, Y_train, test_size=test_size, seed=seed):
    X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size = test_size, random_state=seed)
    return X_train, X_test, Y_train, Y_test
  

def train_test_augmented(X_data, Y_data, batch_size=batch_size):
    
    image_gen_args = dict(samplewise_center=False, 
                     samplewise_std_normalization=False, 
                     rotation_range=45.,
                     width_shift_range=0.2,
                     height_shift_range=0.2,
                     shear_range=0.2,
                     zoom_range=0.2,
                     horizontal_flip=True,
                     vertical_flip=True,
                     fill_mode='reflect')
    
    X_datagen = ImageDataGenerator(**image_gen_args, rescale= 1./ X_train[0].max())
    Y_datagen = ImageDataGenerator(**image_gen_args)
    X_datagen.fit(X_data, augment=True, seed=seed)
    Y_datagen.fit(Y_data, augment=True, seed=seed)
    X_train_augmented = X_datagen.flow(X_data, batch_size=batch_size, shuffle=True, seed=seed)
    Y_train_augmented = Y_datagen.flow(Y_data, batch_size=batch_size, shuffle=True, seed=seed)
    
    for x, y in zip(X_train_augmented, Y_train_augmented):
        yield x, y
        
        
# https://www.kaggle.com/kmader/data-preprocessing-and-unet-segmentation-gpu
def mean_iou(y_true, y_pred):
    
    prec = []
    
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)

# Custom loss function
def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 0.5 * keras.losses.binary_crossentropy(y_true, y_pred) - dice_coef(y_true, y_pred)
  
def printm():
    process = psutil.Process(os.getpid())
    print("\nGen RAM Free: " + humanize.naturalsize( psutil.virtual_memory().available ), " | Proc size: " + humanize.naturalsize( process.memory_info().rss))
    print("GPU RAM Free: {0:.0f}MB | Used: {1:.0f}MB | Util {2:3.0f}% | Total {3:.0f}MB".format(gpu.memoryFree, gpu.memoryUsed, gpu.memoryUtil*100, gpu.memoryTotal))

seed = 101
batch_size = 32
num_epoch = 50
num_chunks = 10
test_size = 0.2

GPUs = GPU.getGPUs()
gpu = GPUs[0]

ids_cracks = os.listdir('Data')
outputs_filename = []
inputs_filename = []

for i in ids_cracks:
    outputs_filename.append(os.path.join('Data_mask', i[:-4] + '_mask' + i[-4:]))
    inputs_filename.append(os.path.join('Data', i))

inputs = Input((256, 256, 3))
x = BatchNormalization()(inputs)
x = Dropout(0.5)(x)

c1 = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
c1 = Conv2D(8, (3, 3), activation='relu', padding='same')(c1)
p1 = MaxPooling2D((2, 2)) (c1)

c2 = Conv2D(16, (3, 3), activation='relu', padding='same')(p1)
c2 = Conv2D(16, (3, 3), activation='relu', padding='same')(c2)
p2 = MaxPooling2D((2, 2)) (c2)

c3 = Conv2D(32, (3, 3), activation='relu', padding='same')(p2)
c3 = Conv2D(32, (3, 3), activation='relu', padding='same')(c3)
p3 = MaxPooling2D((2, 2)) (c3)

c4 = Conv2D(64, (3, 3), activation='relu', padding='same')(p3)
c4 = Conv2D(64, (3, 3), activation='relu', padding='same')(c4)
p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

c5 = Conv2D(128, (3, 3), activation='relu', padding='same')(p4)
c5 = Conv2D(128, (3, 3), activation='relu', padding='same')(c5)

u6 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c5)
u6 = concatenate([u6, c4])
c6 = Conv2D(64, (3, 3), activation='relu', padding='same')(u6)
c6 = Conv2D(64, (3, 3), activation='relu', padding='same')(c6)

u7 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c6)
u7 = concatenate([u7, c3])
c7 = Conv2D(32, (3, 3), activation='relu', padding='same')(u7)
c7 = Conv2D(32, (3, 3), activation='relu', padding='same')(c7)

u8 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c7)
u8 = concatenate([u8, c2])
c8 = Conv2D(16, (3, 3), activation='relu', padding='same')(u8)
c8 = Conv2D(16, (3, 3), activation='relu', padding='same')(c8)

u9 = Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same')(c8)
u9 = concatenate([u9, c1], axis=3)
c9 = Conv2D(8, (3, 3), activation='relu', padding='same')(u9)
c9 = Conv2D(8, (3, 3), activation='relu', padding='same')(c9)

outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)

model = Model(inputs=[inputs], outputs=[outputs])
model.compile(optimizer = 'adam', loss = dice_coef_loss, metrics = [dice_coef, 'binary_accuracy', 'mse'])

checkpoint = ModelCheckpoint('unet.h5', monitor='val_loss', verbose=1,save_best_only=True, mode='min')
reduceLROnPlat = ReduceLROnPlateau(monitor='loss', factor=0.8, patience=10, verbose=1, mode='auto', epsilon=0.0001, cooldown=5, min_lr=0.0001)
early = EarlyStopping(monitor="val_loss", mode="min", patience=10)
callbacks_list = [checkpoint, reduceLROnPlat, early]

X_train_dat = get_X_data(inputs_filename, output_shape=(256, 256))
Y_train_dat = get_Y_data(outputs_filename, output_shape=(256, 256))
data_slice = len(X_train_dat) // num_chunks 

for i in tqdm(range(num_epoch)):
    print(f'\nEpoch: {i+1}/{num_epoch}')
    for n in tqdm(range(num_chunks)):
        print(f'\tChunks: {n+1}/{num_chunks}')
        
        X_train_slice = X_train_dat[n * data_slice : (n+1) * data_slice]
        Y_train_slice = Y_train_dat[n * data_slice : (n+1) * data_slice]

        X_train, X_test, Y_train, Y_test = split_train_test(X_train_slice, Y_train_slice)
        train_gen = train_test_augmented(X_train.reshape(-1, 256, 256, 3), Y_train.reshape(-1, 256, 256, 1), batch_size=batch_size)
        valid_gen = train_test_augmented(X_test.reshape(-1, 256, 256, 3), Y_test.reshape(-1, 256, 256, 1), batch_size = batch_size)
        
        model.fit_generator(train_gen, validation_data=valid_gen, validation_steps=256//batch_size, steps_per_epoch=2048//batch_size, 
                            epochs=1, verbose=1, callbacks = callbacks_list)
        
        del X_train, X_test, Y_train, Y_test, X_train_slice,Y_train_slice
        gc.collect()
          
    printm()

