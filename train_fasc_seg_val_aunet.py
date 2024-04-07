import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image 
plt.style.use("ggplot")

from tqdm import tqdm
from skimage.transform import resize


from keras import backend as K
from keras.models import Model, load_model
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout, Multiply
from keras.layers.core import Lambda, RepeatVector, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D, GlobalMaxPool2D
from keras.layers.merge import concatenate, add
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


# Convolution block
def conv2d_block(input_tensor, n_filters, kernel_size = 3, batchnorm = True):
    """Function to add 2 convolutional layers with the parameters passed to it"""
    # first layer
    x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size),\
              kernel_initializer = 'he_normal', padding = 'same')(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # second layer
    x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size),\
              kernel_initializer = 'he_normal', padding = 'same')(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    return x


def attention_block(input_tensor, skip_tensor, n_filters):
    """Attention block which computes attention coefficients"""
    g = Conv2D(n_filters, kernel_size=1, strides=1, padding='same')(skip_tensor)
    g = BatchNormalization()(g)

    x = Conv2D(n_filters, kernel_size=1, strides=1, padding='same')(input_tensor)
    x = BatchNormalization()(x)

    f = Activation('relu')(add([g, x]))
    psi = Conv2D(1, kernel_size=1, strides=1, padding='same')(f)
    psi = BatchNormalization()(psi)
    psi = Activation('sigmoid')(psi)

    return Multiply()([input_tensor, psi])


def get_attention_unet(input_img, n_filters=64, dropout=0.1, batchnorm=True):
    """Function to define the Attention U-Net Model"""

    # Contracting Path (Encoder)
    c1 = conv2d_block(input_img, n_filters * 1, kernel_size=3, batchnorm=batchnorm)
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(dropout)(p1)

    c2 = conv2d_block(p1, n_filters * 2, kernel_size=3, batchnorm=batchnorm)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)

    c3 = conv2d_block(p2, n_filters * 4, kernel_size=3, batchnorm=batchnorm)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)

    c4 = conv2d_block(p3, n_filters * 8, kernel_size=3, batchnorm=batchnorm)
    p4 = MaxPooling2D((2, 2))(c4)
    p4 = Dropout(dropout)(p4)

    # Bottleneck
    c5 = conv2d_block(p4, n_filters=n_filters * 16, kernel_size=3, batchnorm=batchnorm)

    # Expansive Path (Decoder)
    u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, attention_block(c4, u6, n_filters * 8)])
    u6 = Dropout(dropout)(u6)
    c6 = conv2d_block(u6, n_filters * 8, kernel_size=3, batchnorm=batchnorm)

    u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, attention_block(c3, u7, n_filters * 4)])
    u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters * 4, kernel_size=3, batchnorm=batchnorm)

    u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, attention_block(c2, u8, n_filters * 2)])
    u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters * 2, kernel_size=3, batchnorm=batchnorm)

    u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, attention_block(c1, u9, n_filters * 1)])
    u9 = Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters * 1, kernel_size=3, batchnorm=batchnorm)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    model = Model(inputs=[input_img], outputs=[outputs])
    return model


# Create u-net model
def get_unet(input_img, n_filters = 64, dropout = 0.1, batchnorm = True):
    """Function to define the UNET Model"""
    
    # Contracting Path (Encoder)
    c1 = conv2d_block(input_img, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(dropout)(p1)
    
    c2 = conv2d_block(p1, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)
    
    c3 = conv2d_block(p2, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)
    
    c4 = conv2d_block(p3, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)
    p4 = MaxPooling2D((2, 2))(c4)
    p4 = Dropout(dropout)(p4)
    
    # Bottleneck
    c5 = conv2d_block(p4, n_filters = n_filters * 16, kernel_size = 3, batchnorm = batchnorm)
    
    # Expansive Path (Decoder)
    u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides = (2, 2), padding = 'same')(c5)
    u6 = concatenate([u6, c4])
    u6 = Dropout(dropout)(u6)
    c6 = conv2d_block(u6, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)
    
    u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides = (2, 2), padding = 'same')(c6)
    u7 = concatenate([u7, c3])
    u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)
    
    u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides = (2, 2), padding = 'same')(c7)
    u8 = concatenate([u8, c2])
    u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
    
    u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides = (2, 2), padding = 'same')(c8)
    u9 = concatenate([u9, c1])
    u9 = Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
    
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    model = Model(inputs=[input_img], outputs=[outputs])
    return model

# Compute Intersection over union (IoU), a measure of labelling accuracy
# NOTE: This is sometimes also called Jaccard score
def IoU(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    union = K.sum(y_true,-1) + K.sum(y_pred,-1) - intersection
    iou = (intersection + smooth) / ( union + smooth)
    return iou

# ================================Start of Fascicle Training================================

# Load training data
# list of names of all images in the given path
im_width = 512
im_height = 512
idsF = next(os.walk("fasc_images_S"))[2] 
print("Total no. of fascicle images = ", len(idsF))
X_trainF = np.zeros((len(idsF), im_height, im_width, 1), dtype=np.float32)
y_trainF = np.zeros((len(idsF), im_height, im_width, 1), dtype=np.float32)

# Load training images and corresponding fascicle masks
# tqdm is used to display the progress bar
for n, id_ in tqdm(enumerate(idsF), total=len(idsF)):
    # Load images
    imgF = load_img("fasc_images_S/"+id_, color_mode = 'grayscale')
    x_imgF = img_to_array(imgF)
    x_imgF = resize(x_imgF, (512, 512, 1), mode = 'constant', preserve_range = True)
    # Load masks
    maskF = img_to_array(load_img("fasc_masks_S/"+id_, color_mode = 'grayscale'))
    maskF = resize(maskF, (512, 512, 1), mode = 'constant', preserve_range = True)
    # Normalise and store images
    X_trainF[n] = x_imgF/255.0
    y_trainF[n] = maskF/255.0


# Load validation data
idsF_valid = next(os.walk("fasc_images_WW"))[2] 
print("Total no. of fascicle images for validation = ", len(idsF_valid))
X_validF = np.zeros((len(idsF_valid), im_height, im_width, 1), dtype=np.float32)
y_validF = np.zeros((len(idsF_valid), im_height, im_width, 1), dtype=np.float32)

# Load validation images and masks
for n, id_ in tqdm(enumerate(idsF_valid), total=len(idsF_valid)):
    # Load images
    imgF = load_img("fasc_images_WW/"+id_, color_mode='grayscale')
    imgF = imgF.transpose(Image.FLIP_LEFT_RIGHT)
    x_imgF = img_to_array(imgF)
    x_imgF = resize(x_imgF, (512, 512, 1), mode='constant', preserve_range=True)
    # Load masks
    mskF = load_img("fasc_masks_WW/"+id_, color_mode='grayscale')
    mskF = mskF.transpose(Image.FLIP_LEFT_RIGHT)
    maskF = img_to_array(mskF)
    maskF = resize(maskF, (512, 512, 1), mode='constant', preserve_range=True)
    # Normalize and store images
    X_validF[n] = x_imgF / 255.0
    y_validF[n] = maskF / 255.0


# Set up fascicle training
# Split data into training and validation
# X_trainF, X_validF, y_trainF, y_validF = train_test_split(XF, yF, test_size=0.1)

# Compile the model
input_imgF = Input((im_height, im_width, 1), name='img')
# modelF = get_unet(input_imgF, n_filters=32, dropout=0.25, batchnorm=True)
modelF = get_attention_unet(input_imgF, n_filters=32, dropout=0.25, batchnorm=True)
modelF.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy", IoU])

# Set some training parameters (e.g. the name you want to give to your trained model)
callbacksF = [
    EarlyStopping(patience=7, verbose=1),
    ReduceLROnPlateau(factor=0.1, patience=7, min_lr=0.00001, verbose=1),
    ModelCheckpoint('model-fasc-WW-aunet111.h5', verbose=1, save_best_only=True, save_weights_only=False), # Name your model (the .h5 part)
    CSVLogger('fasc2_training_losses_aunet111.csv', separator=',', append=False)
]

# Train the fascicle model
resultsF = modelF.fit(X_trainF, y_trainF, batch_size=2, epochs=20, callbacks=callbacksF,\
                    validation_data=(X_validF, y_validF))

