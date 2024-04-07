import os, argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image 
plt.style.use("ggplot")

from tqdm import tqdm
from skimage.transform import resize
from sklearn.model_selection import train_test_split

from keras import backend as K
from keras.models import Model
from keras.layers import Input, BatchNormalization, Activation, Dropout
from keras.layers import Conv2D, BatchNormalization, Activation, MaxPooling2D, Dropout, Conv2DTranspose, concatenate, Multiply

from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate, add
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array, load_img


parser = argparse.ArgumentParser(description = "Train_Seg_Model")
parser.add_argument('--image_dir',  type=str, required=True, default='', 
                    help='The directory of original images in the training dataset')
parser.add_argument('--mask_dir', type=str, required=True, default='',
                    help='The directory of binary masks in the training dataset')
parser.add_argument("--seg_type", type=str, required=True, default='',
                    help="Specify the segmentation task: 'aponeurosis' or 'fascicle'.")
parser.add_argument("--model_type", type=str, default='u-net',
                    help="Specify the architecture of the neural network: 'u-net' or 'attention-u-net'.")
parser.add_argument("--model_name", type=str, default='model-seg',
                    help="The model name for saving after training.")


opt = parser.parse_args()
image_dir = opt.image_dir
mask_dir = opt.mask_dir
seg_type = opt.seg_type
model_type = opt.model_type
model_name = opt.model_name

'''
python train_fasc_seg.py --image_dir apo_images --mask_dir apo_masks --seg_type aponeurosis --model_type u-net --model_name test_train_fasc
'''

# Handle user input error
# Check if the image directory exists
if not os.path.exists(image_dir):
    raise argparse.ArgumentError(None, f"The specified image directory '{image_dir}' does not exist.")

# Check if the mask directory exists
if not os.path.exists(mask_dir):
    raise argparse.ArgumentError(None, f"The specified mask directory '{mask_dir}' does not exist.")

# Check if the model type is valid
if model_type not in ['u-net', 'attention-u-net']:
    raise argparse.ArgumentError(None, f"Invalid model type '{model_type}'. The model type must be either 'u-net' or 'attention-u-net'.")


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


if __name__=="__main__":

    # Aponeurosis segmentation model training
    if seg_type == 'aponeurosis':
        # ================================Start of Aponeurosis Training================================
        # Load training data
        # list of all images in the path
        im_width = 512
        im_height = 512
        border = 5
        # ids = next(os.walk("apo_images"))[2] 
        ids = next(os.walk(image_dir))[2] 
        print("Total no. of aponeurosis images = ", len(ids))
        X = np.zeros((len(ids), im_height, im_width, 1), dtype=np.float32)
        y = np.zeros((len(ids), im_height, im_width, 1), dtype=np.float32)

        # Load training images and corresponding aponeurosis masks
        # tqdm is used to display the progress bar
        for n, id_ in tqdm(enumerate(ids), total=len(ids)):
            # Load images
            # img = load_img("apo_images/"+id_, color_mode='grayscale')
            img = load_img(os.path.join(image_dir, id_), color_mode='grayscale')
            x_img = img_to_array(img)
            x_img = resize(x_img, (512, 512, 1), mode = 'constant', preserve_range = True)
            # Load masks
            # mask = img_to_array(load_img("apo_masks/"+id_, color_mode='grayscale'))
            mask = img_to_array(load_img(os.path.join(mask_dir, id_), color_mode='grayscale'))
            mask = resize(mask, (512, 512, 1), mode = 'constant', preserve_range = True)
            # Normalise and store images
            X[n] = x_img/255.0
            y[n] = mask/255.0

        # Split data into training and validation
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1) # i.e. 90% training / 10% test split

        # Compile the aponeurosis model
        input_img = Input((im_height, im_width, 1), name='img')
        # Load the selected neural network architecture based on the specified model type.
        if model_type == 'u-net':
            model_apo = get_unet(input_img, n_filters=64, dropout=0.25, batchnorm=True)
            print("==========Using U-net for aponeurosis segmentation model training")
        elif model_type == 'attention-u-net':
            model_apo = get_attention_unet(input_img, n_filters=64, dropout=0.25, batchnorm=True)
            print("==========Using Attention U-net for aponeurosis segmentation model training")
        model_apo.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy", IoU])

        # Set some training parameters
        callbacks = [
            EarlyStopping(patience=8, verbose=1),
            ReduceLROnPlateau(factor=0.1, patience=10, min_lr=0.00001, verbose=1),
            ModelCheckpoint(model_name+'.h5', verbose=1, save_best_only=True, save_weights_only=False), 
            CSVLogger(model_name+'.csv', separator=',', append=False)
        ]

        # Train the aponeurosis model
        results = model_apo.fit(X_train, y_train, batch_size=2, epochs=60, callbacks=callbacks, validation_data=(X_valid, y_valid))

    # Fascicle segmentation model training
    elif seg_type == 'fascicle':

        # ================================Start of Fascicle Training================================

        # Load training data
        # list of names of all images in the given path
        im_width = 512
        im_height = 512
        idsF = next(os.walk(image_dir))[2] 
        print("Total no. of fascicle images = ", len(idsF))
        XF = np.zeros((len(idsF), im_height, im_width, 1), dtype=np.float32)
        yF = np.zeros((len(idsF), im_height, im_width, 1), dtype=np.float32)

        # Load training images and corresponding fascicle masks
        # tqdm is used to display the progress bar
        for n, id_ in tqdm(enumerate(idsF), total=len(idsF)):
            # Load images
            # imgF = load_img("fasc_images_S/"+id_, color_mode = 'grayscale')
            imgF = load_img(os.path.join(image_dir, id_), color_mode='grayscale')
            x_imgF = img_to_array(imgF)
            x_imgF = resize(x_imgF, (512, 512, 1), mode = 'constant', preserve_range = True)
            # Load masks
            # maskF = img_to_array(load_img("fasc_masks_S/"+id_, color_mode = 'grayscale'))
            maskF = img_to_array(load_img(os.path.join(mask_dir, id_), color_mode='grayscale'))
            maskF = resize(maskF, (512, 512, 1), mode = 'constant', preserve_range = True)
            # Normalise and store images
            XF[n] = x_imgF/255.0
            yF[n] = maskF/255.0

        # Split data into training and validation
        X_train, X_valid, y_train, y_valid = train_test_split(XF, yF, test_size=0.1) # i.e. 90% training / 10% test split


        # Compile the model
        input_imgF = Input((im_height, im_width, 1), name='img')
        # Load the selected neural network architecture based on the specified model type.
        if model_type == 'u-net':
            modelF = get_unet(input_imgF, n_filters=32, dropout=0.25, batchnorm=True)
            print("==========Using U-net for fascicle segmentation model training")
        elif model_type == 'attention-u-net':
            modelF = get_attention_unet(input_imgF, n_filters=32, dropout=0.25, batchnorm=True)
            print("==========Using Attention U-net for fascicle segmentation model training")
        modelF.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy", IoU])

        # Set some training parameters (e.g. the name you want to give to your trained model)
        callbacksF = [
            EarlyStopping(patience=7, verbose=1),
            ReduceLROnPlateau(factor=0.1, patience=7, min_lr=0.00001, verbose=1),
            ModelCheckpoint(model_name+'.h5', verbose=1, save_best_only=True, save_weights_only=False), # Name your model (the .h5 part)
            CSVLogger(model_name+'.csv', separator=',', append=False)
        ]

        # Train the fascicle model
        resultsF = modelF.fit(X_train, y_train, batch_size=2, epochs=10, callbacks=callbacksF,\
                            validation_data=(X_valid, y_valid))

