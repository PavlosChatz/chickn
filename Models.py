import tensorflow as tf
from tensorflow.keras import applications
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Input, Flatten, Dense, Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint


def model_helper(X_train, Y_train, X_val, Y_val):
    model = vgg16_model()
    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

    epochs = 20
    batch_size = 48 

    train_datagen = ImageDataGenerator(
        rotation_range = 30,
        width_shift_range = 0.1, 
        height_shift_range = 0.1,
        horizontal_flip = True)
    train_datagen.fit(X_train)

    history = model.fit(
        train_datagen.flow(X_train, Y_train, batch_size = batch_size),
        steps_per_epoch = X_train.shape[0] // batch_size,
        epochs = epochs,
        validation_data = (X_val, Y_val),
        callbacks = [ModelCheckpoint('VGG16-model', monitor = 'val_acc')]
    )
    return 

def vgg16_model():

    
    dims = (64, 64, 3)
    conv_model = applications.VGG16(include_top = False, weights = 'imagenet', input_tensor = None, input_shape = dims, pooling = 'max')
    conv_model.trainable = False
    inputs = Input(shape = dims)
    vgg16_conv_outputs = conv_model(inputs)
    x = Flatten(name = 'flatten')(vgg16_conv_outputs)
    x = Dense(1000, activation = 'relu', name = 'fc1') (x)
    x = Dense(1000, activation = 'relu', name = 'fc2') (x)
    x = Dense(10, activation = 'softmax', name = 'predictions') (x)

    model = Model(inputs = inputs, outputs = x)
    model.summary()
    return model

def cnn_model():
    model = None
    return model