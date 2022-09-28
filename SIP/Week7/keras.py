import numpy

from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, InputLayer
from tensorflow.keras.optimizers import Adam

## Configure the network

# batch_size to train
batch_size = 20 * 256
# number of output classes
nb_classes = 135
# number of epochs to train
nb_epoch = 400

# number of convolutional filters to use
nb_filters = 20
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 5

model = Sequential([
    InputLayer(input_shape=(29, 29, 1)),
    Conv2D(filters=nb_filters, kernel_size=nb_conv, activation='relu'),
    MaxPool2D(pool_size=(nb_pool, nb_pool)),
    Dropout(0.5),
    Conv2D(filters=nb_filters, kernel_size=nb_conv, activation='relu'),
    MaxPool2D(pool_size=(nb_pool, nb_pool)),
    Dropout(0.25),
    Flatten(),
    Dense(units=4000, activation='relu'),
    Dense(units=nb_classes, activation='softmax'),
])
    
optimizer = Adam(lr=1e-4, epsilon=1e-08)

model.compile(optimizer=optimizer,
             loss='categorical_crossentropy',
             metrics=['accuracy'])


## Train model - uncoment to perform the training yourself
#

#train = numpy.load('train.npz')
#x_train = train['x_train'].reshape((-1, 29, 29, 1))
#y_train = train['y_train']
#
#early_stopping = EarlyStopping(patience=10)
#history = model.fit(x_train, y_train, epochs=nb_epoch, batch_size=batch_size,
#                    callbacks=[early_stopping], validation_split=0.2)
#model.save_weights('keras.h5')

## Load the pretrained network
model.load_weights('keras.h5') 
