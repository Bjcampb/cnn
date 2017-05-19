import scipy.io as sio
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPooling1D, GlobalAveragePooling1D
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras import regularizers
from keras.utils import np_utils
from keras import optimizers

###############################################################################
# Define
###############################################################################
def LoadStackPeak(fileloc):
    '''Loads Matlab file and stacks data in proper format'''
    
    file = sio.loadmat(str(fileloc))
    PeakData = np.vstack((file['data']['water'][0,0],
                          file['data']['fat'][0,0],
                          file['data']['peak_1'][0,0],
                          file['data']['peak_4'][0,0],
                          file['data']['peak_6'][0,0],
                          file['data']['peak_8'][0,0],
                          file['data']['peak_others'][0,0],
                          file['data']['r2'][0,0]))
    return PeakData



###############################################################################
# Load Surface Coil Mice Data
###############################################################################

# Mouse 1
SM1IGprewat = LoadStackPeak('/home/brandon/Documents/FrontiersConference/FullPeakData/Surface Coil 082016/Mouse 1/Igwat/'
                           'Mouse_1_Pre_Igwat_WAT.mat')
SM1IGpremus = LoadStackPeak('/home/brandon/Documents/FrontiersConference/FullPeakData/Surface Coil 082016/Mouse 1/Igwat/'
                           'Mouse_1_Pre_Igwat_Muscle.mat')
SM1ICprewat = LoadStackPeak('/home/brandon/Documents/FrontiersConference/FullPeakData/Surface Coil 082016/Mouse 1/Intercap/'
                           'Mouse_1_Pre_Intercap_WAT.mat')

SM1ICprebat = LoadStackPeak('/home/brandon/Documents/FrontiersConference/FullPeakData/Surface Coil 082016/Mouse 1/Intercap/'
                           'Mouse_1_Pre_Intercap_BAT.mat')
SM1ICpremus = LoadStackPeak('/home/brandon/Documents/FrontiersConference/FullPeakData/Surface Coil 082016/Mouse 1/Intercap/'
                           'Mouse_1_Pre_Intercap_Muscle.mat')

# Mouse 2
SM2IGprewat = LoadStackPeak('/home/brandon/Documents/FrontiersConference/FullPeakData/Surface Coil 082016/Mouse 2/Igwat/'
                           'Mouse_2_Pre_Igwat_WAT.mat')

SM2IGpremus = LoadStackPeak('/home/brandon/Documents/FrontiersConference/FullPeakData/Surface Coil 082016/Mouse 2/Igwat/'
                           'Mouse_2_Pre_Igwat_Muscle.mat')
SM2ICprewat = LoadStackPeak('/home/brandon/Documents/FrontiersConference/FullPeakData/Surface Coil 082016/Mouse 2/Intercap/'
                           'Mouse_2_Pre_Intercap_WAT.mat')
SM2ICprebat = LoadStackPeak('/home/brandon/Documents/FrontiersConference/FullPeakData/Surface Coil 082016/Mouse 2/Intercap/'
                           'Mouse_2_Pre_Intercap_BAT.mat')
SM2ICpremus = LoadStackPeak('/home/brandon/Documents/FrontiersConference/FullPeakData/Surface Coil 082016/Mouse 2/Intercap/'
                           'Mouse_2_Pre_Intercap_Muscle.mat')

###############################################################################
# Load Volume Coil Mice Data
###############################################################################

#Mouse 1
VM1IGprewat = LoadStackPeak('/home/brandon/Documents/FrontiersConference/FullPeakData/Volume Coil 022016/Mouse 1/Igwat/'
                           'Mouse_1_Pre_Igwat_WAT.mat')
VM1IGpremus = LoadStackPeak('/home/brandon/Documents/FrontiersConference/FullPeakData/Volume Coil 022016/Mouse 1/Igwat/'
                           'Mouse_1_Pre_Igwat_Muscle.mat')

VM1ICprewat = LoadStackPeak('/home/brandon/Documents/FrontiersConference/FullPeakData/Volume Coil 022016/Mouse 1/Intercap/'
                           'Mouse_1_Pre_Intercap_WAT.mat')
VM1ICprebat = LoadStackPeak('/home/brandon/Documents/FrontiersConference/FullPeakData/Volume Coil 022016/Mouse 1/Intercap/'
                           'Mouse_1_Pre_Intercap_BAT.mat')
VM1ICpremus = LoadStackPeak('/home/brandon/Documents/FrontiersConference/FullPeakData/Volume Coil 022016/Mouse 1/Intercap/'
                           'Mouse_1_Pre_Intercap_Muscle.mat')

# Mouse 3
VM3IGprewat = LoadStackPeak('/home/brandon/Documents/FrontiersConference/FullPeakData/Volume Coil 022016/Mouse 3/Igwat/'
                           'Mouse_3_Pre_Igwat_WAT.mat')
VM3IGpremus = LoadStackPeak('/home/brandon/Documents/FrontiersConference/FullPeakData/Volume Coil 022016/Mouse 3/Igwat/'
                           'Mouse_3_Pre_Igwat_Muscle.mat')

# Mouse 4
VM4ICprewat = LoadStackPeak('/home/brandon/Documents/FrontiersConference/FullPeakData/Volume Coil 022016/Mouse 4/Intercap/'
                           'Mouse_4_Pre_Intercap_WAT.mat')
VM4ICprebat = LoadStackPeak('/home/brandon/Documents/FrontiersConference/FullPeakData/Volume Coil 022016/Mouse 4/Intercap/'
                           'Mouse_4_Pre_Intercap_BAT.mat')
VM4ICpremus = LoadStackPeak('/home/brandon/Documents/FrontiersConference/FullPeakData/Volume Coil 022016/Mouse 4/Intercap/'
                           'Mouse_4_Pre_Intercap_Muscle.mat')

# Mouse 5
VM5ICprewat = LoadStackPeak('/home/brandon/Documents/FrontiersConference/FullPeakData/Volume Coil 022016/Mouse 5/Intercap/'
                           'Mouse_5_Pre_Intercap_WAT.mat')
VM5ICprebat = LoadStackPeak('/home/brandon/Documents/FrontiersConference/FullPeakData/Volume Coil 022016/Mouse 5/Intercap/'
                           'Mouse_5_Pre_Intercap_BAT.mat')
VM5ICpremus = LoadStackPeak('/home/brandon/Documents/FrontiersConference/FullPeakData/Volume Coil 022016/Mouse 5/Intercap/'
                           'Mouse_5_Pre_Intercap_Muscle.mat')

###############################################################################
# Standardize and Organize
###############################################################################

min_max_scaler = preprocessing.MinMaxScaler()

WAT_training = np.transpose(np.hstack((SM1ICprewat,SM2ICprewat, SM2IGprewat,
                                       VM1ICprewat, VM5ICprewat)))
WAT_test = np.transpose(np.hstack((SM1IGprewat, VM1IGprewat, VM3IGprewat,
                                   VM4ICprewat)))

BAT_training = np.transpose(np.hstack((SM1ICprebat, SM2ICprebat, VM1ICprebat)))
BAT_training = BAT_training[0:np.shape(WAT_training)[0],:] # Reduced size of BAT to be equal to WAT
BAT_test = np.transpose(np.hstack((VM4ICprebat, VM5ICprebat)))


MUS_training = np.transpose(np.hstack((SM1IGpremus, SM2ICpremus, SM2IGpremus, 
                                       VM1IGpremus, VM5ICpremus)))
MUS_training = MUS_training[0:np.shape(WAT_training)[0],:]
MUS_test = np.transpose(np.hstack((SM1ICpremus, VM1ICpremus, VM4ICpremus)))

# Create classes
WAT_train_class = np.zeros((WAT_training.shape[0],1), dtype=int)
WAT_test_class = np.zeros((WAT_test.shape[0],1), dtype=int)

BAT_train_class = np.zeros((BAT_training.shape[0],1), dtype=int)+1
BAT_test_class = np.zeros((BAT_test.shape[0],1), dtype=int)+1

MUS_train_class = np.zeros((MUS_training.shape[0],1), dtype=int)+2
MUS_test_class = np.zeros((MUS_test.shape[0],1), dtype=int)+2

#############################################################################
## Create class
##############################################################################
X_train = np.concatenate((WAT_training, BAT_training, MUS_training), axis=0)
y_train = np.concatenate((WAT_train_class, BAT_train_class, MUS_train_class))
y_train = np.ravel(y_train)

X_test = np.concatenate((WAT_test, BAT_test, MUS_test), axis=0)
y_test = np.concatenate((WAT_test_class, BAT_test_class, MUS_test_class))
y_test = np.ravel(y_test)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# Normalize training and testing

scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)


X_test = scaler.transform(X_test)


# Switch order of matrix dimensions to fix keras (expects (samples, steps, input_dim))
#(real/imag, samples, echos) - (samples, echos, real/imag)
X_train = np.expand_dims(X_train, axis=0)
X_test = np.expand_dims(X_test, axis=0)
X_train = np.einsum('ijk -> jki', X_train)
X_test = np.einsum('ijk -> jki', X_test)

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

###############################################################################
##         Create the model
###############################################################################

# Signal Parameters
echos = 8    # Total amount of time steps
depth = 1  # Real and Imag have been combined
weight_decay = 0.0001#0.00001

num_filter = [32, 16]
length_filter = [4, 4]
pool_length = [2,2]

# CNN NETWORK
model = Sequential() # Initializes model to sequientially add layers

model.add(Conv1D(filters = num_filter[0],
                        kernel_size = length_filter[0],
                        padding = 'same',
                        activation = 'relu',
                        kernel_regularizer=regularizers.l2(weight_decay),
                        input_shape = (echos, depth)))
model.add(BatchNormalization(axis=-1, momentum=0.99))

model.add(MaxPooling1D(pool_size=pool_length[0]))

model.add(Dropout(0.25))


model.add(Conv1D(filters = num_filter[1],
                        kernel_size = length_filter[1],
                        padding = 'same',
                        activation = 'relu',
                        kernel_regularizer=regularizers.l2(weight_decay)))

model.add(BatchNormalization(axis=-1, momentum=0.99))

model.add(MaxPooling1D(pool_size=pool_length[1]))

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(3, activation='softmax')) # Regular Neural Net hidden layer

# Compile model
sgd = optimizers.SGD(lr=0.01, clipvalue=0.5, momentum=.6, decay=0.001)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])


# Fit model on training data
model.fit(X_train, y_train, batch_size=25, epochs=10, verbose=1,
          validation_data=(X_test, y_test))

# Evaluate model on test data
Train_Accuracy = model.evaluate(X_train, y_train, verbose=0)[1]
Validation_Accuracy = model.evaluate(X_test, y_test, verbose=0)[1]






