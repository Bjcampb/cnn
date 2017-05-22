import scipy.io as sio
import numpy as np
from sklearn import preprocessing

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras import regularizers
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras import optimizers

###############################################################################
# Define
###############################################################################
def LoadStackPeak(fileloc):
    '''Loads Matlab file and stacks data in proper format'''
    
    file = sio.loadmat(str(fileloc))
    PeakData = file['dataset']
    return PeakData

def Standardize(dataset):
    """ Dataset format (Samples, Row, Col, Depth) """
    z_score = np.zeros(np.shape(dataset))
    for i in range(np.shape(dataset)[3]):
        z_score[:,:,:,i] = (dataset[:,:,:,i] - dataset[:,:,:,i].mean()) / dataset[:,:,:,i].std()
    
    return z_score
###############################################################################
# Load Data
###############################################################################

# Volume Coil Data
VCM1IG_mus = LoadStackPeak('aug_mouse_1_pre_igwat_mus.mat')
VCM1IG_wat = LoadStackPeak('aug_mouse_1_pre_igwat_wat.mat')
VCM1IC_bat = LoadStackPeak('aug_mouse_1_pre_intercap_bat.mat')
VCM1IC_mus = LoadStackPeak('aug_mouse_1_pre_intercap_muscle.mat')
VCM2IG_mus = LoadStackPeak('aug_mouse_2_pre_igwat_muscle.mat')
VCM2IG_wat = LoadStackPeak('aug_mouse_2_pre_igwat_wat.mat')
VCM2IC_bat = LoadStackPeak('aug_mouse_2_pre_intercap_bat.mat')
VCM2IC_mus = LoadStackPeak('aug_mouse_2_pre_intercap_muscle.mat')

# Surface Coil Data
SCM1IG_mus = LoadStackPeak('feb_mouse_1_pre_igwat_muscle.mat')
SCM1IG_wat = LoadStackPeak('feb_mouse_1_pre_igwat_wat.mat')
SCM1IC_bat = LoadStackPeak('feb_mouse_1_pre_intercap_bat.mat')
SCM1IC_mus = LoadStackPeak('feb_mouse_1_pre_intercap_muscle.mat')
SCM1IC_wat = LoadStackPeak('feb_mouse_1_pre_intercap_wat.mat')
SCM3IG_mus = LoadStackPeak('feb_mouse_3_pre_igwat_muscle.mat')
SCM3IG_wat = LoadStackPeak('feb_mouse_3_pre_igwat_wat.mat')
SCM4IC_bat = LoadStackPeak('feb_mouse_4_pre_intercap_bat.mat')
SCM4IC_mus = LoadStackPeak('feb_mouse_4_pre_intercap_muscle.mat')
SCM4IC_wat = LoadStackPeak('feb_mouse_4_pre_intercap_wat.mat')
SCM5IC_bat = LoadStackPeak('feb_mouse_5_pre_intercap_bat.mat')
SCM5IC_mus = LoadStackPeak('feb_mouse_5_pre_intercap_muscle.mat')
SCM5IC_wat = LoadStackPeak('feb_mouse_5_pre_intercap_wat.mat')

###############################################################################
# Standardize and Organize
###############################################################################

WAT_training = np.concatenate((VCM1IG_wat, SCM1IG_wat, SCM1IC_wat,
                               SCM3IG_wat, SCM5IC_wat), axis=2)
WAT_validation = np.concatenate((VCM2IG_wat, SCM4IC_wat), axis=2)

BAT_training = np.concatenate((VCM1IC_bat, SCM1IC_bat, SCM5IC_bat), axis=2)
BAT_training = BAT_training[:,:,0:np.shape(WAT_training)[2],:] # Reduce size of BAT to be equal to WAT
BAT_validation = np.concatenate((VCM2IC_bat, SCM4IC_bat), axis=2)

MUS_training = np.concatenate((VCM1IG_mus, VCM1IC_mus, SCM1IG_mus, SCM1IC_mus,
                               SCM3IG_mus, SCM5IC_mus), axis=2)
MUS_training = MUS_training[:,:,0:np.shape(WAT_training)[2],:] # Reduce size of MUS to be equal to WAT
MUS_validation = np.concatenate((VCM2IG_mus, VCM2IC_mus, SCM4IC_mus), axis=2)


###############################################################################
# Create class
###############################################################################
WAT_train_class = np.zeros((WAT_training.shape[2],1), dtype=int)
WAT_validation_class = np.zeros((WAT_validation.shape[2],1), dtype=int)

BAT_train_class = np.zeros((BAT_training.shape[2],1), dtype=int)+1
BAT_validation_class = np.zeros((BAT_validation.shape[2],1), dtype=int)+1

MUS_train_class = np.zeros((MUS_training.shape[2],1), dtype=int)+2
MUS_validation_class = np.zeros((MUS_validation.shape[2],1), dtype=int)+2

X_train = np.concatenate((WAT_training, BAT_training, MUS_training), axis=2)
y_train = np.concatenate((WAT_train_class, BAT_train_class, MUS_train_class))
y_train = np.ravel(y_train)

X_validation= np.concatenate((WAT_validation, BAT_validation, 
                              MUS_validation), axis=2)
y_validation= np.concatenate((WAT_validation_class, BAT_validation_class, 
                              MUS_validation_class))
y_validation= np.ravel(y_validation)

X_train = X_train.astype('float32')
X_validation= X_validation.astype('float32')

X_train = np.einsum('ijkl -> kijl', X_train)
X_validation = np.einsum('ijkl -> kijl', X_validation)

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_validation = np_utils.to_categorical(y_validation)
num_classes = y_validation.shape[1]

# Standardize Data
X_train = Standardize(X_train)
X_validation = Standardize(X_validation)

###############################################################################
#        Create the model
###############################################################################

input_shape = (33,33,8)
weight_decay = 0.0001

num_filter = [64,32]
length_filter = [(6, 6),(4,4)]


model = Sequential()

model.add(Conv2D(filters = num_filter[0],
                 kernel_size = length_filter[0],
                 strides=(1,1),
                 padding='valid',
                 activation='relu',
                 kernel_regularizer=regularizers.l2(weight_decay),
                 input_shape=input_shape))

model.add(MaxPooling2D(pool_size=(2,2),
                       padding='valid'))

model.add(BatchNormalization(axis=-1, momentum=0.99))

model.add(Dropout(0.50))

model.add(Conv2D(filters = num_filter[1],
                 kernel_size = length_filter[1],
                 strides=(1,1),
                 padding='valid',
                 activation='relu',
                 kernel_regularizer=regularizers.l2(weight_decay),
                 input_shape=input_shape))

model.add(MaxPooling2D(pool_size=(2,2),
                       padding='valid'))

model.add(BatchNormalization(axis=-1, momentum=0.99))

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(25, activation='relu'))

model.add(Dense(3, activation='softmax'))

# Compile model
sgd = optimizers.SGD(lr=0.01, clipvalue=0.5, momentum=.6, decay=0.001)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
metrics=['accuracy'])

# Fit model on training data
model.fit(X_train, y_train, batch_size=25, epochs=20, verbose=1,
          validation_data=(X_validation, y_validation))

# Evaluate model on test data
Train_Accuracy = model.evaluate(X_train, y_train, verbose=0)[1]
Validation_Accuracy = model.evaluate(X_validation, y_validation, verbose=0)[1]



