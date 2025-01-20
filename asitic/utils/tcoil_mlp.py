import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras import regularizers
from tensorflow.keras.layers.experimental import preprocessing

"""

    L, W, S, N, tap -->[ML Algorithm]--> La, Ra, Lb, Rb, k, Cbr  

"""



def mlp(tcoil_x_train, tcoil_y_train):   
    
    # normalize = preprocessing.Normalization()
    # normalize.adapt(tcoil_x_train) 
    tcoil_model = tf.keras.Sequential([
        #normalize,
        layers.Dense(512, activation = 'elu', kernel_regularizer=regularizers.l2(0.0001)),
        #layers.Dropout(0.1),
        layers.Dense(1024, activation = 'elu', kernel_regularizer=regularizers.l2(0.0001)),
        layers.Dropout(0.1),
        layers.Dense(1024, activation = 'elu', kernel_regularizer=regularizers.l2(0.0001)),
        layers.Dense(1024, activation = 'elu', kernel_regularizer=regularizers.l2(0.0001)),
        layers.Dense(512, activation = 'elu', kernel_regularizer=regularizers.l2(0.0001)),
        layers.Dense(256, activation = 'elu', kernel_regularizer=regularizers.l2(0.0001)),
        layers.Dense(np.shape(tcoil_y_train)[1])
        ])
    
    
    epochs = 600

    loss_1 = tf.keras.losses.MeanAbsoluteError()
    loss_2 = tf.keras.losses.MeanSquaredError()

    tcoil_model.compile(loss = loss_1,
                        #loss_weights=[0.7, 0.3],
                        optimizer = tf.keras.optimizers.Adam(),
                        metrics= ['mean_absolute_error'])
    
    
    tcoil_model.fit(tcoil_x_train, tcoil_y_train, validation_split = 0.2, epochs = epochs)
        

    return tcoil_model

