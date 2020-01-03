import tensorflow as tf
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(gpus[0], 'GPU')

class haltCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('loss') <= 0.12):
            print("\n\n\nReached 0.12 loss value so cancelling training!\n\n\n")
            self.model.stop_training = True

class stopCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('loss') > 20):
            print("\n\n\nDNN failed\n\n\n")
            self.model.stop_training = True

def create_nn(x_train_shape,dropout=0):
    dnn = tf.keras.Sequential()
    dnn.add(tf.keras.layers.Flatten())
    dnn.add(tf.keras.layers.Dense(50, activation='tanh', input_shape=x_train_shape))
    dnn.add(tf.keras.layers.Dropout(dropout))
    dnn.add(tf.keras.layers.Dense(50, activation='sigmoid'))
    dnn.add(tf.keras.layers.Dropout(dropout))
    dnn.add(tf.keras.layers.Dense(50, activation='selu'))
    dnn.add(tf.keras.layers.Dense(1, activation='relu'))
    
    dnn.compile(optimizer="nadam", loss='mean_squared_logarithmic_error', metrics=['mean_squared_logarithmic_error'])
    return dnn




nn_scalers = {}
chp_path = r'models\nn\checkpoints\cp.ckpt'
train = pd.read_csv(r'data\train_scale0_synth.csv')


callback = 0
errors = []

all_cent = len(train.center_id.unique())
i = 0

for center in train.center_id.unique():
    progress = i/all_cent
    print('*******************************'+'\n'+
          f'{center}'+'\n'+
          '*******************************'+'\n'+
          f'progress = {progress}%')
 
    train = pd.read_csv(r'data/train_scale0_synth.csv')
    train.drop(['price_ratio','price_dummy'],axis=1, inplace=True)
    train = train[train.center_id == center]
    
    dummies = pd.get_dummies(train.meal_id)
    train = pd.concat([train, dummies], axis = 1)
    
    x = train.copy()
    x.drop(['meal_id', 'num_orders', 'week', 'id', 'center_id'], axis=1, inplace=True)
    
    nn_scaler = MinMaxScaler()
    nn_scaler.fit(x)
    x = nn_scaler.transform(x)
    
    y = np.asarray(train['num_orders'])
    
    annealer = tf.keras.callbacks.LearningRateScheduler(lambda x: 1e-3 * 0.95 ** x)
    early_stop = haltCallback()
    stop = stopCallback()
    callback = tf.keras.callbacks.EarlyStopping(monitor='mean_squared_logarithmic_error', patience=4)
    
    nn = create_nn(x.shape)
    
    batch = 5
    epoch = 50
    with tf.device('/device:GPU:0'):
        nn.fit(x, y,
               batch_size=batch, epochs=epoch, callbacks=[annealer, early_stop, stop, callback],
               verbose=2)
        loss = nn.evaluate(x,y, verbose=0)[0]
        counter = 0
        while loss > 0.5:
            counter += 1
            errors.append(f'{center} nn failed to learn, loss {loss}')
            nn = create_nn(x.shape,dropout=0)
            nn.fit(x, y,
               batch_size=batch, epochs=epoch, callbacks=[annealer, early_stop, stop, callback],
               verbose=2)
            loss = nn.evaluate(x,y, verbose=0)[0]
            errors.append(f'{center}, training: {loss}')
            if counter > 5:
                errors.append(f'{center}, broken while, training: {loss}')
                break
            
    nn.save(f'E:\\Hackathons\\Food demand\\models\\nn\\{center}.h5')
    nn_scalers.update({f'{center}':nn_scaler})
    i += 1
    
pickle.dump(nn_scalers, open('E:\\Hackathons\\Food demand\\models\\nn\\scalers.pkl', 'wb'))

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    