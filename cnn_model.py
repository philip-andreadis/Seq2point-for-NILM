import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from keras.callbacks import CSVLogger
import math
from utils import CustomEarlyStopping

class CNNModel():
    def __init__(self, optimizer, loss='mse', metrics=['mse', 'mae']):
        """
        Initialization function, it creates an instance of the CNN model and compiles it

        Parameters:
        ----------
        loss : str
            Loss function to use for training.
        optimizer : str
            Optimizer to use for training.
        metrics : list
            List of metrics to use for training.
        
        Returns:
        ----------
        model : keras.Sequential
            The compiled CNN model.
            
        """
        
        self.MODEL_NAME = 'CNN'
        # Create the CNN
        self.model = keras.Sequential(
                 [
                    keras.Input(shape=(599,1)),
                    layers.Conv1D(30, 10, strides=1, activation=layers.LeakyReLU(), padding='same'),
                    layers.Conv1D(30, 8, strides=1, padding='same', activation=layers.LeakyReLU()),
                    layers.Conv1D(40, 6, strides=1, padding='same', activation=layers.LeakyReLU()),
                    layers.Conv1D(50, 5, strides=1, padding='same', activation=layers.LeakyReLU()),
                    layers.Conv1D(50, 5, strides=1, padding='same', activation=layers.LeakyReLU()),
                    layers.Flatten(), 
                    layers.Dense(1024, activation=layers.LeakyReLU()),
                    layers.Dense(1, activation='linear')
                 ]
        )

        # Early stopping callback
        self.early_stopping = tf.keras.callbacks.EarlyStopping(
                                    monitor='val_loss',
                                    min_delta=0,
                                    patience=5,
                                    verbose=1,
                                    mode='auto',
                                    baseline=None,
                                    restore_best_weights=False,
                                    start_from_epoch=14
                                )
        
        # Custom early stopping
        self.custom_early_stopping = CustomEarlyStopping(patience=5)
        
        # Learning rate scheduler
        def scheduler(epoch, lr):
            return lr * math.exp(-0.01*epoch) 

        # Define scheduler
        self.lr_scheduler = keras.callbacks.LearningRateScheduler(scheduler, verbose=1)
          
        # Compile the model
        self.model.compile(loss=loss, optimizer=optimizer, metrics=metrics)


    def train(self, X_train, Y_train, X_val, Y_val, dir, filename, i, epochs=20, batch_size=32, verbose=2):

        # Check if GPU is available
        print("\n ---Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

        # Csv logger
        self.csv_logger = CSVLogger(dir + '/' + 'training_history_' + filename + '(' + str(i) + ')' + '.csv', separator=',', append=False)

        # Choose callbacks
        # callbacks = [self.csv_logger, self.early_stopping, self.lr_scheduler]
        callbacks = [self.csv_logger, self.early_stopping]
        # callbacks = [self.csv_logger]

        # Fit the model    
        self.history = self.model.fit(X_train, Y_train, validation_data=(X_val,Y_val), epochs=epochs, batch_size=batch_size, shuffle=True, validation_split=0.1, callbacks=callbacks, verbose=verbose)
        
    def freeze_conv(self):
        for layer in self.model.layers[:-2]:
            layer.trainable = False
        # Recompile the model
        self.model.compile(loss=self.model.loss, optimizer=self.model.optimizer, metrics=['mse', 'mae'])

    def disaggregate(self,X_test):

        # Make predictions
        predictions = self.model.predict(X_test, verbose=2)
        return predictions

    def save_model(self, dir, filename):
        # Save the model as keras file
        # self.model.save(dir + '/' + filename + '.keras')

        # Save the model as h5 and json files
        model_json = self.model.to_json()
        with open(dir + "/" + filename + '.json', "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights(dir + "/" + filename + '.h5')
        print("Saved model to disk")

