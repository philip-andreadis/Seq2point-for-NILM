import numpy as np
import pandas as pd
import math
import tensorflow as tf



def threshold(consumption, device):
    """
    This function implements thresholding of the target appliance consumption based on predefined on-state threshold values.
    Each timestep is classified to either off-state (0) or on-state (1).

    Parameters
    ----------
    consumption : numpy array
        The consumption signal to be thresholded.
    device : str
        The device that the consumption signal belongs to.
    
    """

    thres_vals = {
    "refrigerator": 50,
    "dishwaser": 10,
    "microwave": 200,
    "lighting": 20,
    "kettle": 2000,
    "heatpump": 200
    }
    thres_cons = np.where(consumption > thres_vals[device], 1, 0)
    return thres_cons



def downsample(house, dataset,frequency = 10):
    """
    This function downsamples the provided house dataframe to the given frequency.

    Parameters
    ----------
    house : pandas dataframe
        House df.
    frequency : int
        Target frequency.

    Returns
    -------
    house_dsed : pandas dataframe
        The downsampled dataframe.
    
    """

    if dataset == 'redd':
        # Use time column as datetime index
        house.index = pd.to_datetime(house.time)
        house = house.drop(columns=['time'])

        # Drop nas (converts to 3 sec freq)
        house = house.dropna()
        
        # Downsample to the required frequency
        if frequency == 10:
            house_dsed = house.resample('10S').mean()
            house_dsed = house_dsed.dropna()
        elif frequency == 6:
            house_dsed = house.resample('6S').mean()
            house_dsed = house_dsed.dropna()
    elif dataset == 'ukdale':
        # Downsample to the required frequency
        house.index = pd.to_datetime(house.time)
        house = house.drop(columns=['time'])
        print(house.shape)
        house_dsed = house.resample(f'{frequency}S').mean()
        print(house_dsed.shape)
        house_dsed = house_dsed.dropna()
        print(house_dsed.shape)
        

    return house_dsed


def avg_metrics(final_metrics, loops):
    """ 
    This function computes the average of the metrics over the different loops.

    Parameters
    ----------
    final_metrics : list
        List of the metrics of the different loops.
    loops : int
        Number of training loops.

    Returns
    -------
    metrics_avg : list
        List of the average metrics.
    
    """

    metrics_avg = []
    for i in range(len(final_metrics[0])):
        sum = 0
        for j in range(loops):
            sum += float(final_metrics[j][i].split(': ')[1])
        avg = sum / loops
        metrics_avg.append(avg)

    return metrics_avg




class CustomEarlyStopping(tf.keras.callbacks.Callback):
    """ 
    This class implements a custom early stopping callback that stops training when the validation loss 
    does not improve for a certain number of epochs.
    """
    def __init__(self, patience=5):
        super(CustomEarlyStopping, self).__init__()
        self.patience = patience
        self.wait = 0
        self.prev_loss = None
        self.start_from_epoch = 14 

    def on_epoch_end(self, epoch, logs=None):
        current_loss = logs.get('val_loss')

        # warm-up period
        if epoch < self.start_from_epoch:
            return
        
        if current_loss is None:
            return
        
        if self.prev_loss is not None and current_loss >= self.prev_loss:
            self.wait += 1
        else:
            self.wait = 0
        
        if self.wait >= self.patience:
            print(f'\nEarly stopping at epoch {epoch} as validation loss has not improved for {self.patience} consecutive epochs.')
            self.model.stop_training = True

        self.prev_loss = current_loss