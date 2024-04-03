import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from utils import downsample
import sys


def process_redd(houses, ds, nas, device):
    """
        Helper function to preprocess redd.
    """

    # Load multiple houses for training
    c = 1
    for i in houses:
        house = pd.read_csv('redd/house_{}.csv'.format(i))
        if ds>3:
            house = downsample(house,frequency=ds,dataset='redd')
            house = house.reset_index(drop=True)
        if c == 1:
            df = house
        else:
            df = pd.concat([df, house], join="inner")
        c += 1

    #deal with nas
    if not(ds > 3):                 # No downsample chosen. Proceed with drop/interpolate
        if nas == 'interpolate':
            # if the gap is less than 2 minutes, then interpolate, else leave it as nan
            df = df.interpolate(method='linear', limit=120)
        elif nas == 'drop':
            df = df.dropna()

    # Keep only the main and the device columns
    if device in df.columns:
        df_device = df[['main', device]]

    # Drop remaining nas
    df_device = df_device.dropna()

    return df_device



def process_ukdale(houses,ds, device):
    """
        Helper function to preprocess ukdale.
    """

    # Load multiple houses for training
    c = 1
    for i in houses:
        house = pd.read_csv('ukdale/house_{}.csv'.format(i))
        # Keep only the main and the device columns
        if device in house.columns:
            house = house[['main', device,'time']]
        else:
            print(f'Error: device not present in house {i}')
            sys.exit(1)

        # Drop nan values 
        house = house.dropna()

        # Downsample house to ds
        if ds>6:
            house = downsample(house,frequency=ds,dataset='ukdale')

        house = house.reset_index(drop=True) # remove datetime index in order to concat multiple houses in df

        if c == 1:
            df = house
        else:
            df = pd.concat([df, house], join="inner")
        c += 1

    return df


def process_pc(device):
    """
        Helper function to process the powerchainger dataset.
    
    """
    house = pd.read_csv('combinedDataset.csv')

    # Keep only the main and the device columns
    if device in house.columns:
        house = house[['main', device]]
    else:
        print(f'Error: device not present in house')
        sys.exit(1)

    # Drop nan values 
    house = house.dropna()

    return house



def seq2point(houses, test_houses, source_domain, target_domain, device, w, nas='drop', standardize=True, ds=1):
    """
    Parameters
    ----------
    houses : list
        list of the numbers of the houses to be loaded for training, e.g. [1, 2, 3, 4, 5].
    test_houses : list
        list of the numbers of the houses to be loaded for testing, e.g. [1, 2, 3, 4, 5].
    standardize : bool
        Whether to standardize the data or not.
    ds : int
        Downsample the dataset frequency to ds seconds. Must be lower than 3 secs otherwise nas='drop' is selected.
    device : str
        Device which the model is going to be trained for, e.g. 'microwave'.
    w : int
        Window size.
    nas : str
        How to deal with missing values. Options are 'interpolate' and 'drop'.
    source_domain: str
        The source domain dataset. Options are 'redd', 'ukdale' and 'pc'.
    target_domain: str
        The target domain dataset. Options are 'redd', 'ukdale' and 'pc'.
    
    
    Returns
    -------
    x : np.array
        Array of shape (num_samples, window_size). The input of the model in seq2point format.
    y : np.array
        Array of shape (num_samples,). The target of the model in seq2point format.
    x_test : np.array
        Array of shape (num_samples, window_size). The test set in seq2point format.
    y_test : np.array
        Array of shape (num_samples,). The test targets in seq2point format.
    """
    if source_domain == 'redd':
        df_device = process_redd(houses,ds,nas,device)
    elif source_domain == 'ukdale':
        df_device = process_ukdale(houses,ds, device)
    elif source_domain == 'pc':
        df_device = process_pc(device)
    

    #rename 'main' column to 'power'
    df_device.rename(columns={'main': 'power'}, inplace=True)

    # Standardize the data
    input_scaler = StandardScaler()
    output_scaler = StandardScaler()
    if standardize:
        # Scale aggregated power   
        agg =  input_scaler.fit_transform(df_device['power'].values.reshape(-1, 1))
        df_device['power'] = agg.reshape(-1, 1)

        # Scale device power
        dev = output_scaler.fit_transform(df_device[device].values.reshape(-1, 1))
        df_device[device] = dev.reshape(-1, 1)

    # Pad power with w/2 zeros to the left and right
    power = np.pad(df_device['power'], w//2, mode='constant', constant_values=0)

    # Pad device with w/2 zeros to the left and right
    device_power = np.pad(df_device[device], w//2, mode='constant', constant_values=0)
    
    # Get x in seq2point format. Slide a window of length w over power by 1 timestep and store in x
    x = np.lib.stride_tricks.as_strided(power, shape=(len(power) - w + 1, w), strides=(power.strides[0], power.strides[0]))

    # Get y in seq2point format. Slide a window of length w over device by 1 timestep and store the midpoint of the window in y
    l = len(df_device)
    y = np.zeros(l)
    for i in range(len(y)):
        y[i] = device_power[i+w//2] # the equivalent is to just use the original device sequence

    
    # Load multiple houses for testing
    if test_houses:
        if target_domain == 'redd':
            df_device_test = process_redd(test_houses,ds,nas,device)
        elif target_domain == 'ukdale':
            df_device_test = process_ukdale(test_houses,ds, device)
        elif target_domain == 'pc':
            df_device_test = process_pc(device)
        

        #rename 'main' column to 'power'
        df_device_test.rename(columns={'main': 'power'}, inplace=True)

        # Standardize the data
        if standardize:
            # Scale aggregated power
            agg =  input_scaler.transform(df_device_test['power'].values.reshape(-1, 1))
            df_device_test['power'] = agg.reshape(-1, 1)

            # Scale device power
            dev = output_scaler.transform(df_device_test[device].values.reshape(-1, 1))
            df_device_test[device] = dev.reshape(-1, 1)

        # Pad power with w/2 zeros to the left and right
        power_test = np.pad(df_device_test['power'], w//2, mode='constant', constant_values=0)

        # Pad device with w/2 zeros to the left and right
        device_power_test = np.pad(df_device_test[device], w//2, mode='constant', constant_values=0)
        
        # Get x in seq2point format. Slide a window of length w over power by 1 timestep and store in x
        x_test = np.lib.stride_tricks.as_strided(power_test, shape=(len(power_test) - w + 1, w), strides=(power_test.strides[0], power_test.strides[0]))

        # Get y in seq2point format. Slide a window of length w over device by 1 timestep and store the midpoint of the window in y
        l = len(df_device_test)
        y_test = np.zeros(l)
        for i in range(len(y_test)):
            y_test[i] = device_power_test[i+w//2]
        


    return x, y, x_test, y_test, output_scaler


