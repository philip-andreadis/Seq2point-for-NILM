import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
from utils import downsample
import sys


def process_redd(houses, ds, nas, device):
    """
        Helper function to preprocess redd.
    """

    # If single house
    if type(houses) == int:
        houses = [houses]

    # Load multiple houses for training
    c = 1
    for i in houses:
        house = pd.read_csv('data/redd/house_{}.csv'.format(i))
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
    # If single house
    if type(houses) == int:
        houses = [houses]

    # Load multiple houses for training
    c = 1
    for i in houses:
        house = pd.read_csv('data/ukdale/house_{}.csv'.format(i))
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
    house = pd.read_csv('data/combinedDataset.csv')

    # Keep only the main and the device columns
    if device in house.columns:
        house = house[['main', device]]
    else:
        print(f'Error: device not present in house')
        sys.exit(1)

    # Drop nan values 
    house = house.dropna()

    return house


def process_heatpump(houses):
    # If single house
    if type(houses) == int:
        houses = [houses]

    c = 1
    for i in houses:
        house = pd.read_csv('data/heatpump_dataset/SFH{}.csv'.format(i))
        house = house.iloc[:len(house)//3,:] # Get a fraction of each house
        if c == 1:
            df = house[['main','heatpump']]
        else:
            df = pd.concat([df, house], join="inner")
        c += 1
    return df


def seq2last_point(agg_power, device_power, w):
    """
    Parameters
    ----------
    agg_power : pd.Series
        Aggregated power.
    device_power : pd.Series
        Power of the device.
    w : int
        Window size.

    Returns
    -------
    x : np.array
        Array of shape (num_samples, window_size). The input of the model in seq2point format.
    y : np.array
        Array of shape (num_samples,). The target of the model in seq2last_point format.
    """

    # Pad power with w-1 zeros to the left
    power = np.pad(agg_power, (w-1, 0), mode='constant')

    # Get x in seq2point format. Slide a window of length w over power by 1 timestep and store in x
    x = np.lib.stride_tricks.as_strided(power, shape=(len(power) - w + 1, w), strides=(power.strides[0], power.strides[0]))

    # This is redundant, just do y = df_device[device]
    # # Pad device with w-1 zeros to the left
    # device_power = np.pad(df_device[device], (w-1, 0), mode='constant')

    # # Get y in seq2point format. Slide a window of length w over device by 1 timestep and store the last element of the window in y
    # y = np.zeros(len(df_device) - w + 1)
    # for i in range(len(y)):
    #     y[i] = df_device[device][i+w-1]

    y = np.array(device_power) 

    return x, y


def seq2seq(agg_power, device_power, w):
    """
    Parameters
    ----------
    agg_power : pd.Series
        Aggregated power.
    device_power : pd.Series
        Power of the device.
    w : int
        Window size.

    Returns
    -------
    x : np.array
        Array of shape (num_samples, window_size). The input of the model in seq2seq format.
    y : np.array
        Array of shape (num_samples, window_size). The target of the model in seq2seq format.
    """
    # Get x in seq2seq format. Slide a window of length w over aggregated power by 1 timestep and store in x (no padding needed)
    x = np.lib.stride_tricks.as_strided(agg_power, shape=(len(agg_power) - w + 1, w), strides=(agg_power.strides[0], agg_power.strides[0]))

    # Get y in seq2seq format. Slide a window of length w over device power by 1 timestep and store in y (no padding needed)
    y = np.lib.stride_tricks.as_strided(device_power, shape=(len(device_power) - w + 1, w), strides=(device_power.strides[0], device_power.strides[0])) 

    return x, y

def seq2point(agg_power, device_power, w):
    """
    Parameters
    ----------
    agg_power : pd.Series
        Aggregated power.
    device_power : pd.Series
        Power of the device.
    w : int
        Window size.

    Returns
    -------
    x : np.array
        Array of shape (num_samples, window_size). The input of the model in seq2point format.
    y : np.array
        Array of shape (num_samples,). The target of the model in seq2point format.
    """
    # Pad power with w/2 zeros to the left and right
    power = np.pad(agg_power, w//2, mode='constant', constant_values=0)

    # Get x in seq2point format. Slide a window of length w over power by 1 timestep and store in x
    x = np.lib.stride_tricks.as_strided(power, shape=(len(power) - w + 1, w), strides=(power.strides[0], power.strides[0]))

    # Pad device with w/2 zeros to the left and right
    # device_power = np.pad(device_power, w//2, mode='constant', constant_values=0)
    
    # # Get y in seq2point format. Slide a window of length w over device by 1 timestep and store the midpoint of the window in y
    # l = len(device_power)
    # y = np.zeros(l)
    # for i in range(len(y)):
    #     y[i] = device_power[i+w//2] # the equivalent is to just use the original device sequence

    y = np.array(device_power)

    return x, y


def preprocess(houses, test_houses, source_domain, target_domain, device, w, nas='drop', standardize=True, ds=1, mode='midpoint'):
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
    X_train : np.array
        Array of shape (num_samples, window_size). The training input in 'mode' format.
    Y_train : np.array
        Array of shape (num_samples,). The training targets in 'mode' format.
    X_val : np.array
        Array of shape (num_samples, window_size). The validation input in 'mode' format.
    Y_val : np.array
        Array of shape (num_samples,). The validation targets in 'mode' format.
    x_test : np.array
        Array of shape (num_samples, window_size). The test set in 'mode' format.
    y_test : np.array
        Array of shape (num_samples,). The test targets in 'mode' format.
    """

    # Prepare training set
    train_houses = []
    for house in houses:
        if source_domain == 'redd':
            train_houses.append(process_redd(house,ds,nas,device))
        elif source_domain == 'ukdale':
            train_houses.append(process_ukdale(house,ds, device))
        elif source_domain == 'pc':
            df_device = process_pc(device) # unmaintained code
        elif source_domain == 'heatpump':
            train_houses.append(process_heatpump(house))
    
    # Prepare scalers
    input_scaler = StandardScaler()
    output_scaler = StandardScaler()
    # Concatenate train houses and fit scalers on them
    if standardize:
        # Scale aggregated power   
        # agg =  input_scaler.fit_transform(df_device['power'].values.reshape(-1, 1))
        # df_device['power'] = agg.reshape(-1, 1)

        # # Scale device power
        # dev = output_scaler.fit_transform(df_device[device].values.reshape(-1, 1))
        # df_device[device] = dev.reshape(-1, 1)

        # for house in train_houses:
        #     # Fit scaler on main power
        #     input_scaler.partial_fit(house['main'].values.reshape(-1, 1))
        #     # Fit scaler on device power
        #     output_scaler.partial_fit(house[device].values.reshape(-1, 1))

        
        # Concatenate all houses
        concat_houses = pd.concat(train_houses, axis=0)
        
        # Fit scalers
        input_scaler.fit(concat_houses['main'].values.reshape(-1, 1))
        output_scaler.fit(concat_houses[device].values.reshape(-1, 1))

    x_train = []
    y_train = []
    x_val = []
    y_val = []
    for house in train_houses:

        if standardize:
            # Scale main load
            main_load = input_scaler.transform(house['main'].values.reshape(-1, 1))
            main_load = main_load.reshape(-1, 1).flatten()
            # Scale device load
            device_load = output_scaler.transform(house[device].values.reshape(-1, 1))
            device_load = device_load.reshape(-1, 1).flatten()
        else:
            main_load = house['main']
            device_load = house[device]

        

        # Get validation set
        main_load_val = main_load[-int(len(main_load)*0.1):]
        device_load_val = device_load[-int(len(device_load)*0.1):]
        main_load = main_load[:-int(len(main_load)*0.1)]
        device_load = device_load[:-int(len(device_load)*0.1)]

        # Seq2LastPoint
        if mode == 'last_point':
            x, y = seq2last_point(main_load, device_load, w)
            x = np.float32(x)                           # Convert to smaller data type for memory efficiency
            y = np.float32(y)
            x_train.append(x)
            y_train.append(y)
            x, y = seq2last_point(main_load_val, device_load_val, w)
            x = np.float32(x)                           
            y = np.float32(y)
            x_val.append(x)
            y_val.append(y)
        # Seq2seq
        elif mode == 'sequence':
            x, y = seq2seq(main_load.values, device_load.values, w)
            x = np.float32(x)                           # Convert to smaller data type for memory efficiency
            y = np.float32(y)
            x_train.append(x)
            y_train.append(y)
            x, y = seq2seq(main_load_val.values, device_load_val.values, w)
            x = np.float32(x)                           
            y = np.float32(y)
            x_val.append(x)
            y_val.append(y)
        # Seq2Point
        elif mode == 'midpoint':
            x, y = seq2point(main_load, device_load, w)
            x = np.float32(x)                           # Convert to smaller data type for memory efficiency
            y = np.float32(y)
            x_train.append(x)
            y_train.append(y)
            x, y = seq2point(main_load_val, device_load_val, w)
            x = np.float32(x)
            y = np.float32(y)
            x_val.append(x)
            y_val.append(y)
    
    # Concatenate X_train and Y_train
    X_train = np.concatenate(x_train, axis=0)
    Y_train = np.concatenate(y_train, axis=0)
    
    # Concatenate X_val and Y_val
    X_val = np.concatenate(x_val, axis=0)
    Y_val = np.concatenate(y_val, axis=0)
    
    # Prepare test set
    if test_houses:
        if target_domain == 'redd':
            df_device_test = process_redd(test_houses,ds,nas,device)
        elif target_domain == 'ukdale':
            df_device_test = process_ukdale(test_houses,ds, device)
        elif target_domain == 'pc':
            df_device_test = process_pc(device)
        elif target_domain == 'heatpump':
            df_device_test = process_heatpump(test_houses)
        

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


        # Seq2LastPoint
        if mode == 'last_point':
            x_test, y_test = seq2last_point(df_device_test['power'], df_device_test[device], w)
        # Seq2seq
        elif mode == 'sequence':
            x_test,_ = seq2seq(df_device_test['power'].values, df_device_test[device].values, w)
            y_test = np.array(df_device_test[device]) # y_test here is the original device power (in scalars) and not windowed as y_train and y_val
        # Seq2Point
        elif mode == 'midpoint':
            x_test, y_test = seq2point(df_device_test['power'], df_device_test[device], w)

    print(f'X_train shape: {X_train.shape}')  
    print(f'Y_train shape: {Y_train.shape}')
    print(f'X_val shape: {X_val.shape}')
    print(f'Y_val shape: {Y_val.shape}')
    print(f'X_test shape: {x_test.shape}')
    print(f'Y_test shape: {y_test.shape}')
        
    # # Save scaler
    # joblib.dump(input_scaler, "scaler_heatpump_houses34.save")
    # joblib.dump(output_scaler, "output_scaler_heatpump_house12.save")

    return X_train, Y_train, X_val, Y_val, x_test, y_test, output_scaler


