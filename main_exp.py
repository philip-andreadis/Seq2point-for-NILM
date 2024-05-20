import pandas as pd
import time
from seq2point import seq2point
from cnn_model import CNNModel
from utils import threshold, avg_metrics
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from keras.callbacks import CSVLogger
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, f1_score
import argparse
from parser import create_arg_parser
import sys
import math
import os



def compute_metrics(Y_test, predictions, i, device):
    """Compute the metrics of the model.
    
    Parameters:
    ----------
    Y_test : np.array
        Ground truth.
    predictions : np.array
        Predictions.
    i : int
        Loop number.
    device : str
        Target device.
        
    Returns:
    ----------
    scores : list
        List of metrics.
    """
    scores = []

    # Compute regression metrics
    mse = mean_squared_error(Y_test, predictions)
    mae = mean_absolute_error(Y_test, predictions)
    rmse = math.sqrt(mean_squared_error(Y_test, predictions))
    scores.append('MAE: %.3f' % mae)
    scores.append('MSE: %.3f' % mse)
    scores.append('RMSE: %.3f' % rmse)

    # Threshold predictions and true consumption
    predictions_thres = threshold(predictions, device)
    Y_test_thres = threshold(Y_test, device)

    # Compute class metrics
    acc = accuracy_score(Y_test_thres,predictions_thres)
    f1 = f1_score(Y_test_thres,predictions_thres)
    scores.append('Accuracy: %.3f' % acc)
    scores.append('Fscore: %.3f' % f1)

    print(f"Results of loop {i}:")
    print('MAE: %.3f' % mae)
    print('MSE: %.3f' % mse)
    print('RMSE: %.3f' % rmse)
    print('Accuracy: %.3f' % acc)
    print('Fscore: %.3f' % f1)
    
    return scores


def get_args():
    
    # Get arguments
    args = create_arg_parser()

    # Dictionary of arguments
    params = {'source': args.source_domain,
            'target': args.target_domain,
            'houses': args.train_houses,
            'houses_test': args.test_houses,
            'device': args.device,
            'scale': args.standardise,
            'lr': args.learning_rate,
            'sr': args.sampling_rate,
            'nas': args.nas,
            'epochs': args.epochs,
            'batch_size': args.batch,
            'loss': args.loss,
            'tl': args.transfer_learning,
            'md': args.mode}
    return params



if __name__ == "__main__":

    # Get arguments
    params = get_args()

    # Filename parameters
    houses_str = ''.join(str(e) for e in params['houses'])
    houses_test_str = ''.join(str(e) for e in params['houses_test'])
    if params['scale']:
        sflag = 'scale'
    else:
        sflag = 'noscale'
    if params['tl']:
        tflag = 'tl'
    else:
        tflag = ''


    # Training loops    
    loops = 3 # set the number of training iterations
    final_metrics = [] # store the final metrics for each loop
    save = True # save the model
    mode = params['md'] 

    # Filename
    filename = f'{params["device"]}_house{houses_str}on{houses_test_str}_{sflag}_{params["nas"]}_{params["sr"]}_{params["lr"]}_{params["loss"]}_{params["epochs"]}epoch_{params["batch_size"]}b_{tflag}'
    dir = f'results_{params["md"]}/results_{params["source"]}_on_{params["target"]}/{filename}'
    print('filename:', filename)
    print('dir:', dir)

    # Create directory to store results
    os.makedirs(dir, exist_ok=True)
    print(f"Directory '{dir}' created.")

    for i in range(loops):
        
        # Get source and target domains, in seq2point format
        X_train, Y_train, X_test, Y_test, output_scaler = seq2point(params['houses'], params['houses_test'], source_domain=params['source'],
                                                                     target_domain=params['target'], device=params['device'], w=599, 
                                                                     standardize=params['scale'], ds=params['sr'], nas=params['nas'], mode=mode)
        
        # Reduce the size of the dataset for script testing (toggle commenting)
        # X_train = X_train[:1000]
        # Y_train = Y_train[:1000]
        # X_test = X_test[:1000]
        # Y_test = Y_test[:1000]

        print("Train/test split:")
        print("X_train shape:", X_train.shape)
        print("Y_train shape:", Y_train.shape)
        print("X_test shape:", X_test.shape)
        print("Y_test shape:", Y_test.shape)
        

        # Instantiate the CNN model
        model = CNNModel(loss=params['loss'],optimizer=tf.keras.optimizers.Adam(learning_rate=params['lr']),metrics=['mse', 'mae'])

        # reshape X_train from [samples, timesteps] into [samples, timesteps, features]
        n_features = 1
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

        # Train network
        print(f'Training on {params["source"]} domain...')
        model.train(X_train, Y_train, dir=dir, filename=filename, epochs=params['epochs'], batch_size=params['batch_size'], verbose=2, i=i)

        # Fine tuning on the target domain
        if params['tl']:
            print(f'Fine tuning on {params["target"]} domain...')
            # Get fine tuning house
            if params['target'] == 'redd':
                house_ft = [5]
            elif params['target'] == 'ukdale':
                house_ft = [6]
            _, _, X_ft, Y_ft, _ = seq2point(params['houses'], house_ft, source_domain=params['source'], target_domain=params['target'],
                                                                    device=params['device'], w=599, standardize=params['scale'], ds=6) 

            # Freeze conv layers
            model.freeze_conv()
            
            # Retrain dense layers on target domain
            X_ft = X_ft.reshape((X_ft.shape[0], X_ft.shape[1], 1))
            model.train(X_ft, Y_ft, dir=dir, filename='fine_tune', epochs=3, batch_size=params['batch_size'], verbose=2, i=i)
        
        # Reshape X_test from [samples, timesteps] into [samples, timesteps, features]
        n_features = 1
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

        # Make predictions
        print(f'Disaggregating on {params["target"]} domain...')
        predictions = model.disaggregate(X_test)
        if params['scale']:
            predictions = output_scaler.inverse_transform(predictions)
            Y_test_original = output_scaler.inverse_transform(Y_test.reshape(-1, 1))
        else:
            Y_test_original = Y_test
        # Save predictions to a txt file
        with open(dir + '/' + 'predictions_' + filename + '(' + str(i) + ')' + '.txt', 'w') as file:
            for prediction in predictions:
                file.write(','.join(map(str, prediction)) + '\n')

        # Comptute metrics
        metrics = compute_metrics(Y_test_original, predictions, i, params['device'])

        # Store metrics of the current training loop
        final_metrics.append(metrics)

        # Save the model 
        if save:
            model.save_model(dir, filename + '(' + str(i) + ')')


    # Compute metrics average
    metrics_avg = avg_metrics(final_metrics, loops)

    # Save final metrics to txt file
    with open(dir + "/" + 'final_metrics_' + filename + '.txt', 'w') as f:
        for i,metrics in enumerate(final_metrics):
            f.write(f"Loop {i}:\n")
            for metric in metrics:
                f.write(f"{metric}\n")
        f.write("\n")
        f.write("Average:\n")
        for avg in metrics_avg:
            f.write(f"{avg}\n")

    
    
    
