import argparse

"""This script handles the arguments that the user can provide. It returns an args object that 
can be used to obtain the provided arguments."""


def create_arg_parser():
    """Adds arguments to the argument parser and parses them in the end"""
    parser = argparse.ArgumentParser()

    # Parameters
    parser.add_argument('--train_houses', nargs='+', type=int, help='<Required> Set flag', required=True)
    
    parser.add_argument('--test_houses', nargs='+', type=int, help='<Required> Set flag', required=True)

    parser.add_argument("--device", default='refrigerator', type=str,
                        help="Target device to train on.")
    
    parser.add_argument("--nas", default='', type=str,
                        help="How to deal with missing values. Possible values [interpolate,drop]")
    
    parser.add_argument("-sr","--sampling_rate", default=1, type=int,
                        help="Target sampling rate.")
    
    parser.add_argument('-st', '--standardise', dest='standardise', default=False, action='store_true', help="Standardise data.")
    
    parser.add_argument("-lr","--learning_rate", default=0.001, type=float,
                        help="Learning rate for the model.")
    
    parser.add_argument("-e","--epochs", default=20, type=int,
                        help="Number of training epochs.")
    
    parser.add_argument("-b","--batch", default=32, type=int,
                        help="Batch size.")
    
    parser.add_argument("--loss", default='mse', type=str,
                        help="Loss function to use for training.")
    
    parser.add_argument("--dataset", default='ukdale', type=str,
                        help="Which dataset will be used. Options are [redd,ukdale]")
    
    parser.add_argument("-sd","--source_domain", default='redd', type=str,
                        help="Which source domain will be used. Options are [redd,ukdale]")
    
    parser.add_argument("-td","--target_domain", default='redd', type=str,
                        help="Which target domain will be used. Options are [redd,ukdale]")

    parser.add_argument('-tl', '--transfer_learning', dest='transfer_learning', default=False, action='store_true', 
                        help="Enable transfer learning.")

    
    args = parser.parse_args()
    return args