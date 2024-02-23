import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import random


def print_statistics(df):
    """
        Helper function that prints various information about a house dataframe.

        Parameters
        ----------
        df : pandas df
            House dataframe.
        
        """

    # Df shape
    print(df.head())
    print(df.shape)
    # Starting/ending date
    print('Starting date:',df.index[0])
    print('Ending date:',df.index[-1])
    # Missing values
    print('Missing values:', df.isna().sum())
    print('\n')

def visualise_house(df):
    """
        Helper function that visualises a random segment of the provided house dataframe.

        Parameters
        ----------
        df : pandas df
            House dataframe.
        
        """
    df = df.dropna()

    # Range
    length = 10000
    start = random.randint(1, len(df)-length)
    end = start + length

    df[start:end].plot()

    labels = []
    for col in df.columns.tolist():
        plt.plot(df[col][start:end], alpha=0.6)
        labels.append(col)

    plt.legend(labels)
    # plt.savefig('temp.png')
    plt.xlabel('Index')
    plt.ylabel('Consumption')
    plt.title('House')
    plt.show()





def ukdale_parser(houses, print_stats=False):
    """
    This function parses the uk-dale raw dataset (.dat format) and returns each selected house in pandas dataframe. 
    The columns are aligned according to their datetime index hence can include missing values. 

    Parameters
    ----------
    houses : list
        List of uk-dale houses to be parsed, e.g. [1, 2, 3, 4, 5].
    
    Returns
    -------
    final_dfs : list
        List of the parsed house dfs.
    
    """

    # List of houses in ukdale
    houses = houses

    # Structure of labels dict: 
    # labels[num_of_house][channel_of_appliance] gives label of appliance - the order of the keys is important - 
    labels = {
        1 : {
            12: "refrigerator",
            5: "washing_machine",
            10: "kettle",
            1: "main",
            13: "microwave",
            6: "dishwaser",
        },
        2 : {
            15: "microwave",
            14: "refrigerator",
            13: "dishwaser",
            12: "washing_machine",
            8: "kettle",
            1: "main"
        },
        3 : {
         2: "kettle",
         1: "main"
        },
        4 : {
         5: "refrigerator",  # freezer (~refrigerator)
         1: "main"
        },
        5 : {
         19: "refrigerator", # fridge_freezer (~refrigerator)
         22: "dishwaser",
         23: "microwave",
         1: "main",
         18: "kettle"
        }
    }
    
    final_dfs = []

    # Iterate houses
    for house in houses:
        house_dict = labels[house] # get house labels dict
        c = 0
        # Iterate channels/devices of house
        for channel in list(house_dict.keys()):
            dev_name = house_dict[channel] # get channel's name
            device = pd.read_csv(f'NILM/ukdale/house_{house}/channel_{channel}.dat', names=[dev_name])

            # Split timestamp and power measurements
            device[['time', dev_name]] = device[dev_name].str.split(' ', expand=True)
            device[dev_name] = device[dev_name].apply(int)
            device['time'] = device['time'].apply(int)

            # Use time column as datetime index
            device.time = pd.to_datetime(device.time,unit='s')
            device.set_index('time', inplace=True)

            if print_stats:
                print_statistics(device)

            # Merge the device dfs
            print(c)
            if c == 0:
                merged_df = device
            else:
                merged_df = pd.merge_asof(merged_df, device, left_index=True, right_index=True, direction='nearest', tolerance=pd.Timedelta(seconds=2))
                # merged_df = merged_df.dropna()
            print_statistics(merged_df)
            c = c + 1

        final_dfs.append(merged_df)

    return final_dfs
            

if __name__ == "__main__":
    list_of_houses = [2]
    house_dfs = ukdale_parser(houses=list_of_houses,print_stats=False)
    for i in house_dfs:
        print_statistics(i)
        # visualise_house(i)
    
    # # Save house dfs to csv
    # for i, h in enumerate(house_dfs):
    #     h.to_csv(f'NILM/ukdale_csv/house_{list_of_houses[i]}.csv')
    