import argparse


def args():
    """
    Function to create command line arguments

    There are two arguments:
        -dn <str> (data_name) name of the data to import form the data folder
            they are: breast-cancer, glass, iris, soybean, vote
        -rs <int> (random_seed) seed used for data split. Defaults to 1. All submitted output uses random_seed 1
        -nb (naive_bayes) Switch to the Naive Bayes classifier. Defaults to Winnow2
    """
    # Initialize the parser
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument('-dn', '--data_name', help='Specify data name to extract and process')
    parser.add_argument('-rs', '--random_state', default=1, type=int,
                        help='Specify a seed to pass to the data splitter')
    parser.add_argument('-kt', '--knn_type', type=str,
                        help='Specify type of KNN, ignore for the original KNN, otherwise pass <edited> or <condensed>')

    # Parse arguments
    command_args = parser.parse_args()

    # Return the parsed arguments
    return command_args
