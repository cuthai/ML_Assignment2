from utils.args import args
from etl.etl import ETL
from knn.knn_classifier import KNNClassifier


def main():
    # Parse arguments
    arguments = args()

    # Set up kwargs for ETL
    kwargs = {
        'data_name': arguments.data_name,
        'random_state': arguments.random_state
    }

    # Perform ETL
    etl = ETL(**kwargs)

    # KNN
    knn_model = KNNClassifier(etl)

    knn_model.tune()

    pass


if __name__ == '__main__':
    main()
