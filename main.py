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
    etl = ETL(**kwargs)

    # Set up kwargs for KNN
    kwargs = {
        'etl': etl,
        'knn_type': arguments.knn_type
    }
    # KNN
    knn_model = KNNClassifier(**kwargs)

    knn_model.fit()

    knn_model.tune()

    if arguments.knn_type in ['edited', 'condensed']:
        knn_model.fit_modified()

    knn_model.predict()

    pass


if __name__ == '__main__':
    main()
