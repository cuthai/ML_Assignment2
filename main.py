from utils.args import args
from etl.etl import ETL
from knn.knn_classifier import KNNClassifier
from knn.knn_regressor import KNNRegressor


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
    if arguments.data_name in ['glass', 'segmentation', 'vote']:
        knn_model = KNNClassifier(**kwargs)
    else:
        knn_model = KNNRegressor(**kwargs)

    knn_model.fit()

    knn_model.tune()

    if arguments.knn_type in ['edited', 'condensed']:
        if arguments.epsilon:
            epsilon_range = [arguments.epsilon]
        else:
            epsilon_range = None
        knn_model.fit_modified(epsilon_range=epsilon_range)

    if arguments.k:
        knn_model.k = arguments.k
    if arguments.sigma:
        knn_model.sigma = arguments.sigma
    knn_model.predict()

    knn_model.output()

    if arguments.k == arguments.sigma == arguments.epsilon is None:
        knn_model.visualize_tune()


if __name__ == '__main__':
    main()
