from utils.args import args
from etl.etl import ETL
from winnow2.multi_winnow2 import MultiWinnow2
from winnow2.winnow2 import Winnow2
from bayes.naive_bayes import NaiveBayes


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

    # Winnow2 Model
    if not arguments.naive_bayes:
        # Single Class Winnow2 Model (2 total classes, 1 dummied)
        if etl.classes == 2:
            # Split data set, create object
            etl.train_test_split()
            winnow_model = Winnow2(etl)

            # Tune
            winnow_model.tune()
            winnow_model.visualize_tune()

            # Fit
            winnow_model.fit()

            # Predict
            winnow_model.predict()

            # Output
            winnow_model.create_and_save_summary()
            winnow_model.save_csv_results()

        # Multi Class Winnow2
        else:
            # Create object
            multi_winnow_model = MultiWinnow2(etl)

            # Split data set
            multi_winnow_model.split_etl()

            # Create, tune, fit, and predict for individual Winnow2 Classes
            multi_winnow_model.individual_class_winnow2()

            # Synthesize results into a multi class
            multi_winnow_model.multi_class_winnow2()

            # Saving
            multi_winnow_model.create_and_save_summary()
            multi_winnow_model.save_csv_results()

    # Naive Bayes model
    else:
        # Split data set
        etl.train_test_split()

        # Create Object
        naive_bayes = NaiveBayes(etl)

        # Tune
        naive_bayes.tune()
        naive_bayes.visualize_tune()

        # Fit
        naive_bayes.fit()

        # Predict
        naive_bayes.predict()

        # Save
        naive_bayes.create_and_save_summary()
        naive_bayes.save_csv_results()


if __name__ == '__main__':
    main()
