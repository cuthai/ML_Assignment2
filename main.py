from utils.args import args
from etl.etl import ETL


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

    pass


if __name__ == '__main__':
    main()
