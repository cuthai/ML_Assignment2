import pandas as pd
import numpy as np


class ETL:
    """
    Class ETL to handle the ETL of the data.

    This class really only does the extract and transform functions of ETL. The data is then received downstream by the
        algorithms for processing.
    """
    def __init__(self, data_name, random_state=1):
        """
        Init function. Takes a data_name and extracts the data and then transforms.

        All data comes from the data folder. The init function calls to both extract and transform for processing

        :param data_name: str, name of the data file passed at the command line. Below are the valid names:
            glass
            segmentation
            vote
            abalone
            machine (assignment name: computer hardware)
            forest-fires
        :param random_state: int, seed for data split
        """
        # Set the attributes to hold our data
        self.data = None
        self.transformed_data = None
        self.data_split = {}

        # Meta attributes
        self.data_name = data_name
        self.random_state = random_state
        self.classes = 0
        self.problem_type = None

        # Extract
        self.extract()

        # Transform
        self.transform()

        # Split
        pass

    def extract(self):
        """
        Function to extract data based on data_name passed

        :return self.data: DataFrame, untransformed data set
        """
        # glass
        if self.data_name == 'glass':
            column_names = ['ID', 'Refractive_Index', 'Sodium', 'Magnesium', 'Aluminum', 'Silicon', 'Potassium',
                            'Calcium', 'Barium', 'Iron', 'Class']
            self.data = pd.read_csv('data\\glass.data', names=column_names)

        # segmentation
        elif self.data_name == 'segmentation':
            column_names = ['Class', 'Region_Centroid_Col', 'Region_Centroid_Row', 'Region_Pixel_Count',
                            'Short_Line_Density_5', 'Short_Line_Density_2', 'Vedge_Mean', 'Vedge_SD', 'Hedge_Mean',
                            'Hedge_SD', 'Intensity_Mean', 'Raw_Red_Mean', 'Raw_Blue_Mean', 'Raw_Green_Mean',
                            'Ex_Red_Mean', 'Ex_Blue_Mean', 'Ex_Green_Mean', 'Value_Mean', 'Saturation_Mean', 'Hue_Mean']
            self.data = pd.read_csv('data\\segmentation.data', names=column_names, skiprows=5)

        # vote
        elif self.data_name == 'vote':
            column_names = ['Class', 'Handicapped_Infants', 'Water_Project_Cost_Sharing', 'Adoption_Budget_Resolution',
                            'Physician_Fee_Freeze', 'El_Salvador_Aid', 'Religious_Groups_School',
                            'Anti_Satellite_Test_Ban', 'Aid_Nicaraguan_Contras', 'MX_Missile', 'Immigration',
                            'Synfuels_Corporation_Cutback', 'Education_Spending', 'Superfund_Right_To_Sue', 'Crime',
                            'Duty_Free_Exports', 'Export_Administration_Act_South_Africa']
            self.data = pd.read_csv('data\\house-votes-84.data', names=column_names)

        # abalone
        elif self.data_name == 'abalone':
            column_names = ['Sex', 'Length', 'Diameter', 'Height', 'Whole_Weight', 'Shucked_Weight', 'Viscera_Weight',
                            'Shell_Weight', 'Rings']
            self.data = pd.read_csv('data\\abalone.data', names=column_names)

        # machine
        elif self.data_name == 'machine':
            column_names = ['Vendor', 'Model_Name', 'MYCT', 'MMIN', 'MMAX', 'CACH', 'CHMIN', 'CHMAX', 'PRP', 'ERP']
            self.data = pd.read_csv('data\\machine.data', names=column_names)

        # forest-fires
        elif self.data_name == 'forest-fires':
            self.data = pd.read_csv('data\\forestfires.data')

        # If an incorrect data_name was specified we'll raise an error here
        else:
            raise NameError('Please specify a predefined name for one of the 5 data sets (breast-cancer, glass, iris, '
                            'soybean, vote)')

    def transform(self):
        """
        Function to transform the specified data

        This is a manager function that calls to the actual helper transform function.
        """
        # glass
        if self.data_name == 'glass':
            self.transform_glass()

        # segmentation
        elif self.data_name == 'segmentation':
            self.transform_segmentation()

        # vote
        elif self.data_name == 'vote':
            self.transform_vote()

        # The extract function should catch this but lets throw again in case
        else:
            raise NameError('Please specify a predefined name for one of the 5 data sets (breast-cancer, glass, iris, '
                            'soybean, vote)')

    def transform_glass(self):
        """
        Function to transform glass data set

        For this function numeric data is binned into groups of 10, and then dummied, similar to the breast-cancer
            data set. There are no missing values. Since this is a multi class data set, we'll dummy the classes for now
            and let the classifier handle the multi classes.

        :return self.transformed_data: DataFrame, transformed data set
        :return self.classes: int, num of classes
        """
        # We'll make a deep copy of our data set
        temp_df = pd.DataFrame.copy(self.data, deep=True)

        # We don't need ID so let's drop that
        temp_df.drop(columns='ID', inplace=True)

        # Normalize Data
        normalized_temp_df = (temp_df - temp_df.mean()) / temp_df.std()

        # Set the class back, the normalize above would have normalized the class as well
        normalized_temp_df['Class'] = temp_df['Class']

        # Set attributes for ETL object, there are 6 total classes so this is a multi classifier
        self.classes = 6
        self.transformed_data = normalized_temp_df

    def transform_segmentation(self):
        """
        Function to transform vote data set

        For this function a question mark is treated as a distinct category, since abstaining from a vote might tell us
            about the rep's party. Since the data is somewhat categorical, there was no binning done. The data is
            dummied based on the 3 possible values in each column.

        :return self.transformed_data: DataFrame, transformed data set
        :return self.classes: int, num of classes
        """
        # We'll make a deep copy of our data set
        temp_df = pd.DataFrame.copy(self.data, deep=True)

        # Region pixel count is always 9 and is not useful for our algorithms
        temp_df.drop(columns='Region_Pixel_Count', inplace=True)

        # Normalize Data
        normalized_temp_df = (temp_df - temp_df.mean()) / temp_df.std()

        # Set the class back, the normalize above would have normalized the class as well
        normalized_temp_df['Class'] = temp_df['Class']

        # Set attributes for ETL object, there are two total classes so this is a singular classifier
        self.classes = 7
        self.transformed_data = normalized_temp_df

    def transform_vote(self):
        """
        Function to transform vote data set

        For this function a question mark is treated as a distinct category, since abstaining from a vote might tell us
            about the rep's party. Since the data is somewhat categorical, there was no binning done. The data is
            dummied based on the 3 possible values in each column.

        :return self.transformed_data: DataFrame, transformed data set
        :return self.classes: int, num of classes
        """
        # We'll make a deep copy of our data set
        temp_df = pd.DataFrame.copy(self.data, deep=True)

        # Get dummies of the binned data
        binned_temp_df = pd.get_dummies(temp_df, columns=['Handicapped_Infants', 'Water_Project_Cost_Sharing',
                                                          'Adoption_Budget_Resolution', 'Physician_Fee_Freeze',
                                                          'El_Salvador_Aid', 'Religious_Groups_School',
                                                          'Anti_Satellite_Test_Ban', 'Aid_Nicaraguan_Contras',
                                                          'MX_Missile', 'Immigration', 'Synfuels_Corporation_Cutback',
                                                          'Education_Spending', 'Superfund_Right_To_Sue', 'Crime',
                                                          'Duty_Free_Exports', 'Export_Administration_Act_South_Africa']
                                        )

        # Set attributes for ETL object, there are two total classes so this is a singular classifier
        self.classes = 2
        self.transformed_data = binned_temp_df

    def train_test_split(self):
        """
        Function to split our transformed data into 10% tune, 60% train, 30% test

        This function randomizes a number and also the order of the 3 resulting data sets. This ensures that our fit
            is on random ordering as well. The split DataFrames are then added back as a dictionary to the ETL object.
            For the soybean data set, 10% tune would be only 4 data points so the tune and train are set to 70% vs 30%
            test.

        :return self.data_split: dict (of DataFrames), dictionary with keys (tune, train, test) referring to the split
            transformed data
        """
        # Define base data size and size of tune and train
        data_size = len(self.transformed_data)
        tune_size = int(data_size / 10)
        train_size = int(data_size * (6 / 10))

        # Check and set the random seed
        if self.random_state:
            np.random.seed(self.random_state)

        # Generate a list of the size of our data and randomly pick 60% without replacement for train
        train_splitter = np.random.choice(a=data_size, size=train_size, replace=False)

        # Determine the remaining index that weren't picked for train
        remainder = list(set(self.transformed_data.index) - set(train_splitter))

        # Sample from the remainder 10%
        tune_splitter = np.random.choice(a=remainder, size=tune_size, replace=False)

        # Determine of the remainder what was picked for tune, leaving us with 30% for test
        test_splitter = list(set(remainder) - set(tune_splitter))

        # If we are working with soybean, we need to add the train and tune together otherwise tune is too small
        # We'll use the same data set for both train and tune
        if self.data_name == 'soybean':
            train_splitter = np.concatenate([train_splitter, tune_splitter])
            tune_splitter = train_splitter

        # Update our attribute with the dictionary defining the train, tune, and test data sets
        self.data_split.update({
            'train': self.transformed_data.iloc[train_splitter],
            'tune': self.transformed_data.iloc[tune_splitter],
            'test': self.transformed_data.iloc[test_splitter]
        })

        # Update our attribute with the dictionary defining the train, tune, and test data sets for class column only
        self.class_data_split.update({
            'train': self.class_column.iloc[train_splitter],
            'tune': self.class_column.iloc[tune_splitter],
            'test': self.class_column.iloc[test_splitter]
        })

    def update_data_name(self, class_name):
        """
        Helper function for multi class

        Updates the data name to include the class name if there are multiple classes

        :param class_name: str, class name to append to the data name
        :return self.data_name: str, updated data name
        """
        self.data_name = f'{self.data_name}_{class_name}'
