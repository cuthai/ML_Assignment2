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
        if self.classes == 0:
            self.cv_split_regression()
        else:
            self.cv_split_classification()

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
            raise NameError('Please specify a predefined name for one of the 6 data sets (glass, segmentation, vote,'
                            'abalone, machine, forest-fires)')

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

        # abalone
        elif self.data_name == 'abalone':
            self.transform_abalone()

        # machine
        elif self.data_name == 'machine':
            self.transform_machine()

        # forest-fires
        elif self.data_name == 'forest-fires':
            self.transform_forest_fires()

        # The extract function should catch this but lets throw again in case
        else:
            raise NameError('Please specify a predefined name for one of the 6 data sets (glass, segmentation, vote,'
                            'abalone, machine, forest-fires)')

    def transform_glass(self):
        """
        Function to transform glass data set

        For this function all numeric data is normalized using mean/standard deviation

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
        Function to transform segmentation data set

        For this function all numeric data is normalized using mean/standard deviation

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
        normalized_temp_df.drop(columns='Class', inplace=True)
        normalized_temp_df['Class'] = temp_df['Class']

        # Set attributes for ETL object, there are two total classes so this is a singular classifier
        self.classes = 7
        self.transformed_data = normalized_temp_df

    def transform_vote(self):
        """
        Function to transform vote data set

        For this function a question mark is treated as a distinct category, since abstaining from a vote might tell us
            about the rep's party. The data is dummied based on the 3 possible values in each column.

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

        # Set the class back, the normalize above would have normalized the class as well
        binned_temp_df.drop(columns='Class', inplace=True)
        binned_temp_df['Class'] = temp_df['Class']

        # Set attributes for ETL object, there are two total classes so this is a singular classifier
        self.classes = 2
        self.transformed_data = binned_temp_df

    def transform_abalone(self):
        """
        Function to transform abalone data set

        For this function all numeric data is normalized using mean/standard deviation

        :return self.transformed_data: DataFrame, transformed data set
        :return self.classes: int, num of classes
        """
        # We'll make a deep copy of our data set
        temp_df = pd.DataFrame.copy(self.data, deep=True)

        # Normalize Data
        normalized_temp_df = (temp_df - temp_df.mean()) / temp_df.std()

        # Add Binned variables for sex
        normalized_temp_df = normalized_temp_df.join(pd.get_dummies(temp_df['Sex'], columns=['Sex']))

        # We'll remove the old binned variables and reorder our target
        normalized_temp_df.drop(columns=['Rings', 'Sex'], inplace=True)
        normalized_temp_df['Rings'] = temp_df['Rings']

        # Set attributes for ETL object, there are two total classes so this is a singular classifier
        self.transformed_data = normalized_temp_df

    def transform_machine(self):
        """
        Function to transform machine data set

        For this function all numeric data is normalized using mean/standard deviation

        :return self.transformed_data: DataFrame, transformed data set
        :return self.classes: int, num of classes
        """
        # We'll make a deep copy of our data set
        temp_df = pd.DataFrame.copy(self.data, deep=True)

        # Normalize Data
        normalized_temp_df = (temp_df - temp_df.mean()) / temp_df.std()

        # We'll remove unneeded variables as well as denormalize the target
        normalized_temp_df.drop(columns=['Vendor', 'Model_Name', 'ERP'], inplace=True)
        normalized_temp_df['PRP'] = temp_df['PRP']

        # Set attributes for ETL object, there are two total classes so this is a singular classifier
        self.transformed_data = normalized_temp_df

    def transform_forest_fires(self):
        """
        Function to transform forest-fires data set

        For this function all numeric data is normalized using mean/standard deviation

        :return self.transformed_data: DataFrame, transformed data set
        :return self.classes: int, num of classes
        """
        # We'll make a deep copy of our data set
        temp_df = pd.DataFrame.copy(self.data, deep=True)

        # Normalize Data
        normalized_temp_df = (temp_df - temp_df.mean()) / temp_df.std()

        # Add Binned variables for sex
        normalized_temp_df = normalized_temp_df.join(pd.get_dummies(temp_df[['month', 'day']],
                                                                    columns=['month', 'day']))

        # We'll remove the old binned variables and reorder our target
        normalized_temp_df.drop(columns=['month', 'day', 'area'], inplace=True)
        normalized_temp_df['area'] = temp_df['area']

        # Set attributes for ETL object, there are two total classes so this is a singular classifier
        self.transformed_data = normalized_temp_df

    def cv_split_classification(self):
        """
        Function to split our transformed data into 10% tune and 5 cross validation splits for classification

        First this function randomizes a number between one and 10 to split out a tune set. After a number is randomized
            and the data is sorted over the class and random number. The index of the data is then mod by 5 and each
            remainder represents a set for cv splitting.

        :return self.data_split: dict (of DataFrames), dictionary with keys (tune, 0, 1, 2, 3, 4) referring to the
            split transformed data
        """
        # Define base data size and size of tune
        data_size = len(self.transformed_data)
        tune_size = int(data_size / 10)

        # Check and set the random seed
        if self.random_state:
            np.random.seed(self.random_state)

        # Sample for tune
        tune_splitter = []

        # Randomize a number between 0 and 10 and multiply by the index to randomly pick observations over data set
        for index in range(tune_size):
            tune_splitter.append(np.random.choice(a=10) + (10 * index))
        self.data_split.update({
            'tune': self.transformed_data.iloc[tune_splitter]
        })

        # Determine the remaining index that weren't picked for tune
        remainder = list(set(self.transformed_data.index) - set(tune_splitter))
        remainder_df = pd.DataFrame(self.transformed_data.iloc[remainder]['Class'])

        # Assign a random number
        remainder_df['Random_Number'] = np.random.randint(0, len(remainder), remainder_df.shape[0])

        # Sort over class and the random number
        remainder_df.sort_values(by=['Class', 'Random_Number'], inplace=True)
        remainder_df.reset_index(inplace=True)

        # Sample for CV
        for index in range(5):
            # Mod the index by 5 and there will be 5 remainder groups for the CV split
            splitter = remainder_df.loc[remainder_df.index % 5 == index]['index']

            # Update our attribute with the dictionary for this index
            self.data_split.update({
                index: self.transformed_data.iloc[splitter]
            })

    def cv_split_regression(self):
        """
        Function to split our transformed data into 10% tune and 5 cross validation splits for regression

        First this function splits out a tune set. The remainder is sampled 5 times to produce 5 cv splits.

        :return self.data_split: dict (of DataFrames), dictionary with keys (tune, 0, 1, 2, 3, 4) referring to the
            split transformed data
        """
        # Define base data size and size of tune and splits
        data_size = len(self.transformed_data)
        tune_size = int(data_size / 10)
        cv_size = int((data_size - tune_size) / 5)

        # The extra data will go to the first splits. The remainder of the length divided by 5 defines how many extra
        extra_data = int((data_size - tune_size) % 5)

        # Check and set the random seed
        if self.random_state:
            np.random.seed(self.random_state)

        # Sample for tune
        tune_splitter = np.random.choice(a=data_size, size=tune_size, replace=False)
        self.data_split.update({
            'tune': self.transformed_data.iloc[tune_splitter]
        })

        # Determine the remaining index that weren't picked for tune
        remainder = list(set(self.transformed_data.index) - set(tune_splitter))

        # CV Split
        for index in range(5):
            # For whatever number of extra data, we'll add one to each of those index
            if (index + 1) <= extra_data:
                # Sample for the CV size if extra data
                splitter = np.random.choice(a=remainder, size=(cv_size + 1), replace=False)

            else:
                # Sample for the CV size
                splitter = np.random.choice(a=remainder, size=cv_size, replace=False)

            # Define the remaining unsampled data points
            remainder = list(set(remainder) - set(splitter))

            # Update our attribute with the dictionary for this index
            self.data_split.update({
                index: self.transformed_data.iloc[splitter]
            })
