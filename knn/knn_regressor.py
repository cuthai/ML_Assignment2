import pandas as pd
import numpy as np
import math


class KNNRegressor:
    def __init__(self, etl, knn_type):
        self.etl = etl
        self.data_split = etl.data_split
        self.knn_type = knn_type

        self.train_data = {}

        self.tune_results = {}
        self.k = 1

        self.test_results = {}

    def fit(self):
        for index in range(1, 6):
            train_index = [train_index for train_index in [1, 2, 3, 4, 5] if train_index != index]

            train_data = pd.DataFrame()
            for data_split_index in train_index:
                train_data = train_data.append(self.data_split[data_split_index])

            self.train_data.update({index: train_data})

    def tune(self, k_range=None, sigma_range=None):
        if not k_range:
            k_range = list(range(3, 22, 2))

        if not sigma_range:
            sigma_range = np.linspace(.5, 3, 6)

        tune_data = self.data_split['tune']
        tune_x = tune_data.iloc[:, :-1]
        tune_y = tune_data.iloc[:, -1]

        tune_results = {k: [0, 0, 0, 0, 0] for k in k_range}

        for index in range(1, 6):
            train_data = self.train_data[index]
            train_x = train_data.iloc[:, :-1]
            train_y = train_data.iloc[:, -1]

            for row_index, row in tune_x.iterrows():
                distances = ((train_x - row) ** 2).sum(axis=1).sort_values()
                kernel = distances.apply(lambda row_distance: math.exp((1 / 2 * sigma_range[0]) * row_distance))

                for k in tune_results.keys():
                    neighbors = distances[:k].index.to_list()
                    neighbors_kernel = kernel.loc[neighbors]
                    neighbors_r = train_y.loc[neighbors]

                    prediction = sum(neighbors_kernel * neighbors_r) / sum(neighbors_kernel)
                    actual = tune_y.loc[row_index]

                    tune_results[k][index - 1] += (actual - prediction) ** 2

        for k in tune_results.keys():
            tune_results[k] = sum(tune_results[k]) / (len(tune_data) * 5)

        self.tune_results = tune_results
        self.k = min(tune_results, key=tune_results.get)

    def fit_modified(self):
        import datetime

        for index in range(5):
            print(index)
            print(datetime.datetime.today())
            if self.knn_type == 'edited':
                self.edit(index)
            else:
                self.condense(index)
            print(datetime.datetime.today())

    def edit(self, index, k=None):
        if not k:
            k = self.k

        train_data = self.train_data[index]
        train_x = train_data.iloc[:, :-1]

        edit_out_list = []

        for row_index, row in train_data.iterrows():
            distances = ((train_x - row) ** 2).sum(axis=1).sort_values()[1:]

            neighbors = distances[:k].index.to_list()
            classes = train_data.loc[neighbors, 'Class']

            class_occurrence = classes.mode()
            if len(class_occurrence) > 1:
                classification = train_data.loc[neighbors[0], 'Class']
            else:
                classification = class_occurrence[0]

            if classification != train_data.loc[row_index, 'Class']:
                edit_out_list.append(row_index)

        train_data = train_data.loc[~train_data.index.isin(edit_out_list)]

        self.train_data.update({index: train_data})

        if len(edit_out_list) > 0:
            self.edit(index)

    def condense(self, index, z_data=None):
        temp_train_data = self.train_data[index]

        if z_data is None:
            z_data = pd.DataFrame(temp_train_data.iloc[0, :]).T
        z_data_x = z_data.iloc[:, :-1]

        condense_in_count = 0

        for row_index, row in temp_train_data.iterrows():
            distances = ((z_data_x - row) ** 2).sum(axis=1).sort_values()

            neighbor = distances.index.to_list()[0]
            classification = z_data.loc[neighbor, 'Class']

            if classification != temp_train_data.loc[row_index, 'Class']:
                condense_in_count += 1
                z_data = z_data.append(temp_train_data.loc[row_index])
                z_data_x = z_data.iloc[:, :-1]

        if condense_in_count > 0:
            self.condense(index, z_data)
        else:
            self.train_data.update({index: z_data})

    def predict(self, k=None):
        if not k:
            k = self.k

        test_results = {index: 0 for index in range(5)}
        test_classification = pd.DataFrame()

        for index in range(5):
            train_data = self.train_data[index]
            train_x = train_data.iloc[:, :-1]

            test_data = self.data_split[index]
            test_x = test_data.iloc[:, :-1]

            for test_row_index, row in test_x.iterrows():
                distances = ((train_x - row) ** 2).sum(axis=1).sort_values()

                neighbors = distances[:k].index.to_list()
                classes = train_data.loc[neighbors, 'Class']

                class_occurrence = classes.mode()
                if len(class_occurrence) > 1:
                    classification = train_data.loc[neighbors[0], 'Class']
                else:
                    classification = class_occurrence[0]

                if classification != test_data.loc[test_row_index, 'Class']:
                    test_results[index] += 1

            test_results[index] = test_results[index] / len(test_data)

        self.test_results = test_results
