import pandas as pd
import numpy as np
import copy
import json
import matplotlib.pyplot as plt


class KNNRegressor:
    def __init__(self, etl, knn_type):
        self.etl = etl
        self.data_name = self.etl.data_name
        self.data_split = etl.data_split

        self.knn_type = 'regular'

        if knn_type:
            self.knn_type = knn_type

        self.train_data = {}

        self.tune_results = {}
        self.k = 1
        self.sigma = 1
        self.epsilon = .05

        self.test_results = {}

        self.summary = {}
        self.summary_prediction = None

    def fit(self):
        for index in range(5):
            train_index = [train_index for train_index in [0, 1, 2, 3, 4] if train_index != index]

            train_data = pd.DataFrame()
            for data_split_index in train_index:
                train_data = train_data.append(self.data_split[data_split_index])

            self.train_data.update({index: train_data})

    def tune(self, k_range=None, k=None, sigma_range=None, sigma=None):
        self.tune_k(k_range, sigma)

        self.tune_sigma(k, sigma_range)

    def tune_k(self, k_range=None, sigma=None):
        if not k_range:
            k_range = list(range(3, 22, 2))

        if not sigma:
            sigma = self.sigma

        tune_data = self.data_split['tune']
        tune_x = tune_data.iloc[:, :-1]
        tune_y = tune_data.iloc[:, -1]

        tune_results = {'k': {k: [0, 0, 0, 0, 0] for k in k_range}}

        for index in range(5):
            train_data = self.train_data[index]
            train_x = train_data.iloc[:, :-1]
            train_y = train_data.iloc[:, -1]

            for row_index, row in tune_x.iterrows():
                distances = ((train_x - row) ** 2).sum(axis=1).sort_values()

                for k in tune_results['k'].keys():
                    neighbors = distances[:k]

                    kernel = np.exp((1 / (2 * sigma)) * neighbors)
                    neighbors_r = train_y.loc[neighbors.index.to_list()]

                    prediction = sum(kernel * neighbors_r) / sum(kernel)
                    actual = tune_y.loc[row_index]

                    tune_results['k'][k][index] += (actual - prediction) ** 2

        for k in tune_results['k'].keys():
            tune_results['k'][k] = sum(tune_results['k'][k]) / (len(tune_data) * 5)

        self.tune_results.update(tune_results)
        self.k = min(tune_results['k'], key=tune_results['k'].get)

    def tune_sigma(self, k=None, sigma_range=None):
        if not k:
            k = self.k

        if not sigma_range:
            sigma_range = np.linspace(.5, 3, 6)

        tune_data = self.data_split['tune']
        tune_x = tune_data.iloc[:, :-1]
        tune_y = tune_data.iloc[:, -1]

        tune_results = {'sigma': {sigma: [0, 0, 0, 0, 0] for sigma in sigma_range}}

        for index in range(5):
            train_data = self.train_data[index]
            train_x = train_data.iloc[:, :-1]
            train_y = train_data.iloc[:, -1]

            for row_index, row in tune_x.iterrows():
                distances = ((train_x - row) ** 2).sum(axis=1).sort_values()

                for sigma in tune_results['sigma'].keys():
                    neighbors = distances[:k]

                    kernel = np.exp((1 / (2 * sigma)) * neighbors)
                    neighbors_r = train_y.loc[neighbors.index.to_list()]

                    prediction = sum(kernel * neighbors_r) / sum(kernel)
                    actual = tune_y.loc[row_index]

                    tune_results['sigma'][sigma][index] += (actual - prediction) ** 2

        for sigma in tune_results['sigma'].keys():
            tune_results['sigma'][sigma] = sum(tune_results['sigma'][sigma]) / (len(tune_data) * 5)

        self.tune_results.update(tune_results)
        self.sigma = min(tune_results['sigma'], key=tune_results['sigma'].get)

    def fit_modified(self, epsilon_range=None):
        if not epsilon_range:
            epsilon_range = [.5, .25, .1, .05, .01]

        min_mse = 1000000
        temp_train_data = {epsilon: {} for epsilon in epsilon_range}
        min_train_data = None

        self.tune_results['epsilon'] = {}

        for epsilon in epsilon_range:
            epsilon_mse = 0

            for index in range(5):
                if self.knn_type == 'edited':
                    train_data, mse = self.edit(copy.deepcopy(self.train_data[index]), epsilon=epsilon)

                    epsilon_mse += mse
                    temp_train_data[epsilon].update({index: train_data})

                else:
                    train_data, mse = self.condense(copy.deepcopy(self.train_data[index]), epsilon=epsilon)

                    epsilon_mse += mse
                    temp_train_data[epsilon].update({index: train_data})

            average_mse = epsilon_mse / 5

            if average_mse < min_mse:
                min_mse = average_mse
                min_train_data = temp_train_data[epsilon]
                self.epsilon = epsilon

            self.tune_results['epsilon'].update({epsilon: average_mse})

        self.train_data = min_train_data

    def edit(self, temp_train_data, k=None, sigma=None, epsilon=None):
        if len(temp_train_data) <= 1:
            return temp_train_data, self.tune_epsilon(temp_train_data)

        if not k:
            k = self.k

        if not sigma:
            sigma = self.sigma

        if not epsilon:
            epsilon = self.epsilon

        train_data = temp_train_data
        train_x = train_data.iloc[:, :-1]
        train_y = train_data.iloc[:, -1]

        edit_out_list = []

        for row_index, row in train_data.iterrows():
            distances = ((train_x - row) ** 2).sum(axis=1).sort_values()[1:]

            neighbors = distances[:k]

            kernel = np.exp((1 / (2 * sigma)) * neighbors)
            neighbors_r = train_y.loc[neighbors.index.to_list()]

            prediction = sum(kernel * neighbors_r) / sum(kernel)
            actual = train_y.loc[row_index]

            if actual == 0:
                percent_different = 1
            else:
                percent_different = abs((prediction - actual) / actual)

            if percent_different > epsilon:
                edit_out_list.append(row_index)

        train_data = train_data.loc[~train_data.index.isin(edit_out_list)]

        if len(train_data) == 0:
            return train_data, 1000001
        if len(edit_out_list) > 0:
            train_data, mse = self.edit(train_data, epsilon=epsilon)
        else:
            mse = self.tune_epsilon(train_data)

        return train_data, mse

    def condense(self, temp_train_data, sigma=None, epsilon=None, z_data=None):
        if not sigma:
            sigma = self.sigma

        if not epsilon:
            epsilon = self.epsilon

        if z_data is None:
            z_data = pd.DataFrame(temp_train_data.iloc[0, :]).T
        z_data_x = z_data.iloc[:, :-1]
        z_data_y = z_data.iloc[:, -1]

        condense_in_count = 0

        temp_train_data = temp_train_data.loc[~temp_train_data.index.isin(z_data.index)]

        for row_index, row in temp_train_data.iterrows():
            distances = ((z_data_x - row) ** 2).sum(axis=1).sort_values()

            neighbors = distances[:1]

            kernel = np.exp((1 / (2 * sigma)) * neighbors)
            neighbors_r = z_data_y.loc[neighbors.index.to_list()]

            prediction = sum(kernel * neighbors_r) / sum(kernel)
            actual = temp_train_data.loc[row_index][-1]

            if actual == 0:
                if prediction > 0:
                    percent_different = 1
                else:
                    percent_different = 0
            else:
                percent_different = abs((prediction - actual) / actual)

            if percent_different > epsilon:
                condense_in_count += 1
                z_data = z_data.append(temp_train_data.loc[row_index])
                z_data_x = z_data.iloc[:, :-1]
                z_data_y = z_data.iloc[:, -1]

        if condense_in_count > 0:
            z_data, mse = self.condense(temp_train_data, epsilon=epsilon, z_data=z_data)
        else:
            mse = self.tune_epsilon(z_data)

        return z_data, mse

    def tune_epsilon(self, train_data, k=None, sigma=None):
        if not k:
            k = self.k

        if not sigma:
            sigma = self.sigma

        tune_data = self.data_split['tune']
        tune_x = tune_data.iloc[:, :-1]
        tune_y = tune_data.iloc[:, -1]

        train_x = train_data.iloc[:, :-1]
        train_y = train_data.iloc[:, -1]

        error = 0

        for row_index, row in tune_x.iterrows():
            distances = ((train_x - row) ** 2).sum(axis=1).sort_values()[:k]

            neighbors = distances[:k]

            kernel = np.exp((1 / (2 * sigma)) * neighbors)
            neighbors_r = train_y.loc[neighbors.index.to_list()]

            prediction = sum(kernel * neighbors_r) / sum(kernel)
            actual = tune_y.loc[row_index]

            error += (actual - prediction) ** 2

        mse = error / len(tune_data)

        return mse

    def predict(self, k=None, sigma=None):
        if not k:
            k = self.k

        if not sigma:
            sigma = self.sigma

        test_results = {
            index: {
                'mse': 0,
                'prediction': []
            } for index in range(5)
        }

        for index in range(5):
            train_data = self.train_data[index]
            train_x = train_data.iloc[:, :-1]
            train_y = train_data.iloc[:, -1]

            test_data = self.data_split[index]
            test_x = test_data.iloc[:, :-1]
            test_y = test_data.iloc[:, -1]

            for row_index, row in test_x.iterrows():
                distances = ((train_x - row) ** 2).sum(axis=1).sort_values()

                neighbors = distances[:k]

                kernel = np.exp((1 / (2 * sigma)) * neighbors)
                neighbors_r = train_y.loc[neighbors.index.to_list()]

                prediction = sum(kernel * neighbors_r) / sum(kernel)
                actual = test_y.loc[row_index]

                test_results[index]['mse'] += (actual - prediction) ** 2

                test_results[index]['prediction'].append(prediction)

            test_results[index]['mse'] = test_results[index]['mse'] / len(test_data)

        self.test_results = test_results

    def output(self):
        mse = sum([self.test_results[index]['mse'] for index in range(5)])

        self.summary = {
            'tune': {
                'k': self.k,
                'sigma': self.sigma
            },
            'test': {
                'mse': mse / 5
            }
        }

        if self.knn_type != 'regular':
            self.summary['tune'].update({'epsilon': self.epsilon})

        with open(f'output_{self.data_name}\\{self.data_name}_{self.knn_type}_summary.json', 'w') as file:
            json.dump(self.summary, file)

        summary_prediction = pd.DataFrame()
        for index in range(5):
            temp_summary_prediction = self.data_split[index]
            temp_summary_prediction['prediction'] = self.test_results[index]['prediction']
            summary_prediction = summary_prediction.append(temp_summary_prediction)

        summary_prediction.to_csv(f'output_{self.data_name}\\{self.data_name}_{self.knn_type}_prediction.csv')
        self.summary_prediction = summary_prediction

    def visualize_tune(self):
        """
        Tune visualization function

        This function uses the results of the tune function to create a plot graph

        :return: matplotlib saved jpg in output folder
        """
        tune_list = ['k', 'sigma']

        if self.knn_type != 'regular':
            tune_list.append('epsilon')

        for parameter in tune_list:
            if parameter == 'k':
                optimal = self.k
            elif parameter == 'sigma':
                optimal = self.sigma
            else:
                optimal = self.epsilon

            # Figure / axis set up
            fig, ax = plt.subplots()

            # We'll plot the list of params and their accuracy
            ax.plot(self.tune_results[parameter].keys(), self.tune_results[parameter].values())

            # Title
            ax.set_title(rf'{self.data_name} {parameter} Tune Results - Optimal: {parameter} {optimal}')

            # X axis
            ax.set_xlabel(parameter)
            ax.set_xticks(list(self.tune_results[parameter].keys()))
            ax.set_xticklabels(list(self.tune_results[parameter].keys()), rotation=45, fontsize=6)

            # Y axis
            ax.set_ylabel('MSE')

            # Saving
            if parameter == 'epsilon':
                plt.savefig(f'output_{self.data_name}\\{self.data_name}_{self.knn_type}_{parameter}_tune.jpg')
            else:
                plt.savefig(f'output_{self.data_name}\\{self.data_name}_{parameter}_tune.jpg')
