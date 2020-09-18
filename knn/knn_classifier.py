import pandas as pd


class KNNClassifier:
    def __init__(self, etl, knn_type):
        self.etl = etl
        self.data_split = etl.data_split
        self.knn_type = knn_type

        self.train_data = {}

        self.tune_results = {}
        self.k = 1

        self.test_results = {}

    def tune(self, k_range=None):
        if not k_range:
            k_range = list(range(3, 22, 2))

        tune_data = self.data_split['tune']
        tune_x = tune_data.iloc[:, :-1]

        tune_results = {k: [0, 0, 0, 0, 0] for k in k_range}

        for index in range(5):
            train_data = self.train_data[index]
            train_x = train_data.iloc[:, :-1]

            for tune_row_index, row in tune_x.iterrows():
                distances = ((train_x - row) ** 2).sum(axis=1).sort_values()

                for k in tune_results.keys():
                    neighbors = distances[:k].index.to_list()
                    classes = train_data.loc[neighbors, 'Class']

                    class_occurrence = classes.mode()
                    if len(class_occurrence) > 1:
                        classification = train_data.loc[neighbors[0], 'Class']
                    else:
                        classification = class_occurrence[0]

                    if classification != tune_data.loc[tune_row_index, 'Class']:
                        tune_results[k][index] += 1

        for k in tune_results.keys():
            tune_results[k] = sum(tune_results[k]) / (len(tune_data) * 5)

        self.tune_results = tune_results
        self.k = min(tune_results, key=tune_results.get)

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

    def fit(self):
        if self.knn_type == 'edited':
            self.fit_edited()
        else:
            self.fit_regular()

    def fit_regular(self):
        for index in range(5):
            train_index = [train_index for train_index in [0, 1, 2, 3, 4] if train_index != index]

            train_data = pd.DataFrame()
            for data_split_index in train_index:
                train_data = train_data.append(self.data_split[data_split_index])

            self.train_data.update({index: train_data})

    def fit_edited(self):
        pass
