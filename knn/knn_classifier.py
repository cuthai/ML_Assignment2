import pandas as pd


class KNNClassifier:
    def __init__(self, etl):
        self.etl = etl
        self.data_split = etl.data_split

        self.tune_results = {}
        self.k = 1

    def tune(self):
        tune_data = self.data_split['tune']
        tune_x = tune_data.iloc[:, :-1]

        tune_results = {k: [0, 0, 0, 0, 0] for k in range(1, 22, 2)}

        for index in range(5):
            test_index = [test_index for test_index in [0, 1, 2, 3, 4] if test_index != index]

            test_data = pd.DataFrame()
            for data_split_index in test_index:
                test_data = test_data.append(self.data_split[data_split_index])
            test_x = test_data.iloc[:, :-1]

            for tune_row_index, row in tune_x.iterrows():
                distances = ((test_x - row) ** 2).sum(axis=1).sort_values()

                for k in tune_results.keys():
                    neighbors = distances[:k].index.to_list()
                    classes = test_data.loc[neighbors, 'Class']

                    class_occurrence = classes.mode()
                    if len(class_occurrence) > 1:
                        classification = test_data.loc[neighbors[0], 'Class']
                    else:
                        classification = class_occurrence[0]

                    if classification != tune_data.loc[tune_row_index, 'Class']:
                        tune_results[k][index] += 1

        for k in tune_results.keys():
            tune_results[k] = sum(tune_results[k]) / (len(tune_data) * 5)

        self.tune_results = tune_results
        self.k = min(tune_results, key=tune_results.get)
