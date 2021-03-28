from tensorflow.keras import layers, Sequential, losses
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from algothon2021ml import ML
import datetime as dt



class Windowing:
    def __init__(self, dataframe, window_size=10, ratio=0.9):
        self.data = {}
        self.ratio = ratio
        self.window_size = window_size
        self.dataframe = dataframe

    def generate_data_sets(self):
        current_label = 'bond_1'
        return_arr = []
        time_arr = []
        date_arr = []
        for arr in self.dataframe.itertuples():
            if current_label == arr[3]:
                return_arr.append(arr[4])
                time_arr.append(arr[2])
                date_arr.append(arr[1])

            if current_label != arr[3]:
                print(len(date_arr))
                self.data[current_label] = {'date':date_arr, 'time':time_arr, 'return':return_arr}
                current_label = arr[3]
                return_arr = [arr[4]]
                time_arr = [arr[2]]
                date_arr = [arr[1]]
        self.data[current_label] = {'date':date_arr, 'time':time_arr, 'return':return_arr}


    def plot_data(self):
        for key, val in self.data.items():
            plt.plot(val['return'])
            plt.show()
    def analyse_data(self):
        for key, val in self.data.items():
            print(len(val['return']))


    def generate_slices(self, lengths=[25]):
        slice_dict = {}
        label_dict = {}
        opened = False
        previous_key = ''
        for i, (key, val) in enumerate(self.data.items()):
            print(f'Working on {key}')
            labels = val['return']
            dates = val['date']
            if key != previous_key:
                print(key)
                if key == 'bond_1' or key == 'bond_2':
                    print('reading bond data')
                    prices = pd.read_csv('Machine Learning Data/bond-prices.csv').fillna(0)
                    volumes = pd.read_csv('Machine Learning Data/bond-volumes.csv').fillna(0)

                if key == 'commodity_1' or key == 'commodity_2':
                    print('reading commodity data')
                    prices = pd.read_csv('Machine Learning Data/commodity-prices.csv').fillna(0)
                    volumes = pd.read_csv('Machine Learning Data/commodity-volumes.csv').fillna(0)
                if key == 'currency_1' or key == 'currency_2':
                    print('reading currency data')
                    prices = pd.read_csv('Machine Learning Data/currency-prices.csv').fillna(0)
                    volumes = pd.read_csv('Machine Learning Data/currency-volumes.csv').fillna(0)
                if key == 'stock_1' or key == 'stock_2':
                    print('reading stock data')
                    prices = pd.read_csv('Machine Learning Data/stock-prices.csv').fillna(0)
                    volumes = pd.read_csv('Machine Learning Data/stock-volume.csv').fillna(0)
                previous_key = key
            volume_arr = []
            price_arr = []
            prices['Date'] = pd.to_datetime(prices['Date'], format='%Y-%m-%d')
            prices_dates = prices['Date'].to_list()
            for date in dates:
                dateobj = dt.datetime.strptime(date, '%Y-%m-%d').date()

                while True:
                    try:
                        ind = prices_dates.index(dateobj)
                        break
                    except ValueError:
                        dateobj = dateobj - dt.timedelta(days=1)
                price_slice = prices.iloc[max(ind - self.window_size, 0):ind].dropna(axis=1)
                del price_slice['Date']
                volume_slice = volumes.iloc[max(ind - self.window_size, 0):ind].dropna(axis=1)
                del volume_slice['Date']
                price_slice = price_slice.to_numpy()
                volume_slice = volume_slice.to_numpy()
                volume_arr.append(volume_slice)
                price_arr.append(price_slice)
            print(np.array(price_arr).shape)
            print(np.array(volume_arr).shape)
            slice_dict[key] = {'prices': np.array(price_arr), 'volumes': np.array(volume_arr)}
            label_dict[key] = labels
        return slice_dict, label_dict

    def split_data(self, slice_dict, label_dict):
        model_size = {}
        training_data, training_labels = {}, {}
        validation_data, validation_labels = {}, {}
        for key, val in slice_dict.items():
            ind = int(len(label_dict[key]) * self.ratio)
            price_arr = val['prices']
            print(price_arr.shape)
            volume_arr = val['volumes']
            labels = label_dict[key]
            label_arr = []
            for label in labels:
                if label > 0:
                    label_arr.append(0)
                elif label < 0:
                    label_arr.append(1)
                else:
                    label_arr.append(2)

            model_size[key] = price_arr[0].shape
            print(model_size[key])
            print(np.array(price_arr[:ind]).shape)
            train_data = [[[[pv, vv] for pv, vv in zip(pa, pa)] for pa, va in zip(p, v)] for p, v in zip(price_arr[:ind], volume_arr[:ind])]
            print(f'Train Shape: {np.array(train_data).shape}')
            train_labels = label_arr[:ind]
            print(f'Train Labels Shape: {np.array(train_labels).shape}')
            print(np.array(price_arr[ind:]).shape)
            val_data = [[[[pv, vv] for pv, vv in zip(pa, pa)] for pa, va in zip(p, v)] for p, v in zip(price_arr[ind:], volume_arr[ind:])]

            val_labels = label_arr[ind:]
            print(f'Validation Labels Shape: {np.array(val_labels).shape}')
            training_data[key] = train_data
            training_labels[key] = train_labels
            validation_data[key] = val_data
            validation_labels[key] = val_labels
        print(model_size)

        return training_data, training_labels, validation_data, validation_labels, model_size


class LSTM(ML):
    def __init__(self, training_data, training_labels, valdiation_data, validation_labels, model_size, **parms):
        super(LSTM, self).__init__(**parms)
        self.model_size = model_size
        self.training_data = training_data
        self.training_labels = training_labels
        self.validation_data = valdiation_data
        self.validation_labels = validation_labels
        self.previous_key = ''
        self.window_size = self.model_size['bond_1'][0]
        self.models = {k:self.create_model(v[0], v[1]) for (k, v) in self.model_size.items()}

    def create_model(self, window_length, feature_size, data_points=2):
        model = Sequential()
        model.add(layers.Conv2D(64, kernel_size=(window_length, 1), activation='relu', input_shape=(window_length, feature_size, data_points)))
        model.add(layers.Flatten())
        model.add(layers.Dense(512, activation='relu'))
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dense(3, activation='linear'))
        model.summary()
        return model

    def predict(self, data):
        model = self.models[self.assetid]
        model.load_weights(f'{self.assetid}-dense.h5')
        date = data['Trade_date'].values[0]
        if self.assetid != self.previous_key:
            print('loading_weights')
            model.summary()
            self.previous_key = self.assetid
            if self.assetid == 'bond_1' or self.assetid == 'bond_2':
                print('reading bond data')
                self.prices = pd.read_csv('Machine Learning Data/bond-prices.csv').fillna(0)
                self.volumes = pd.read_csv('Machine Learning Data/bond-volumes.csv').fillna(0)

            if self.assetid == 'commodity_1' or self.assetid == 'commodity_2':
                print('reading commodity data')
                self.prices = pd.read_csv('Machine Learning Data/commodity-prices.csv').fillna(0)
                self.volumes = pd.read_csv('Machine Learning Data/commodity-volumes.csv').fillna(0)
            if self.assetid == 'currency_1' or self.assetid == 'currency_2':
                print('reading currency data')
                self.prices = pd.read_csv('Machine Learning Data/currency-prices.csv').fillna(0)
                self.volumes = pd.read_csv('Machine Learning Data/currency-volumes.csv').fillna(0)
            if self.assetid == 'stock_1' or self.assetid == 'stock_2':
                print('reading stock data')
                self.prices = pd.read_csv('Machine Learning Data/stock-prices.csv').fillna(0)
                self.volumes = pd.read_csv('Machine Learning Data/stock-volume.csv').fillna(0)
        self.prices['Date'] = pd.to_datetime(self.prices['Date'], format='%Y-%m-%d')
        prices_dates = self.prices['Date'].to_list()
        dateobj = dt.datetime.strptime(date, '%Y-%m-%d').date()
        while True:
            try:
                ind = prices_dates.index(dateobj)
                break
            except ValueError:
                dateobj = dateobj - dt.timedelta(days=1)
        price_slice = self.prices.iloc[max(ind - self.window_size, 0):ind].dropna(axis=1)
        del price_slice['Date']
        volume_slice = self.volumes.iloc[max(ind - self.window_size, 0):ind].dropna(axis=1)
        del volume_slice['Date']
        price_slice = price_slice.to_numpy()
        volume_slice = volume_slice.to_numpy()
        pv_arr = [[[[pv, vv] for pv, vv in zip(p, v)] for p, v in zip(price_slice, volume_slice)]]
        out = model.predict(pv_arr)
        lookup = [1, -1, 0]
        return lookup[np.argmax(out)]


    def train_model(self):
        for key in self.training_data.keys():
            train_data = self.training_data[key]
            train_labels = self.training_labels[key]
            val_data = self.validation_data[key]
            val_labels = self.validation_labels[key]
            for _ in range(2):
                train_data.extend(self.training_data[key])
                train_labels.extend(self.training_labels[key])
                val_data.extend(self.validation_data[key])
                val_labels.extend(self.validation_labels[key])
            model = self.models[key]
            model.compile(optimizer='adam', loss=losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
            model.fit(np.array(train_data), np.array(train_labels), epochs=30,
                      validation_data=(np.array(val_data), np.array(val_labels)))
            model.save_weights(f'{key}-dense.h5')








if __name__ == "__main__":
    window = Windowing(pd.read_csv('ml_train.csv'))
    window.generate_data_sets()
    slice_dict, label_dict = window.generate_slices()
    training_data, training_labels, validation_data, validation_labels, model_size = window.split_data(slice_dict, label_dict)

    lstm = LSTM(training_data, training_labels, validation_data, validation_labels, model_size)
    lstm.train_model()
    validation = lstm.run_validate()
    test = lstm.run_test()