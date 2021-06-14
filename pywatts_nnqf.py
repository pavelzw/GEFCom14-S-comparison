from itertools import product

import numpy as np
import pandas
import pandas as pd
import xarray as xr

from nnqf import nnqf_filter
from pywatts.callbacks import CSVCallback

from pywatts.core.pipeline import Pipeline
from pywatts.modules import Sampler
from pywatts.wrapper import FunctionModule, SKLearnWrapper
from sklearn.ensemble import BaggingRegressor
from sklearn.neural_network import MLPRegressor


def load_data():
    predictors = pd.read_csv(f'data/Task {task}/predictors{task}.csv', parse_dates=['TIMESTAMP']) \
        [['ZONEID', 'TIMESTAMP', 'VAR164', 'VAR169', 'VAR175', 'VAR178', 'VAR228']].set_index('TIMESTAMP')
    train = pd.read_csv(f'data/Task {task + 1}/train{task + 1}.csv', parse_dates=['TIMESTAMP']).set_index('TIMESTAMP')
    benchmark = pd.read_csv(f"data/Task {task}/benchmark{'0' + str(task) if task < 10 else task}.csv",
                            parse_dates=['TIMESTAMP']).set_index('TIMESTAMP')

    solar_plants = [train[train['ZONEID'] == i][['POWER']].rename({'POWER': f'ZONEID {i}'}, axis='columns')
                    for i in [1, 2, 3]]
    train_data = pd.concat(solar_plants, axis=1)

    predictors_categories = [
        predictors[predictors['ZONEID'] == i][['VAR169', 'VAR175', 'VAR178', 'VAR228', 'VAR164']]
            .rename({'VAR169': f'SSRD {i}',
                     'VAR175': f'STRD {i}',
                     'VAR178': f'TSR {i}',
                     'VAR228': f'TP {i}',
                     'VAR164': f'TCC {i}'
                     }, axis='columns')
        for i in [1, 2, 3]]
    predictor_data = pd.concat(predictors_categories, axis=1)

    # decumulate data
    for name, zoneid in product(['SSRD', 'STRD', 'TSR'], range(1, 4)):
        subtract = predictor_data[f'{name} {zoneid}'].copy()
        subtract.iloc[1:] = subtract[:-1]
        subtract.iloc[::24] = 0

        predictor_data[f'{name} {zoneid}'] -= subtract

    gefcom14_metadata = {
        'num_series': 3,
        'num_steps': len(predictor_data),
        'prediction_length': len(benchmark) // 3,
        'start': predictor_data.index[0]
    }

    for i in range(1, 4):
        predictor_data[f'SSRD {i}'] /= predictor_data[f'SSRD {i}'].max()
        predictor_data[f'STRD {i}'] /= predictor_data[f'STRD {i}'].max()
        predictor_data[f'TSR {i}'] /= predictor_data[f'TSR {i}'].max()

    predictor_zones = []
    train_data_zones = []

    for i in range(1, 4):
        predictor = predictor_data[[f'SSRD {i}',
                                    f'STRD {i}',
                                    f'TSR {i}',
                                    f'TP {i}',
                                    f'TCC {i}']]
        predictor_zones.append(predictor)
        train_data_zone = train_data[[f'ZONEID {i}']]
        train_data_zones.append(train_data_zone)

    return predictor_zones, train_data_zones, gefcom14_metadata


def nnqf_transform(x_input, y_output):
    yq_output = nnqf_filter(x_input=np.array(x_input), y_output=np.array(y_output),
                            q_quantile=[p / 100 for p in range(1, 100)],
                            num_neighbors=100).T

    return xr.DataArray(yq_output, indexes=y_output.indexes)


def remove_quantile_crossing(quantiles):
    prediction = np.maximum(quantiles, 0)
    prediction.values.sort()
    prediction = prediction.rename({'time': 'TIMESTAMP', 'dim_0': 'quantiles'})
    prediction.coords['quantiles'] = [str(x / 100) for x in range(1, 100)]
    return prediction


def pinball_loss(actual, prediction):
    actual = np.array(actual)
    prediction = np.array(prediction)

    percentiles = np.empty((actual.shape[0], 99))
    for i in range(1, 100):
        percentiles[:, i - 1] = i
    loss = np.where(actual < prediction,
                    (1 - percentiles / 100) * (prediction - actual),
                    percentiles / 100 * (actual - prediction))
    return xr.DataArray(np.array([np.mean(loss)])).rename({'dim_0': 'pinball loss'})


if __name__ == '__main__':
    results = pd.DataFrame(columns=range(4, 16), index=['loss'])

    for task in range(4, 16):
        losses = {}
        for zone in range(1, 4):
            print(f'Task {task} Zone {zone}')

            predictor_zones, train_data_zones, gefcom14_metadata = load_data()
            x_input = predictor_zones[zone - 1]
            y_output = train_data_zones[zone - 1]

            data_train = {'x_input': xr.DataArray(x_input[:-gefcom14_metadata['prediction_length']]),
                          'y_output': xr.DataArray(y_output[:-gefcom14_metadata['prediction_length']])}
            data_test = {'x_input': xr.DataArray(x_input[-gefcom14_metadata['prediction_length']:]),
                         'y_output': xr.DataArray(y_output[-gefcom14_metadata['prediction_length']:])}

            pipeline_train = Pipeline(path='results/nnqf/train')

            nnqf_output = FunctionModule(name='NNQF transformer', transform_method=nnqf_transform)(
                x_input=pipeline_train['x_input'], y_output=pipeline_train['y_output']
            )

            x_horizon_sampler = Sampler(sample_size=24)
            x_horizon_train = x_horizon_sampler(
                x=pipeline_train['x_input']
            )

            neural_network = SKLearnWrapper(module=BaggingRegressor(
                    base_estimator=MLPRegressor(
                        hidden_layer_sizes=(100,)),
                    n_estimators=1),
                name='neural network')
            nn_output_train = neural_network(
                x=x_horizon_train, target=nnqf_output
            )

            print('Training NNQF model...')
            pipeline_train.train(data_train)
            print('Training finished.')

            # testing
            pipeline_test = Pipeline(path='results/nnqf/test')
            x_horizon_test = x_horizon_sampler(
                x=pipeline_test['x_input']
            )
            nn_output_test = neural_network(
                x=x_horizon_test
            )
            post_processed = FunctionModule(name='prediction', transform_method=remove_quantile_crossing)(
                quantiles=nn_output_test,
                callbacks=[CSVCallback(f'task{task} zone{zone}')]
            )
            pinball_loss_evaluated = FunctionModule(name='pinball loss', transform_method=pinball_loss)(
                actual=pipeline_test['y_output'], prediction=post_processed,
                callbacks=[CSVCallback(f'task{task} zone{zone}')]
            )

            output = pipeline_test.test(data_test)
            losses[zone] = output['pinball loss'].data[0]
        loss = round((losses[1] + losses[2] + losses[3])/3, 5)
        results[task] = loss
        print(f'---------------------------\nLoss for task {task}: {loss}\n'
              f'---------------------------')
    print(results)
    print(f'Average loss: {round(results.mean(axis="columns").iloc[0], 5)}')
    results.to_csv('results/nnqf/losses.csv')
