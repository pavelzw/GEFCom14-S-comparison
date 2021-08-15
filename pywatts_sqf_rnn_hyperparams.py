import pandas as pd
from itertools import product
import xarray as xr
import numpy as np
from gluonts.model.deepar import DeepAREstimator
from gluonts.mx.distribution import PiecewiseLinearOutput
from gluonts.mx.trainer import Trainer

from gluonts_wrapper import GluonTSWrapper
from multi_dim_csv_callback import MultiDimCSVCallback
from number_callback import NumberCallback
from pywatts.core.pipeline import Pipeline
from pywatts.wrapper import FunctionModule

from hyperopt import fmin, tpe, hp, STATUS_OK, STATUS_FAIL, Trials


def load_data(task):
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
        'freq': '1H',
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


def pinball_loss(actual, prediction):
    losses = np.empty(prediction.shape)
    for zone in range(3):
        actual_zone = np.array(actual)
        prediction_zone = np.array(prediction)[:, zone, :]
        actual_zone = actual_zone[-prediction.shape[0]:, zone, :]

        percentiles = np.empty((actual_zone.shape[0], 99))
        for i in range(1, 100):
            percentiles[:, i - 1] = i
        losses[:, zone, :] = np.where(actual_zone < prediction_zone,
                                (1 - percentiles / 100) * (prediction_zone - actual_zone),
                                percentiles / 100 * (actual_zone - prediction_zone))
    return xr.DataArray([np.mean(losses)]).rename({'dim_0': 'pinball loss'})


def evaluate_task(task, args):
    num_pieces, num_cells, num_layers, context_length, \
        cell_type, dropoutcell_type, dropout_rate = args

    predictor_zones, train_data_zones, gefcom14_metadata = load_data(task)

    x_input1 = xr.DataArray(predictor_zones[0])
    x_input2 = xr.DataArray(predictor_zones[1])
    x_input3 = xr.DataArray(predictor_zones[2])
    x_input = xr.DataArray(coords=[('TIMESTAMP', x_input1.TIMESTAMP.data),
                                   ('ZONE', [1, 2, 3]),
                                   ('dim_0', x_input1.dim_1.data)])
    x_input[:, 0, :] = x_input1
    x_input[:, 1, :] = x_input2
    x_input[:, 2, :] = x_input3

    y_output1 = xr.DataArray(train_data_zones[0])
    y_output2 = xr.DataArray(train_data_zones[1])
    y_output3 = xr.DataArray(train_data_zones[2])
    y_output = xr.DataArray(coords=[('TIMESTAMP', y_output1.TIMESTAMP.data),
                                    ('ZONE', [1, 2, 3]),
                                    ('dim_0', y_output1.dim_1.data)])
    y_output[:, 0, :] = y_output1
    y_output[:, 1, :] = y_output2
    y_output[:, 2, :] = y_output3

    data_train = {'x_input': x_input[:-gefcom14_metadata['prediction_length'], :, :],
                  'y_output': y_output[:-gefcom14_metadata['prediction_length'], :, :]}
    data_test = {'x_input': x_input,
                 'y_output': y_output}

    pipeline_train = Pipeline(path='results/sqf-rnn/test')

    neural_network = GluonTSWrapper(module=DeepAREstimator(
        freq=gefcom14_metadata['freq'],
        prediction_length=gefcom14_metadata['prediction_length'],
        num_cells=num_cells,
        num_layers=num_layers,
        dropout_rate=dropout_rate,
        context_length=context_length,
        cell_type=cell_type,
        dropoutcell_type=dropoutcell_type,
        distr_output=PiecewiseLinearOutput(num_pieces=num_pieces),  # SQF-RNN
        use_feat_dynamic_real=True,
        trainer=Trainer(epochs=5)
    ), start=gefcom14_metadata['start'],
       freq=gefcom14_metadata['freq'],
       name='neural network')

    nn_output_train = neural_network(
        x=pipeline_train['x_input'], target=pipeline_train['y_output']
    )

    pipeline_train.train(data_train)

    # testing
    pipeline_test = Pipeline(path='results/sqf-rnn/test')

    nn_output_test = neural_network(
        x=pipeline_test['x_input'],
        callbacks=[MultiDimCSVCallback(f'task{task}')]
    )

    pinball_loss_evaluated = FunctionModule(name='pinball loss', transform_method=pinball_loss)(
        actual=pipeline_test['y_output'], prediction=nn_output_test,
        callbacks=[NumberCallback(f'task{task}', (0,))]
    )

    output = pipeline_test.test(data_test)
    loss = round(output['pinball loss'].data[0], 5)
    return loss


def evaluate(args):
    results = pd.DataFrame(columns=range(4, 16), index=['loss'])

    for task in range(4, 16):
        loss = evaluate_task(task, args)
        results[task] = loss
        print(f'Loss for task {task}: {loss}')
    print(results)
    avg = round(results.mean(axis="columns").iloc[0], 5)
    print(f'Hyperparameters: {args}')
    print(f'Average loss: {avg}')
    results.to_csv(f'results/sqf-rnn/losses_{args}.csv')
    return {'loss': avg,
            'status': STATUS_OK}


if __name__ == '__main__':
    space = [
        hp.choice('num_pieces', range(2, 8)),
        hp.choice('num_cells', range(20, 100)),
        hp.choice('num_layers', range(2, 6)),
        hp.choice('context_length', range(24 * 60)),
        hp.choice('cell_type', ['lstm', 'gru']),
        hp.choice('dropoutcell_type',
                  ['ZoneoutCell', 'RNNZoneoutCell', 'VariationalDropoutCell', 'VariationalZoneoutCell']),
        hp.uniform('dropout_rate', 0.01, 0.5)
    ]

    trials = Trials()
    best = fmin(fn=evaluate,
                space=space,
                algo=tpe.suggest,
                trials=trials,
                max_evals=1)

    print("---------------BEST---------------")
    print(best)
    print('trials.best_trial')
    print(trials.best_trial)
    print('trials.argmin')
    print(trials.argmin)
    import pickle

    pickle.dump(trials, open(f'results/sqf-rnn/trials.p', 'wb'))
