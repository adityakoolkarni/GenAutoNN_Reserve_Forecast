import os
import time
import json
import datetime
import mxnet as mx
from pprint import pprint
import matplotlib.pyplot as plt
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.evaluation import Evaluator

from utils.parser import configparser
from data.dataloader import dssmDataloader
from models.dssm import DeepStateSpaceModel


def plot_forecasts(forecast, series, name, context_len, pred_len):
    plot_length = context_len + pred_len
    prediction_intervals = (50.0, 90.0)
    legend = ['observations', 'median prediction']
    legend += [f'{k} prediction interval' for k in prediction_intervals][::-1]

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    series[-plot_length:].plot(ax=ax)  # plot the time series
    forecast.plot(prediction_intervals=prediction_intervals, color='g')
    plt.grid(which="both")
    plt.legend(legend, loc="upper left")
    ax.set_title(name)
    plt.savefig('plots/' + name + '.png')


    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    mean_forecast = np.mean(forecast,axis=0)
    hourly_error = mean_forecast - series[-pred_len:]
    ax.plot(np.arange(24),hourly_error)
    ax.set_title(name+'_hourly_error')


def log_eval(configs, agg_metrics):
    if not os.path.exists('logs'):
        os.makedirs('logs')
    filename = datetime.datetime.now().isoformat(timespec='seconds') + '.json'
    log_content = {
        'configuration': vars(configs),
        'results': agg_metrics
        }
    with open(os.path.join('logs', filename), 'w') as f:
        json.dump(log_content, f)


def train(model, train_ds, metadata):
    """
    Get a probabilistic estimator, train this to produce a predictor, and then
    save the predictor.
    """
    estimator = model.get_estimator(metadata)

    tic = time.time()
    predictor = estimator.train(
        training_data=train_ds
        )

    print("*"*64)
    training_time = int(time.time() - tic)
    print(f"Training took {training_time} seconds.")
    model.save_model(predictor)
    print("Predictor saved.")
    print("*"*64)


def eval(model, validation_ds, metadata, configs, names):
    predictor = model.load_model()

    forecast_it, series_it = make_evaluation_predictions(
        dataset=validation_ds,  # test dataset
        predictor=predictor,  # predictor
        num_samples=128  # number of sample paths we want for evaluation
        )

    evaluator = Evaluator(quantiles=[0.10, 0.50, 0.90])
    agg_metrics, item_metrics = evaluator(
        series_it,
        forecast_it,
        num_series=len(validation_ds)
        )
    log_eval(configs, agg_metrics)
    print('\nConfiguration and results logged.')

    forecasts_ls = list(forecast_it)
    series_ls = list(series_it)
    for i in range(len(forecasts_ls)):
        forecast = forecasts_ls[i]
        series = series_ls[i]
        name = names[i]
        plot_forecasts(forecast, series, name,
                       configs.context_len, configs.pred_len)
    print('\nTarget series forecast plots saved.')


if __name__ == '__main__':
    mx.random.seed(1)
    ctx = mx.cpu()
    configs = configparser.parse_args()
    pprint(vars(configs))

    loader = dssmDataloader(configs)
    splits = [0.80, 0.10]
    (
        clipped_datasets,
        full_datasets,
        metadata,
        target_names
    ) = loader.load_data(splits)

    # make this very clear...
    train_ds_clipped = clipped_datasets[0]
    train_ds_full = full_datasets[0]
    validation_ds_clipped = clipped_datasets[1]
    validation_ds_full = full_datasets[1]
    test_ds_clipped = clipped_datasets[2]
    test_ds_full = full_datasets[2]

    # take a look at what an entry of each ListDataset object looks like
    train_ds_entry = next(iter(train_ds))
    validation_ds_entry = next(iter(validation_ds))
    print('\n')
    print('Shape of train_ds_entry target:')
    print(train_ds_entry['target'].shape)
    print('Shape of validation_ds_entry target:')
    print(validation_ds_entry['target'].shape)
    print('\n')

    model = DeepStateSpaceModel(configs, ctx)

    #### Step 1: Train the model on clipped dataset ####
    train(model, train_ds_clipped, metadata)
    #### Step 2: Evaluate the model on full dataset ####
    eval(model,train_ds_full,metadata,configs,target_names) 

    ###### Use these metrics to see tune the hyperparameters ######
    #### Step 3: Train the model on clipped dataset ####
    train(model, validation_ds_clipped, metadata)
    #### Step 4: Evaluate the model on full dataset ####
    eval(model,validation_ds_full,metadata,configs,target_names) 

    ###### Use these metrics to see generalizability of the hyperparameters ######
    #### Step 3: Train the model on clipped dataset ####
    train(model, test_ds_clipped, metadata)
    #### Step 4: Evaluate the model on full dataset ####
    eval(model,test_ds_full,metadata,configs,target_names) 


