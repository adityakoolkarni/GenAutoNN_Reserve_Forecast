import os
import time
import json
import datetime
import mxnet as mx
from pprint import pprint
import matplotlib.pyplot as plt
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.evaluation import Evaluator
import numpy as np

from pathlib import Path
from utils.parser import configparser
from data.dataloader import dssmDataloader
from models.dssm import DeepStateSpaceModel
from itertools import tee


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


def plot_forecasts(forecast, agg_metrics, series, name, configs):
    context_len, pred_len = configs.context_len, configs.pred_len
    plot_length = context_len + pred_len
    prediction_intervals = (50.0, 90.0)
    legend = ['observations', 'median prediction']
    legend += [f'{k} prediction interval' for k in prediction_intervals][::-1]

    plot_path = os.path.join(configs.model_save_path, 'plots')
    if not os.path.isdir(plot_path):
        os.mkdir(plot_path)

    # first plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    series[-plot_length:].plot(ax=ax)  # plot the time series
    forecast.plot(prediction_intervals=prediction_intervals, color='g')
    plt.grid(which="both")
    plt.legend(legend, loc="upper left")
    ax.set_title(name)
    plt.savefig(os.path.join(plot_path, name + ".png"))

    # second plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    ax.plot(
        series.index[-configs.pred_len:],
        agg_metrics[name + '_error_perc'] * 100
        )
    ax.set_title(f"Percentage Error Plot for {name}")
    plt.savefig(os.path.join(plot_path, name + '_error_perc.png'))


def log_eval(configs, agg_metrics):
    # filename = datetime.datetime.now().isoformat(timespec='seconds')
    # log_content = {
    #     'configuration': vars(configs),
    #     'results': agg_metrics
    #     }
    with open(os.path.join(configs.model_save_path, 'logs.txt'), 'w') as f:
        f.writelines(['Configuration Settings ', '\n'])
        for k, v in vars(configs).items():
            f.writelines([str(k), '\t', str(v), '\n'])
    with open(os.path.join(configs.model_save_path, 'logs.txt'), 'a') as f:
        f.writelines(['Evaluation Metrics ', '\n'])
        for k, v in agg_metrics.items():
            f.writelines([str(k), '\t', str(v), '\n'])


def combine_forecast_components(forecasts_ls, series_ls):
    caiso_net_load_ramp_mean = (
        np.mean(forecasts_ls[0].samples, axis=0) -
        np.mean(forecasts_ls[2].samples, axis=0) -
        np.mean(forecasts_ls[4].samples, axis=0)
        )
    caiso_net_load_ramp_median = (
        np.median(forecasts_ls[0].samples, axis=0) -
        np.median(forecasts_ls[2].samples, axis=0) -
        np.median(forecasts_ls[4].samples, axis=0)
        )
    eia_net_load_ramp_mean = (
        np.mean(forecasts_ls[1].samples, axis=0) -
        np.mean(forecasts_ls[3].samples, axis=0) -
        np.mean(forecasts_ls[5].samples, axis=0)
        )
    eia_net_load_ramp_median = (
        np.median(forecasts_ls[1].samples, axis=0) -
        np.median(forecasts_ls[3].samples, axis=0) -
        np.median(forecasts_ls[5].samples, axis=0)
        )

    combined_forecast_samples = [
        caiso_net_load_ramp_mean,
        caiso_net_load_ramp_median,
        eia_net_load_ramp_mean,
        eia_net_load_ramp_median
        ]
    labels = [
        "caiso_net_load_ramp_mean",
        "caiso_net_load_ramp_median",
        "eia_net_load_ramp_mean",
        "eia_net_load_ramp_median"
        ]

    return combined_forecast_samples, labels


def eval(model, validation_ds, metadata, configs, names,
         caiso_net_load_ramp, eia_net_load_ramp):
    predictor = model.load_model()

    forecasts_it, series_it = make_evaluation_predictions(
        dataset=validation_ds,  # test dataset
        predictor=predictor,  # predictor
        num_samples=128  # number of sample paths we want for evaluation
        )

    forecasts_list_it, forecast_it = tee(forecasts_it)
    series_list_it, series_it = tee(series_it)
    forecasts_ls = list(forecasts_list_it)
    series_ls = list(series_list_it)

    evaluator = Evaluator(quantiles=[0.10, 0.50, 0.90])
    agg_metrics, item_metrics = evaluator(
        series_it,
        forecast_it,
        num_series=len(validation_ds)
        )

    combined_forecast_samples, labels = combine_forecast_components(
        forecasts_ls,
        series_ls
        )

    for i in range(len(forecasts_ls)):
        forecast = forecasts_ls[i]
        name = names[i].lower()
        print("*-"*40)
        print("*"*40, end='')
        print(f"Forecast {name}", "*"*30)
        print(f"Number of sample paths: {forecast.num_samples}")
        print(f"Dimension of samples: {forecast.samples.shape}")
        print(f"Start date of the forecast window: {forecast.start_date}")
        print(f"Frequency of the time series: {forecast.freq}")
        print("*-"*40)
        series = series_ls[i]
        agg_metrics[name + '_error_perc'] = (
            (np.mean(forecast.samples, axis=0) -
                series[-configs.pred_len:].to_numpy().reshape(-1)) /
            series[-configs.pred_len:].to_numpy().reshape(-1))
        plot_forecasts(forecast, agg_metrics, series, name, configs)

    # now do a similar calc for the combined forecasts
    # not yet logging this!!
    for i in range(len(combined_forecast_samples)):
        cur_samples = combined_forecast_samples[i]
        cur_label = labels[i]
        if "caiso" in cur_label:
            cur_target = caiso_net_load_ramp
        else:
            cur_target = eia_net_load_ramp
        cur_percent_error = (
            (cur_samples -
             cur_target[-configs.pred_len:].reshape(-1)) /
            cur_target[-configs.pred_len:].reshape(-1))
        print(f"{cur_label} percent error: {cur_percent_error}")

    print('\nTarget series forecast plots saved.')

    log_eval(configs, agg_metrics)
    print('\nConfiguration and results logged.')


if __name__ == '__main__':
    mx.random.seed(1)
    ctx = mx.cpu()
    configs = configparser.parse_args()
    sim_number = len(os.listdir('./saved_models'))
    configs.model_save_path += f'/simulation_num_{sim_number}'
    configs.model_save_path = Path(configs.model_save_path)
    if not os.path.isdir(configs.model_save_path):
        os.mkdir(configs.model_save_path)
    pprint(vars(configs))

    loader = dssmDataloader(configs)
    # ratios of train and validation data, with the remainder as the test set
    splits = [0.80, 0.10]
    (
        clipped_datasets,
        full_datasets,
        metadatas,
        target_names,
        caiso_net_load_ramp,
        eia_net_load_ramp
    ) = loader.load_data(splits)

    # make this very clear...
    train_ds = clipped_datasets[0]
    train_metadata = metadatas[0]
    validation_ds = full_datasets[1]
    validation_metadata = metadatas[1]
    test_ds = full_datasets[2]
    test_metadata = metadatas[2]

    # take a look at what an entry of each ListDataset object looks like
    train_ds_entry = next(iter(train_ds))
    validation_ds_entry = next(iter(validation_ds))
    print('\n')
    print('Shape of train_ds_entry target:')
    print(train_ds_entry['target'].shape)
    print("Train_ds metadata:")
    print(json.dumps(train_metadata, indent=4))
    print('Shape of validation_ds_entry target:')
    print(validation_ds_entry['target'].shape)
    print("validation_ds metadata:")
    print(json.dumps(validation_metadata, indent=4))
    print('\n')

    configs.train_len = train_ds_entry['target'].shape[0]
    model = DeepStateSpaceModel(configs, ctx)

    train(model, train_ds, train_metadata)
    eval(model, validation_ds, validation_metadata, configs, target_names,
         caiso_net_load_ramp, eia_net_load_ramp)
