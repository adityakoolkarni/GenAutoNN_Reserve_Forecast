"""
Dataloader class to load CAISO and EIA data into gluonts model
for training and predicting CA net load ramp.
"""

import numpy as np
import pandas as pd
import mxnet as mx
from mxnet import gluon

from multiprocessing import cpu_count
from gluonts.dataset.common import ListDataset
from gluonts.dataset.field_names import FieldName
from gluonts.transform import (
    Chain,
    ExpectedNumInstanceSampler,
    InstanceSplitter,
    )


class dssmDataloader():
    def __init__(self, configs):
        self.configs = configs
        self.df = pd.read_csv(configs.data_path)

    def extract_data(self):
        """Func used to grab relevant fields from the df."""
        caiso_target = np.asarray(self.df['caiso_ramp'])
        eia_target = np.asarray(self.df['eia_ramp'])
        targets = np.stack([caiso_target, eia_target], axis=0)

        feature_labels = ['cat_hour', 'cat_day', 'cat_month']
        feature_cols = []
        for label in feature_labels:
            feature_cols.append(
                np.asarray([int(i) for i in self.df[label]])
                )
        features = np.asarray(feature_cols)

        return targets, features

    def make_transformation(self, freq, context_length, pred_length):
        """Func to create a chain obj that transforms the dataset."""
        return Chain(
            [
                InstanceSplitter(
                    target_field=FieldName.TARGET,
                    is_pad_field=FieldName.IS_PAD,
                    start_field=FieldName.START,
                    forecast_start_field=FieldName.FORECAST_START,
                    train_sampler=ExpectedNumInstanceSampler(num_instances=1),
                    past_length=context_length,
                    future_length=pred_length,
                    time_series_fields=[FieldName.FEAT_DYNAMIC_CAT]
                )
            ]
        )

    def load_data(self):
        """Note that targets is n_series x n_steps, and features are made
        to be n_series x n_features x n_steps for the zip func below."""
        targets, features = self.extract_data()
        for _ in range(targets.shape[0]-1):
            features = np.stack([features, features])
        dates = np.asarray(self.df['dt'])

        ds_metadata = {
            'num_series': targets.shape[0],
            'num_steps': targets.shape[1],
            'context_length': int(self.configs.seq_len),
            'pred_length': int(self.configs.seq_len / 2),
            'freq': '1H',
            'start': [dates[0] for _ in range(targets.shape[0])]
        }

        train_ds = ListDataset(
            [{FieldName.TARGET: target,
              FieldName.START: start,
              FieldName.FEAT_DYNAMIC_CAT: np.squeeze(fdc)}
             for (target, start, fdc) in zip(
                targets[:, :-ds_metadata['pred_length']],
                ds_metadata['start'],
                features[:, :, :-ds_metadata['pred_length']])],
            freq=ds_metadata['freq']
            )
        test_ds = ListDataset(
            [{FieldName.TARGET: target,
              FieldName.START: start,
              FieldName.FEAT_DYNAMIC_CAT: np.squeeze(fdc)}
             for (target, start, fdc) in zip(
                targets,
                ds_metadata['start'],
                features)],
            freq=ds_metadata['freq']
            )

        transformation = self.make_transformation(
            ds_metadata['freq'],
            ds_metadata['context_length'],
            ds_metadata['pred_length']
            )

        train_tf = transformation(iter(train_ds), is_train=True)
        test_tf = transformation(iter(test_ds), is_train=False)

        return train_tf, test_tf
