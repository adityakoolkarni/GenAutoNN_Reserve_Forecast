"""
Dataloader class to load CAISO and EIA data into GluonTS model
for training and predicting CA net load ramp.
"""

import numpy as np
import pandas as pd

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
        """Extracts target and feature series from df."""
        caiso_load_target = np.asarray(self.df['caiso_load_ramp'])
        caiso_solar_target = np.asarray(self.df['caiso_solar_ramp'])
        caiso_wind_target = np.asarray(self.df['caiso_wind_ramp'])
        eia_load_target = np.asarray(self.df['eia_load_ramp'])
        eia_solar_target = np.asarray(self.df['eia_solar_ramp'])
        eia_wind_target = np.asarray(self.df['eia_wind_ramp'])
        target_names = [
            'CAISO Load Ramp',
            'EIA Load Ramp',
            'CAISO Solar Ramp',
            'EIA Solar Ramp',
            'CAISO Wind Ramp',
            'EIA Wind Ramp'
            ]
        targets = np.stack([
            caiso_load_target,
            eia_load_target,
            caiso_solar_target,
            eia_solar_target,
            caiso_wind_target,
            eia_wind_target
            ], axis=0)
        feature_labels = ['cat_hour', 'cat_day', 'cat_month', 'cat_year']
        feature_cols = []
        for label in feature_labels:
            feature_cols.append(
                np.asarray([int(i) for i in self.df[label]])
                )
        features = np.asarray(feature_cols)

        return targets, features, target_names

    def make_transformation(self, freq, context_len, pred_len):
        """Creates a chain obj that can transform the dataset. Note that
        currently this isn't used as the estimator.train method from GluonTS
        performs its own transform on the data.
        """
        return Chain(
            [
                InstanceSplitter(
                    target_field=FieldName.TARGET,
                    is_pad_field=FieldName.IS_PAD,
                    start_field=FieldName.START,
                    forecast_start_field=FieldName.FORECAST_START,
                    train_sampler=ExpectedNumInstanceSampler(num_instances=1),
                    past_length=context_len,
                    future_length=pred_len,
                    time_series_fields=[FieldName.FEAT_DYNAMIC_CAT]
                )
            ]
        )

    def build_ds_iterables(self, markers, targets,
                           dynamic_features, static_features,
                           ds_metadata, clip_flag):
        """Builds the GluonTS ListDataset iterables... both with the ending
        clipped (clipped or train) and without (full or test).
        """
        ds_iterables = []
        clip = self.configs.pred_len if clip_flag else 0
        for i in range(len(markers) - 1):
            ds_iterables.append(
                ListDataset(
                    [
                        {
                            FieldName.TARGET: target,
                            FieldName.START: start,
                            #FieldName.FEAT_DYNAMIC_CAT: np.squeeze(fdc),
                            FieldName.FEAT_STATIC_CAT: [fsc]
                        }
                        for (target, start, fdc, fsc) in zip(
                            targets[
                                        :,
                                        markers[i]:markers[i+1]-clip
                                    ],
                            ds_metadata['start'],
                            #dynamic_features[
                            #                    :,
                            #                    :,
                            #                    markers[i]:markers[i+1]-clip
                            #                ],
                            static_features
                            )
                    ],
                    freq=self.configs.freq
                )
            )

        if len(ds_iterables) == 1:
            return ds_iterables[0]
        else:
            return ds_iterables

    def load_data(self, splits):
        """Loads GluonTS compatible train, validation, and test datasets, each
        of which has a training and test component. Note that targets is of
        shape n_series x n_steps, and thus features are made to be of shape
        n_series x n_features x n_steps to allow them to be zipped together to
        form the dataset lists.
        """
        targets, dynamic_features, target_names = self.extract_data()
        dynamic_features = np.expand_dims(dynamic_features, axis=0)
        init_dynamic_features = dynamic_features
        for _ in range(targets.shape[0]-1):
            dynamic_features = np.concatenate([dynamic_features,
                                               init_dynamic_features],
                                              axis=0)
        # 0s for load ramps, 1s for solar ramps, and 2s for wind ramps
        static_features = [0, 0, 1, 1, 2, 2]
        dates = np.asarray(self.df['dt'])

        ds_metadata = {
            'num_series': targets.shape[0],
            'num_steps': targets.shape[1],
        }

        assert len(splits) == 2, 'Incorrect train/validation/test split.'

        """No splits currently given as GluonTS doesn't appear to allow for
        validation and test sets with fewer steps than the training set.
        """
        markers = [
            0,
            ds_metadata['num_steps'] * splits[0],
            ds_metadata['num_steps'] * (splits[0] + splits[1]),
            ds_metadata['num_steps']
            ]

        ### Take start data according to the split of time stamp
        ds_metadata['start'] = [dates[markers[0]],
                                            dates[markers[1]],
                                            dates[markers[2]]]
        clipped_datasets = self.build_ds_iterables(markers, targets,
                                                   dynamic_features,
                                                   static_features,
                                                   ds_metadata,
                                                   clip_flag=True)
        full_datasets = self.build_ds_iterables(markers, targets,
                                                dynamic_features,
                                                static_features,
                                                ds_metadata,
                                                clip_flag=False)

        return clipped_datasets, full_datasets, ds_metadata, target_names
