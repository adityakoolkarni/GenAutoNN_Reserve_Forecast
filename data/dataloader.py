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
        print("Extarcatmog", self.configs.six_ramps)
        if self.configs.six_ramps:
            caiso_load_target = np.asarray(self.df['caiso_load_ramp'])
            caiso_solar_target = np.asarray(self.df['caiso_solar_ramp'])
            caiso_wind_target = np.asarray(self.df['caiso_wind_ramp'])
            eia_load_target = np.asarray(self.df['eia_load_ramp'])
            eia_solar_target = np.asarray(self.df['eia_solar_ramp'])
            eia_wind_target = np.asarray(self.df['eia_wind_ramp'])
            targets = np.stack([
                caiso_load_target,
                eia_load_target,
                caiso_solar_target,
                eia_solar_target,
                caiso_wind_target,
                eia_wind_target
                ], axis=0)
            target_names = [
                'CAISO_Load_Ramp',
                'EIA_Load_Ramp',
                'CAISO_Solar_Ramp',
                'EIA_Solar_Ramp',
                'CAISO_Wind_Ramp',
                'EIA_Wind_Ramp'
                ]
            caiso_net_load_ramp = np.asarray(self.df["caiso_ramp"])
            eia_net_load_ramp = np.asarray(self.df["eia_ramp"])

        else:
            targets = np.asarray(self.df['eia_ramp']).reshape(1, -1)
            target_names = ['EIA_Total_Ramp']
            caiso_net_load_ramp, eia_net_load_ramp = None, None

        return targets, target_names, caiso_net_load_ramp, eia_net_load_ramp

    def build_ds_iterables(self, markers, targets,
                           static_features,
                           ds_metadatas, clip_flag):
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
                            FieldName.FEAT_STATIC_CAT: [fsc]
                        }
                        for (target, start, fsc) in zip(
                            targets[
                                        :,
                                        markers[i]:markers[i+1]-clip
                                    ],
                            ds_metadatas[i]['start'],
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
        assert len(splits) == 2, 'Incorrect train/validation/test split.'

        (
            targets,
            target_names,
            caiso_net_load_ramp,
            eia_net_load_ramp
        ) = self.extract_data()
        # 0s for load ramps, 1s for solar ramps, and 2s for wind ramps
        if self.configs.six_ramps:
            static_features = [0, 0, 1, 1, 2, 2]
        else:
            static_features = [0]
        dates = np.asarray(self.df['dt'])
        num_steps = targets.shape[1]

        markers = [
            0,
            int(num_steps * splits[0]),
            int(num_steps * (splits[0] + splits[1])),
            num_steps
            ]

        ds_metadatas = []
        for i in range(len(markers) - 1):
            cur_start = markers[i]
            ds_metadatas.append(
                {
                    'num_series': targets.shape[0],
                    'num_steps': markers[i + 1] - markers[i],
                    'start': [
                        dates[cur_start]
                        for _
                        in range(targets.shape[0])
                        ]
                }
            )

        clipped_datasets = self.build_ds_iterables(markers, targets,
                                                   static_features,
                                                   ds_metadatas,
                                                   clip_flag=True)
        full_datasets = self.build_ds_iterables(markers, targets,
                                                static_features,
                                                ds_metadatas,
                                                clip_flag=False)

        return (
            clipped_datasets,
            full_datasets,
            ds_metadatas,
            target_names,
            caiso_net_load_ramp,
            eia_net_load_ramp
            )
