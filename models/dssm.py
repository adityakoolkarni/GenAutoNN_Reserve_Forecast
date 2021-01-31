from gluonts.model.deepstate import DeepStateEstimator
from gluonts.model.predictor import Predictor
from gluonts.trainer import Trainer
from pathlib import Path
import mxnet as mx
from gluonts.model.deepstate.issm import ISSM, CompositeISSM,SeasonalityISSM

class DeepStateSpaceModel:
    def __init__(self,configs,ctx):

        self.configs = configs
        self.ctx = ctx

    def get_estimator(self,metadata):

        self.estimator = DeepStateEstimator(
            freq = self.configs.freq,
            prediction_length=self.configs.pred_len,
            cardinality=[3 if self.configs.six_ramps else 1], ##one per sequence type 1)demand ramp 2) solar ramp 3) wind ramp
            issm = OurSeasonality.get_seasonality(self.configs),
            use_feat_static_cat= True if self.configs.six_ramps else False,
            add_trend=True,
            past_length=self.configs.context_len,
            num_layers = self.configs.num_layers,
            num_cells=self.configs.num_hidden,
            trainer=Trainer(ctx=self.ctx,
                        epochs=self.configs.num_epochs,
                        learning_rate=self.configs.learning_rate,
                        hybridize=False,
                        patience=5,
                        num_batches_per_epoch = self.configs.train_len // self.configs.batch_size,
                        batch_size=self.configs.batch_size
                       )
            )

        return self.estimator

    def save_model(self,predictor):
        predictor.serialize(Path(self.configs.model_save_path))

    def load_model(self):
        print("Model path " ,Path(self.configs.model_save_path))
        import os
        print(os.listdir(Path(self.configs.model_save_path)))

        predictor_deserialized = Predictor.deserialize(Path(self.configs.model_save_path),ctx=mx.cpu())
        return predictor_deserialized

class OurSeasonality(CompositeISSM):
    @classmethod
    def get_seasonality(cls,configs):
        if configs.seasonality == "M":
            seasonal_issms = [
                SeasonalityISSM(num_seasons=12)  # month-of-year seasonality
            ]
        elif configs.seasonality == "W":
            seasonal_issms = [
                SeasonalityISSM(num_seasons=53)  # week-of-year seasonality
            ]
        elif configs.seasonality == "D":
            seasonal_issms = [
                SeasonalityISSM(num_seasons=7)
            ]  # day-of-week seasonality
        elif configs.seasonality == "H":
            seasonal_issms = [
                SeasonalityISSM(num_seasons=24)
            ]  # day-of-week seasonality
        elif configs.seasonality == "HD":
            seasonal_issms = [
                SeasonalityISSM(num_seasons=24),  # hour-of-day seasonality
                SeasonalityISSM(num_seasons=7),  # day-of-week seasonality
            ]
        elif configs.seasonality == "HM":
            seasonal_issms = [
                SeasonalityISSM(num_seasons=24),  # hour-of-day seasonality
                SeasonalityISSM(num_seasons=12),  # month-of-year seasonality
            ]
        elif configs.seasonality == "HW":
            seasonal_issms = [
                SeasonalityISSM(num_seasons=24),  # hour-of-day seasonality
                SeasonalityISSM(num_seasons=53),  # week-of-year seasonality
            ]
        else:
            RuntimeError(f"Unsupported frequency {configs.seasonality}")

        return cls(seasonal_issms=seasonal_issms, add_trend=True)

