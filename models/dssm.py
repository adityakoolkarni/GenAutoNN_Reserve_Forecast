from gluonts.model.deepstate import DeepStateEstimator
from gluonts.model.predictor import Predictor
from gluonts.trainer import Trainer
from pathlib import Path

class DeepStateSpaceModel:
    def __init__(self,configs,ctx):

        self.configs = configs
        self.ctx = ctx

    def get_estimator(self,metadata):

        self.estimator = DeepStateEstimator(
            freq = self.configs.freq,
            prediction_length=self.configs.pred_len,
            cardinality=[3 if self.configs.six_ramps else 2], ##one per sequence type 1)demand ramp 2) solar ramp 3) wind ramp
            add_trend=True,
            past_length=self.configs.context_len,
            num_layers = self.configs.num_layers,
            num_cells=self.configs.num_hidden,
            trainer=Trainer(ctx=self.ctx,
                        epochs=self.configs.num_epochs,
                        learning_rate=self.configs.learning_rate,
                        hybridize=False,
                        #num_batches_per_epoch = self.configs.train_len // self.configs.batch_size,
                        batch_size=self.configs.batch_size
                       )
            )

        return self.estimator

    def save_model(self,predictor):
        predictor.serialize(Path(self.configs.model_save_path))

    def load_model(self):
        predictor_deserialized = Predictor.deserialize(Path(self.configs.model_save_path))
        return predictor_deserialized

