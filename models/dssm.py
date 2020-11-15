from gluonts.model.deepstate import DeepStateEstimator
from gluonts.trainer import Trainer

class DeepStateSpaceModel:
    def __init__(self,configs,ctx):

        self.configs = configs
        self.ctx = ctx

    def get_estimator(self):
        self.estimator = DeepStateEstimator(
        freq = custom_ds_metadata['freq'],
        prediction_length=custom_ds_metadata['prediction_length'],
        cardinality=[2],
        add_trend=True,
        past_length=6*custom_ds_metadata['prediction_length'],
        trainer=Trainer(ctx=ctx,
                    epochs=5,
                    learning_rate=1e-3,
                    hybridize=False,
                    num_batches_per_epoch=100
                   )
        )

        return self.estimator

    def save_model(self,predictor):
        predictor.serialize(self.configs.model_save_path)


    def load_model(self):
        from gluonts.model.predictor import Predictor
        predictor_deserialized = Predictor.deserialize(Path("/tmp/"))

