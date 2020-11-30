from gluonts.model.deepstate import DeepStateEstimator
from gluonts.model.predictor import Predictor
from gluonts.trainer import Trainer

class DeepStateSpaceModel:
    def __init__(self,configs,ctx):

        self.configs = configs
        self.ctx = ctx

    def get_estimator(self,metadata):

        self.estimator = DeepStateEstimator(
            freq = metadata['freq'],
            prediction_length=metadata['pred_length'],
            cardinality=[24,7,12],
            add_trend=True,
            past_length=metadata['context_length'],
            num_cells=self.configs.num_hidden,
            use_feat_static_cat=False,
            use_feat_dynamic_real=False,
            trainer=Trainer(ctx=self.ctx,
                        epochs=self.configs.num_epochs,
                        learning_rate=self.configs.learning_rate,
                        hybridize=False,
                        batch_size=self.configs.batch_size,
                        num_batches_per_epoch=metadata['num_steps']//self.configs.batch_size
                       )
            )

        return self.estimator

    def save_model(self,predictor):
        predictor.serialize(self.configs.model_save_path)

    def load_model(self):
        predictor_deserialized = Predictor.deserialize(self.configs.model_save_path)
        return predictor_deserialized

