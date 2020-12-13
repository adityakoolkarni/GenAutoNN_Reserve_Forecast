from gluonts.model.forecast import SampleForecast

class ForecastMe(SampleForecast):
    def __init__(self, **kwargs):
        super().__init__(kwargs)
