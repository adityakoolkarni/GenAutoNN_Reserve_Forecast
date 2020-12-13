from glounts.models.forecast import SampleForecast

class ForecastMe(SampleForecast):
    def __init__(self, **kwargs):
        super().__init__(kwargs)
