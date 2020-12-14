from gluonts.model.forecast import SampleForecast, Quantile
import numpy as np
import pandas as pd

class ForecastMe(SampleForecast):
    def plot(self, prediction_intervals=(50.0, 90.0), show_mean=False, color="b", label=None, output_file=None, *args, **kwargs,
    ):
        """
        Plots the median of the forecast as well as confidence bounds.
        (requires matplotlib and pandas).
        Parameters
        ----------
        prediction_intervals : float or list of floats in [0, 100]
            Confidence interval size(s). If a list, it will stack the error
            plots for each confidence interval. Only relevant for error styles
            with "ci" in the name.
        show_mean : boolean
            Whether to also show the mean of the forecast.
        color : matplotlib color name or dictionary
            The color used for plotting the forecast.
        label : string
            A label (prefix) that is used for the forecast
        output_file : str or None, default None
            Output path for the plot file. If None, plot is not saved to file.
        args :
            Other arguments are passed to main plot() call
        kwargs :
            Other keyword arguments are passed to main plot() call
        """

        # matplotlib==2.0.* gives errors in Brazil builds and has to be
        # imported locally
        print("Inside my plotting")
        print("**"*40)
        import matplotlib.pyplot as plt

        label_prefix = "" if label is None else label + "-"

        for c in prediction_intervals:
            assert 0.0 <= c <= 100.0

        ps = [50.0] + [
            50.0 + f * c / 2.0
            for c in prediction_intervals
            for f in [-1.0, +1.0]
        ]
        percentiles_sorted = sorted(set(ps))

        def alpha_for_percentile(p):
            return (p / 100.0) ** 0.3

        ps_data = [self.quantile(p / 100.0) for p in percentiles_sorted]
        i_p50 = len(percentiles_sorted) // 2

        p50_data = ps_data[i_p50]
        p50_series = pd.Series(data=p50_data, index=self.index)
        p50_series.plot(color='black', ls=":", label=f"{label_prefix}median")

        if show_mean:
            mean_data = np.mean(self._sorted_samples, axis=0)
            pd.Series(data=mean_data, index=self.index).plot(
                color=color,
                ls=":",
                label=f"{label_prefix}mean",
                *args,
                **kwargs,
            )

        for i in range(len(percentiles_sorted) // 2):
            ptile = percentiles_sorted[i]
            alpha = alpha_for_percentile(ptile)
            plt.fill_between(
                self.index,
                ps_data[i],
                ps_data[-i - 1],
                facecolor=color,
                alpha=alpha,
                interpolate=True,
                *args,
                **kwargs,
            )
            # Hack to create labels for the error intervals.
            # Doesn't actually plot anything, because we only pass a single data point
            pd.Series(data=p50_data[:1], index=self.index[:1]).plot(
                color=color,
                alpha=alpha,
                linewidth=10,
                label=f"{label_prefix}{100 - ptile * 2}%",
                *args,
                **kwargs,
            )

    def quantile(self, q):
        q = Quantile.parse(q).value
        sample_idx = int(np.round((self.num_samples - 1) * q))
        return self._sorted_samples[sample_idx, :]
