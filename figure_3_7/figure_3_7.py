from typing import List, Tuple, Optional

import bokeh.layouts
import bokeh.models
import bokeh.plotting
import dataclasses
import numpy as np
import scipy.stats

# Global constants
FIG_SIZE = 400

# Axis limits
X_LIMITS = X_MIN, X_MAX = -1, 1
Y_LIMITS = Y_MIN, Y_MAX = -1, 1

# Parameter grid
GRID_SIZE = 300
X = np.linspace(X_MIN, X_MAX, GRID_SIZE)
Y = np.linspace(Y_MIN, Y_MAX, GRID_SIZE)
XX, YY = np.meshgrid(X, Y)
GRID_POINTS = np.vstack([XX.ravel(), YY.ravel()]).T

# Initial prior
M_0 = np.array([0, 0])
COV_0 = np.array([
    [1, 0],
    [0, 1],
])

# Model parameters
ALPHA: float = 2.0
SIGMA: float = 0.2
BETA: float = SIGMA ** -2

# Parameters of the true line
TRUE_SLOPE = 0.5
TRUE_INTERCEPT = -0.3


@dataclasses.dataclass(frozen=True)
class ObservationGenerator:
    slope: float = TRUE_SLOPE
    y_intercept: float = TRUE_INTERCEPT
    x_dist: scipy.stats.rdist = scipy.stats.uniform(X_MIN, X_MAX - X_MIN)
    noise_dist: scipy.stats.rdist = scipy.stats.norm(0, SIGMA)

    def sample(self, size=1):
        x = self.x_dist.rvs(size=size)
        noise = self.noise_dist.rvs(size=size)
        return np.array([x, self.slope * x + self.y_intercept + noise]).T


def distribution_to_array(distribution: scipy.stats.rdist):
    d = distribution.pdf(GRID_POINTS)
    return d.T.reshape(XX.shape)


def bokeh_figure():
    return bokeh.plotting.figure(
        x_range=(X_MIN, X_MAX),
        y_range=(Y_MIN, Y_MAX),
        plot_width=FIG_SIZE,
        plot_height=FIG_SIZE,
        tooltips=[("X", "$X"), ("Y", "$Y"), ("value", "@image")]
    )


@dataclasses.dataclass
class BayesianLinearRegression:
    prior_posterior: scipy.stats.rdist = None
    likelihood: Optional[np.ndarray] = None

    points: List[Tuple[float, float]] = None

    m_n: np.ndarray = None
    cov_n: np.ndarray = None

    def __post_init__(self):
        self.reset_state()

    def add_observation(self, raw_samples):
        x_obs = raw_samples[:, 0]
        y_obs = raw_samples[:, 1]

        phi = np.vstack([np.ones(len(raw_samples)), x_obs]).T

        x_last_obs, y_last_obs = raw_samples[-1]
        gaussian = scipy.stats.norm(XX + YY * x_last_obs, SIGMA)
        self.likelihood = gaussian.pdf(y_last_obs)

        for row in raw_samples:
            self.points.append(tuple([*row]))

        m_0 = self.m_n
        cov_0 = self.cov_n

        # Update posterior:
        self.cov_n = np.linalg.inv(
            np.linalg.inv(cov_0) + BETA * (phi.T.dot(phi))
        )
        self.m_n = self.cov_n.dot(np.linalg.inv(cov_0).dot(m_0) + BETA * phi.T.dot(y_obs))

        print(f"    Updated Mean:  {self.m_n}")
        print(f"    Updated Cov:   {list(map(list, self.cov_n))}")

        self._update_prior_posterior()

    def sample_lines(self, size=1):
        line_data = self.prior_posterior.rvs(size)
        for sample in line_data:
            yield (X, sample[0] + sample[1] * X)

    def reset_state(self):
        self.m_n = M_0.copy()
        self.cov_n = COV_0.copy()
        self._update_prior_posterior()
        self.likelihood = None
        self.points = []

    def _update_prior_posterior(self):
        self.prior_posterior = scipy.stats.multivariate_normal(
            mean=self.m_n, cov=self.cov_n,
        )

    def get_plot_row(self, new_samples=1):
        print(f"Adding {new_samples} new observation(s)")
        raw_samples = ObservationGenerator().sample(new_samples)
        if new_samples > 0:
            self.add_observation(raw_samples)

        likelihood_plot = bokeh_figure()
        if self.likelihood is not None:
            likelihood_plot.image(
                image=[self.likelihood],
                x=X_MIN,
                y=Y_MIN,
                dw=X_MAX - X_MIN,
                dh=Y_MAX - Y_MIN,
                palette="Viridis256"
            )

        prior_plot = bokeh_figure()
        prior_plot.image(
            image=[distribution_to_array(self.prior_posterior)],
            x=X_MIN,
            y=Y_MIN,
            dw=X_MAX - X_MIN,
            dh=Y_MAX - Y_MIN,
            palette="Viridis256"
        )

        sampled_lines_plot = bokeh_figure()
        sampled_lines_plot.scatter(
            x=[p[0] for p in self.points],
            y=[p[1] for p in self.points],
        )
        for x_data, y_data in self.sample_lines(10):
            sampled_lines_plot.line(
                x=x_data,
                y=y_data,
            )
        return [likelihood_plot, prior_plot, sampled_lines_plot]


def titlerow():
    style = {"text-align": "center"}
    return [
        bokeh.models.Div(text="Likelihood of last point", style=style),
        bokeh.models.Div(text="Prior / Posterior", style=style),
        bokeh.models.Div(text="Data Space", style=style),
    ]


def main():
    regression = BayesianLinearRegression()

    bokeh.plotting.output_file("image.html", title="Bishop Figure 3.7")

    rows = [titlerow()] + [regression.get_plot_row(i) for i in [0, 1, 1, 18]]

    layout = bokeh.layouts.gridplot(
        rows, plot_width=FIG_SIZE,
    )

    bokeh.plotting.show(layout)  # open a browser


if __name__ == '__main__':
    main()
