import sqlalchemy
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels \
    import RBF, WhiteKernel, RationalQuadratic, ExpSineSquared, ConstantKernel
import numpy as np
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
import os
from pathlib import Path

def gp_line_plot(pred, std, df):
    plt.plot(df.index.values, pred)
    plt.fill_between(df.index.values, pred - std, pred + std, alpha=0.5)
    plt.plot(df.index.values, df['weight'].values, 'k.')

    plt.ylabel('Weight [kg]')

def gp_distribution_plot(samples):
    weight_differential = samples[1, :] - samples[0, :]
    
    kde = KernelDensity(kernel='gaussian', bandwidth=0.01).fit(weight_differential.reshape(-1, 1))

    plot_x = np.linspace(np.min(weight_differential), np.max(weight_differential), 1000)

    plot_y = np.exp(kde.score_samples(plot_x.reshape(-1, 1)))

    plt.plot(plot_x, plot_y)

    # Data with weight loss
    loss_idx = plot_x < 0

    plt.fill_between(plot_x[loss_idx], plot_y[loss_idx], np.zeros_like(plot_x[loss_idx]), alpha=0.5)

    loss_prob = np.trapz(plot_x[loss_idx], plot_y[loss_idx])
    full_prob = np.trapz(plot_x, plot_y)

    plt.text(np.min(plot_x), np.max(plot_y) * 0.9, 'Weight loss chance {:.2f} %'.format(loss_prob / full_prob * 100))
    plt.xlabel('Weight [kg]')


def gp_fit(df):
    # Convert time to float in days
    all_times = df.index.values
    time_x = df.index.to_julian_date().values
    time_x -= time_x.min()

    kernel = 1**2 * RBF(length_scale=7, length_scale_bounds=(1, 20)) + WhiteKernel(noise_level=0.1**2, noise_level_bounds=(1e-7, 40))

    gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-10,
                                normalize_y=True, n_restarts_optimizer=10)

    gp.fit(time_x.reshape(-1, 1), df['weight'].values)

    y_pred, y_std = gp.predict(time_x.reshape(-1, 1), return_std=True)

    gp_samples = gp.sample_y(np.array([time_x[0], time_x[-1]]).reshape(-1, 1), n_samples=100000)

    return y_pred, y_std, gp_samples

async def make_plots():
    db = sqlalchemy.create_engine(r'postgresql+psycopg2://{}:{}@{}:5432/weight'.format(os.environ['SQL_USER'], os.environ['SQL_PASS'], os.environ['SQL_SERVER']))

    cur_dir = Path(__file__).parent
    out_dir = cur_dir.joinpath('static')

    data = pd.read_sql('select * from weight', db).set_index(['time'])
    fig = data.where(data['name'] == 'fig').dropna().sort_values(by='time')
    como = data.where(data['name'] == 'como').dropna().sort_values(by='time')

    fig_pred, fig_std, fig_gp_samples = gp_fit(fig)
    como_pred, como_std, como_gp_samples = gp_fit(como)

    plt.figure(figsize=(24, 4))
    plt.subplot(1, 2, 1)
    gp_line_plot(como_pred, como_std, como)
    plt.title('Co-mo')

    plt.subplot(1, 2, 2)
    gp_line_plot(fig_pred, fig_std, fig)

    plt.title('Fig')
    plt.tight_layout()
    plt.savefig(out_dir.joinpath('weight_fit.png').as_posix(), dpi=250, bbox_inches='tight')

    plt.figure(figsize=(24, 4))
    plt.subplot(1, 2, 1)
    gp_distribution_plot(como_gp_samples)
    plt.title('Co-mo')

    plt.subplot(1, 2, 2)
    gp_distribution_plot(fig_gp_samples)

    plt.title('Fig')
    plt.tight_layout()
    plt.savefig(out_dir.joinpath('weight_distribution.png').as_posix(), dpi=250, bbox_inches='tight')
