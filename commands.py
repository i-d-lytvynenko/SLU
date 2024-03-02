import fire
from matplotlib.pyplot import savefig
from src.visualizations import (
    PlotActivations,
    PlotSLU
)


def plot_activations():
    PlotActivations().evaluate()
    savefig('logs/activations.png')


def plot_SLU():
    PlotSLU(x_min=-20, x_max=20).evaluate()
    savefig('logs/slu_20.png')
    PlotSLU(x_min=-3, x_max=3).evaluate()
    savefig('logs/slu_3.png')


def plot_loss():
    pass


def plot_gradients():
    pass


if __name__ == "__main__":
    fire.Fire()

