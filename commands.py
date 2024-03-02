import fire
from src.visualizations import (
    PlotActivations,
    PlotSLU
)


def plot_activations():
    PlotActivations().evaluate()


def plot_SLU():
    PlotSLU().evaluate()


def plot_loss():
    pass


def plot_gradients():
    pass


if __name__ == "__main__":
    fire.Fire()

