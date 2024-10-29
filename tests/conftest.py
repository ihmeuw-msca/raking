from pathlib import Path

import pandas as pd
import pytest

EXAMPLES = Path(__file__).resolve().parent / "examples"


class Example1D:
    def __init__(self):
        self.df_obs = pd.read_csv(EXAMPLES / "example_1D" / "observations.csv")
        self.df_margin = pd.read_csv(EXAMPLES / "example_1D" / "margin.csv")


class Example1D_draws:
    def __init__(self):
        self.df_obs = pd.read_csv(
            EXAMPLES / "example_1D_draws" / "observations.csv"
        )
        self.df_margin = pd.read_csv(
            EXAMPLES / "example_1D_draws" / "margin.csv"
        )


class Example2D:
    def __init__(self):
        self.df_obs = pd.read_csv(EXAMPLES / "example_2D" / "observations.csv")
        self.df_margins_1 = pd.read_csv(
            EXAMPLES / "example_2D" / "margins_1.csv"
        )
        self.df_margins_2 = pd.read_csv(
            EXAMPLES / "example_2D" / "margins_2.csv"
        )


class Example2D_draws:
    def __init__(self):
        self.df_obs = pd.read_csv(
            EXAMPLES / "example_2D_draws" / "observations.csv"
        )
        self.df_margins_1 = pd.read_csv(
            EXAMPLES / "example_2D_draws" / "margins_1.csv"
        )
        self.df_margins_2 = pd.read_csv(
            EXAMPLES / "example_2D_draws" / "margins_2.csv"
        )


class Example3D:
    def __init__(self):
        self.df_obs = pd.read_csv(EXAMPLES / "example_3D" / "observations.csv")
        self.df_margins_1 = pd.read_csv(
            EXAMPLES / "example_3D" / "margins_1.csv"
        )
        self.df_margins_2 = pd.read_csv(
            EXAMPLES / "example_3D" / "margins_2.csv"
        )
        self.df_margins_3 = pd.read_csv(
            EXAMPLES / "example_3D" / "margins_3.csv"
        )


class Example3D_draws:
    def __init__(self):
        self.df_obs = pd.read_csv(
            EXAMPLES / "example_3D_draws" / "observations.csv"
        )
        self.df_margins_1 = pd.read_csv(
            EXAMPLES / "example_3D_draws" / "margins_1.csv"
        )
        self.df_margins_2 = pd.read_csv(
            EXAMPLES / "example_3D_draws" / "margins_2.csv"
        )
        self.df_margins_3 = pd.read_csv(
            EXAMPLES / "example_3D_draws" / "margins_3.csv"
        )


class ExampleUSHD:
    def __init__(self):
        self.df_obs = pd.read_csv(
            EXAMPLES / "example_USHD" / "observations.csv"
        )
        self.df_margins = pd.read_csv(EXAMPLES / "example_USHD" / "margins.csv")


class ExampleUSHD_draws:
    def __init__(self):
        self.df_obs = pd.read_csv(
            EXAMPLES / "example_USHD_draws" / "observations.csv"
        )
        self.df_margins = pd.read_csv(
            EXAMPLES / "example_USHD_draws" / "margins.csv"
        )


@pytest.fixture
def example_1D():
    return Example1D()


@pytest.fixture
def example_1D_draws():
    return Example1D_draws()


@pytest.fixture
def example_2D():
    return Example2D()


@pytest.fixture
def example_2D_draws():
    return Example2D_draws()


@pytest.fixture
def example_3D():
    return Example3D()


@pytest.fixture
def example_3D_draws():
    return Example3D_draws()


@pytest.fixture
def example_USHD():
    return ExampleUSHD()


@pytest.fixture
def example_USHD_draws():
    return ExampleUSHD_draws()
