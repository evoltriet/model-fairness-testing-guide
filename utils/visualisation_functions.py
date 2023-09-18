# Source: https://github.com/valeria-io/bias-in-credit-models

import pandas as pd
import numpy as np
from math import pi
from scipy import stats

from bokeh.plotting import figure
from bokeh.io import export_svgs
from bokeh.models import ColumnDataSource, NumeralTickFormatter, FactorRange, LinearAxis, Range1d
import warnings

warnings.filterwarnings('ignore')


def save_plot(p, file_name, path='../static/images/'):
    """
    Saves Bokeh plot figure as svg
    :param p: Bokeh plot figure
    :param file_name: name for the plot
    :param path: path where file is saved
    """
    p.output_backend = "svg"
    export_svgs(p, filename=path + file_name + '.svg')

def plot_dual_axis_bar_line(df: pd.DataFrame, title: str, groups_name: str, bar_target_name_variable0: str,
                            bar_target_name_variable1: str, bar_variables: list,
                            line_target_name: str, left_axis_y_label: str, right_axis_y_label: str,
                            bar_colours: list=["#8c9eff", "#536dfe"], plot_height=300, plot_width=700):

    """

    :param df: wide dataframe with data for each bar, the categorical valriables, the grouping and line
    :param title: title of plot
    :param groups_name: name for the column where the groups are
    :param bar_target_name_variable0: name for the bar chart of the first variable
    :param bar_target_name_variable1: name for the bar chart of the second variable
    :param bar_variables: names of the variables used as a list
    :param line_target_name: name of the column for the line chart
    :param left_axis_y_label: label name for the left axis (related to the bar chart)
    :param right_axis_y_label: label name for the right axis (related to the line chart)
    :param bar_colours: colours used for each variable
    :param plot_height: height of the plot
    :param plot_width: width of the plot
    :return: figure with bar chart in left axis and line chart in right axis
    """
    df = df.copy()

    groups = df[groups_name].unique()

    tp_rates = [[df.loc[index, bar_target_name_variable0],
                 df.loc[index, bar_target_name_variable1]]
                for index, row in df.iterrows()]

    tp_rates = [item for sublist in tp_rates for item in sublist]

    index_tuple = [(group_, bar_variable) for group_ in groups for bar_variable in bar_variables]
    colours = bar_colours * len(groups)

    p = figure(x_range=FactorRange(*index_tuple), plot_height=plot_height, plot_width=plot_width,
               title=title, tools="save")

    """ Bar chart specific """
    source = ColumnDataSource(data=dict(x=index_tuple, counts=tp_rates, profits=list(df[line_target_name]),
                                        colours=colours))
    p.vbar(x='x', top='counts', width=0.9, source=source, color='colours')

    """ Line chart specific """
    p.line(x=list(groups), y=list(df[line_target_name]), y_range_name=right_axis_y_label, line_color="#ffca28",
           line_width=2)
    p.circle(x=list(groups), y=list(df[line_target_name]), y_range_name=right_axis_y_label, color="#ffca28", size=7)

    """ Axis specific """
    p.y_range = Range1d(0, 1)
    p.yaxis.axis_label = left_axis_y_label
    p.extra_y_ranges = {right_axis_y_label: Range1d(start=0, end=max(df[line_target_name])*1.2)}
    p.add_layout(LinearAxis(y_range_name=right_axis_y_label, axis_label=right_axis_y_label), 'right')
    p.yaxis[0].formatter = NumeralTickFormatter(format='0 %')

    p.x_range.range_padding = 0.1
    p.xaxis.major_label_orientation = 1
    p.xgrid.grid_line_color = None

    return p





