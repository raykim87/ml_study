import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor

master_historical_df = pd.read_csv('/Users/raykim/Projects/historical_price/master_historical_price.csv', index_col=[0])
master_historical_df.index = pd.to_datetime(master_historical_df.index)


def plot_by_industry(ind: str, start_date: str):
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(15, 20))
    # closing value
    df = master_historical_df[master_historical_df.index >= pd.Timestamp(start_date)]

    for comp in df[df.industry == ind]['title'].unique():
        axes[0].plot(df[df.title == comp]['Close'], label=comp)
    #         axes[0].legend(bbox_to_anchor=(1, 1))

    # closing value normalised by the first value
    for comp in df[df.industry == ind]['title'].unique():
        df2 = df[df.title == comp]['Close'].sort_index()
        df2 = df2 / df2[0]
        axes[1].plot(df2, label=comp)

    cursor = Cursor(ax=axes[0])
#         df2.plot(label=comp, ax=axes[1])
#         axes[1].legend(bbox_to_anchor=(1, 1))
#     datacursor()
    plt.show()

plot_by_industry('energy', '2019-01-01')