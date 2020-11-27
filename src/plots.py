import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


plt.style.use('seaborn-whitegrid')
plt.rcParams['axes.edgecolor'] = '#333F4B'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['xtick.color'] = '#333F4B'
plt.rcParams['ytick.color'] = '#333F4B'


SMALL = 10
MEDIUM = 12
LARGE = 18

plt.rc('xtick', labelsize=SMALL)
plt.rc('ytick', labelsize=SMALL)
plt.rc('axes', labelsize=MEDIUM)

plt.rc('axes', titlesize=LARGE)
plt.rc('figure', titlesize=LARGE)


class Plot:
    def __init__(self):
        pass

    def num_sales_per_store(self, df: pd.DataFrame, item_name: str = "___"):
        """
        Visualize number of sales per store

        """

        df = df.groupby(by='store_id').agg('sum').reset_index()
        ix = df['store_id'].values
        val = df['sales'].values

        plt.figure(figsize=[12, 5])
        plt.title(f"Total sales per store for {item_name}")

        plt.bar(ix, val, width=0.3)

        plt.xlabel('store_id')
        plt.ylabel('num_sales')
        plt.show()

    def view_item_sales_store(self, df: pd.DataFrame, item_name: str = "___"):
        """
        Visualize sales of item for each store

        Args:
            df (pd.DataFrame): DataFrame with details of sales of an item
            item_name (str, optional): Name of item being analyzed. Defaults to "___".
        """

        # group by stores
        df_group = df.groupby(by='store_id')
        keys = list(df_group.groups.keys())

        # set subplot params
        nrows = 5
        ncols = 2
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols)
        fig.set_figheight(15)
        fig.set_figwidth(12)
        plt.suptitle(f"Item demand of stores for {item_name}", fontsize=18)

        # initialize subplots
        plt.subplots_adjust(wspace=0.3, hspace=0.3)

        # initialize group keys index
        k = 0

        for row_no in range(nrows):

            for col_no in range(ncols):
                df_temp = df_group.get_group(keys[k])

                # group by day and sum all sales
                per_day_sales = df_temp.groupby(
                    by='day').agg('sum').reset_index()

                # rename all days
                per_day_sales['day'] = per_day_sales['day'].str.replace(
                    'd_', '').astype(int)

                # sort by days
                per_day_sales = per_day_sales.sort_values(by='day')

                axes[row_no, col_no].title.set_text(f"Store {keys[k]}")
                sns.lineplot(x='day',
                             y='sales',
                             data=per_day_sales,
                             ax=axes[row_no, col_no])

                # increment key index of group
                k += 1

        # plot
        fig.tight_layout()
        plt.show()

    def view_avg_item_sales(self,
                            df: pd.DataFrame,
                            df_cal: pd.DataFrame,
                            period='month',
                            item_name: str = "___"):
        """
        Visualize average item sales for each store over various periods : week, month, year

        Args:
            df (pd.DataFrame): DataFrame containing sales data for an item
            df_cal (pd.DataFrame): Calendar DataFrame
            period (str, optional): Period to do averaging. Defaults to 'month'.
            item_name (str, optional): Name of item being analyzed. Defaults to "___".
        """

        # group by stores
        df_group = df.groupby(by='store_id')
        keys = list(df_group.groups.keys())

        # set subplot params
        nrows = 5
        ncols = 2
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols)
        fig.set_figheight(15)
        fig.set_figwidth(12)
        plt.suptitle(
            f"Average sales per {period} for {item_name}", fontsize=18)

        # initialize subplots
        plt.subplots_adjust(wspace=0.3, hspace=0.3)

        # initialize group keys index
        k = 0

        for row_no in range(nrows):

            for col_no in range(ncols):
                df_temp = df_group.get_group(keys[k])

                # group by day and sum all sales
                per_day_sales = df_temp.groupby(
                    by='day').agg('sum').reset_index()

                # rename all days
                per_day_sales['day'] = per_day_sales['day'].str.replace(
                    'd_', '').astype(int)

                # sort by days
                per_day_sales = per_day_sales.sort_values(by='day')
                per_day_sales['day'] = 'd_' + per_day_sales['day'].astype(str)

                # merge on calendar
                per_day_sales = per_day_sales.merge(right=df_cal,
                                                    how='left',
                                                    left_on='day',
                                                    right_on='d')
                per_day_sales['date'] = pd.to_datetime(per_day_sales['date'])
                per_day_sales['month'] = per_day_sales['date'].apply(
                    lambda x: x.month)

                # average sales per week
                if period == 'week':
                    # get per month average monthly sales
                    per_period_avg_sale = per_day_sales.groupby(
                        by=['year', 'month', 'wday'])['sales'].agg(
                            'mean').reset_index()

                # average sales per month
                if period == 'month':
                    # get per month average monthly sales
                    per_period_avg_sale = per_day_sales.groupby(
                        by=['year', 'month'])['sales'].agg('mean').reset_index()

                # average sales per year
                if period == 'year':
                    # get per month average monthly sales
                    per_period_avg_sale = per_day_sales.groupby(
                        by=['year'])['sales'].agg('mean').reset_index()

                # set x_ticks according to period
                per_period_avg_sale[f"{period}_num"] = np.arange(
                    per_period_avg_sale.shape[0])
                axes[row_no, col_no].title.set_text(f"Store {keys[k]}")

                # line plot average sales per period
                sns.lineplot(x=f"{period}_num",
                             y="sales",
                             data=per_period_avg_sale,
                             ax=axes[row_no, col_no])

                # increment key index of group
                k += 1

        # plot
        fig.tight_layout()
        plt.show()
