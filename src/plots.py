import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import calplot

# supress warnings
import warnings
warnings.filterwarnings("ignore")

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

    # ********************************************** #
    # ********************************************** #
    # ********************************************** #
    # ********************************************** #
    # *************** STORE LEVEL EDA ************** #
    # ********************************************** #
    # ********************************************** #
    # ********************************************** #
    # ********************************************** #

    ############################################################
    ################ Number of sales per store #################
    ############################################################

    def num_sales_per_store(self, df: pd.DataFrame):
        """
        Visualize total number of sales per store

        """

        df = df.groupby(by='store_id').agg('sum').reset_index()
        ix = df['store_id'].values
        val = df['sales'].values

        plt.figure(figsize=[12, 5])
        plt.title(f"Total sales per store ")

        plt.bar(ix, val, width=0.3)

        plt.xlabel('store_id')
        plt.ylabel('num_sales')
        plt.show()

    ############################################################
    ################## Daily sales per store ###################
    ############################################################

    def view_daily_sales_per_store(self, df: pd.DataFrame, df_cal: pd.DataFrame, window: int = 30):
        """
        View daily sales for each store

        Args:
            df (pd.DataFrame) : sales train melted data frame containing store_id and sales
            df_cal (pd.DataFrame) : calendar DataFrame
            window (int) : window for smoothing daily sales

        """
        # group by store_id
        group_stm = df.groupby(by='store_id')
        group_keys = list(group_stm.groups.keys())

        daily_sales = {}

        plt.figure(figsize=[16, 7])
        plt.title(f"Daily sales for each store")

        # plot sales of each store
        for K in group_keys:
            # get daily sales for each store
            res = group_stm.get_group(name=K).groupby(
                by='day')['sales'].agg('sum').reset_index()

            # merge with actual calendar
            res = res.merge(df_cal, how='left', left_on='day', right_on='d')

            # convert date to datetime format
            res['date'] = pd.to_datetime(res['date'])

            # sort days by day number
            res['day'] = res['day'].str.replace('d_', '').astype(int)
            res.sort_values(by='day', inplace=True)

            # store results in dictionary
            daily_sales[K] = res
            ds = daily_sales[K]

            # plot current plot
            plt.plot(ds['date'], ds['sales'])

        plt.legend(group_keys)
        plt.show()

        print("\n\n")
        print(f"As the data is highly uneven and it is difficult to distinguish between graphs")
        print(
            f"We will do smoothing of daily sales of each store with window of {window} days")
        print("\n\n")

        plt.figure(figsize=[16, 7])
        plt.title(f"Daily sales for each store smooth={window} days")

        for K in group_keys:
            # get daily sales for each store
            res = group_stm.get_group(name=K).groupby(
                by='day')['sales'].agg('sum').reset_index()

            # merge with actual calendar
            res = res.merge(df_cal, how='left', left_on='day', right_on='d')

            # convert date to datetime format
            res['date'] = pd.to_datetime(res['date'])

            # sort days by day number
            res['day'] = res['day'].str.replace('d_', '').astype(int)
            res.sort_values(by='day', inplace=True)

            # rolling mean
            res['sales'] = res['sales'].rolling(window).mean()

            # store results in dict
            daily_sales[K] = res
            ds = daily_sales[K]

            # plot current plot
            plt.plot(ds['date'], ds['sales'])

        plt.legend(group_keys)
        plt.show()

    ############################################################
    ############ View calendar heatmap for a store #############
    ############################################################

    def view_calplot_heatmap(self, df: pd.DataFrame, df_cal: pd.DataFrame, store_id: str = 'CA_1', window: int = None):
        """
        View daily sales for each store

        Args:
            df (pd.DataFrame) : sales train melted data frame containing store_id and sales
            df_cal (pd.DataFrame) : calendar DataFrame
            store_id (str) : unique store_id
            window (int, optional) : window for smoothing daily sales over period of time

        """
        pd.set_option('mode.chained_assignment', None)

        group_stm = df.groupby(by='store_id')
        K = store_id

        # get daily sales for each store
        res = group_stm.get_group(name=K).groupby(
            by='day')['sales'].agg('sum').reset_index()
        del group_stm

        # sort by days
        res['day'] = res['day'].str.replace('d_', '').astype(int)
        res.sort_values(by='day', inplace=True)

        res['day'] = 'd_' + res['day'].astype(str)

        # merge on calendar
        res = res.merge(right=df_cal, how='left', left_on='day', right_on='d')

        # get events i.e sales for each date
        events = res[['date', 'sales']]
        events.iloc[:, 0] = pd.to_datetime(
            events['date'], format="%Y-%m-%d", errors='coerce')
        events.set_index(keys=['date'], inplace=True)
        events = events['sales']

        # if window not None
        if window:
            # smooth using rolling window average
            events = events.rolling(window).mean()
            print(
                f"Calendar heatmap for sales of store {K} rolling average {window} days period")
        else:
            # do nothing
            print(f"Calendar heatmap for sales of store {K}")

        # do calplot
        calplot.calplot(events)
        plt.show()
        return events

    # ********************************************** #
    # ********************************************** #
    # ********************************************** #
    # ********************************************** #
    # *************** ITEM LEVEL EDA *************** #
    # ********************************************** #
    # ********************************************** #
    # ********************************************** #
    # ********************************************** #

    ############################################################
    ################ Per day sales of an Item ##################
    ############################################################

    def view_per_day_sale_item(self,
                               df: pd.DataFrame,
                               df_cal: pd.DataFrame,
                               item_name: str = "___",
                               view_calplot=True):
        """
        Visualize sales of item across each store

        Args:
            df (pd.DataFrame): DataFrame with details of sales of an item
            df_cal (pd.DataFrame) : calendar DataFrame
            item_name (str, optional): Name of item being analyzed. Defaults to "___".
            view_calplot (bool, optional) : View calendar heatmap. Defaults to True. 
        """

        # group by days and sum all sales of that day for that item
        res = df.groupby(by='day')['sales'].agg('sum').reset_index()

        # sort by days
        res['day'] = res['day'].str.replace('d_', '').astype(int)
        res.sort_values(by='day', inplace=True)

        res['day'] = 'd_' + res['day'].astype(str)

        # merge on calendar
        res = res.merge(right=df_cal, how='left', left_on='day', right_on='d')

        # convert to datetime
        res['date'] = pd.to_datetime(
            res['date'], format="%Y-%m-%d", errors='coerce')

        plt.figure(figsize=[10, 5])
        plt.title(f"Per day sales for {item_name}")
        sns.lineplot(
            x='date',
            y='sales',
            data=res,
        )
        plt.show()

        if view_calplot:
            temp = res
            events = temp[['date', 'sales']]
            events.set_index(keys=['date'], inplace=True)
            events = events['sales']

            print(f"Per day sales for {item_name} Heatmap")

            calplot.calplot(events)
            plt.show()

    ############################################################
    ############# Number of item sales per store ###############
    ############################################################

    def view_item_sales_store(self, df: pd.DataFrame, df_cal: pd.DataFrame, item_name: str = "___"):
        """
        Visualize sales of item for each store

        Args:
            df (pd.DataFrame): DataFrame with details of sales of an item
            df_cal (pd.DataFrame) : calendar DataFrame
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

                # merge with actual calendar
                per_day_sales = per_day_sales.merge(
                    df_cal, how='left', left_on='day', right_on='d')

                # convert date to datetime format
                per_day_sales['date'] = pd.to_datetime(per_day_sales['date'])

                # rename all days
                per_day_sales['day'] = per_day_sales['day'].str.replace(
                    'd_', '').astype(int)

                # sort by days
                per_day_sales = per_day_sales.sort_values(by='day')

                axes[row_no, col_no].title.set_text(f"Store {keys[k]}")
                sns.lineplot(x='date',
                             y='sales',
                             data=per_day_sales,
                             ax=axes[row_no, col_no])

                # increment key index of group
                k += 1

        # plot
        fig.tight_layout()
        plt.show()

    ############################################################
    ############### Average item sales per store ###############
    ############################################################

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
