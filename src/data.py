from genericpath import exists
import os
from os.path import exists as file_exists
from glob import glob
import joblib

import pandas as pd

# custom scripts
import utils


class DataHandle:
    """
    Class for loading data
    """

    def __init__(self, data_root: str) -> None:
        """
        Initialize data handler

        Args:
            data_root (str): Root path for data directory
        """

        self.data_root = data_root
        self.util_handle = utils.Utils()

        print(f"Your data root path : {self.data_root}")
        os.makedirs(self.data_root+"/processed/", exist_ok=True)

    def setup_data_dir(self, data_root="../data"):
        os.makedirs(f"{data_root}/extracted", exist_ok=True)
        os.makedirs(f"{data_root}/processed", exist_ok=True)

        raw_files = glob(f"{data_root}/raw/*.zip")
        for rf in raw_files:
            os.system(f"unzip -d {data_root}/extracted/ {rf}")

    def get_calendar(self, memopt=True) -> pd.DataFrame:
        """
        Get calendar.csv data frame
        Args:
            memopt (bool, optional): do memory optimization. Defaults to True.
        Returns:
            pd.DataFrame: calendar dataframe
        """

        save_path = self.data_root+"/processed/calendar_memopt.bin"

        read_file = "/extracted/calendar.csv"

        if file_exists(save_path):
            print(f"Loading {save_path}")
            df = pd.read_pickle(save_path)
        else:
            df = pd.read_csv(self.data_root+read_file)
            if memopt:
                df = self.util_handle.reduce_memory_usage(df)

                if not file_exists(save_path):
                    print(f"Saving {save_path}")
                    df.to_pickle(save_path, )

            else:
                pass

        return df

    def get_sales_train_val(self, memopt=True) -> pd.DataFrame:
        """
        Get sales_train_validation.csv dataframe
        Args:
            memopt (bool, optional): do memory optimization. Defaults to True.
        Returns:
            pd.DataFrame: sales train validation dataframe
        """

        save_path = self.data_root+"/processed/stv_memopt.bin"

        read_file = "/extracted/sales_train_validation.csv"

        if file_exists(save_path):
            print(f"Loading {save_path}")
            df = pd.read_pickle(save_path)
        else:
            df = pd.read_csv(self.data_root+read_file)
            if memopt:
                df = self.util_handle.reduce_memory_usage(df)

                if not file_exists(save_path):
                    print(f"Saving {save_path}")
                    df.to_pickle(save_path, )

            else:
                pass

        return df

    def get_sales_train_eval(self, memopt=True) -> pd.DataFrame:
        """
        Get sales_train_evaluation dataframe
        Args:
            memopt (bool, optional): do memory optimization. Defaults to True.
        Returns:
            pd.DataFrame: sales train eval dataframe
        """

        save_path = self.data_root+"/processed/ste_memopt.bin"

        read_file = "/extracted/sales_train_evaluation.csv"

        if file_exists(save_path):
            print(f"Loading {save_path}")
            df = pd.read_pickle(save_path)
        else:
            df = pd.read_csv(self.data_root+read_file)
            if memopt:
                df = self.util_handle.reduce_memory_usage(df)

                if not file_exists(save_path):
                    print(f"Saving {save_path}")
                    df.to_pickle(save_path, )

            else:
                pass

        return df

    def get_sell_prices(self, memopt=True) -> pd.DataFrame:
        """
        Get sell_prices.csv dataframe
        Args:
            memopt (bool, optional): do memory optimization. Defaults to True.
        Returns:
            pd.DataFrame: sell prices dataframe
        """

        save_path = self.data_root+"/processed/slp_memopt.bin"

        read_file = "/extracted/sell_prices.csv"

        if file_exists(save_path):
            print(f"Loading {save_path}")
            df = pd.read_pickle(save_path)
        else:
            df = pd.read_csv(self.data_root+read_file)
            if memopt:
                df = self.util_handle.reduce_memory_usage(df)

                if not file_exists(save_path):
                    print(f"Saving {save_path}")
                    df.to_pickle(save_path, )

            else:
                pass

        return df

    def get_submission(self, memopt=True) -> pd.DataFrame:
        """
        Get sample_submission.csv dataframe
        Args:
            memopt (bool, optional): do memory optimization. Defaults to True.
        Returns:
            pd.DataFrame: submission dataframe
        """

        save_path = self.data_root+"/processed/sub_memopt.bin"

        read_file = "/extracted/sample_submission.csv"

        if file_exists(save_path):
            print(f"Loading {save_path}")
            df = pd.read_pickle(save_path)
        else:
            df = pd.read_csv(self.data_root+read_file)
            if memopt:
                df = self.util_handle.reduce_memory_usage(df)

                if not file_exists(save_path):
                    print(f"Saving {save_path}")
                    df.to_pickle(save_path, )
            else:
                pass

        return df

    def load_data(self, load_eval=False, load_sub=False) -> dict:
        """
        Load data for training 

        Args:
            load_eval (bool, optional): Load evaluation data. Defaults to False.
            load_sub (bool, optional) : load submission data. Defaults to False.    

        Returns:
            dict: Dictionary of loaded dataframes
        """

        cal_df = self.get_calendar(memopt=False)
        stv_df = self.get_sales_train_val(memopt=True)
        sel_df = self.get_sell_prices(memopt=True)

        data_dict = {
            "calendar": cal_df,
            "sales_train": stv_df,
            "sell_prices": sel_df,
        }

        if load_eval:
            ste_df = self.get_sales_train_eval(memopt=True)
            data_dict["sales_eval"] = ste_df

        if load_sub:
            sub_df = self.get_submission(memopt=True)
            data_dict["submission"] = sub_df

        return data_dict


class DataPrepare(DataHandle):
    """
    Prepare data for training

    Args:
        DataHandle (DataHandle): Object of DataHandle
    """

    def __init__(self, data_root: str) -> None:
        """
        Initialize DataPrepare class

        Args:
            data_root (str): Root path for data directory
        """

        super().__init__(data_root=data_root)

        self.data_root = data_root
        self.setup_data_dir(self.data_root)

    def unpivot_data(self, data: dict) -> dict:
        """
        Unpivot the data for sales_train_validation.csv, submissions.csv

        Args:
            data (dict): dictionary with loaded dataframes

        Returns:
            data (dict): dictionary with unpivoted dataframes

        Reference
            [1] Melt operation
                https://www.kaggle.com/beezus666/end-to-end-data-wrangling-simple-random-forest
        """
        df_cal = data['calendar']
        df_slt = data['sales_train']
        df_slp = data['sell_prices']
        df_sub = data['submission']

        id_vars = ['id', 'item_id', 'dept_id',
                   'cat_id', 'store_id', 'state_id']

        # unpivot sales_train_validation.csv
        df_slt_melted = pd.melt(
            df_slt,
            id_vars=id_vars,
            var_name='day',
            value_name='sales',
        )

        # drop redundant columns from sales_train_validation.csv
        df_slt_melted.drop(
            ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'],
            axis=1,
            inplace=True)

        # prepare sales train dataframe
        df_slt_melted['id'] = df_slt_melted['id'].str.replace(
            '_validation', '')

        # unpivot submissions.csv
        df_sub_melted = pd.melt(
            df_sub,
            id_vars=['id'],
            value_vars=df_sub.drop(['id'], axis=1).columns,
            var_name='day',
            value_name='sales'
        )

        # changing names of day columns from 'Fx' to 'x'
        df_sub_melted['day'] = df_sub_melted['day'].str.replace('F', '')

        # after 1913rd day we need to forecast
        df_sub_melted['day'] = pd.to_numeric(
            df_sub_melted['day'], errors='coerce')
        df_sub_melted['day'] += 1913
        df_sub_melted = df_sub_melted.applymap(str)
        df_sub_melted['day'] = 'd_' + df_sub_melted['day'].astype(str)

        # prepare sell prices dataframe
        df_slp['id'] = df_slp['item_id'].astype(str) + '_' \
            + df_slp['store_id'].astype(str)

        df_slp.drop(columns=['item_id', 'store_id'], inplace=True)

        # combine data
        data_melted = {
            "calendar": self.util_handle.reduce_memory_usage(df_cal),
            "sales_train_melted": self.util_handle.reduce_memory_usage(df_slt_melted),
            "sell_prices": self.util_handle.reduce_memory_usage(df_slp),
            "submission_melted": self.util_handle.reduce_memory_usage(df_sub_melted)
        }

        return data_melted

    def prepare_data(self, eda: bool = True, eda_params: dict = None) -> dict:
        """
        Loads all data files and unpivots data
        Args:
            eda (bool) : load data for EDA
            eda_params (dict) : parameters for getting parts of data for EDA
        Returns:
            dict: Dictionary with melted dataframes
        """
        save_path = self.data_root+"/processed/data_melted.bin"

        if file_exists(save_path):
            print(f"Loading {save_path}")
            data_melted = joblib.load(save_path)
        else:
            data_dict = self.load_data(load_eval=False, load_sub=True)
            data_melted = self.unpivot_data(data_dict)
            print(f"Saving {save_path}")

            joblib.dump(data_melted, save_path, compress=5)

        if eda:
            df_stm = data_melted['sales_train_melted']
            # create columns for EDA
            df_stm['item_id'] = df_stm['id'].str[:-5]
            df_stm['dept_id'] = df_stm['id'].str[:-9]
            df_stm['store_id'] = df_stm['id'].str[-4:]

            # put sales column in the end
            sales = df_stm['sales']
            df_stm.drop(columns=['sales'], inplace=True)
            df_stm['sales'] = sales

            data_melted['sales_train_melted'] = df_stm

            if eda_params:
                # get lot of items that needs to be analysed
                lot = eda_params.get("keep_items", -1)
                save_name = eda_params.get("save_name", "eda_default_lot")
                save_path = f"../data/processed/{save_name}.bin"

                if file_exists(save_path):
                    data_melted = joblib.load(save_path)
                    print(f"Loading {save_path}")

                else:
                    # initialize dictionary to hold data for specific items
                    lot_data = {}

                    # get parts of dataframe with respect ot item id
                    if lot:
                        for k in lot:
                            df_part = df_stm[df_stm['item_id'] == k]
                            lot_data[k] = df_part

                    del data_melted['sales_train_melted']
                    data_melted['lot_data'] = lot_data

                    joblib.dump(data_melted, save_path)
                    print(f"Saving {save_path}")

        return data_melted

    def get_train_data(self) -> pd.DataFrame:
        pass
