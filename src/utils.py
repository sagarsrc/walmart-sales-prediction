
import numpy as np
import pandas as pd

from tqdm import tqdm


class Utils:
    """
    Utility tools for quick development
    """

    def __init__(self):
        pass

    def reduce_memory_usage(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Reduce memory usage of dataframe

        Args:
            df (pd.DataFrame): Input dataframe
            loglevel (str, optional): Set log level : "DEBUG", "INFO". Defaults to None.

        Returns:
            pd.DataFrame: Memory optimized dataframe


        Reference:
        [1] Reducing memory usage
                https://www.kaggle.com/gemartin/load-data-reduce-memory-usage
        """

        prev_memory = df.memory_usage().sum() / 1024**2

        dtypes_int = [
            np.uint8, np.uint16, np.uint32, np.uint64,
            np.int8, np.int16, np.int32, np.int64,
        ]

        dtypes_float = [np.float16, np.float32, np.float64, np.float128]

        for i in tqdm(list(df.columns)):
            col_type = df[i].dtype
            dstr = str(col_type).split('.')[-1][:-2]

            if col_type in dtypes_int or col_type in dtypes_float:

                # get column min max if type of column is numerical
                col_min = df[i].max()
                col_max = df[i].min()

                for d in dtypes_int:
                    # check if column type belongs to int
                    if 'int' in dstr:
                        if col_min > np.iinfo(d).min and col_max < np.iinfo(d).max:
                            df[i] = df[i].astype(d)

                            # as dtypes are arranged in increasing order
                            # do not check with other dtypes with larger range
                            break
                    else:
                        break

                for d in dtypes_float:
                    # check if column type belongs to float
                    if 'float' in dstr:
                        if col_min > np.finfo(d).min and col_max < np.finfo(d).max:
                            df[i] = df[i].astype(d)

                            # as dtypes are arranged in increasing order
                            # do not check with other dtypes with larger range
                            break
                    else:
                        break

            else:

                # if type of column is not numerical
                df[i] = df[i].astype('category')

        new_memory = df.memory_usage().sum() / 1024**2
        p_red = 100*(abs(prev_memory-new_memory))/prev_memory
        print(f"Previous memory usage {prev_memory:.2f} MB")
        print(f"Optimized memory usage {new_memory:.2f} MB")
        print(f"Percentage reduction in memory {p_red:.2f}%")

        return df
