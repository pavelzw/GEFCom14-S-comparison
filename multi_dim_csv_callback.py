from typing import Optional, Dict

import pandas as pd
import xarray as xr

from pywatts.callbacks.base_callback import BaseCallback


class MultiDimCSVCallback(BaseCallback):
    """
    Callback class to save csv files.

    :param BaseCallback: Base callback as parent class.
    :type BaseCallback: BaseCallback
    """

    def __init__(self, prefix: str, use_filemanager: Optional[bool] = None):
        """
        Initialise csv callback class given a prefix and optional use_filemanager flag.

        :param prefix: Prefix of the CSV file that should be written.
        :type prefix: str
        :param use_filemanager: Optional flag to set if the filemanager of the pipeline should be used.
        :type use_filemanager: Optional[bool]
        """
        if use_filemanager is None:
            # use base class default if use_filemanager is not set
            super().__init__()
        else:
            super().__init__(use_filemanager)
        self.prefix = prefix
        self.name = 'predictions'

    def __call__(self, data_dict: Dict[str, xr.DataArray]):
        """
        Implementation of abstract base __call__ method
        to write the csv file to a given location based on the filename.

        :param data_dict: Dict of DataArrays that should be written to CSV files.
        :type data_dict: Dict[str, xr.DataArray]
        """
        for key in data_dict:
            df = data_dict[key].to_dataframe(name=self.name).reset_index()
            export = pd.DataFrame({**{
                'TIMESTAMP': pd.Series([], dtype='int'),
                'ZONE': pd.Series([], dtype='int')
            }, **{
                p / 100: pd.Series([], dtype='float')
                for p in range(1, 100)
            }})

            for _, row in df[df['quantiles'] == '0.01'].iterrows():
                quantiles = df[(df['TIMESTAMP'] == row["TIMESTAMP"]) & (df['ZONE'] == row['ZONE'])]
                quantiles2 = quantiles[['quantiles', self.name]].reset_index(drop=True).T
                quantiles2.columns = [p / 100 for p in range(1, 100)]
                quantiles2.insert(loc=0, column='ZONE', value=row['ZONE'])
                quantiles2.insert(loc=0, column='TIMESTAMP', value=row['TIMESTAMP'])

                export = export.append(quantiles2.iloc[1], ignore_index=True)
            export = export.reset_index(drop=True).set_index(['TIMESTAMP', 'ZONE'])
            export.to_csv(self.get_path(f"{self.prefix}_{key}.csv"))
