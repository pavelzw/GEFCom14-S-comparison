import numpy as np
import pandas as pd
from gluonts.dataset.common import ListDataset
from gluonts.dataset.field_names import FieldName

from pywatts.wrapper.base_wrapper import BaseWrapper
import gluonts
import xarray as xr
from typing import Dict


class GluonTSWrapper(BaseWrapper):
    """
    A wrapper class for GluonTS. Should only used internal by the pipeline itself
    :param module: The gluon-ts estimator to wrap
    :type module: gluonts.mx.model.estimator.GluonEstimator
    :param name: The name of the module
    :type name: str
    """

    def get_params(self) -> Dict[str, object]:
        return {'predictor': self.predictor}

    def set_params(self, *args, **kwargs):
        self.predictor = kwargs['predictor']

    def __init__(self, module: gluonts.mx.model.estimator.GluonEstimator,
                 name: str = None,
                 start=None,
                 freq=None):
        if name is None:
            name = module.__class__.__name__
        self.start = start
        self.freq = freq
        self.predictor = None
        self.target = None
        super().__init__(name)
        self.module = module
        self.targets = []

    def fit(self, **kwargs):
        """
        Fit the gluon-ts module
        :param x: input data
        :param target: target data
        """
        input = np.array(kwargs['x']).T
        target = np.squeeze(np.array(kwargs['target']))
        train_ds = ListDataset([{
            FieldName.TARGET: target[:, i],
            FieldName.START: self.start,
            FieldName.FEAT_DYNAMIC_REAL: input[:, i, :],
        } for i in range(3)], freq=self.freq)

        # hotfix: the model needs the target data for testing, but only the x data is allowed in the transform method
        self.target = target

        self.predictor = self.module.train(train_ds)
        self.is_fitted = True

    def transform(self, **kwargs: xr.DataArray) -> xr.DataArray:
        """
        Transforms a dataset or predicts the result with the wrapped gluon-ts module
        :param x: the input dataset
        :return: the transformed output
        """
        if not self.is_fitted:
            raise ValueError("Module needs to be fitted first.")
        x = kwargs['x']

        if self.target.shape[0] == x.shape[0]:
            return xr.DataArray(np.zeros((kwargs['x'].shape[0], 99)))

        assert self.predictor is not None
        x = kwargs['x'].T

        x_ds = ListDataset([{
            FieldName.TARGET: self.target[:, i],
            FieldName.START: pd.Timestamp(x.TIMESTAMP.data[0]),
            FieldName.FEAT_DYNAMIC_REAL: x[:, i, :]
        } for i in range(3)], freq=self.freq)
        forecasts = list(self.predictor.predict(x_ds))

        predictions = xr.DataArray(coords=[('TIMESTAMP', x.TIMESTAMP.data[-self.module.prediction_length:]),
                                   ('ZONE', [1, 2, 3]),
                                   ('quantiles', [str(p/100) for p in range(1, 100)])])
        for i in range(3):
            predictions[:, i, :] = pd.concat([np.maximum(forecasts[i].quantile_ts(p / 100), 0)
                                              for p in range(1, 100)], axis=1)
        return predictions
