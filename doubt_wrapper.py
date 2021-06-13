import xarray as xr
from pywatts.wrapper.base_wrapper import BaseWrapper


class DoubtWrapper(BaseWrapper):
    """
    A wrapper class for doubt modules. Should only used internal by the pipeline itself
    :param module: The doubt module to wrap
    :param name: The name of the module
    :type name: str
    """

    def __init__(self, module, quantiles=None, name: str = None):
        if name is None:
            name = module.__class__.__name__
        super().__init__(name)
        self.module = module
        if quantiles is None:
            self.quantiles = [0.5]
        else:
            self.quantiles = quantiles

    def get_params(self):
        """
        Return the parameter of the doubt module
        :return:
        """
        return self.module.get_params()

    def set_params(self, **kwargs):
        """
        Set the parameter of the internal sklearn module
        :param kwargs: The parameter of the internal sklearn module
        :return:
        """
        return self.module.set_params(**kwargs)

    def fit(self, **kwargs):
        """
        Fit the sklearn module
        :param x: input data
        :param target: target data
        """
        x = kwargs['x'].data
        target = kwargs['target'].data
        self.module.fit(x, target)
        self.is_fitted = True

    def transform(self, **kwargs: xr.DataArray) -> xr.DataArray:
        """
        Transforms a dataset or predicts the result with the wrapped doubt module
        :param x: the input dataset
        :return: the transformed output
        """
        x = kwargs['x'].data
        _, predictions = self.module.predict(x, quantiles=self.quantiles)
        return xr.DataArray(predictions)
