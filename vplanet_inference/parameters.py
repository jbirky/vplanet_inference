import numpy as np

__all__ = ["VplanetParameters"]


class VplanetParameters(object):

    def __init__(self, names, units, bounds=None, true=None, data=None, labels=None, uncertainty=None):

        # required
        self.names = names
        self.units = units

        # optional
        self.bounds = bounds
        self.data = data
        self.true = true
        self.uncertainty = uncertainty

        if (labels is None) or (labels[0] is None):
            self.labels = self.names
        else:
            self.labels = labels

        self.num = len(self.names)

        if self.units is not None:
            self.dict_units = dict(zip(self.names, self.units))
        if self.bounds is not None:
            self.dict_bounds = dict(zip(self.names, self.bounds))
        if self.true is not None:
            self.dict_true = dict(zip(self.names, self.true))
        if self.data is not None:
            self.dict_data = dict(zip(self.names, self.data))
        if self.labels is not None:
            self.dict_labels = dict(zip(self.names, self.labels))


    def set_data(self, true):

        self.true = true
        self.data = np.array([self.true, self.uncertainty]).T

        self.dict_true = dict(zip(self.names, self.true))
        self.dict_data = dict(zip(self.names, self.data))


    def get_true_units(self, param_name):

        return self.true[param_name] * self.units[param_name]