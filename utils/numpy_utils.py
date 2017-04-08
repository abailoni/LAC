import numpy as np

class DynamicRecArray(object):
    """
    EXAMPLE OF USAGE:

        y = DynamicRecArray(('a4,int32,float64'))
        y.extend([("xyz", 12, 3.2), ("abc", 100, 0.2)])
        y.append(("123", 1000, 0))
        print y.data
        for i in xrange(100):
            y.append((str(i), i, i+0.1))

    """
    def __init__(self, dtype, resizing_frequency=50):
        self.dtype = np.dtype(dtype)
        self.length = 0
        self.res_freq = resizing_frequency
        self.size = self.res_freq
        self._data = np.empty(self.size, dtype=self.dtype)

    def __len__(self):
        return self.length

    def __call__(self, *args, **kwargs):
        return self.data()

    def append(self, rec):
        if self.length == self.size:
            self.size = int(self.size+self.res_freq)
            self._data = np.resize(self._data, self.size)
        self._data[self.length] = rec
        self.length += 1

    def extend(self, recs):
        for rec in recs:
            self.append(rec)

    @property
    def data(self):
        return self._data[:self.length]

