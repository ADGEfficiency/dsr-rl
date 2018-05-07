import numpy as np

#  use epsilon to catch div/0 errors
epsilon = 1e-5


class Processor(object):
    """
    A base class for Processor objects.

    args
        length (int) the length of the array
                      this corresponds to the second dimension of the shape
                      shape = (num_samples, length)
    """
    def __init__(self, length):
        self.length = length

    def __call__(self, batch):
        return self.transform(batch)

    def transform(self, batch):
        """
        Preprocesses array into 2 dimensions.

        args
            batch (np.array)

        returns
            transformed_batch (np.array) shape=(num_samples, length)
        """
        if batch.ndim > 2:
            raise ValueError('batch dimension is greater than 2')

        #  reshape into two dimensions
        batch = np.reshape(batch, (-1, self.length))

        return self._transform(batch)


class Normalizer(Processor):
    """
    Processor object for performing normalization to range [0,1]
    Normalization = (val - min) / (max - min)

    args
        length (int) the length of the array
                      this corresponds to the second dimension of the shape
                      shape=(num_samples, length)
    """
    def __init__(self, length):
        #  initialize the parent Processor class
        super().__init__(length)

        self.mins = None
        self.maxs = None

    def _transform(self, batch):
        """
        Transforms a batch (shape=(num_samples, length))

        if we are using a space -> use mins & maxs set in __init__
        if using history -> min & max over history + batch
        else -> min & max over batch

        args
            batch (np.array)
        returns
            transformed_batch (np.array) shape=(num_samples, length)
        """
        #  use statistics from the batch
        self.mins = batch.min(axis=0).reshape(1, self.length)
        self.maxs = batch.min(axis=0).reshape(1, self.length)

        #  perform the min & max normalization
        return (batch - self.mins) / (self.maxs - self.mins + epsilon)


class Standardizer(Processor):
    """
    Processor object for performing standardization
    Standardization = scaling for zero mean, unit variance

    args
        length (int) the length of the array
                      this corresponds to the second dimension of the shape
                      shape=(num_samples, length)
    """
    def __init__(self, length):
        #  initialize the parent Processor class
        super().__init__(length)

        #  setup initial statistics
        self.count = 0
        self.sum = np.zeros(shape=(1, self.length))
        self.sum_sq = np.zeros(shape=(1, self.length))

    def _transform(self, batch):
        """
        Transforms a batch shape=(num_samples, length).

        If we are using history, we make use of the three counters.

        Else we process the batch using the mean & standard deviation
        from the batch.

        args
            batch (np.array)

        returns
            transformed_batch (np.array) shape=(num_samples, length)
        """
        #  calculate mean & standard deivation from the batch
        self.means, self.stds = batch.mean(axis=0), batch.std(axis=0)

        #  perform the de-meaning & scaling by standard deivation
        return (batch - self.means) / (self.stds + epsilon)
