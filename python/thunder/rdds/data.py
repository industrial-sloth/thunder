class Data(object):
    """
    Generic base class for data types in thunder.

    All data types are backed by an RDD of key-value pairs
    where the key is a tuple identifier and the value is an array

    This base class mainly provides convenience functions for accessing
    properties of the object using the appropriate RDD methods.

    Attributes
    ----------

    rdd: Spark RDD
        The Spark Resilient Distributed Dataset wrapped by this Data object.
        Standard pyspark RDD methods on a data instance `obj` that are not already
        directly exposed by the Data object can be accessed via `obj.rdd`.
    """
    _metadata = []

    def __init__(self, rdd):
        self.rdd = rdd

    def populateParamsFromFirstRecord(self):
        """Calls first() on the underlying rdd, using the returned record to determine appropriate attribute settings
        for this object (for instance, setting self.dtype to match the dtype of the underlying rdd records).

        This method is expected to be overridden by subclasses. Subclasses should first call
        super(cls, self).populateParamsFromFirstRecord(), then use the returned record to set any additional attributes.

        Returns the result of calling self.rdd.first().
        """
        return self.rdd.first()

    def __finalize__(self, other):
        """
        Lazily propagate attributes from other to self, only if attributes
        are not already defined in self

        Parameters
        ----------
        other : the object from which to get the attributes that we are going
            to propagate

        """
        if isinstance(other, Data):
            for name in self._metadata:
                if (getattr(other, name, None) is not None) and (getattr(self, name, None) is None):
                    object.__setattr__(self, name, getattr(other, name, None))
        return self

    @property
    def _constructor(self):
        raise NotImplementedError

    def first(self):
        """ Return first record.

        This calls the Spark first() method on the underlying RDD. As a side effect, any attributes on this object that
        can be set based on the values of the first record will be set (see populateParamsFromFirstRecord).
        """
        return self.populateParamsFromFirstRecord()

    def take(self, *args, **kwargs):
        """ Take samples

        This calls the Spark take() method on the underlying RDD.
        """
        return self.rdd.take(*args, **kwargs)

    def values(self):
        """ Return values, ignoring keys

        This calls the Spark values() method on the underlying RDD.
        """
        return self.rdd.values()

    def keys(self):
        """ Return keys, ignoring values

        This calls the Spark keys() method on the underlying RDD.
        """
        return self.rdd.keys()

    def collect(self):
        """ Return all records to the driver

        This will be slow for large datasets, and may exhaust the available memory on the driver.

        This calls the Spark collect() method on the underlying RDD.
        """
        return self.rdd.collect()

    def count(self):
        """ Mean of values, ignoring keys

        This calls the Spark count() method on the underlying RDD.
        """
        return self.rdd.count()

    def cache(self):
        """ Enable in-memory caching

        This calls the Spark cache() method on the underlying RDD.
        """
        self.rdd.cache()
        return self

    def filterOnKeys(self, func):
        """ Filter records by applying a function to keys """
        return self._constructor(self.rdd.filter(lambda (k, v): func(k))).__finalize__(self)._resetCounts()

    def filterOnValues(self, func):
        """ Filter records by applying a function to values """
        return self._constructor(self.rdd.filter(lambda (k, v): func(v))).__finalize__(self)._resetCounts()


class NumpyData(Data):
    """Abstract base class for Data objects whose values are backed by numpy arrays

    Subclassing implementations should override populateParamsFromFirstRecord() to set
    the _dtype attribute, and should implement astype().
    """
    _metadata = Data._metadata + ['_dtype']

    def __init__(self, rdd, dtype=None):
        super(NumpyData, self).__init__(rdd)
        # 'if dtype' is False here when passed a numpy dtype object.
        self._dtype = dtype

    @property
    def dtype(self):
        if not self._dtype:
            self.populateParamsFromFirstRecord()
        return self._dtype

    def _astypeImpl(self, dtype, casting='safe'):
        """Implementations should return a new RDD, with the numpy array data updated
        to the passed dtype by the requested casting rules.
        """
        raise NotImplementedError("Subclasses should override this")

    def astype(self, dtype, casting='safe'):
        """Cast values to specified numpy dtype

        Calls numpy's astype() method.

        If the string 'smallfloat' is passed, then the values will be cast to the smallest floating point representation
        to which they can be cast safely, as determined by the thunder.utils.common smallest_float_type function.
        Typically this will be a float type larger than a passed integer type (for instance, float16 for int8 or uint8).

        If the passed dtype is the same as the current dtype, or if 'smallfloat' is passed when values are already
        in floating point, then this method will return immediately, returning self.

        Parameters
        ----------
        dtype: numpy dtype or dtype specifier, or string 'smallfloat', or None
            Data type to which RDD values are to be cast. Will return immediately, performing no cast, if None is passed.

        casting: 'no'|'equiv'|'safe'|'same_kind'|'unsafe', optional, default 'safe'
            Casting method to pass on to numpy's astype() method; see numpy documentation for details.

        Returns
        -------
        New Data object, of same type as self, with values cast to the requested dtype; or self if no cast is performed.
        """
        if dtype is None:
            return self
        if dtype == 'smallfloat':
            # get the smallest floating point type that can be safely cast to from our current type
            from thunder.utils.common import smallest_float_type
            dtype = smallest_float_type(self.dtype)
        if str(dtype) == str(self.dtype):
            # no cast required
            return self
        nextrdd = self._astypeImpl(dtype, casting)
        return self._constructor(nextrdd, dtype=dtype).__finalize__(self)

    def numpyValues(self):
        """Return an RDD of the numpy arrays backing this data.

        The returned RDD will not have keys, only numpy array values.
        """
        raise NotImplementedError("Subclasses should override this")

    def mean(self, dtype='smallfloat', casting='safe'):
        """ Mean of values, ignoring keys

        If dtype is not None, then the values will first be cast to the requested type before the operation is
        performed. See Data.astype() for details.

        obj.mean() is equivalent to obj.astype(dtype, casting).rdd.values().mean().
        """
        out = self.astype(dtype, casting)
        return out.numpyValues().mean()

    def sum(self, dtype=None, casting='safe'):
        """ Sum of values, ignoring keys

        If dtype is not None, then the values will first be cast to the requested type before the operation is
        performed. See Data.astype() for details.

        obj.sum() is equivalent to obj.astype(dtype, casting).rdd.values().sum().
        """
        out = self.astype(dtype, casting)
        return out.numpyValues().sum()

    def variance(self, dtype='smallfloat', casting='safe'):
        """ Variance of values, ignoring keys

        If dtype is not None, then the values will first be cast to the requested type before the operation is
        performed. See Data.astype() for details.

        obj.variance() is equivalent to obj.astype(dtype, casting).rdd.values().variance()."""
        out = self.astype(dtype, casting)
        return out.numpyValues().variance()

    def stdev(self, dtype='smallfloat', casting='safe'):
        """ Standard deviation of values, ignoring keys

        If dtype is not None, then the values will first be cast to the requested type before the operation is
        performed. See Data.astype() for details.

        obj.stdev() is equivalent to obj.astype(dtype, casting).rdd.values().stdev().
        """
        out = self.astype(dtype, casting)
        return out.numpyValues().stdev()

    def stats(self, dtype='smallfloat', casting='safe'):
        """ Stats of values, ignoring keys

        If dtype is not None, then the values will first be cast to the requested type before the operation is
        performed. See Data.astype() for details.

        obj.stats() is equivalent to obj.astype(dtype, casting).rdd.values().stats().
        """
        out = self.astype(dtype, casting)
        return out.numpyValues().stats()

    def max(self):
        """ Maximum of values, ignoring keys """
        from numpy import maximum
        return self.numpyValues().reduce(maximum)

    def min(self):
        """ Minimum of values, ignoring keys """
        from numpy import minimum
        return self.numpyValues().reduce(minimum)


class NumpyArrayValuedData(NumpyData):
    """Abstract base class for Data objects whose values are numpy arrays
    """
    def populateParamsFromFirstRecord(self):
        record = super(NumpyArrayValuedData, self).populateParamsFromFirstRecord()
        self._dtype = str(record[1].dtype)
        return record

    def _astypeImpl(self, dtype, casting='safe'):
        return self.rdd.mapValues(lambda v: v.astype(dtype, casting=casting))

    def numpyValues(self):
        return self.rdd.values()


class NumpyArrayAttributeDataValue(object):
    """Superclass for values of a NumpyArrayAttributeData object

    This class declares that instances have a read-only property `values`. By convention,
    this `values` attribute should be a numpy array.
    """
    @property
    def values(self):
        """Returns a numpy array
        """
        raise NotImplementedError("Subclasses should override this")

    def withValues(self, newValues):
        """Returns a new instance of this class, identical to self but with its `values`
        member replaced by the passed `newValues`.
        """
        raise NotImplementedError("Subclasses should override this")


class NumpyArrayAttributeData(NumpyData):
    """Abstract base class for Data objects whose values have a numpy array as their own `.values` attribute
    """
    def populateParamsFromFirstRecord(self):
        record = super(NumpyArrayAttributeData, self).populateParamsFromFirstRecord()
        self._dtype = str(record[1].values.dtype)
        return record

    def _astypeImpl(self, dtype, casting='safe'):
        return self.rdd.mapValues(
            lambda v: v.withValues(v.values.astype(dtype, casting=casting))
        )

    def numpyValues(self):
        return self.rdd.map(lambda (k, v): v.values)
