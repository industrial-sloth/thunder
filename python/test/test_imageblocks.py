import itertools
from numpy import allclose, arange, array, array_equal, concatenate, dtype, prod
import unittest
from nose.tools import assert_equals, assert_true, assert_almost_equal, assert_raises
from thunder.rdds.imageblocks import ImageBlocks, ImageBlockValue, PaddedImageBlockValue, _BlockMemoryAsReversedSequence
from test_utils import PySparkTestCase


class TestImageBlockStats(PySparkTestCase):
    def generateTestImageBlocks(self):
        ary = arange(16, dtype=dtype('uint8')).reshape((4, 4))
        value1 = ImageBlockValue.fromArrayBySlices(ary, (slice(2), slice(None)))
        value2 = ImageBlockValue.fromArrayBySlices(ary, (slice(2, 4), slice(None)))
        keys = [(0, 0), (2, 0)]
        rdd = self.sc.parallelize(zip(keys, (value1, value2)), 2)
        return ImageBlocks(rdd, (4, 4), 1, 'uint8')

    def setUp(self):
        super(TestImageBlockStats, self).setUp()
        self.blocks = self.generateTestImageBlocks()
        self.arys = self.blocks.numpyValues().collect()

    def test_mean(self):
        from test_utils import elementwise_mean
        meanval = self.blocks.mean()

        expected = elementwise_mean(self.arys)
        assert_true(allclose(expected, meanval))
        assert_equals('float16', str(meanval.dtype))

    def test_sum(self):
        from numpy import add
        sumval = self.blocks.sum(dtype="uint32")

        arys = [ary.astype('uint32') for ary in self.arys]
        expected = reduce(add, arys)
        assert_true(array_equal(expected, sumval))
        assert_equals('uint32', str(sumval.dtype))

    def test_variance(self):
        from test_utils import elementwise_var
        varval = self.blocks.variance()

        expected = elementwise_var(self.arys)
        assert_true(allclose(expected, varval, atol=0.001))
        assert_equals('float16', str(varval.dtype))

    def test_stddev(self):
        from test_utils import elementwise_stdev
        stdval = self.blocks.stdev()

        expected = elementwise_stdev(self.arys)
        assert_true(allclose(expected, stdval, atol=0.001))
        assert_equals("float32", str(stdval.dtype))  # see equivalent test in test_images.py

    def test_stats(self):
        from test_utils import elementwise_mean, elementwise_var
        statsval = self.blocks.stats()

        floatarys = [ary.astype('float16') for ary in self.arys]
        expectedmean = elementwise_mean(floatarys)
        expectedvar = elementwise_var(floatarys)
        assert_true(allclose(expectedmean, statsval.mean()))
        assert_true(allclose(expectedvar, statsval.variance()))

    def test_max(self):
        from numpy import maximum
        maxval = self.blocks.max()
        assert_true(array_equal(reduce(maximum, self.arys), maxval))

    def test_min(self):
        from numpy import minimum
        minval = self.blocks.min()
        assert_true(array_equal(reduce(minimum, self.arys), minval))


class TestImageBlockValue(unittest.TestCase):

    def test_fromArrayByPlane(self):
        values = arange(12, dtype=dtype('int16')).reshape((3, 4), order='C')

        planedim = 0
        planedimidx = 1
        imageblock = ImageBlockValue.fromArrayByPlane(values, planedim=planedim, planeidx=planedimidx)

        assert_equals(values.shape, imageblock.origshape)
        assert_equals(slice(planedimidx, planedimidx+1, 1), imageblock.origslices[planedim])
        assert_equals(slice(None), imageblock.origslices[1])
        assert_true(array_equal(values[planedimidx, :].flatten(order='C'), imageblock.values.flatten(order='C')))

    def test_fromArrayBySlices(self):
        values = arange(12, dtype=dtype('int16')).reshape((3, 4), order='C')

        slices = [[slice(0, 3)], [slice(0, 2), slice(2, 4)]]
        slicesiter = itertools.product(*slices)

        imageblocks = [ImageBlockValue.fromArrayBySlices(values, sls) for sls in slicesiter]
        assert_equals(2, len(imageblocks))
        assert_equals((3, 2), imageblocks[0].values.shape)
        assert_true(array_equal(values[(slice(0, 3), slice(0, 2))], imageblocks[0].values))

    def test_fromPlanarBlocks(self):
        values = arange(36, dtype=dtype('int16')).reshape((3, 4, 3), order='F')

        imageblocks = [ImageBlockValue.fromArrayByPlane(values, -1, i) for i in xrange(values.shape[2])]

        recombblock = ImageBlockValue.fromPlanarBlocks(imageblocks, planarDim=-1)

        assert_true(array_equal(values, recombblock.values))
        assert_equals([slice(None)] * values.ndim, recombblock.origslices)
        assert_equals(values.shape, recombblock.origshape)

    def test_addDimension(self):
        values = arange(12, dtype=dtype('int16')).reshape((3, 4), order='C')
        morevalues = arange(12, 24, dtype=dtype('int16')).reshape((3, 4), order='C')

        origshape = values.shape
        origslices = [slice(None)] * values.ndim
        newdimsize = 2
        initimageblock = ImageBlockValue(origshape=origshape, origslices=origslices, values=values)
        anotherinitimageblock = ImageBlockValue(origshape=origshape, origslices=origslices, values=morevalues)

        imageblock = initimageblock.addDimension(newdimidx=0, newdimsize=newdimsize)
        anotherimageblock = anotherinitimageblock.addDimension(newdimidx=1, newdimsize=newdimsize)

        expectedorigshape = tuple([newdimsize] + list(initimageblock.origshape))
        assert_equals(expectedorigshape, imageblock.origshape)
        assert_equals(expectedorigshape, anotherimageblock.origshape)

        expectednslices = len(expectedorigshape)
        assert_equals(expectednslices, len(imageblock.origslices))
        assert_equals(expectednslices, len(anotherimageblock.origslices))

        assert_equals(slice(0, 1, 1), imageblock.origslices[0])
        assert_equals(slice(1, 2, 1), anotherimageblock.origslices[0])

        expectedshape = tuple([1] + list(values.shape))
        assert_equals(expectedshape, imageblock.values.shape)
        assert_equals(expectedshape, anotherimageblock.values.shape)

        # check that straight array concatenation works as expected in this particular case
        expectedcatvals = arange(24, dtype=dtype('int16'))
        actualcatvals = concatenate((imageblock.values, anotherimageblock.values), axis=0).flatten(order='C')
        assert_true(array_equal(expectedcatvals, actualcatvals))

    def test_toSeriesIter(self):
        sh = 2, 3, 4
        sz = int(prod(sh))
        ary = arange(sz, dtype=dtype('int16')).reshape(sh)
        imageblock = ImageBlockValue.fromArray(ary)

        series = list(imageblock.toSeriesIter(-1))

        # this was less confusing when a series could be created by
        # a straight linear read of a binary array...
        expectedseries = [
            ((0, 0), array([0, 1, 2, 3], dtype='int16')),
            ((1, 0), array([12, 13, 14, 15], dtype='int16')),
            ((0, 1), array([4, 5, 6, 7], dtype='int16')),
            ((1, 1), array([16, 17, 18, 19], dtype='int16')),
            ((0, 2), array([8, 9, 10, 11], dtype='int16')),
            ((1, 2), array([20, 21, 22, 23], dtype='int16')),
        ]

        for actual, expected in zip(series, expectedseries):
            # check key equality
            assert_equals(expected[0], actual[0])
            # check value equality
            assert_true(array_equal(expected[1], actual[1]))

    def test_toSeriesIter2(self):
        # add singleton dimension on end of (2, 4) shape to be "time":
        ary = arange(8, dtype=dtype('int16')).reshape((2, 4, 1))

        imageblock = ImageBlockValue.fromArray(ary)

        seriesvals = list(imageblock.toSeriesIter(-1))

        # check ordering of keys
        assert_equals((0, 0), seriesvals[0][0])  # first key
        assert_equals((1, 0), seriesvals[1][0])  # second key
        assert_equals((0, 1), seriesvals[2][0])
        assert_equals((1, 1), seriesvals[3][0])
        assert_equals((0, 2), seriesvals[4][0])
        assert_equals((1, 2), seriesvals[5][0])
        assert_equals((0, 3), seriesvals[6][0])
        assert_equals((1, 3), seriesvals[7][0])

        # check that values are in original order
        collectedvals = array([kv[1] for kv in seriesvals], dtype=dtype('int16')).ravel()
        assert_true(array_equal(ary.ravel(order='F'), collectedvals))


class TestPaddedImageBlockValue(unittest.TestCase):
    def test_fromArrayBySlices_fullArray(self):
        ary = arange(8, dtype=dtype('int16')).reshape((2, 4))
        slices = [slice(None)] * 2
        pibv = PaddedImageBlockValue.fromArrayBySlices(ary, slices, 2)

        assert_true(array_equal(ary, pibv.values))
        assert_equals(slices[0], pibv.coreimgslices[0])
        assert_equals(slices[1], pibv.coreimgslices[1])
        assert_equals(slice(0, 2, 1), pibv.corevalslices[0])
        assert_equals(slice(0, 4, 1), pibv.corevalslices[1])
        # since there's no actual padding here, padded slices are equal to core slices
        assert_equals(slice(0, 2, 1), pibv.padimgslices[0])
        assert_equals(slice(0, 4, 1), pibv.padimgslices[1])

    def test_fromArrayBySlices_padded1(self):
        ary = arange(16, dtype=dtype('int16')).reshape((4, 4))
        slices = [slice(1, 3, 1)] * 2
        # ask for 2 pixels padding, but will only get 1 due to edge:
        pibv = PaddedImageBlockValue.fromArrayBySlices(ary, slices, 2)

        assert_true(array_equal(ary, pibv.values))
        assert_equals(slices[0], pibv.coreimgslices[0])
        assert_equals(slices[1], pibv.coreimgslices[1])
        assert_equals(slice(1, 3, 1), pibv.corevalslices[0])
        assert_equals(slice(1, 3, 1), pibv.corevalslices[1])
        assert_equals(slice(0, 4, 1), pibv.padimgslices[0])
        assert_equals(slice(0, 4, 1), pibv.padimgslices[1])

    def test_fromArrayBySlices_padded2(self):
        ary = arange(16, dtype=dtype('int16')).reshape((4, 4))
        slices = [slice(2, 4, 1)] * 2
        pibv = PaddedImageBlockValue.fromArrayBySlices(ary, slices, 1)

        assert_true(array_equal(ary[1:, 1:], pibv.values))
        assert_equals(slices[0], pibv.coreimgslices[0])
        assert_equals(slices[1], pibv.coreimgslices[1])
        assert_equals(slice(1, 3, 1), pibv.corevalslices[0])
        assert_equals(slice(1, 3, 1), pibv.corevalslices[1])
        assert_equals(slice(1, 4, 1), pibv.padimgslices[0])
        assert_equals(slice(1, 4, 1), pibv.padimgslices[1])

    def test_stackPlanarBlocks(self):
        ary = arange(32, dtype=dtype('uint8')).reshape(2, 4, 4)
        slices = [slice(2, 4, 1)] * 2
        pibv1 = PaddedImageBlockValue.fromArrayBySlices(ary, [slice(0, 1, 1)]+slices, (0, 1, 1))
        pibv2 = PaddedImageBlockValue.fromArrayBySlices(ary, [slice(1, 2, 1)]+slices, (0, 1, 1))
        underTest = PaddedImageBlockValue.stackPlanarBlocks((pibv1, pibv2))

        assert_true(array_equal(ary[:, 1:, 1:], underTest.values))
        assert_equals(slice(None), underTest.coreimgslices[0])
        assert_equals(slices[0], underTest.coreimgslices[1])
        assert_equals(slices[1], underTest.coreimgslices[2])
        assert_equals(slice(None), underTest.corevalslices[0])
        assert_equals(slice(1, 3, 1), underTest.corevalslices[1])
        assert_equals(slice(1, 3, 1), underTest.corevalslices[2])
        assert_equals(slice(None), underTest.padimgslices[0])
        assert_equals(slice(1, 4, 1), underTest.padimgslices[1])
        assert_equals(slice(1, 4, 1), underTest.padimgslices[2])


class TestBlockMemoryAsSequence(unittest.TestCase):
    def test_range(self):
        dims = (2, 2)
        undertest = _BlockMemoryAsReversedSequence(dims)

        assert_equals(3, len(undertest))
        assert_equals((2, 2), undertest.indtosub(0))
        assert_equals((1, 2), undertest.indtosub(1))
        assert_equals((1, 1), undertest.indtosub(2))
        assert_raises(IndexError, undertest.indtosub, 3)
