import numpy as np
import os
from nose.tools import assert_equals, assert_true

from test_utils import PySparkTestCaseWithOutputDir
from thunder import ThunderContext


class TestContextLoading(PySparkTestCaseWithOutputDir):
    def setUp(self):
        super(TestContextLoading, self).setUp()
        self.tsc = ThunderContext(self.sc)

    def test_loadImagesAsSeriesNoShuffle(self):
        rangeary = np.arange(64*128, dtype=np.dtype('int16'))
        rangeary.shape = (64, 128)
        filepath = os.path.join(self.outputdir, "rangeary.stack")
        rangeary.tofile(filepath)
        range_series_noshuffle = self.tsc.loadImagesAsSeries(filepath, dims=(64, 128))
        range_series_noshuffle_ary = range_series_noshuffle.pack()

        assert_equals((64, 128), range_series_noshuffle.dims.count)
        assert_equals((64, 128), range_series_noshuffle_ary.shape)
        assert_true(np.array_equal(rangeary, range_series_noshuffle_ary))

    def test_loadImagesAsSeriesWithShuffle(self):
        rangeary = np.arange(64*128, dtype=np.dtype('int16'))
        rangeary.shape = (64, 128)
        filepath = os.path.join(self.outputdir, "rangeary.stack")
        rangeary.tofile(filepath)
        range_series_withshuffle = self.tsc.loadImagesAsSeries(filepath, dims=(64, 128), shuffle=True)
        range_series_withshuffle_ary = range_series_withshuffle.pack()

        assert_equals((64, 128), range_series_withshuffle.dims.count)
        assert_equals((64, 128), range_series_withshuffle_ary.shape)
        assert_true(np.array_equal(rangeary, range_series_withshuffle_ary))