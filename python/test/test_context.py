import numpy as np
import os
import unittest
from nose.tools import assert_equals, assert_true

from test_utils import PySparkTestCaseWithOutputDir
from thunder import ThunderContext

_have_image = False
try:
    from PIL import Image
    _have_image = True
except ImportError:
    # PIL not available; skip tests that require it
    Image = None


class TestContextLoading(PySparkTestCaseWithOutputDir):
    def setUp(self):
        super(TestContextLoading, self).setUp()
        self.tsc = ThunderContext(self.sc)

    def test_loadStacksAsSeriesNoShuffle(self):
        rangeary = np.arange(64*128, dtype=np.dtype('int16'))
        rangeary.shape = (64, 128)
        filepath = os.path.join(self.outputdir, "rangeary.stack")
        rangeary.tofile(filepath)
        range_series_noshuffle = self.tsc.loadImagesAsSeries(filepath, dims=(64, 128))
        range_series_noshuffle_ary = range_series_noshuffle.pack()

        assert_equals((64, 128), range_series_noshuffle.dims.count)
        assert_equals((64, 128), range_series_noshuffle_ary.shape)
        assert_true(np.array_equal(rangeary, range_series_noshuffle_ary))

    def test_loadStacksAsSeriesWithShuffle(self):
        rangeary = np.arange(64*128, dtype=np.dtype('int16'))
        rangeary.shape = (64, 128)
        filepath = os.path.join(self.outputdir, "rangeary.stack")
        rangeary.tofile(filepath)
        range_series_withshuffle = self.tsc.loadImagesAsSeries(filepath, dims=(64, 128), shuffle=True)
        range_series_withshuffle_ary = range_series_withshuffle.pack()

        assert_equals((64, 128), range_series_withshuffle.dims.count)
        assert_equals((64, 128), range_series_withshuffle_ary.shape)
        assert_true(np.array_equal(rangeary, range_series_withshuffle_ary))

    def test_loadMultipleStacksAsSeriesNoShuffle(self):
        rangeary = np.arange(64*128, dtype=np.dtype('int16'))
        rangeary.shape = (64, 128)
        filepath = os.path.join(self.outputdir, "rangeary01.stack")
        rangeary.tofile(filepath)
        rangeary2 = np.arange(64*128, 2*64*128, dtype=np.dtype('int16'))
        rangeary2.shape = (64, 128)
        filepath = os.path.join(self.outputdir, "rangeary02.stack")
        rangeary2.tofile(filepath)

        range_series_noshuffle = self.tsc.loadImagesAsSeries(self.outputdir, dims=(64, 128))
        range_series_noshuffle_ary = range_series_noshuffle.pack()

        assert_equals((64, 128), range_series_noshuffle.dims.count)
        assert_equals((2, 64, 128), range_series_noshuffle_ary.shape)
        assert_true(np.array_equal(rangeary, range_series_noshuffle_ary[0]))
        assert_true(np.array_equal(rangeary2, range_series_noshuffle_ary[1]))

    def test_loadMultipleStacksAsSeriesWithShuffle(self):
        rangeary = np.arange(64*128, dtype=np.dtype('int16'))
        rangeary.shape = (64, 128)
        filepath = os.path.join(self.outputdir, "rangeary01.stack")
        rangeary.tofile(filepath)
        rangeary2 = np.arange(64*128, 2*64*128, dtype=np.dtype('int16'))
        rangeary2.shape = (64, 128)
        filepath = os.path.join(self.outputdir, "rangeary02.stack")
        rangeary2.tofile(filepath)

        range_series_shuffle = self.tsc.loadImagesAsSeries(self.outputdir, dims=(64, 128), shuffle=True)
        range_series_shuffle_ary = range_series_shuffle.pack()

        assert_equals((64, 128), range_series_shuffle.dims.count)
        assert_equals((2, 64, 128), range_series_shuffle_ary.shape)

        assert_true(np.array_equal(rangeary, range_series_shuffle_ary[0]))
        assert_true(np.array_equal(rangeary2, range_series_shuffle_ary[1]))

    @unittest.skipIf(not _have_image, "PIL/pillow not installed or not functional")
    def test_loadTifAsSeriesNoShuffle(self):
        tmpary = np.arange(60*120, dtype=np.dtype('uint16'))
        rangeary = np.mod(tmpary, 255).astype('uint8').reshape((60, 120))
        pilimg = Image.fromarray(rangeary)
        filepath = os.path.join(self.outputdir, "rangetif01.tif")
        pilimg.save(filepath)
        del pilimg, tmpary

        range_series_noshuffle = self.tsc.loadImagesAsSeries(self.outputdir, inputformat="tif-stack")
        range_series_noshuffle_ary = range_series_noshuffle.pack()

        assert_equals((60, 120, 1), range_series_noshuffle.dims.count)
        assert_equals((60, 120), range_series_noshuffle_ary.shape)
        assert_true(np.array_equal(rangeary, range_series_noshuffle_ary))

    @unittest.skipIf(not _have_image, "PIL/pillow not installed or not functional")
    def test_loadTifAsSeriesWithShuffle(self):
        tmpary = np.arange(60*120, dtype=np.dtype('uint16'))
        rangeary = np.mod(tmpary, 255).astype('uint8').reshape((60, 120))
        pilimg = Image.fromarray(rangeary)
        filepath = os.path.join(self.outputdir, "rangetif01.tif")
        pilimg.save(filepath)
        del pilimg, tmpary

        range_series_shuffle = self.tsc.loadImagesAsSeries(self.outputdir, inputformat="tif-stack",
                                                           shuffle=True)
        range_series_shuffle_ary = range_series_shuffle.pack()

        assert_equals((60, 120, 1), range_series_shuffle.dims.count)
        assert_equals((60, 120), range_series_shuffle_ary.shape)
        assert_true(np.array_equal(rangeary, range_series_shuffle_ary))

    @unittest.skipIf(not _have_image, "PIL/pillow not installed or not functional")
    def test_loadMultipleTifsAsSeriesNoShuffle(self):
        tmpary = np.arange(60*120, dtype=np.dtype('uint16'))
        rangeary = np.mod(tmpary, 255).astype('uint8').reshape((60, 120))
        pilimg = Image.fromarray(rangeary)
        filepath = os.path.join(self.outputdir, "rangetif01.tif")
        pilimg.save(filepath)

        tmpary = np.arange(60*120, 2*60*120, dtype=np.dtype('uint16'))
        rangeary2 = np.mod(tmpary, 255).astype('uint8').reshape((60, 120))
        pilimg = Image.fromarray(rangeary2)
        filepath = os.path.join(self.outputdir, "rangetif02.tif")
        pilimg.save(filepath)

        del pilimg, tmpary

        range_series_noshuffle = self.tsc.loadImagesAsSeries(self.outputdir, inputformat="tif-stack")
        range_series_noshuffle_ary = range_series_noshuffle.pack()

        assert_equals((60, 120, 1), range_series_noshuffle.dims.count)
        assert_equals((2, 60, 120), range_series_noshuffle_ary.shape)
        assert_true(np.array_equal(rangeary, range_series_noshuffle_ary[0]))
        assert_true(np.array_equal(rangeary2, range_series_noshuffle_ary[1]))
