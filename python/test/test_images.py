from collections import Counter
import glob
import struct
import unittest
import os
from operator import mul
from numpy import arange, array, array_equal, dtype, prod, zeros
import itertools
from nose.tools import assert_equals, assert_true, assert_almost_equal, assert_raises

from thunder.rdds.fileio.imagesloader import ImagesLoader
from thunder.rdds.imageblocks import ImageBlocksPartitioningStrategy
from test_utils import PySparkTestCase, PySparkTestCaseWithOutputDir


_have_image = False
try:
    from PIL import Image
    _have_image = True
except ImportError:
    # PIL not available; skip tests that require it
    pass


def _generate_test_arrays(narys, dtype='int16'):
    sh = 4, 3, 3
    sz = prod(sh)
    arys = [arange(i, i+sz, dtype=dtype).reshape(sh) for i in xrange(0, sz * narys, sz)]
    return arys, sh, sz


class TestImages(PySparkTestCase):

    def evaluate_series(self, arys, series, sz):
        assert_equals(sz, len(series))
        for serieskey, seriesval in series:
            expectedval = array([ary[serieskey] for ary in arys], dtype='int16')
            assert_true(array_equal(expectedval, seriesval))

    def test_toSeries(self):
        # create 3 arrays of 4x3x3 images (C-order), containing sequential integers
        narys = 3
        arys, sh, sz = _generate_test_arrays(narys)

        imagedata = ImagesLoader(self.sc).fromArrays(arys)
        strategy = ImageBlocksPartitioningStrategy(splitsPerDim=(4, 1, 1))
        series = imagedata.partition(strategy).toSeries().collect()

        self.evaluate_series(arys, series, sz)

    def test_toSeriesBySlices(self):
        narys = 3
        arys, sh, sz = _generate_test_arrays(narys)

        imagedata = ImagesLoader(self.sc).fromArrays(arys)
        imagedata.cache()

        test_params = [
            (1, 1, 1), (1, 1, 2), (1, 1, 3), (1, 2, 1), (1, 2, 2), (1, 2, 3),
            (1, 3, 1), (1, 3, 2), (1, 3, 3),
            (2, 1, 1), (2, 1, 2), (2, 1, 3), (2, 2, 1), (2, 2, 2), (2, 2, 3),
            (2, 3, 1), (2, 3, 2), (2, 3, 3)]
        for bpd in test_params:
            strategy = ImageBlocksPartitioningStrategy(splitsPerDim=bpd)
            series = imagedata.partition(strategy).toSeries().collect()

            self.evaluate_series(arys, series, sz)


class TestImagesUsingOutputDir(PySparkTestCaseWithOutputDir):

    def _run_tstSaveAsBinarySeries(self, testidx, narys_, valdtype, groupingdim_):
        """Pseudo-parameterized test fixture, allows reusing existing spark context
        """
        paramstr = "(groupingdim=%d, valuedtype='%s')" % (groupingdim_, valdtype)
        arys, aryshape, arysize = _generate_test_arrays(narys_, dtype=valdtype)
        outdir = os.path.join(self.outputdir, "anotherdir%02d" % testidx)

        images = ImagesLoader(self.sc).fromArrays(arys)

        slicesPerDim = [1]*arys[0].ndim
        slicesPerDim[groupingdim_] = arys[0].shape[groupingdim_]
        strategy = ImageBlocksPartitioningStrategy(splitsPerDim=slicesPerDim)
        images.partition(strategy).saveAsBinarySeries(outdir)

        ndims = len(aryshape)
        # prevent padding to 4-byte boundaries: "=" specifies no alignment
        unpacker = struct.Struct('=' + 'h'*ndims + dtype(valdtype).char*narys_)

        def calcExpectedNKeys(aryshape__, groupingdim__):
            tmpshape = list(aryshape__[:])
            del tmpshape[groupingdim__]
            return prod(tmpshape)
        expectednkeys = calcExpectedNKeys(aryshape, groupingdim_)

        def byrec(f_, unpacker_, nkeys_):
            rec = True
            while rec:
                rec = f_.read(unpacker_.size)
                if rec:
                    allrecvals = unpacker_.unpack(rec)
                    yield allrecvals[:nkeys_], allrecvals[nkeys_:]

        outfilenames = glob.glob(os.path.join(outdir, "*.bin"))
        assert_equals(aryshape[groupingdim_], len(outfilenames))
        for outfilename in outfilenames:
            with open(outfilename, 'rb') as f:
                nkeys = 0
                for keys, vals in byrec(f, unpacker, ndims):
                    nkeys += 1
                    assert_equals(narys_, len(vals))
                    for validx, val in enumerate(vals):
                        assert_equals(arys[validx][keys], val, "Expected %g, got %g, for test %d %s" %
                                      (arys[validx][keys], val, testidx, paramstr))
                assert_equals(expectednkeys, nkeys)

        confname = os.path.join(outdir, "conf.json")
        assert_true(os.path.isfile(confname))
        with open(os.path.join(outdir, "conf.json"), 'r') as fconf:
            import json
            conf = json.load(fconf)
            assert_equals(outdir, conf['input'])
            assert_equals(tuple(aryshape), tuple(conf['dims']))
            assert_equals(len(aryshape), conf['nkeys'])
            assert_equals(narys_, conf['nvalues'])
            assert_equals(valdtype, conf['valuetype'])
            assert_equals('int16', conf['keytype'])

        assert_true(os.path.isfile(os.path.join(outdir, 'SUCCESS')))

    def test_saveAsBinarySeries(self):
        narys = 3
        arys, aryshape, _ = _generate_test_arrays(narys)

        outdir = os.path.join(self.outputdir, "anotherdir")
        os.mkdir(outdir)
        dummystrat = ImageBlocksPartitioningStrategy(splitsPerDim=(1, 1, 1))
        assert_raises(ValueError, ImagesLoader(self.sc).fromArrays(arys).partition(dummystrat)
                      .saveAsBinarySeries, outdir)

        groupingdims = xrange(len(aryshape))
        dtypes = ('int16', 'int32', 'float32')
        paramiters = itertools.product(groupingdims, dtypes)

        for idx, params in enumerate(paramiters):
            gd, dt = params
            self._run_tstSaveAsBinarySeries(idx, narys, dt, gd)


if __name__ == "__main__":
    if not _have_image:
        print "NOTE: Skipping PIL/pillow tests as neither seem to be installed and functional"
    unittest.main()
    if not _have_image:
        print "NOTE: PIL/pillow tests were skipped as neither seem to be installed and functional"
