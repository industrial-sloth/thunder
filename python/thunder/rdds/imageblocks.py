"""Classes relating to ImageBlocks, the default / reference implementation of an Images partitioning strategy.
"""
import cStringIO as StringIO
import itertools
from numpy import zeros, reshape
import struct
from thunder.rdds.images import PartitioningStrategy, PartitionedImages
from thunder.rdds.series import Series


class ImageBlocksPartitioningStrategy(PartitioningStrategy):
    """A PartitioningStrategy that groups Images into nonoverlapping, roughly equally-sized blocks.

    The number and dimensions of image blocks are specified as "splits per dimension", which is for each
    spatial dimension of the original Images the number of partitions to generate along that dimension. So
    for instance, given a 12 x 12 Images object, an ImageBlocksPartitioningStrategy with splitsPerDim=(2,2)
    would yield PartitionedImages with 4 blocks, each 6 x 6.
    """
    def __init__(self, splitsPerDim, numSparkPartitions=None):
        """Returns a new ImageBlocksPartitioningStrategy.

        Parameters
        ----------
        splitsPerDim : n-tuple of positive int, where n = dimensionality of image
            Specifies that intermediate blocks are to be generated by splitting the i-th dimension
            of the image into splitsPerDim[i] roughly equally-sized partitions.
            1 <= splitsPerDim[i] <= self.dims[i]
        """
        super(ImageBlocksPartitioningStrategy, self).__init__()
        self._splitsPerDim = ImageBlocksPartitioningStrategy.__normalizeSplits(splitsPerDim)
        self._slices = None
        self._npartitions = reduce(lambda x, y: x * y, self._splitsPerDim, 1) if not numSparkPartitions \
            else int(numSparkPartitions)

    @classmethod
    def generateFromBlockSize(cls, blockSize, dims, nimages, datatype, numSparkPartitions=None):
        """Returns a new ImageBlocksPartitioningStrategy, that yields blocks
        closely matching the requested size in bytes.

        Parameters
        ----------
        blockSize : positive int or string
            Requests an average size for the intermediate blocks in bytes. A passed string should
            be in a format like "256k" or "150M" (see util.common.parseMemoryString). If blocksPerDim
            or groupingDim are passed, they will take precedence over this argument. See
            imageblocks._BlockMemoryAsSequence for a description of the partitioning strategy used.

        Returns
        -------
        n-tuple of positive int, where n == len(self.dims)
            Each value in the returned tuple represents the number of splits to apply along the
            corresponding dimension in order to yield blocks close to the requested size.
        """
        import bisect
        from numpy import dtype
        minseriessize = nimages * dtype(datatype).itemsize

        memseq = _BlockMemoryAsReversedSequence(dims)
        tmpidx = bisect.bisect_left(memseq, blockSize / float(minseriessize))
        if tmpidx == len(memseq):
            # handle case where requested block is bigger than the biggest image
            # we can produce; just give back the biggest block size
            tmpidx -= 1
        splitsPerDim = memseq.indtosub(tmpidx)
        return cls(splitsPerDim, numSparkPartitions=numSparkPartitions)

    @property
    def npartitions(self):
        """The number of Spark partitions across which the resulting RDD is to be distributed.
        """
        return self._npartitions

    @staticmethod
    def __normalizeSplits(splitsPerDim):
        splitsPerDim = map(int, splitsPerDim)
        if any((nsplits <= 0 for nsplits in splitsPerDim)):
            raise ValueError("All numbers of blocks must be positive; got " + str(splitsPerDim))
        return splitsPerDim

    def __validateSplitsForImage(self):
        dims = self.dims
        splitsPerDim = self._splitsPerDim
        ndim = len(dims)
        if not len(splitsPerDim) == ndim:
            raise ValueError("splitsPerDim length (%d) must match image dimensionality (%d); " %
                             (len(splitsPerDim), ndim) +
                             "have splitsPerDim %s and image shape %s" % (str(splitsPerDim), str(dims)))

    @staticmethod
    def __generateSlices(splitsPerDim, dims):
        # slices will be sequence of sequences of slices
        # slices[i] will hold slices for ith dimension
        slices = []
        for nsplits, dimsize in zip(splitsPerDim, dims):
            blocksize = dimsize / nsplits  # integer division
            blockrem = dimsize % nsplits
            st = 0
            dimslices = []
            for blockidx in xrange(nsplits):
                en = st + blocksize
                if blockrem:
                    en += 1
                    blockrem -= 1
                dimslices.append(slice(st, min(en, dimsize), 1))
                st = en
            slices.append(dimslices)
        return slices

    def setImages(self, images):
        super(ImageBlocksPartitioningStrategy, self).setImages(images)
        self.__validateSplitsForImage()
        self._slices = ImageBlocksPartitioningStrategy.__generateSlices(self._splitsPerDim, self.dims)

    def partitionFunction(self, timePointIdxAndImageArray):
        tpidx, imgary = timePointIdxAndImageArray
        totnumimages = self.nimages
        slices = self._slices

        ret_vals = []
        sliceproduct = itertools.product(*slices)
        for blockslices in sliceproduct:
            blockval = ImageBlockValue.fromArrayBySlices(imgary, blockslices, docopy=False)
            blockval = blockval.addDimension(newdimidx=tpidx, newdimsize=totnumimages)
            # resulting key will be (x, y, z) (for 3d data), where x, y, z are starting
            # position of block within image volume
            newkey = [sl.start for sl in blockslices]
            ret_vals.append((tuple(newkey), blockval))
        return ret_vals

    def blockingFunction(self, partitionedSequence):
        return ImageBlockValue.fromPlanarBlocks(partitionedSequence, 0)


class ImageBlocks(PartitionedImages):
    """Intermediate representation used in conversion from Images to Series.

    This class is not expected to be directly used by clients.
    """

    @staticmethod
    def _blockToSeries(blockVal, seriesDim):
        for seriesKey, seriesVal in blockVal.toSeriesIter(seriesDim=seriesDim):
            yield tuple(seriesKey), seriesVal

    def toSeries(self, seriesDim=0):

        def blockToSeriesAdapter(kv):
            blockKey, blockVal = kv
            return ImageBlocks._blockToSeries(blockVal, seriesDim)

        blockedrdd = self._groupIntoSeriesBlocks()

        # returns generator of (z, y, x) array data for all z, y, x
        seriesrdd = blockedrdd.flatMap(blockToSeriesAdapter)
        return Series(seriesrdd)

    @staticmethod
    def getBinarySeriesNameForKey(blockKey):
        """

        Returns
        -------
        string blocklabel
            Block label will be in form "key02_0000k-key01_0000j-key00_0000i" for i,j,k x,y,z indicies as first series
            in block. No extension (e.g. ".bin") is appended to the block label.
        """
        return '-'.join(reversed(["key%02d_%05g" % (ki, k) for (ki, k) in enumerate(blockKey)]))

    def toBinarySeries(self, seriesDim=0):

        blockedrdd = self._groupIntoSeriesBlocks()

        def blockToBinarySeries(kv):
            blockKey, blockVal = kv
            label = ImageBlocks.getBinarySeriesNameForKey(blockKey) + ".bin"
            keypacker = None
            buf = StringIO.StringIO()
            for seriesKey, series in ImageBlocks._blockToSeries(blockVal, seriesDim):
                if keypacker is None:
                    keypacker = struct.Struct('h'*len(seriesKey))
                # print >> sys.stderr, seriesKey, series, series.tostring().encode('hex')
                buf.write(keypacker.pack(*seriesKey))
                buf.write(series.tostring())
            val = buf.getvalue()
            buf.close()
            return label, val

        return blockedrdd.map(blockToBinarySeries)

    def _groupIntoSeriesBlocks(self):
        """Combine blocks representing individual image blocks into a single time-by-blocks volume

        Returns:
        --------
        RDD, key/value: tuple of int, ImageBlockValue
        key:
            spatial indicies of start of block, for instance (x, y, z): (0, 0, 0), (0, 0, 1),... (0, 0, z_max-1)
        value:
            ImageBlockValue with single fully-populated array, dimensions of time by space, for instance (t, x, y, z):
            ary[0:t_max, :, :, z_i]
        """
        # key will be x, y, z for start of block
        # val will be single blockvalue array data with origshape and origslices in dimensions (t, x, y, z)
        # val array data will be t by (spatial block size)
        return self.rdd.groupByKey().mapValues(lambda v: ImageBlockValue.fromPlanarBlocks(v, 0))


class ImageBlockValue(object):
    """
    Helper data structure for transformations from Images to Series.

    Not intended for direct use by clients.

    Attributes
    ----------
    origshape : n-sequence of positive int
        Represents the shape of the overall volume of which this block is a component.

    origslices : n-sequence of slices
        Represents the position of this block of data within the overall volume of which the block is
        a component. Effectively, an assertion that origVolume[self.origslices] == self.values.
        Slices sl in origslices should either be equal to slice(None), representing the whole range
        of the corresponding dimension, or else should have all of sl.start, sl.stop, and sl.step
        specified.

    values : numpy ndarray
        Data making up this block.
        values.ndim will be equal to len(origshape) and len(origslices).
        values.shape will typically be smaller than origshape, but may be the same.
    """
    __slots__ = ('origshape', 'origslices', 'values')

    def __init__(self, origshape, origslices, values):
        self.origshape = origshape
        self.origslices = origslices
        self.values = values

    @classmethod
    def fromArray(cls, ary):
        return ImageBlockValue(origshape=ary.shape, origslices=tuple([slice(None)]*ary.ndim), values=ary)

    @classmethod
    def fromArrayByPlane(cls, imagearray, planedim, planeidx, docopy=False):
        """Extracts a single plane from the passed array as an ImageBlockValue.

        The origshape and origslices parameters of the returned IBV will reflect the
        original size and shape of the passed array.

        Parameters:
        -----------
        imagearray: numpy ndarray

        planedim: integer, 0 <= planedim < imagearray.ndim
            Dimension of passed array along which to extract plane

        planeidx: integer, 0 <= planeidx < imagearray.shape[planedim]
            Point along dimension from which to extract plane

        docopy: boolean, default False
            If true, return a copy of the array data; default is (typically) to return view

        Returns:
        --------
        new ImageBlockValue of plane of passed array
        """
        ndim = imagearray.ndim
        slices = [slice(None)] * ndim
        slices[planedim] = slice(planeidx, planeidx+1, 1)
        return cls.fromArrayBySlices(imagearray, slices, docopy=docopy)

    @classmethod
    def fromArrayBySlices(cls, imagearray, slices, docopy=False):
        slicedary = imagearray[slices].copy() if docopy else imagearray[slices]
        return cls(imagearray.shape, tuple(slices), slicedary)

    @classmethod
    def fromBlocks_orig(cls, blocksIter):
        """Creates a new ImageBlockValue from an iterator over blocks with compatible origshape.

        The new ImageBlockValue created will have values of shape origshape. Each block will
        copy its own data into the newly-created array according to the slicing given by its origslices
        attribute.

        """
        ary = None
        for block in blocksIter:
            if ary is None:
                ary = zeros(block.origshape, dtype=block.values.dtype)

            if not tuple(ary.shape) == tuple(block.origshape):
                raise ValueError("Shape mismatch; blocks specify differing original shapes %s and %s" %
                                 (str(block.origshape), str(ary.shape)))
            ary[block.origslices] = block.values
        return cls.fromArray(ary)

    @classmethod
    def fromPlanarBlocks(cls, blocksIter, planarDim):
        """Creates a new ImageBlockValue from an iterator over IBVs that have at least one singleton dimension.

        The resulting block will effectively be the concatenation of all blocks in the passed iterator,
        grouped as determined by each passed block's origslices attribute. This method assumes that
        the passed blocks will all have either the first or the last slice in origslices specifying only
        a single index. This index will be used to sort the consituent blocks into their proper positions
        within the returned ImageBlockValue.

        Parameters
        ----------
        blocksIter : iterable over ImageBlockValue
            All blocks in blocksIter must have identical origshape, identical values.shape, and origslices
            that are identical apart from origslices[planarDim]. For each block, origslices[planarDim]
            must specify only a single index.

        planarDim : integer
            planarDim must currently be either 0, len(origshape)-1, or -1, specifying the first, last, or
            last dimension of the block, respectively, as the dimension across which blocks are to be
            combined. origslices[planarDim] must specify a single index.

        """
        def getBlockSlices(slices, planarDim_):
            compatslices = list(slices[:])
            del compatslices[planarDim_]
            return tuple(compatslices)

        def _initialize_fromPlanarBlocks(firstBlock, planarDim_):
            # set up collection array:
            fullndim = len(firstBlock.origshape)
            if not (planarDim_ == 0 or planarDim_ == len(firstBlock.origshape)-1 or planarDim_ == -1):
                raise ValueError("planarDim must specify either first or last dimension, got %d" % planarDim_)
            if planarDim_ < 0:
                planarDim_ = fullndim + planarDim_

            newshape = list(firstBlock.values.shape[:])
            newshape[planarDim_] = block.origshape[planarDim_]

            ary = zeros(newshape, dtype=block.values.dtype)
            matchingslices = getBlockSlices(block.origslices, planarDim_)
            return ary, matchingslices, fullndim, block.origshape[:], planarDim_

        def allequal(seqA, seqB):
            return all([a == b for (a, b) in zip(seqA, seqB)])

        ary = None
        matchingslices = None
        fullndim = None
        origshape = None
        for block in blocksIter:
            if ary is None:
                # set up collection array:
                ary, matchingslices, fullndim, origshape, planarDim = \
                    _initialize_fromPlanarBlocks(block, planarDim)

            # check for compatible slices (apart from slice on planar dimension)
            blockslices = getBlockSlices(block.origslices, planarDim)
            if not allequal(matchingslices, blockslices):
                raise ValueError("Incompatible slices; got %s and %s" % (str(matchingslices), str(blockslices)))
            if not allequal(origshape, block.origshape):
                raise ValueError("Incompatible original shapes; got %s and %s" % (str(origshape), str(block.origshape)))

            # put values into collection array:
            targslices = [slice(None)] * fullndim
            targslices[planarDim] = block.origslices[planarDim]

            ary[targslices] = block.values

        # new slices should be full slice for formerly planar dimension, plus existing block slices
        newslices = list(getBlockSlices(block.origslices, planarDim))
        newslices.insert(planarDim, slice(None))

        return cls(block.origshape, newslices, ary)

    def addDimension(self, newdimidx=0, newdimsize=1):
        """Returns a new ImageBlockValue embedded in a space of dimension n+1

        Given that self is an ImageBlockValue of dimension n, returns a new
        ImageBlockValue of dimension n+1. The new IBV's origshape and origslices
        attributes will be consistent with this new dimensionality, treating the new
        IBV's values as an embedding within the new higher dimensional space.

        For instance, if ImageBlockValues are three-dimensional volumes, calling
        addDimension would recast them each as a single 'time point' in a 4-dimensional
        space.

        The new ImageBlockValue will always have the new dimension in the first position
        (index 0). ('Time' will be at self.origshape[0] and self.origslices[0]).

        Parameters:
        -----------
        newdimidx: nonnegative integer, default 0
            The zero-based index of the current block within the newly added dimension.
            (For instance, the timepoint of the volume within a new time series.)

        newdimsize: positive integer, default 1
            The total size of the newly added dimension.
            (For instance, the total number of time points in the higher-dimensional volume
            in which the new block is embedded.)

        Returns:
        --------
            a new ImageBlockValue object, with origslices and origshape modified as
            described above. New block value may be a view onto self.value.
        """
        newshape = list(self.origshape)
        newshape.insert(0, newdimsize)
        newslices = list(self.origslices)
        # todo: check array ordering here and insert at back if in 'F' order, to preserve contiguous ordering?
        newslices.insert(0, slice(newdimidx, newdimidx+1, 1))
        newblockshape = list(self.values.shape)
        newblockshape.insert(0, 1)
        newvalues = reshape(self.values, tuple(newblockshape))
        return type(self)(tuple(newshape), tuple(newslices), newvalues)

    def toSeriesIter(self, seriesDim=0):
        """Returns an iterator over key,array pairs suitable for casting into a Series object.

        Returns:
        --------
        iterator< key, series >
        key: tuple of int
        series: 1d array of self.values.dtype
        """
        rangeiters = self._get_range_iterators()
        # remove iterator over dimension where we are requesting series
        del rangeiters[seriesDim]
        # correct for original dimensionality if inserting at end of list
        insertDim = seriesDim if seriesDim >= 0 else len(self.origshape) + seriesDim
        # reverse rangeiters twice to ensure that first dimension is most rapidly varying
        for idxSeq in itertools.product(*reversed(rangeiters)):
            idxSeq = tuple(reversed(idxSeq))
            expandedIdxSeq = list(idxSeq)
            expandedIdxSeq.insert(insertDim, None)
            slices = []
            for d, (idx, origslice) in enumerate(zip(expandedIdxSeq, self.origslices)):
                if idx is None:
                    newslice = slice(None)
                else:
                    # correct slice into our own value for any offset given by origslice:
                    start = idx - origslice.start if not origslice == slice(None) else idx
                    newslice = slice(start, start+1, 1)
                slices.append(newslice)

            series = self.values[slices].squeeze()
            yield tuple(idxSeq), series

    def _get_range_iterators(self):
        """Returns a sequence of iterators over the range of the slices in self.origslices

        When passed to itertools.product, these iterators should cover the original image
        volume represented by this block.
        """
        iters = []
        noneSlice = slice(None)
        for sliceidx, sl in enumerate(self.origslices):
            if sl == noneSlice:
                it = xrange(self.origshape[sliceidx])
            else:
                it = xrange(sl.start, sl.stop, sl.step)
            iters.append(it)
        return iters

    def __repr__(self):
        return "ImageBlockValue(origshape=%s, origslices=%s, values=%s)" % \
               (repr(self.origshape), repr(self.origslices), repr(self.values))

class _BlockMemoryAsSequence(object):
    """Helper class used in calculation of slices for requested block partitions of a particular size.

    The partitioning strategy represented by objects of this class is to split into N equally-sized
    subdivisions along each dimension, starting with the rightmost dimension.

    So for instance consider an Image with spatial dimensions 5, 10, 3 in x, y, z. The first nontrivial
    partition would be to split into 2 blocks along the z axis:
    splits: (1, 1, 2)
    In this example, downstream this would turn into two blocks, one of size (5, 10, 2) and another
    of size (5, 10, 1).

    The next partition would be to split into 3 blocks along the z axis, which happens to
    corresponding to having a single block per z-plane:
    splits: (1, 1, 3)
    Here these splits would yield 3 blocks, each of size (5, 10, 1).

    After this the z-axis cannot be partitioned further, so the next partition starts splitting along
    the y-axis:
    splits: (1, 2, 3)
    This yields 6 blocks, each of size (5, 5, 1).

    Several other splits are possible along the y-axis, going from (1, 2, 3) up to (1, 10, 3).
    Following this we move on to the x-axis, starting with splits (2, 10, 3) and going up to
    (5, 10, 3), which is the finest subdivision possible for this data.

    Instances of this class represent the average size of a block yielded by this partitioning
    strategy in a linear order, moving from the most coarse subdivision (1, 1, 1) to the finest
    (x, y, z), where (x, y, z) are the dimensions of the array being partitioned.

    This representation is intended to support binary search for the partitioning strategy yielding
    a block size closest to a requested amount.
    """
    def __init__(self, dims):
        self._dims = dims

    def indtosub(self, idx):
        """Converts a linear index to a corresponding partition strategy, represented as
        number of splits along each dimension.
        """
        dims = self._dims
        ndims = len(dims)
        sub = [1] * ndims
        for didx, d in enumerate(dims[::-1]):
            didx = ndims - (didx + 1)
            delta = min(dims[didx]-1, idx)
            if delta > 0:
                sub[didx] += delta
                idx -= delta
            if idx <= 0:
                break
        return tuple(sub)

    def blockMemoryForSplits(self, sub):
        """Returns the average number of cells in a block generated by the passed sequence of splits.
        """
        from operator import mul
        sz = [d / float(s) for (d, s) in zip(self._dims, sub)]
        return reduce(mul, sz)

    def __len__(self):
        return sum([d-1 for d in self._dims]) + 1

    def __getitem__(self, item):
        sub = self.indtosub(item)
        return self.blockMemoryForSplits(sub)


class _BlockMemoryAsReversedSequence(_BlockMemoryAsSequence):
    """A version of _BlockMemoryAsSequence that represents the linear ordering of splits in the
    opposite order, starting with the finest partitioning allowable for the array dimensions.

    This can yield a sequence of block sizes in increasing order, which is required for binary
    search using python's 'bisect' library.
    """
    def _reverseIdx(self, idx):
        l = len(self)
        if idx < 0 or idx >= l:
            raise IndexError("list index out of range")
        return l - (idx + 1)

    def indtosub(self, idx):
        return super(_BlockMemoryAsReversedSequence, self).indtosub(self._reverseIdx(idx))
