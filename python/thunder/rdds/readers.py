import glob
import os


class LocalFSReader(object):
    def __init__(self, sparkcontext):
        self.sc = sparkcontext
        self.lastnrecs = -1

    @staticmethod
    def listFiles(datapath, ext=None, startidx=None, stopidx=None):

        if os.path.isdir(datapath):
            if ext:
                files = sorted(glob.glob(os.path.join(datapath, '*.' + ext)))
            else:
                files = sorted(os.listdir(datapath))
        else:
            files = sorted(glob.glob(datapath))

        if len(files) < 1:
            raise IOError('cannot find files of type "%s" in %s' % (ext if ext else '*', datapath))

        if startidx or stopidx:
            if startidx is None:
                startidx = 0
            if stopidx is None:
                stopidx = len(files)
            files = files[startidx:stopidx]

        return files

    def read(self, datapath, ext=None, startidx=None, stopidx=None):
        """Returns RDD of int, buffer k/v pairs
        """
        filepaths = self.listFiles(datapath, ext=ext, startidx=startidx, stopidx=stopidx)

        def readfcn(filepath):
            buf = None
            with open(filepath, 'rb') as f:
                buf = f.read()
            return buf

        lfilepaths = len(filepaths)
        self.lastnrecs = lfilepaths
        return self.sc.parallelize(enumerate(filepaths), lfilepaths).map(lambda (k, v): (k, readfcn(v)))
