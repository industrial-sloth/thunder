package thunder.util.io.hadoop

import java.io.IOException

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FSDataInputStream, FileSystem, Path}
import org.apache.hadoop.io.compress.CompressionCodecFactory
import org.apache.hadoop.io.{Text, IOUtils, BytesWritable, LongWritable}
import org.apache.hadoop.mapreduce.lib.input.FileSplit
import org.apache.hadoop.mapreduce.{TaskAttemptContext, InputSplit, RecordReader}


class FullFileBinaryRecordReader extends RecordReader[Text, BytesWritable] {

  override def initialize(inputSplit: InputSplit, context: TaskAttemptContext) {

    // the file input
    fileSplit = inputSplit.asInstanceOf[FileSplit]

    // the actual file we will be reading from
    val file = fileSplit.getPath

    // job configuration
    conf = context.getConfiguration

    // check compression
    val codec = new CompressionCodecFactory(conf).getCodec(file)
    if (codec != null) {
      throw new IOException("FullFileBinaryRecordReader does not support reading compressed files")
    }

    val whitelistProp: java.lang.String = conf.get("whitelist", "")
    if (! whitelistProp.isEmpty) {
      whitelistSet = Some(whitelistProp.split(",").toSet)
    }

  }

  override def getCurrentKey: Text = {
    recordKey
  }

  override def getCurrentValue: BytesWritable = {
    recordValue
  }

  override def getProgress: Float = {
    if (processed) 1.0f else 0.0f
  }

  override def nextKeyValue(): Boolean = {

    if (!processed) {

      val filePath: Path = fileSplit.getPath
      if (whitelistSet.isDefined && (! whitelistSet.get.contains(filePath.getName))) {
        // we have passed in a whitelist of filenames to read, and this file isn't in it
        processed = true
        return false
      }

      if (recordKey == null) {
        recordKey = new Text(fileSplit.getPath.getName)
        recordValue = new BytesWritable
      }

      val contents = new Array[Byte](fileSplit.getLength.toInt)


      val fs: FileSystem = filePath.getFileSystem(conf)

      var fileInputStream: FSDataInputStream = null
      try {
        fileInputStream = fs.open(filePath)
        IOUtils.readFully(fileInputStream, contents, 0, contents.length)
        recordValue.set(contents, 0, contents.length)
      } finally {
        IOUtils.closeStream(fileInputStream)
      }

      processed = true
      return true
    }

  false
  }

  override def close() {
    // already closed in nextKeyValue
  }

  var conf: Configuration = null
  var fileSplit: FileSplit = null
  var recordKey: Text = null
  var recordValue: BytesWritable = null
  var whitelistSet: Option[Set[String]] = None
  var processed: Boolean = false
}
