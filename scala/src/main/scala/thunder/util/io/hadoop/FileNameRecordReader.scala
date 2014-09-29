package thunder.util.io.hadoop

import org.apache.hadoop.io.{NullWritable, Text}
import org.apache.hadoop.mapreduce.lib.input.FileSplit
import org.apache.hadoop.mapreduce.{InputSplit, RecordReader, TaskAttemptContext}


class FileNameRecordReader extends RecordReader[Text, NullWritable] {

  override def initialize(inputSplit: InputSplit, context: TaskAttemptContext) {

    // the file input
    fileSplit = inputSplit.asInstanceOf[FileSplit]

  }

  override def getCurrentKey: Text = {
    recordKey
  }

  override def getCurrentValue: NullWritable = {
    recordValue
  }

  override def getProgress: Float = {
    if (processed) 1.0f else 0.0f
  }

  override def nextKeyValue(): Boolean = {

    if (recordKey == null) {
      recordKey = new Text(fileSplit.getPath.getName)
      recordValue = NullWritable.get
    }

    processed = true
    true
  }

  override def close() {
    // do nothing
  }

  var fileSplit: FileSplit = null
  var recordKey: Text = null
  var recordValue: NullWritable = null
  var processed: Boolean = false
}
