package thunder.util.io.hadoop

import org.apache.hadoop.io.{IntWritable, NullWritable, Text}
import org.apache.hadoop.mapreduce.lib.input.FileSplit
import org.apache.hadoop.mapreduce.{InputSplit, RecordReader, TaskAttemptContext}


class FileNameRecordReader extends RecordReader[Text, IntWritable] {

  override def initialize(inputSplit: InputSplit, context: TaskAttemptContext) {

    // the file input
    fileSplit = inputSplit.asInstanceOf[FileSplit]

  }

  override def getCurrentKey: Text = {
    recordKey
  }

  override def getCurrentValue: IntWritable = {
    recordValue
  }

  override def getProgress: Float = {
    if (processed) 1.0f else 0.0f
  }

  override def nextKeyValue(): Boolean = {

    if (! processed) {
      if (recordKey == null) {
        recordKey = new Text(fileSplit.getPath.getName)
        recordValue = new IntWritable(0)
      }

      processed = true
      return true
    }
    false
  }

  override def close() {
    // do nothing
  }

  var fileSplit: FileSplit = null
  var recordKey: Text = null
  var recordValue: IntWritable = null
  var processed: Boolean = false
}
