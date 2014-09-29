package thunder.util.io.hadoop

import org.apache.hadoop.fs.Path
import org.apache.hadoop.io.{NullWritable, Text}
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat
import org.apache.hadoop.mapreduce.{InputSplit, JobContext, RecordReader, TaskAttemptContext}


class FileNameInputFormat extends FileInputFormat[Text, NullWritable] {

  override def isSplitable(context: JobContext, filename: Path): Boolean = {
    false
  }

  override def createRecordReader(split: InputSplit, context: TaskAttemptContext):
  RecordReader[Text, NullWritable] = {
    new FileNameRecordReader
  }
}
