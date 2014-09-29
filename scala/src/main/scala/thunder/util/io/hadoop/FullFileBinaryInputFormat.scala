package thunder.util.io.hadoop

import org.apache.hadoop.fs.Path
import org.apache.hadoop.io.{Text, BytesWritable}
import org.apache.hadoop.mapreduce.{RecordReader, TaskAttemptContext, InputSplit, JobContext}
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat


class FullFileBinaryInputFormat extends FileInputFormat[Text, BytesWritable] {

  override def isSplitable(context: JobContext, filename: Path): Boolean = {
    false
  }

  override def createRecordReader(split: InputSplit, context: TaskAttemptContext):
  RecordReader[Text, BytesWritable] = {
    new FullFileBinaryRecordReader
  }
}
