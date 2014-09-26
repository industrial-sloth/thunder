package thunder.util.io.hadoop

import org.apache.hadoop.fs.Path
import org.apache.hadoop.io.{BytesWritable, LongWritable}
import org.apache.hadoop.mapreduce.{RecordReader, TaskAttemptContext, InputSplit, JobContext}
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat


class FullFileBinaryInputFormat extends FileInputFormat[LongWritable, BytesWritable] {

  override def isSplitable(context: JobContext, filename: Path): Boolean = {
    false
  }

  override def createRecordReader(split: InputSplit, context: TaskAttemptContext):
  RecordReader[LongWritable, BytesWritable] = {
    new FullFileBinaryRecordReader
  }
}
