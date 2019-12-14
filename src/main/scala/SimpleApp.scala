import org.apache.spark.sql.SparkSession

/* SimpleApp.scala */

object SimpleApp {
  def main(args: Array[String]) {
    try {
      // An error will be thrown if the env does not exist
      val DATASET_FOLDER_PATH: String = sys.env("DATASET_FOLDER_PATH")

      val spark = SparkSession.builder
        .appName("Simple Application")
        .config("spark.master", "local")
        .getOrCreate()

      val datasetsDF = spark.read
        .format("csv")
        .option("inferSchema", "true")
        .option("header", "true")
        .load(DATASET_FOLDER_PATH + "/1996.csv")

      LinearRegression.start(datasetsDF)

      spark.stop()
    } catch {
      case _: NoSuchElementException => {
        println(
          "Exception: You must provide the environment variable DATASET_FOLDER_PATH with the path to the datasets."
        )
      }
    }

  }
}
