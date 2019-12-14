import org.apache.spark.sql.SparkSession
import preprocessing.PreProcessDataset

/**
  * Predictor for ArrDelay target variable in the flights dataset.
  */
object ArrDelayPredictor {
  def main(args: Array[String]) {
    try {
      // An error will be thrown if the env variable does not exist
      val DATASET_FOLDER_PATH: String = sys.env("DATASET_FOLDER_PATH");

      // Variable to specify if the predictor should be interactive. (Allowing the user to select ML models...)
      val interactiveMode = true;

      val spark = SparkSession.builder
        .appName("Flight Arrival Prediction")
        .config("spark.master", "local") // Work in local only
        .getOrCreate()

      // Read dataset dataFrame
      val datasetsDF = spark.read
        .format("csv")
        .option("inferSchema", "true")
        .option("header", "true")
        .load(DATASET_FOLDER_PATH + "/1996.csv")

      // Preprocess data
      val processedDatasetsDF = PreProcessDataset.start(datasetsDF);

      // Execute ML model by choice of user
      val supportedMlModels = Array("lr");
      var mlModelSelected = if (interactiveMode) "" else "lr";

      // If interactive mode, allow user to select a custom machine learning technique
      while (mlModelSelected == "") {
        println(
          "Choose ML to use between: " + supportedMlModels
            .mkString("[", ", ", "]")
        );

        val userInput = scala.io.StdIn.readLine().trim()
        if (supportedMlModels.contains(userInput)) {
          mlModelSelected = userInput;
        } else {
          println(
            s"The machine learning model selected '${userInput}' does not exist."
          )
        }
      }

      mlModelSelected match {
        case "lr" =>
          LinearRegression.start(processedDatasetsDF)
      }

      // TODO: Show accuracy of the ML chosen

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
