import java.util.NoSuchElementException

import mlmodels.{
  LinearRegressionCustomModel,
  DecisionTreeModel,
  RandomForestModel
}
import org.apache.spark.sql.SparkSession
import preprocess.PreProcessDataset

/**
  * Predictor for ArrDelay target variable in the flights dataset.
  */
object ArrDelayPredictor {
  def main(args: Array[String]) {
    try {
      // An error will be thrown if the arguments required do not exist
      if (args.length == 0) {
        throw new NoSuchElementException()
      }
      val datasetFolderPath: String = args(0);

      // Variable to specify if the predictor should be interactive. (Allowing the user to select ML models...)
      val interactiveMode =
        if (args.length > 1 && args(1) == "true") true else false;

      val spark = SparkSession.builder
        .appName("Flight Arrival Prediction")
        .config("spark.master", "local") // Work in local only
        .getOrCreate()

      // Read dataset dataFrame
      val datasetsDF = spark.read
        .format("csv")
        .option("inferSchema", "true")
        .option("header", "true")
        .load(datasetFolderPath + "/1996.csv")

      // Preprocess data
      val processedDatasetsDF = PreProcessDataset.start(spark, datasetsDF);
      processedDatasetsDF.show(10);

      // Execute ML model by choice of user
      val supportedMlModels = Array("lr, dt, glr");
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

        case "lr" => {
          println("Linear regression")
          LinearRegressionCustomModel.start(processedDatasetsDF)
        }
        case "dt" => {
          println("Decision tree")
          DecisionTreeModel.start(processedDatasetsDF)
        }
        case "rf" => {
          println("Random Forest")
          RandomForestModel.start(processedDatasetsDF)
        }
      }

      // TODO: Show accuracy of the ML chosen

      spark.stop()
    } catch {
      case _: NoSuchElementException => {
        println(
          "Exception: You must provide the dataset folder path as the first argument of the app. Optionally, a second argument for interactive app can be passed."
        )
      }
    }

  }
}
