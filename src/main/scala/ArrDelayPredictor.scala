import java.util.NoSuchElementException

import mlmodels.{
  DecisionTreeModel,
  LinearRegressionCustomModel,
  RandomForestModel
}
import org.apache.spark.sql.{AnalysisException, DataFrame, SparkSession}
import preprocess.PreProcessDataset

/**
  * Predictor for ArrDelay target variable in the flights dataset.
  */
object ArrDelayPredictor {

  val MergedDatasetPath = Utils.SavePath + "/merged_dataset"
  val MergedPreprocessedDatasetPath = Utils.SavePath + "/merged_processed_dataset"

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

      var datasetsDF: DataFrame = spark.emptyDataFrame

      try {
        datasetsDF = spark.read.load(MergedPreprocessedDatasetPath)

        println("Succesfully read preprocessed")
        datasetsDF.show(100)
      } catch {
        case _: AnalysisException => {
          try {
            datasetsDF = spark.read.load(MergedDatasetPath)

            println("Succesfully read merged dataset")
            datasetsDF.show(100)
          } catch {
            case _: AnalysisException => {
              println("Merged data frame did not exist")
              // Read dataset dataFrame
              datasetsDF = spark.read
                .format("csv")
                .option("header", "true")
                .load(datasetFolderPath)

              // Extract random sample of the input data
              val MaxNumRowTruncated = 1500000.0
              val proportion: Double = MaxNumRowTruncated / datasetsDF
                .count()
                .toDouble
              println(s"Extracting a proportion of ${proportion}...")
              val truncatedDatasetsDF =
                datasetsDF.sample(withReplacement = false, proportion, 1234)

              // Show rows before preprocessing
              println("-Before Preprocessing")
              println(s"Number of rows: ${truncatedDatasetsDF.count()}")
              truncatedDatasetsDF
                .sample(withReplacement = false, 0.1, 1234)
                .show(100)

              // Save dataset to memory
              truncatedDatasetsDF.write.save(MergedDatasetPath)

              // Use short dataset
              datasetsDF = truncatedDatasetsDF
            }
          } finally {
            println("- Preprocessing data")
            // If not processed, preprocess data
            datasetsDF = PreProcessDataset.start(spark, datasetsDF)

            //datasetsDF.write.save(MergedPreprocessedDatasetPath)

            println("Preprocessed data saved is disabled")
          }
        }
      }

      datasetsDF.show(100);

      datasetsDF.cache()

      // Execute ML model by choice of user
      val supportedMlModels = Array("lr, dt, rf");
      var mlModelSelected = if (interactiveMode) "" else "dt";

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
          LinearRegressionCustomModel.start(datasetsDF)
        }
        case "dt" => {
          println("Decision tree")
          DecisionTreeModel.start(datasetsDF)
        }
        case "rf" => {
          println("Random Forest")
          RandomForestModel.start(datasetsDF)
        }
      }
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
