package mlmodels

import org.apache.hadoop.mapred.InvalidInputException
import org.apache.spark.ml.{Pipeline, PipelineModel, PipelineStage}
import org.apache.spark.ml.regression.{LinearRegression, LinearRegressionModel}
import org.apache.spark.sql.DataFrame
import preprocess.PreProcessDataset
import tuning.HyperparameterTuning

object LinearRegressionCustomModel {

  val LINEAR_MODEL_PATH = Utils.MODEL_PATH + "/linear_regression";
  val VALIDATION_PIPELINE_PATH = Utils.MODEL_PATH + "/linear_regression_training";

  def saveValidationDataPipeline(validationDataset: DataFrame): DataFrame = {
    val pipelineStages = PreProcessDataset.getFeaturesPipelineStages();
    val pipeline = new Pipeline()
      .setStages(pipelineStages)
      .fit(validationDataset)

    pipeline.write.overwrite().save(VALIDATION_PIPELINE_PATH);
    return pipeline.transform(validationDataset)
  }

  def getTransformedValidationData(validationDataset: DataFrame): DataFrame = {
    val pipeline = PipelineModel.load(VALIDATION_PIPELINE_PATH);
    return pipeline.transform(validationDataset);
  }

  def trainAndSaveModel(trainingData: DataFrame): LinearRegressionModel = {
    // Declare the linear regression model.
    val lr = new LinearRegression()
      .setFeaturesCol("features")
      .setLabelCol("ArrDelay")
      .setMaxIter(10)

    // Get the preprocessing stages from utils.
    val pipelineStages = PreProcessDataset.getFeaturesPipelineStages()

    // Create the pipeline that is going to fit the model.
    val pipeline = new Pipeline().setStages(pipelineStages ++ Array(lr));

    // Train model. This also runs the indexer.
    val model = pipeline.fit(trainingData)

    // Convert model to Linear regression model
    val linearModel: LinearRegressionModel = model
      .stages(pipelineStages.length)
      .asInstanceOf[LinearRegressionModel]

    // Save model on disk.
    linearModel.write.overwrite().save(LINEAR_MODEL_PATH)
    println(s"Model saved on ${LINEAR_MODEL_PATH}")

    return linearModel;
  }

  def loadModelFromLocalStorage(): LinearRegressionModel = {
    val model =
      LinearRegressionModel.load(Utils.MODEL_PATH + "/linear_regression");
    return model
  }

  def start(dataset: DataFrame): Unit = {
    // Split the data into training and test sets (30% held out for testing).
    val Array(trainingData, testData) = dataset.randomSplit(Array(0.7, 0.3))

    // Get the transformed test data
    var transformedTestData: DataFrame = null
    try {
      transformedTestData = getTransformedValidationData(testData);
    } catch {
      case x: InvalidInputException => {
        transformedTestData = saveValidationDataPipeline(testData);
      }
    }

    // Evaluate the model. if it is on local storage
    // this method loads it and if not it creates it and saves it.
    try {
      println("Loading model from file system...")
      val linearModel = loadModelFromLocalStorage();
      println(
        s"RegressionModel -> ${linearModel.intercept} + ${linearModel.coefficients}"
      )

      val predictions = linearModel.transform(transformedTestData)
      HyperparameterTuning.showModelPrecision(predictions);

    } catch {
      case x: InvalidInputException => {
        println(x.getMessage)
        val linearModel = trainAndSaveModel(trainingData);
        println(
          s"RegressionModel -> ${linearModel.intercept} + ${linearModel.coefficients}"
        )

        val predictions = linearModel.transform(transformedTestData)
        HyperparameterTuning.showModelPrecision(predictions);
      }
    }
  }
}
