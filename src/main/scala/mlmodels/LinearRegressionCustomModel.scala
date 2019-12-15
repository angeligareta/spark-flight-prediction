package mlmodels

import org.apache.hadoop.mapred.InvalidInputException
import org.apache.spark.ml.evaluation.{
  BinaryClassificationEvaluator,
  RegressionEvaluator
}
import org.apache.spark.ml.{Pipeline, PipelineModel, PipelineStage}
import org.apache.spark.ml.regression.{LinearRegression, LinearRegressionModel}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
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

  def start2(dataset: DataFrame): LinearRegressionModel = {
    // Declare the linear regression model.
    val lr = new LinearRegression()
      .setFeaturesCol("features")
      .setLabelCol("ArrDelay")
      .setMaxIter(10)

    // Get the preprocessing stages from utils.
    val pipelineStages = PreProcessDataset.getFeaturesPipelineStages()

    // Create the pipeline that is going to fit the model.
    val pipeline = new Pipeline().setStages(pipelineStages ++ Array(lr));

    // We use a ParamGridBuilder to construct a grid of parameters to search over.
    // With 3 values for hashingTF.numFeatures and 2 values for lr.regParam,
    // this grid will have 3 x 2 = 6 parameter settings for CrossValidator to choose from.
    val paramGrid = new ParamGridBuilder()
      .addGrid(lr.regParam, Array(0.1)) // Array(0.001, 0.01, 0.1, 0.5, 1.0, 2.0)
      //.addGrid(lr.elasticNetParam, Array(0.25, 0.5)) // Array(0.0, 0.25, 0.5, 0.75, 1.0)
      //.addGrid(lr.maxIter, Array(1, 5, 10, 20, 50)) // Array(1, 5, 10, 20, 50)
      .build()

    // We now treat the Pipeline as an Estimator, wrapping it in a CrossValidator instance.
    // This will allow us to jointly choose parameters for all Pipeline stages.
    // A CrossValidator requires an Estimator, a set of Estimator ParamMaps, and an Evaluator.
    // Note that the evaluator here is a BinaryClassificationEvaluator and its default metric
    // is areaUnderROC.
    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(new RegressionEvaluator().setLabelCol("ArrDelay"))
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(2) // Use 3+ in practice
      .setParallelism(2) // Evaluate up to 2 parameter settings in parallel

    // Fit the model using the cross validator.
    val model = cv.fit(dataset)

    /*
    val rm = new RegressionMetrics(
      holdout.rdd
        .map(x => (x(0).asInstanceOf[Double], x(1).asInstanceOf[Double]))
    )
    println("sqrt(MSE): " + Math.sqrt(rm.meanSquaredError))
    println("R Squared: " + rm.r2)
    println("Explained Variance: " + rm.explainedVariance + "\n")
     */
    // Convert the best model to a Linear regression model
    val linearModel: LinearRegressionModel =
      model.bestModel
        .asInstanceOf[PipelineModel]
        .stages(pipelineStages.length)
        .asInstanceOf[LinearRegressionModel]

    // Save model on disk.
    linearModel.write.overwrite().save(LINEAR_MODEL_PATH)
    println(s"Model saved on ${LINEAR_MODEL_PATH}")

    return linearModel;
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

      println(s"Coefficients: ${linearModel.coefficients}")
      println(s"Intercept: ${linearModel.intercept}")

      val trainingSummary = linearModel.summary
      println(s"numIterations: ${trainingSummary.totalIterations}")
      println(s"objectiveHistory: ${trainingSummary.objectiveHistory.toList}")
      trainingSummary.residuals.show()
      println(s"RMSE: ${trainingSummary.rootMeanSquaredError}")
      println(s"r2: ${trainingSummary.r2}")

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
