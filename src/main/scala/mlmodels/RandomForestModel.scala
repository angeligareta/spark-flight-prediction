package mlmodels

import org.apache.hadoop.mapred.InvalidInputException
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.regression.{
  RandomForestRegressionModel,
  RandomForestRegressor
}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.DataFrame
import preprocess.PreProcessDataset
import tuning.HyperparameterTuning

object RandomForestModel {
  def getRandomForestPath() = {
    s"${Utils.getSavePath()}/random_forest";
  }

  def trainAndGetModel(trainingData: DataFrame): RandomForestRegressionModel = {
    val pipelineStages = PreProcessDataset.getFeaturesPipelineStages()

    // Train a RandomForest model.
    val rf = new RandomForestRegressor()
      .setLabelCol(Utils.ResponseVariable)
      .setFeaturesCol("normFeatures")
      .setMaxBins(7500) // At least as large as all categorical variables

    // Pipeline to prepare the data before training the model
    val pipeline = new Pipeline()
      .setStages(pipelineStages ++ Array(rf))

    // Evaluator we want to use to choose the best model.
    val evaluator = new RegressionEvaluator()
      .setLabelCol(Utils.ResponseVariable)
      .setPredictionCol("prediction")
      .setMetricName("rmse")

    // We use a ParamGridBuilder to construct a grid of parameters to search over.
    val paramGrid = new ParamGridBuilder()
    //.addGrid(rf.maxBins, Array(7500))
    //.addGrid(rf.numTrees, Array(100, 200, 300))
    //.addGrid(rf.maxDepth, Array(5, 10))
      .build()

    // Cross Validator will contribute to a better hyperparameter tuning
    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(5)

    // Train model using Cross Validator.
    val model = cv.fit(trainingData)

    // Extract the best model from cross validator models
    val randomForestModel: RandomForestRegressionModel =
      model.bestModel
        .asInstanceOf[PipelineModel]
        .stages(pipelineStages.length)
        .asInstanceOf[RandomForestRegressionModel]

    // Save best model
    randomForestModel.write.overwrite
      .save(getRandomForestPath())
    println(s"Model saved on ${getRandomForestPath()}")

    randomForestModel;
  }

  def start(dataset: DataFrame): Unit = {
    // Split the data into training and test sets (30% held out for testing).
    var Array(trainingData, testData) = dataset.randomSplit(Array(0.8, 0.2))

    // Transform validation data
    val pipelineStages = PreProcessDataset.getFeaturesPipelineStages()
    testData = new Pipeline()
      .setStages(pipelineStages)
      .fit(testData)
      .transform(testData)

    var randomForestModel: RandomForestRegressionModel = null;
    // First try to load random forest from path.
    try {
      randomForestModel =
        RandomForestRegressionModel.load(getRandomForestPath())
      println("Loading Model from file...")

    }
    // Otherwise, train the model with training data and get new model.
    catch {
      case _: InvalidInputException => {
        println("Model file not found, training new model...")
        randomForestModel = trainAndGetModel(trainingData);
      }
    }

    // Make predictions.
    val predictions = randomForestModel.transform(testData)
    HyperparameterTuning.showModelPrecision(predictions)

    println("SELECTED NUM TREES")
    println(randomForestModel.getNumTrees)

    println("SELECTED MAX DEPTH")
    println(randomForestModel.getMaxDepth)
  }
}
