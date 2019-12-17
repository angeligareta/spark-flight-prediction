package mlmodels

import org.apache.hadoop.mapred.InvalidInputException
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.regression.{
  DecisionTreeRegressionModel,
  DecisionTreeRegressor
}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.DataFrame
import preprocess.PreProcessDataset
import tuning.HyperparameterTuning

object DecisionTreeModel {
  val DecisionTreeModelPath = Utils.SavePath + "/decision_tree"

  def trainAndGetModel(trainingData: DataFrame): DecisionTreeRegressionModel = {
    val pipelineStages = PreProcessDataset.getFeaturesPipelineStages()

    // Train a DecisionTree model.
    val dt = new DecisionTreeRegressor()
      .setLabelCol("ArrDelay")
      .setFeaturesCol("features")
      .setMaxBins(3500) // At least as large as all categorical variables

    // Pipeline to prepare the data before training the model
    val pipeline = new Pipeline()
      .setStages(pipelineStages ++ Array(dt))

    // Evaluator we want to use to choose the best model.
    val evaluator = new RegressionEvaluator()
      .setLabelCol("ArrDelay")
      .setPredictionCol("prediction")
      .setMetricName("rmse")

    // We use a ParamGridBuilder to construct a grid of parameters to search over.
    val paramGrid = new ParamGridBuilder()
      .addGrid(dt.maxDepth, Array(5))
      //.addGrid(dt.maxDepth, Array(5, 7, 9, 11))
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
    val decisionTreeModel: DecisionTreeRegressionModel =
      model.bestModel
        .asInstanceOf[PipelineModel]
        .stages(pipelineStages.length)
        .asInstanceOf[DecisionTreeRegressionModel]

    // Save best model
    decisionTreeModel.write
      .overwrite()
      .save(DecisionTreeModelPath)

    decisionTreeModel;
  }

  def start(dataset: DataFrame): Unit = {
    // Split the data into training and test sets (30% held out for testing).
    var Array(trainingData, testData) = dataset.randomSplit(Array(0.7, 0.3))

    // Transform validation data
    val pipelineStages = PreProcessDataset.getFeaturesPipelineStages()
    testData = new Pipeline()
      .setStages(pipelineStages)
      .fit(testData)
      .transform(testData)

    var decisionTreeModel: DecisionTreeRegressionModel = null;
    // First try to load random forest from path.
    try {
      decisionTreeModel =
        DecisionTreeRegressionModel.load(DecisionTreeModelPath)
      println("Loading Model from file...")
    }
    // Otherwise, train the model with training data and get new model.
    catch {
      case _: InvalidInputException => {
        println("Model file not found, training new model...")
        decisionTreeModel = trainAndGetModel(trainingData);
      }
    }

    // Make predictions.
    val predictions = decisionTreeModel.transform(testData)
    HyperparameterTuning.showModelPrecision(predictions);

    println("PARAMS")
    println(decisionTreeModel.explainParams())

    println("Feature Importances")
    println(decisionTreeModel.featureImportances.toString)
  }
}
