package mlmodels

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.regression.DecisionTreeRegressor
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.DataFrame
import preprocess.PreProcessDataset
import tuning.HyperparameterTuning

object DecisionTreeModel {
  def start(dataset: DataFrame): Unit = {
    // Split the data into training and test sets (30% held out for testing).
    val Array(trainingData, testData) = dataset.randomSplit(Array(0.7, 0.3))
    val pipelineStages = PreProcessDataset.getFeaturesPipelineStages()

    // Train a DecisionTree model.
    val dt = new DecisionTreeRegressor()
      .setLabelCol("ArrDelay")
      .setFeaturesCol("features")

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
      .addGrid(dt.maxDepth, Array(5, 7, 9, 11))
      .build()

    // Cross Validator will contribute to a better hyperparameter tuning
    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(5)

    // Train model using Cross Validator.
    val model = cv.fit(trainingData)

    // Make predictions.
    val predictions = model.transform(testData)
    HyperparameterTuning.showModelPrecision(predictions);

    // Save model
    model.write
      .overwrite()
      .save(Utils.MODEL_PATH + "/decision_tree")
  }
}
