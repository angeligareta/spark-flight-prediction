package mlmodels

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.regression.{
  DecisionTreeRegressionModel,
  DecisionTreeRegressor
}
import org.apache.spark.sql.DataFrame
import preprocess.PreProcessDataset

object DecisionTreeModel {
  def start(dataset: DataFrame): Unit = {
    val modelPath = "./models/decision_tree";

    // Train a DecisionTree model.
    var dt = new DecisionTreeRegressor()
      .setLabelCol("ArrDelay")
      .setFeaturesCol("features")

    //    try {
    //      var dt = DecisionTreeRegressor.load(modelPath);
    //
    //    }
    //    catch {
    //      case x: InvalidInputException {
    //
    //    }
    //    }

    // Split the data into training and test sets (30% held out for testing).
    val Array(trainingData, testData) = dataset.randomSplit(Array(0.7, 0.3))

    val pipelineStages = PreProcessDataset.getFeaturesPipelineStages(dataset)

    // Chain indexer and tree in a Pipeline.
    val pipeline = new Pipeline()
      .setStages(pipelineStages ++ Array(dt))

    // Train model. This also runs the indexer.
    val model = pipeline.fit(trainingData)

    // Make predictions.
    val predictions = model.transform(testData)

    // Select example rows to display.
    predictions.select("prediction", "ArrDelay", "features").show(5)

    // Select (prediction, true label) and compute test error
    val evaluator = new RegressionEvaluator()
      .setLabelCol("ArrDelay")
      .setPredictionCol("prediction")
      .setMetricName("rmse")

    val rmse = evaluator.evaluate(predictions)
    println("Root Mean Squared Error (RMSE) on test data = " + rmse)

    val treeModel: DecisionTreeRegressionModel = model
      .stages(pipelineStages.length)
      .asInstanceOf[DecisionTreeRegressionModel]
    println("Learned regression tree model:\n" + treeModel.toDebugString)

    println("Saving model")
    treeModel.save(modelPath);
  }
}
