package mlmodels

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.regression.{
  DecisionTreeRegressionModel,
  DecisionTreeRegressor
}
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

    val pipeline = new Pipeline()
      .setStages(pipelineStages ++ Array(dt))

    // Train model. This also runs the indexer.
    val model = pipeline.fit(trainingData)

    // Save model
    model.write
      .overwrite()
      .save(Utils.MODEL_PATH + "/decision_tree")

    // Make predictions.
    val predictions = model.transform(testData)
    HyperparameterTuning.showModelPrecision(predictions);

    // Show DecisionTree
    val treeModel: DecisionTreeRegressionModel = model
      .stages(pipelineStages.length)
      .asInstanceOf[DecisionTreeRegressionModel]
    println("Learned regression tree model:\n" + treeModel.toDebugString)
  }
}
