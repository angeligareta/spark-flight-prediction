package mlmodels

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.regression.{
  RandomForestRegressionModel,
  RandomForestRegressor
}
import org.apache.spark.sql.DataFrame
import preprocess.PreProcessDataset
import tuning.HyperparameterTuning

object RandomForestModel {
  def start(dataset: DataFrame): Unit = {
    // Split the data into training and test sets (30% held out for testing).
    val Array(trainingData, testData) = dataset.randomSplit(Array(0.7, 0.3))
    val pipelineStages = PreProcessDataset.getFeaturesPipelineStages(dataset)

    // Train a RandomForest model.
    val rf = new RandomForestRegressor()
      .setLabelCol("ArrDelay")
      .setFeaturesCol("features")

    val pipeline = new Pipeline()
      .setStages(pipelineStages ++ Array(rf))

    // Train model. This also runs the indexer.
    val model = pipeline.fit(trainingData)

    // Save model
    model.write
      .overwrite()
      .save(Utils.MODEL_PATH + "/random_forest")

    // Make predictions.
    val predictions = model.transform(testData)
    HyperparameterTuning.showModelPrecision(predictions);

    val rfModel = model
      .stages(pipelineStages.length)
      .asInstanceOf[RandomForestRegressionModel]
    println(s"Learned regression forest model:\n ${rfModel.toDebugString}")
  }
}
