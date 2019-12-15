package mlmodels

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.Normalizer
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.sql.DataFrame
import preprocess.PreProcessDataset

object LinearRegression {
  def start(dataset: DataFrame): Unit = {
    val Array(training, test) =
      dataset.randomSplit(Array(0.9, 0.1), seed = 12345)

    val pipelineStages = PreProcessDataset.getFeaturesPipelineStages(dataset);

    val lr = new LinearRegression()
      .setFeaturesCol("features")
      .setLabelCol("ArrDelay")
      .setMaxIter(10)

    val pipeline = new Pipeline().setStages(pipelineStages :+ lr);

    val model = pipeline.fit(training)

    val predictions = model.transform(test)

    // Select example rows to display.
    predictions.show(25)

    /*val lr = new LinearRegression()
      .setFeaturesCol("features")
      .setLabelCol("ArrDelay")
      .setMaxIter(10)

    val normalizer = new Normalizer()
      .setInputCol("features")
      .setOutputCol("normalizedFeatures")
      .setP(1)*/

//    val pipeline = new Pipeline().setStages(pipelineStages :+ lr);
//    val model = pipeline.fit(training)
//
//    // We could not do prediction
//    val predictions = model.transform(training)
//
//    val regEval = new RegressionEvaluator()
//      .setMetricName("r2")
//      .setPredictionCol("prediction")
//      .setLabelCol("label")
//
//    regEval.evaluate(predictions)

    //model.transform(test).show(10)

//    // We use a ParamGridBuilder to construct a grid of parameters to search over.
//    // TrainValidationSplit will try all combinations of values and determine best model using
//    // the evaluator.
//    val paramGrid = new ParamGridBuilder()
//      .addGrid(lr.regParam, Array(0.1, 0.01))
//      .addGrid(lr.fitIntercept)
//      .addGrid(lr.elasticNetParam, Array(0.0, 0.5, 1.0))
//      .build()
//
//    // In this case the estimator is simply the linear regression.
//    // A TrainValidationSplit requires an Estimator, a set of Estimator ParamMaps, and an Evaluator.
//    val trainValidationSplit = new TrainValidationSplit()
//      .setEstimator(lr)
//      .setEvaluator(new RegressionEvaluator)
//      .setEstimatorParamMaps(paramGrid)
//      // 80% of the data will be used for training and the remaining 20% for validation.
//      .setTrainRatio(0.8)
//      // Evaluate up to 2 parameter settings in parallel
//      .setParallelism(2)
//
//    // Run train validation split, and choose the best set of parameters.
//    val model = trainValidationSplit.fit(training)
//
//    // Make predictions on test data. model is the model with combination of parameters
//    // that performed best.
//    model
//      .transform(test)
//      .show()
  }

}
