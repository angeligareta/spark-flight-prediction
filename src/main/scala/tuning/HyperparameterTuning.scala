package tuning

import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.sql.DataFrame

object HyperparameterTuning {
  def showModelPrecision(predictions: DataFrame): Unit = {
    // Select example rows to display.
    predictions.select("prediction", "ArrDelay", "features").show(5)

    // Evaluator for test error
    var evaluator = new RegressionEvaluator()
      .setLabelCol("ArrDelay")
      .setPredictionCol("prediction")
      .setMetricName("rmse")

    val rmse = evaluator.evaluate(predictions)
    println(s"Root Mean Squared Error (RMSE) on test data = $rmse")

    // New evaluator for R2
    evaluator = new RegressionEvaluator()
      .setLabelCol("ArrDelay")
      .setPredictionCol("prediction")
      .setMetricName("r2")

    val r2 = evaluator.evaluate(predictions)
    println(s"R2 on test data = $r2")
  }
}
