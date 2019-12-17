package tuning

import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.sql.DataFrame

object HyperparameterTuning {
  def showModelPrecision(predictions: DataFrame): Unit = {
    val allowedParams = Array("mse", "rmse", "r2", "mae")
    var estimatorText = Array[String]()

    allowedParams.foreach(param => {
      // Evaluator for test error
      val evaluator = new RegressionEvaluator()
        .setLabelCol(Utils.ResponseVariable)
        .setPredictionCol("prediction")
        .setMetricName(param)

      val estimate = evaluator.evaluate(predictions)

      estimatorText = estimatorText :+ (
        s"${param} on test data = ${estimate}"
      )
    })

    // Select example rows to display.
    println("RESULTS")
    predictions.show(20)
    estimatorText.foreach(println)
  }
}
