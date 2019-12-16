import java.io.{File, PrintWriter}
import java.util

import org.apache.spark.ml.regression.LinearRegressionModel
import preprocess.PreProcessDataset

/**
  * Common utils for the package.
  */
package object Utils {
  val MODEL_PATH = "./models";
  val MODEL_SUMMARY_PATH = s"${MODEL_PATH}/summary.txt"

  def printModelSummary(model: LinearRegressionModel) = {
    val trainingSummary = model.summary

    val pw = new PrintWriter(new File(MODEL_SUMMARY_PATH))

    var toWriteString = s"totalIterations -> ${trainingSummary.totalIterations}"
    pw.println(toWriteString)
    println(toWriteString)

    toWriteString = s"numInstances -> ${trainingSummary.numInstances}"
    pw.println(toWriteString)
    println(toWriteString)

    toWriteString = s"----- MODEL CHOSEN PARAMS -----"
    pw.println(toWriteString)
    println(toWriteString)

    toWriteString = s"regParam -> ${model.getRegParam}"
    pw.println(toWriteString)
    println(toWriteString)

    toWriteString = s"elasticNetParam -> ${model.getElasticNetParam}"
    pw.println(toWriteString)
    println(toWriteString)

    toWriteString = s"elasticNetParam -> ${model.getMaxIter}"
    pw.println(toWriteString)
    println(toWriteString)

    toWriteString = s"----- MODEL SUMMARY -----"
    pw.println(toWriteString)
    println(toWriteString)

    toWriteString = s"r2 -> ${trainingSummary.r2}"
    pw.println(toWriteString)
    println(toWriteString)

    toWriteString = s"r2adj -> ${trainingSummary.r2adj}"
    pw.println(toWriteString)
    println(toWriteString)

    toWriteString = "objectiveHistory -> " + trainingSummary.objectiveHistory
      .mkString(" ")
    pw.println(toWriteString)
    println(toWriteString)

    toWriteString =
      s"rootMeanSquareError -> ${trainingSummary.rootMeanSquaredError}"
    pw.println(toWriteString)
    println(toWriteString)

    toWriteString = s"explainedVariance -> ${trainingSummary.explainedVariance}"
    pw.println(toWriteString)
    println(toWriteString)

    toWriteString = s"degreesOfFreedom -> ${trainingSummary.degreesOfFreedom}"
    pw.println(toWriteString)
    println(toWriteString)

    toWriteString = s"labelCol -> ${trainingSummary.labelCol}"
    pw.println(toWriteString)
    println(toWriteString)

    toWriteString = s"meanSquaredError -> ${trainingSummary.meanSquaredError}"
    pw.println(toWriteString)
    println(toWriteString)

    toWriteString = s"meanAbsoluteError -> ${trainingSummary.meanAbsoluteError}"
    pw.println(toWriteString)
    println(toWriteString)

    toWriteString = "devianceResiduals -> " + trainingSummary.devianceResiduals
      .mkString(" ")
    pw.println(toWriteString)
    println(toWriteString)

    // Print column with t value and p value.
    var listToPrint = List(
      List("Column", "p-value", "t-value", "coefficient standard error"),
      List(
        "Intercept",
        trainingSummary.pValues(0),
        trainingSummary.tValues(0),
        trainingSummary.coefficientStandardErrors(0)
      )
    )

    for (i <- 1 until PreProcessDataset.featuresVariables.length) {
      val column = PreProcessDataset.featuresVariables(i);
      val pvalue = trainingSummary.pValues(i);
      val tvalue = trainingSummary.tValues(i);
      val coefStdErr = trainingSummary.coefficientStandardErrors(i)
      listToPrint = listToPrint :+ List(column, pvalue, tvalue, coefStdErr)
    }

    toWriteString = Tabulator.format(listToPrint)
    pw.println(toWriteString)
    println(toWriteString)
    /*
    toWriteString = "residuals -> " + trainingSummary.residuals
      .collect()
      .mkString(" ")
    pw.println(toWriteString)
    println(toWriteString)
     */

    /*
    toWriteString = "predictions -> " + trainingSummary.predictions
      .collect()
      .mkString(" ")
    pw.println(toWriteString)
    println(toWriteString)*/

    pw.close()
  }
}
