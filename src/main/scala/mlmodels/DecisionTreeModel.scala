package mlmodels

import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.sql.DataFrame

object DecisionTreeModel {
  def start(dataset: DataFrame): Unit = {
    val datasetRDD = dataset.rdd.map(
      row =>
        LabeledPoint(
          row.getAs[Double]("ArrDelay"),
          row.getAs[org.apache.spark.mllib.linalg.Vector]("ArrTime")
      )
    )
    // Split the data into training and test sets (30% held out for testing)
    val splits = datasetRDD.randomSplit(Array(0.7, 0.3))
    val (trainingData, testData) = (splits(0), splits(1))

    // Train a DecisionTree model.
    //  Empty categoricalFeaturesInfo indicates all features are continuous.
    val categoricalFeaturesInfo = Map[Int, Int]()
    val impurity = "variance"
    val maxDepth = 5
    val maxBins = 32

    val model = DecisionTree.trainRegressor(
      trainingData,
      categoricalFeaturesInfo,
      impurity,
      maxDepth,
      maxBins
    )

    // Evaluate model on test instances and compute test error
    val labelsAndPredictions = testData.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }
    val testMSE =
      labelsAndPredictions.map { case (v, p) => math.pow(v - p, 2) }.mean()
    println(s"Test Mean Squared Error = $testMSE")
    println(s"Learned regression tree model:\n ${model.toDebugString}")
  }
}
