package preprocess

import org.apache.spark.ml.PipelineStage
import org.apache.spark.ml.feature.{Normalizer, StringIndexer, VectorAssembler}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, SparkSession}

/**
  * Object with methods to preprocess data
  */
object PreProcessDataset {
  val variablesToDrop = Array(
    "ArrTime",
    "ActualElapsedTime",
    "AirTime",
    "TaxiIn",
    "Diverted",
    "CarrierDelay",
    "WeatherDelay",
    "NASDelay",
    "SecurityDelay",
    "LateAircraftDelay",
    // New variables that we think are unnecessary
    "Cancelled",
    "CancellationCode"
  );

  val categoricalVariables =
    Array(
      "Year",
      "Month",
      "DayofMonth",
      "DayOfWeek",
      "UniqueCarrier",
      "FlightNum",
      "TailNum",
      "Origin",
      "Dest"
    )

  val continuousVariables = Array(
    "DepTime",
    "CRSDepTime",
    "CRSArrTime",
    "CRSElapsedTime",
    "DepDelay",
    "Distance",
    "TaxiOut"
  )

  val newContinuousVariables = Array("DepTimeMin", "CRSDepTimeMin")
  val newCategoricalVariables = Array(
    //"DepTimeDisc", To try without DepTimeMin
    //"CRSDepTimeDisc" To try without CRSDepTimeMin
  )

  val totalContinuousVariables
    : Array[String] = continuousVariables ++ newContinuousVariables
  val totalCategoricalVariables
    : Array[String] = categoricalVariables // ++ newCategoricalVariables
  val indexedCategoricalVariables: Array[String] =
    totalCategoricalVariables.map(v => s"${v}Index")

  val featuresVariables
    : Array[String] = indexedCategoricalVariables ++ totalContinuousVariables

  def getFeaturesPipelineStages(): Array[PipelineStage] = {
    val categoricalIndexers = totalCategoricalVariables
      .map(
        v =>
          new StringIndexer()
            .setInputCol(v)
            .setOutputCol(s"${v}Index")
            .setHandleInvalid("keep")
      )

    val assembler = new VectorAssembler()
      .setInputCols(featuresVariables)
      .setOutputCol("features")
      .setHandleInvalid("keep")

    val featuresNormalizer = new Normalizer()
      .setInputCol("features")
      .setOutputCol("normFeatures")
      .setP(1.0)

    categoricalIndexers ++ Array(assembler, featuresNormalizer)
  }

  def addNewColumns(spark: SparkSession, dataset: DataFrame): DataFrame = {
    // Import implicits to use $
    import spark.implicits._

    // Transformation to convert column from custom time to min
    val transformCustomTimeToMin = udf((time: Double) => {
      val timeInString = time.toInt.toString

      var hour = 0
      if (timeInString.length > 2) {
        hour = timeInString.substring(0, timeInString.length - 2).toInt
      }
      (hour * 60) + timeInString.takeRight(2).toInt
    })

    val discretizeTime = udf((time: Double) => {
      time match {
        case time if time >= 1 && time <= 1125   => "Morning"
        case time if time > 1125 && time <= 1750 => "Afternoon"
        case time if time > 1750 && time <= 2400 => "Evening"
      }
    })

    dataset
      .withColumn(
        "DepTimeMin",
        transformCustomTimeToMin($"DepTime") cast "Double"
      )
      .withColumn(
        "CRSDepTimeMin",
        transformCustomTimeToMin($"CRSDepTime") cast "Double"
      )
      .withColumn("DepTimeDisc", discretizeTime($"DepTime") cast "String")
      .withColumn("CRSDepTimeDisc", discretizeTime($"CRSDepTime") cast "String")
      .withColumn("ArrDelayCubeRoot", cbrt($"ArrDelay") cast "Double")
  }

  def handleNAValues(dataset: DataFrame,
                     columnsToProcess: Array[String]): DataFrame = {
    // First drop na of explanatory variable
    var preProcessDataset = dataset.na.drop(Array("ArrDelay"))

    columnsToProcess.foreach(column => {
      var columnType = "StringType"
      dataset.dtypes.foreach(tuple => {
        if (tuple._1 == column) {
          columnType = tuple._2
        }
      })

      println(s"Column ${column} with type ${columnType}")
      columnType match {
        case "DoubleType" => {
          val columnMean =
            dataset.agg(avg(column)).first().getDouble(0)
          println(s"Mean is ${columnMean}")
          preProcessDataset =
            preProcessDataset.na.fill(columnMean, Array(column))
        }
        case "StringType" => {
          val categoricalColumnMean = s"No_${column}"
          println(s"Categorical Mean is ${categoricalColumnMean}")
          preProcessDataset =
            preProcessDataset.na.fill(categoricalColumnMean, Array(column))
        }
      }
    })
    preProcessDataset
  }

  def castContinuousVariables(spark: SparkSession,
                              dataset: DataFrame,
                              variablesToCast: Array[String]): DataFrame = {
    var preProcessDataset = dataset
    // Import implicits to use $
    import spark.implicits._

    variablesToCast.foreach(continuousVariable => {
      preProcessDataset = preProcessDataset
        .withColumn(continuousVariable, $"${continuousVariable}" cast "Double")
    })

    preProcessDataset
  }

  def start(spark: SparkSession, dataset: DataFrame): DataFrame = {
    // Drop columns that the exercise required.
    var preProcessDataset = dataset.drop(variablesToDrop: _*)

    println("Dataset types")
    println(preProcessDataset.dtypes.mkString(", "))

    // Cast variables
    preProcessDataset = castContinuousVariables(
      spark,
      preProcessDataset,
      continuousVariables ++ Array("ArrDelay")
    )
    println("Preprocessed dataset after cast")
    preProcessDataset.show(100)

    // Handle NA Values
    preProcessDataset = handleNAValues(
      preProcessDataset,
      categoricalVariables ++ continuousVariables ++ Array("ArrDelay")
    )
    println("Preprocessed dataset after handle na values")
    preProcessDataset.show(100)

    // Add new columns
    preProcessDataset = addNewColumns(spark, preProcessDataset)

    println("Preprocessed dataset after add new columns")
    preProcessDataset.show(100)

    preProcessDataset
  }
}
