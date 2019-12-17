package preprocess

import org.apache.spark.ml.PipelineStage
import org.apache.spark.ml.feature.{
  Normalizer,
  OneHotEncoderEstimator,
  StringIndexer,
  VectorAssembler
}
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
      //"DepTimeDisc", To try without DepTimeMin
      //"CRSDepTimeDisc" To try without CRSDepTimeMin
    )

  val continuousVariables = Array(
    "Year",
    "DepTime",
    "CRSDepTime",
    "CRSArrTime",
    "CRSElapsedTime",
    "DepDelay",
    "Distance",
    "TaxiOut",
    "DepTimeMin",
    "CRSDepTimeMin"
  );
  val indexedCategoricalVariables = categoricalVariables.map(v => s"${v}Index")
  val encodedCategoricalVariables = categoricalVariables.map(v => s"${v}Vec");
  val featuresVariables = indexedCategoricalVariables ++ continuousVariables

  def getFeaturesPipelineStages(): Array[PipelineStage] = {
    val categoricalIndexers = categoricalVariables
      .map(
        v =>
          new StringIndexer()
            .setInputCol(v)
            .setOutputCol(s"${v}Index")
            .setHandleInvalid("keep")
      )
    val categoricalEncoder = new OneHotEncoderEstimator()
      .setInputCols(indexedCategoricalVariables)
      .setOutputCols(encodedCategoricalVariables)
    val assembler = new VectorAssembler()
      .setInputCols(featuresVariables)
      .setOutputCol("features")
      .setHandleInvalid("keep")
    val featuresNormalizer = new Normalizer()
      .setInputCol("features")
      .setOutputCol("normFeatures")
      .setP(1.0)

    categoricalIndexers ++ Array(
      categoricalEncoder,
      assembler,
      featuresNormalizer
    )
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
      .withColumn("DepTimeMin", transformCustomTimeToMin($"DepTime"))
      .withColumn("CRSDepTimeMin", transformCustomTimeToMin($"CRSDepTime"))
      .withColumn("DepTimeDisc", discretizeTime($"DepTime"))
      .withColumn("CRSDepTimeDisc", discretizeTime($"CRSDepTime"))
      .withColumn("ArrDelayCubeRoot", cbrt($"ArrDelay"))
  }

  def handleNAValues(dataset: DataFrame): DataFrame = {
    // First drop na of explanatory variable
    var preProcessDataset = dataset.na.drop(Array("ArrDelay"))

    dataset.columns.foreach(column => {
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

  def start(spark: SparkSession, dataset: DataFrame): DataFrame = {
    // Drop columns that the exercise required.
    var preProcessDataset = dataset.drop(variablesToDrop: _*)

    // Import implicits to use $
    import spark.implicits._

    continuousVariables.foreach(continuousVariable => {
      preProcessDataset = preProcessDataset
        .withColumn(continuousVariable, $"${continuousVariable}" cast "Double")
    })
    preProcessDataset = preProcessDataset
      .withColumn("ArrDelay", $"ArrDelay" cast "Double")

    println(preProcessDataset.dtypes.mkString(", "))

    // Handle NA Values
    preProcessDataset = handleNAValues(preProcessDataset)

    // Add new columns
    preProcessDataset = addNewColumns(spark, preProcessDataset)

    preProcessDataset
  }
}
