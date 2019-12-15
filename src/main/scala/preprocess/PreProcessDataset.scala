package preprocess

import org.apache.spark.ml.PipelineStage
import org.apache.spark.ml.feature.{
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
    "LateAircraftDelay"
  );

  val categoricalVariables =
    Array(
      //"Year", if we have only one year could not be categorical
      "Month",
      "DayofMonth",
      "DayOfWeek",
      "UniqueCarrier",
      "FlightNum",
      "TailNum",
      "Origin",
      "Dest"
      // "Cancelled" canceled plane does not affect arrival delay
      //"CancellationCode"
    )

  val continuousVariables = Array(
    "Year",
    "DepTime",
    "CRSDepTime",
    "CRSArrTime",
    "CRSElapsedTime",
    "DepDelay",
    "Distance",
    "TaxiOut"
  );
  val indexedCategoricalVariables = categoricalVariables.map(v => s"${v}Index")
  val encodedCategoricalVariables = categoricalVariables.map(v => s"${v}Vec");
  val featuresVariables = encodedCategoricalVariables ++ continuousVariables

  def getFeaturesPipelineStages(): Array[PipelineStage] = {
    val categoricalIndexers = categoricalVariables
      .map(
        v =>
          new StringIndexer()
            .setInputCol(v)
            .setOutputCol(v + "Index")
            .setHandleInvalid("skip")
      )
    val categoricalEncoder = new OneHotEncoderEstimator()
      .setInputCols(indexedCategoricalVariables)
      .setOutputCols(encodedCategoricalVariables)
    val assembler = new VectorAssembler()
      .setInputCols(featuresVariables)
      .setOutputCol("features")

    categoricalIndexers ++ Array(categoricalEncoder) ++ Array(assembler)
  }

  def handleNAValues(preProcessDataset: DataFrame): Unit = {
    preProcessDataset.columns.foreach(column => {
      var columnType = "StringType"
      preProcessDataset.dtypes.foreach(tuple => {
        if (tuple._1 == column) {
          columnType = tuple._2
        }
      })

      println(s"Column ${column} with type ${columnType}")
      columnType match {
        case "IntegerType" => {
          val columnMean =
            preProcessDataset.agg(avg(column)).first().getDouble(0)
          println(s"Mean is ${columnMean}")
          preProcessDataset.na.fill(columnMean, Array(column))
        }
        case "StringType" => {
          val categoricalColumnMean = s"No_${column}"
          println(s"Categorical Mean is ${categoricalColumnMean}")
          preProcessDataset.na.fill(categoricalColumnMean, Array(column))
        }
      }
    })
  }

  def start(spark: SparkSession, dataset: DataFrame): DataFrame = {
    // Drop columns that the exercise required.
    var preProcessDataset = dataset.drop(variablesToDrop: _*);

    // Import implicits to use $
    import spark.implicits._

    continuousVariables.foreach(continuousVariable => {
      preProcessDataset = preProcessDataset
        .withColumn(continuousVariable, $"${continuousVariable}" cast "Int")
    })

    println(preProcessDataset.dtypes.mkString(", "))
    handleNAValues(preProcessDataset);

    preProcessDataset
  }
}
