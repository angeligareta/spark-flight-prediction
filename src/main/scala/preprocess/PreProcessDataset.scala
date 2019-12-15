package preprocess

import org.apache.spark.ml.PipelineStage
import org.apache.spark.ml.feature.{
  OneHotEncoderEstimator,
  StringIndexer,
  VectorAssembler
}
import org.apache.spark.ml.param.shared.HasHandleInvalid
import org.apache.spark.ml.util.DefaultParamsWritable
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, SparkSession}

/**
  * Object with methods to preprocess data
  */
object PreProcessDataset {
  def getFeaturesPipelineStages()
    : Array[PipelineStage with HasHandleInvalid with DefaultParamsWritable] = {
    val uniqueCarrierIndexer =
      new StringIndexer()
        .setInputCol("UniqueCarrier")
        .setOutputCol("UniqueCarrierIndexed")
        .setHandleInvalid("skip")

    val tailNumIndexer =
      new StringIndexer()
        .setInputCol("TailNum")
        .setOutputCol("TailNumIndexed")
        .setHandleInvalid("skip")

    val originIndexer =
      new StringIndexer()
        .setInputCol("Origin")
        .setOutputCol("OriginIndexed")
        .setHandleInvalid("skip")

    val destIndexer =
      new StringIndexer()
        .setInputCol("Dest")
        .setOutputCol("DestIndexed")
        .setHandleInvalid("skip")

    /*val cancellationCodeIndexer = new StringIndexer()
      .setInputCol("CancellationCode")
      .setOutputCol("CancellationCodeIndexed")*/

    val encoder = new OneHotEncoderEstimator()
      .setInputCols(
        Array(
          tailNumIndexer.getOutputCol,
          originIndexer.getOutputCol,
          destIndexer.getOutputCol
          //cancellationCodeIndexer.getOutputCol
        )
      )
      .setOutputCols(
        Array(
          "UniqueCarrierVec",
          "TailNumVec",
          "OriginVec",
          "DestVec"
          //"CancellationCodeVec"
        )
      )

    val features = new VectorAssembler()
      .setInputCols(
        Array(
          "Year",
          "Month",
          "DayofMonth",
          "DayOfWeek",
          "DepTime",
          "CRSDepTime",
          "UniqueCarrierIndexed",
          "FlightNum",
          "TailNumIndexed",
          "DepDelay",
          "Distance",
          "OriginIndexed",
          "DestIndexed"
          //"CancellationCodeVec"
        )
      )
      .setOutputCol("features")

    return Array(
      uniqueCarrierIndexer,
      tailNumIndexer,
      originIndexer,
      destIndexer,
      //cancellationCodeIndexer,
      // encoder,
      features
    );
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
          val columnMean = preProcessDataset.agg(avg(column)).first().getInt(0)
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
    val columnsToDrop = Array(
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
      "CancellationCode" // All NA in 1996
    );
    var preProcessDataset = dataset.drop(columnsToDrop: _*);

    // Import implicits to use $
    import spark.implicits._

    // Cast columns that are string to correct format
    preProcessDataset = preProcessDataset
      .withColumn("DepTime", $"DepTime" cast "Int")
      .withColumn("CRSElapsedTime", $"CRSElapsedTime" cast "Int")
      .withColumn("ArrDelay", $"ArrDelay" cast "Int")
      .withColumn("DepDelay", $"DepDelay" cast "Int")
      .withColumn("DepDelay", $"DepDelay" cast "Int")

    handleNAValues(preProcessDataset);

    // TODO: More preProcessing
    return preProcessDataset
  }
}
