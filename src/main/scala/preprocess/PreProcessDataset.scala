package preprocess

import org.apache.spark.ml.{Pipeline, PipelineStage}
import org.apache.spark.ml.feature.{
  OneHotEncoderEstimator,
  StringIndexer,
  VectorAssembler
}
import org.apache.spark.ml.param.shared.HasHandleInvalid
import org.apache.spark.ml.util.DefaultParamsWritable
import org.apache.spark.sql.{DataFrame, SparkSession}

/**
  * Object with methods to preprocess data
  */
object PreProcessDataset {

  def getFeaturesPipelineStages(
    dataset: DataFrame
  ): Array[PipelineStage with HasHandleInvalid with DefaultParamsWritable] = {
    val uniqueCarrierIndexer =
      new StringIndexer()
        .setInputCol("UniqueCarrier")
        .setOutputCol("UniqueCarrierIndexed")

    val tailNumIndexer =
      new StringIndexer()
        .setInputCol("TailNumber")
        .setOutputCol("TailNumIndexed")

    val originIndexer =
      new StringIndexer().setInputCol("Origin").setOutputCol("OriginIndexed");

    val destIndexer =
      new StringIndexer().setInputCol("Dest").setOutputCol("DestIndexed")

    val cancellationCodeIndexer = new StringIndexer()
      .setInputCol("CancellationCode")
      .setOutputCol("CancellationCodeIndexed")

    val encoder = new OneHotEncoderEstimator()
      .setInputCols(
        Array(
          uniqueCarrierIndexer.getOutputCol,
          tailNumIndexer.getOutputCol,
          originIndexer.getOutputCol,
          destIndexer.getOutputCol,
          cancellationCodeIndexer.getOutputCol
        )
      )
      .setOutputCols(
        Array(
          "UniqueCarrierVec",
          "TailNumVec",
          "OriginVec",
          "DestVec",
          "CancellationCodeVec"
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
          "UniqueCarrierVec",
          "FlightNum",
          "TailNumVec",
          "DepDelay",
          "Distance",
          "OriginVec",
          "DestVec",
          "CancellationCodeVec"
        )
      )
      .setOutputCol("features")

    return Array(
      uniqueCarrierIndexer,
      tailNumIndexer,
      originIndexer,
      destIndexer,
      cancellationCodeIndexer,
      encoder,
      features
    );
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
      "LateAircraftDelay"
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

    // TODO: More preProcessing
    return preProcessDataset
  }
}