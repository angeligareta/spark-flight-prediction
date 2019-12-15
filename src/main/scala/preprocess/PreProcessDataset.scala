package preprocess

import org.apache.spark.ml.PipelineStage
import org.apache.spark.ml.feature.{
  OneHotEncoderEstimator,
  StringIndexer,
  VectorAssembler
}
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
    "CancellationCode" // All NA in 1996
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

    return categoricalIndexers ++ Array(categoricalEncoder) ++ Array(assembler)
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
    // FIXME: Do not just drop na
    preProcessDataset = preProcessDataset.na.drop()

    // TODO: More preProcessing
    return preProcessDataset
  }
}
