import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.tree.GradientBoostedTrees
import org.apache.spark.mllib.tree.configuration.BoostingStrategy
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vector

import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.rdd.RDD

import org.apache.log4j._

object testGBDT{

	def formatData(sc:SparkContext, name :String):RDD[LabeledPoint] = {
                val inputData = sc.textFile(name).map(_.trim).filter(line => !(line.isEmpty))
                val trainData = inputData.map(line => {
                        val items = line.split('\t')
                        val label = items.head.toDouble
                        val ftr = items.tail.map(item => {
                                val kv = item.split(':')
                                val k = kv(0).toInt
                                val v = kv(1).toDouble
                                (k, v)
                        }).toSeq
                        new LabeledPoint(label,Vectors.sparse(1686,ftr))
                }).cache
		trainData
	}	

	def main(args: Array[String]){
	        val conf = new SparkConf().setAppName("TestGBDT")
        	val sc = new SparkContext(conf)

		val trainData = formatData(sc,args(0))

		val focast_data = formatData(sc,args(1))
	
		val bs = BoostingStrategy.defaultParams("Classification")
		val we= GradientBoostedTrees.train(trainData,bs)

		focast_data.map{lp => 1/(1+math.exp(-we.predict(lp.features)))+"\t"+1+"\t"+lp.label}.saveAsTextFile(args(2))

		sc.stop()
	}
}
