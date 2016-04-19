(ns ^{:doc "Deeplearning4j MNIST example using CMGS. Based on http://deeplearning4j.org/mnist-tutorial"}
    dl4clj.examples.mnist.mnist-cmgs-example
  (:require 
            [dl4clj.nn.api.layer :refer [set-listeners]]
            )

  (:import [org.deeplearning4j.datasets.iterator.impl MnistDataSetIterator]
           [org.deeplearning4j CMGSNet]
           ))


(def numRows 28)
(def numColumns 28)
(def outputNum 10)

(def numSamples 60000)
(def batchSize 500)
(def iterations 50)
(def seed 123)
(def listenerFreq 10)
(def splitTrainNum (int (* batchSize 0.8)))

(def testInput (atom []))
;;List<INDArray> testInput = new ArrayList<>();
(def testLabels (atom []))
;;List<INDArray> testLabels = new ArrayList<>();

;; Train on batches of 100 out of 60000 examples
(def iter (MnistDataSetIterator. 100 numSamples))

(println "Build model....");
(def model (-> (CMGSNet. numRows numColumns outputNum seed iterations)
               .init))

(set-listeners model [(ScoreIterationListener. listenerFreq)])

(println "Train model....")
(doseq [mnist iter]
  (let [
        ;; train set that is the result
        trainTest (.splitTestAndTrain mnist splitTrainNum (Random. seed))
        ;; get feature matrix and labels for training
        trainInput (.getTrain trainTest)]
    (.add testInput (-> trainTest
                        .getTest
                        .getFeatureMatrix))
    
    (.add testLabels
          (-> trainTest
              .getTest
              .getLabels))
    (.fit model trainInput)))

(println "Evaluate model....")

(let [eval (Evaluation. outputNum)]
  (for [i (range 0 (count testInput))]
    (.eval eval
           (nth testLabels i)
           (.output model (nth testInput i)))))


(println eval.stats())
(println "****************Example finished********************")
