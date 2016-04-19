(ns ^{:doc "Deeplearning4j MNIST example using LeNet. Based on http:;; deeplearning4j.org/mnist-tutorial"}
    dl4clj.examples.mnist.mnist-lenet-example
  (:require 
            [dl4clj.nn.api.layer :refer [set-listeners]]
            )

  (:import [org.deeplearning4j.datasets.iterator.impl MnistDataSetIterator]

           [org.deeplearning4j.nn.conf
            NeuralNetConfiguration
            NeuralNetConfiguration$Builder]
           [org.deeplearning4j.eval Evaluation]
           [org.deeplearning4j.nn.api OptimizationAlgorithm]
           [org.deeplearning4j.nn.conf MultiLayerConfiguration NeuralNetConfiguration Updater]
           [org.deeplearning4j.nn.conf.layers
            ConvolutionLayer
            ConvolutionLayer$Builder
            DenseLayer
            DenseLayer$Builder
            OutputLayer
            OutputLayer$Builder
            SubsamplingLayer
            SubsamplingLayer$Builder
            SubsamplingLayer$PoolingType
            ]
           [org.deeplearning4j.nn.conf.layers.setup ConvolutionLayerSetup]
           [org.deeplearning4j.nn.multilayer MultiLayerNetwork]
           [org.deeplearning4j.nn.weights WeightInit]
           [org.deeplearning4j.optimize.listeners ScoreIterationListener]
           [org.nd4j.linalg.api.ndarray INDArray]
           [org.nd4j.linalg.dataset DataSet]
           [org.nd4j.linalg.dataset.api.iterator DataSetIterator]
           [org.nd4j.linalg.lossfunctions
            LossFunctions
            LossFunctions$LossFunction
            ]
           ))

(def nChannels 1)
(def outputNum 10)
(def batchSize 64)
(def nEpochs 10)
(def iterations 1)
(def seed 123)

(println "Load data....")
(def mnistTrain (MnistDataSetIterator. (int batchSize) (int seed) true))
(def mnistTest (MnistDataSetIterator. (int batchSize) (int seed) false))

(println "Build model....")

(defn make-builder []
  (let [builder (doto (new NeuralNetConfiguration$Builder)
                  (.seed seed)
                  (.iterations iterations)
                  (.regularization true)
                  (.l2 0.0005)
                  (.learningRate 0.01)
                  ;; .biasLearningRate(0.02)
                  ;; .learningRateDecayPolicy(LearningRatePolicy.Inverse).lrPolicyDecayRate(0.001).lrPolicyPower(0.75)
                  (.weightInit WeightInit/XAVIER)
                  (.optimizationAlgo OptimizationAlgorithm/STOCHASTIC_GRADIENT_DESCENT)
                  (.updater Updater/NESTEROVS)
                  (.momentum 0.9)
                  (.list 6)
                  (.layer (.build (doto (new ConvolutionLayer$Builder (int-array [5 5]))
                                    (.nIn nChannels)
                                    (.stride (int-array [1 1]))
                                    (.nOut 20)
                                    (.activation "identity")
                                    )))
                  (.layer (.build (doto (SubsamplingLayer$Builder. SubsamplingLayer$PoolingType/MAX)
                                    (.kernelSize (int-array [2 2]))
                                    (.stride (int-array [2 2])))))
                  ;;(.layer 2 (doto (ConvolutionLayer$Builder. 5  5)
                  ;;            (.nIn nChannels)
                  ;;            (.stride 1 1)
                  ;;            (.nOut 50)
                  ;;            (.activation "identity")
                  ;;            (.build)))
                  ;;(.layer 3 (doto (SubsamplingLayer$Builder. SubsamplingLayer$PoolingType/MAX)
                  ;;            (.kernelSize 2 2)
                  ;;            (.stride 2 2)
                  ;;            (.build)))
                  ;;(.layer 4 (doto (DenseLayer$Builder.)
                  ;;            (.activation "relu")
                  ;;            (.nOut 500)
                  ;;            (.build )))
                  ;;(.layer 5 (doto (OutputLayer$Builder. LossFunctions$LossFunction/NEGATIVELOGLIKELIHOOD)
                  ;;            (.nOut outputNum)
                  ;;            (.activation "softmax")
                  ;;            (.build)))
                  )]
    (doto builder
        (.backprop true)
        (.pretrain false))))
  ;;setup (ConvolutionLayerSetup. builder 28 28 1)

(let [builder (make-builder)

      ]
  (println builder)
  )
      

(let [builder (make-builder)
      conf (.build builder)
      model (MultiLayerNetwork. conf)]
  (.init model)

  (println "Train model....")
  (.setListeners model (ScoreIterationListener. 1))

  (doseq [i (range nEpochs)]
    (.fit model mnistTrain)
    (println "*** Completed epoch {} ***", i)
    (println "Evaluate model....")
    (let [^Evaluation eval (Evaluation. outputNum)]
      (doseq [^DataSet ds mnistTest]
        (.eval eval (.getLabels ds)
                     (.output model (.getFeatureMatrix ds))))
      (println (.stats eval))
      (.reset mnistTest))))

  (println "****************Example finished********************")
