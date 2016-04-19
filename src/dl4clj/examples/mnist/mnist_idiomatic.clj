(ns ^{:doc "Deeplearning4j MNIST example using LeNet. Based on http:;; deeplearning4j.org/mnist-tutorial"}
    dl4clj.examples.mnist.mnist-idiomatic
  (:require [dl4clj.examples.rnn.tools :refer [sample-characters-from-network]]
            [dl4clj.examples.example-utils :refer (shakespeare)]
            [dl4clj.examples.rnn.character-iterator :refer (get-shakespeare-iterator)]
            [nd4clj.linalg.dataset.api.iterator.data-set-iterator :refer (input-columns total-outcomes reset)]
            [dl4clj.nn.conf.layers.graves-lstm]
            [dl4clj.nn.conf.layers.rnn-output-layer]
            [dl4clj.nn.conf.distribution.uniform-distribution]
            [nd4clj.linalg.lossfunctions.loss-functions]
            [dl4clj.nn.conf.neural-net-configuration :refer (neural-net-configuration)]
            [dl4clj.nn.multilayer.multi-layer-network :refer (multi-layer-network init get-layers get-layer)]
            [dl4clj.nn.api.model :as model]
            [dl4clj.nn.api.classifier :as classifier]
            [dl4clj.nn.api.layer])
  (:import [java.util Random]))


(def nChannels 1)
(def outputNum 10)
(def batchSize 64)
(def nEpochs 10)
(def iterations 1)
(def seed 123)

(def conf (neural-net-configuration
           {:optimization-algo :stochastic-gradient-descent
            :iterations 1
            :learning-rate 0.1
            :rms-decay 0.95
            :seed 12345
            :regularization true
            :l2 0.001
            :list 6
            :layers {
                     ;;-0 {:graves-lstm
                     ;;    {:n-in (input-columns iter)
                     ;;     :n-out lstm-layer-size
                     ;;     :updater :rmsprop
                     ;;     :activation :tanh
                     ;;     :weight-init :distribution
                     ;;     :dist {:uniform {:lower -0.08, :upper 0.08}}}}
                     ;;-1 {:graves-lstm
                     ;;    {:n-in lstm-layer-size
                     ;;     :n-out lstm-layer-size
                     ;;     :updater :rmsprop
                     ;;     :activation :tanh
                     ;;     :weight-init :distribution
                     ;;     :dist {:uniform {:lower -0.08, :upper 0.08}}}}
                     ;;-2 {:rnnoutput
                     ;;    {:loss-function :mcxent
                     ;;     :activation :softmax
                     ;;     :updater :rmsprop
                     ;;     :n-in lstm-layer-size
                     ;;     :n-out (total-outcomes iter)
                     ;;     :weight-init :distribution
                     ;;     :dist {:uniform {:lower -0.08, :upper 0.08}}}}
                     
                     0 {:convolution {
                                      ;; (new ConvolutionLayer$Builder (int-array [5 5]))
                                      :n-in nChannels
                                      :stride [1 1]
                                      :n-out 20
                                      :activation :identity}}
                     1 { :subsampling {:pooling-type :max 
                                       :kernelSize [2 2]
                                       :stride [2 2]}}
                     2 {:convolution {
                                        ;(ConvolutionLayer$Builder. 5  5)
                                      :nIn nChannels
                                      :stride [1 1]
                                      :n-out 50
                                      :activation :identity}}
                     
                     3 { :subsampling {:pooling-type :max 
                                       :kernelSize [2 2]
                                       :stride [2 2]}}
                     4 {:dense-layer {:activation :relu
                                      :n-out 500}}
                     5 {:output-layer {:loss-function :NEGATIVELOGLIKELIHOOD
                                       :n-out outputNum
                                       .activation :softmax}}}
            :pretrain false
            :backprop true}))

(defn make-builder []
  (let [builder (doto (new NeuralNetConfiguration$Builder)
                  (.seed seed)
                  (.iterations iterations)
                  (.regularization true)
                  (.l2 0.0005)
                  (.learningRate 0.01)
                  (.weightInit WeightInit/XAVIER)
                  (.updater Updater/NESTEROVS)
                  (.momentum 0.9)
                  )]))
