using Machine.Learning.Models;
using System;
using System.Collections.Generic;
using System.Linq;

namespace Machine.Learning
{
    class NeuralNetwork
    {
        //Inithial weight ranges
        private const double MAX_INIT_WEIGHT = 0.5;
        private const double MIN_INIT_WEIGHT = -0.5;

        //Learning rate TODO: possible allow this to be adjusted based on the overall outcome of weights (possibly getting stuck in a "valley")
        private const double LEARNING_RATE = 0.1;

        private readonly Random rand = new Random();
        private Layer[] layers;
        public List<Layer> Layers => layers.ToList();

        //Is a copy of the layers used during backpropogation and then applied to the "layers" once completed...
        private List<Layer> adjustedLayers;

        public void Initialize(int[] _layout)
        {
            this.layers = new Layer[_layout.Length];
            var layerCount = _layout.Length;
            for (var i = 0; i < layerCount; i++)
            {
                if (i > 0)// this is a hidden or output layer
                    this.CreateLayerWithInitialWeights(i, _layout[i]);
                else// this is the Input layer
                    this.CreateInputLayer(_layout[i]);
            }
        }
        public void Initialize(List<Layer> _layers)
        {
            this.layers = _layers.ToArray();
        }
        public void CreateInputLayer(int _numberOfNeurons)
        {
            var inputLayer = new Layer(_numberOfNeurons);
            for (var i = 0; i < _numberOfNeurons; i++)
                inputLayer.Neurons[i] = new Neuron();

            this.layers[0] = inputLayer;
        }

        public void CreateLayerWithInitialWeights(int _index, int _numberOfNeurons)
        {
            var hiddenLayer = new Layer(_numberOfNeurons);
            for (var i = 0; i < _numberOfNeurons; i++)
            {
                int numNeuronsInPreviousLayer = this.layers[_index - 1].Neurons.Length;
                double[] previousWeights = new double[numNeuronsInPreviousLayer];
                for (var j = 0; j < numNeuronsInPreviousLayer; j++)
                {
                    var weight = rand.NextDouble() * ((MAX_INIT_WEIGHT - MIN_INIT_WEIGHT) + MIN_INIT_WEIGHT);
                    previousWeights[j] = weight;
                }
                hiddenLayer.Neurons[i] = new Neuron(previousWeights);
            }
            this.layers[_index] = hiddenLayer;
        }

        public Neuron[] FeedForward(double[] _input)
        {
            int layerIndex = 0;
            for (var i = 0; i < _input.Length; i++)
                this.layers[layerIndex].Neurons[i].ActivationValue = _input[i];

            layerIndex++;

            for (var i = 1; i < this.layers.Length; i++)
            {
                for (var j = 0; j < this.layers[i].Neurons.Length; j++)
                {
                    double value = 0;
                    var previousLayer = this.layers[i - 1];
                    var currentNeuron = this.layers[i].Neurons[j];
                    for (var k = 0; k < currentNeuron.ConnectionWeights.Length; k++)
                    {
                        value += currentNeuron.ConnectionWeights[k] * previousLayer.Neurons[k].ActivationValue;
                    }
                    this.layers[i].Neurons[j].ActivationValue = (value + currentNeuron.Bias).LogSigmoid();
                }
            }
            return this.layers[this.layers.Length - 1].Neurons;
        }

        /// <summary>
        /// Pass in inputs and what you expect to be outputed to train the network.
        /// </summary>
        /// <param name="_inputs">provide test inputs</param>
        /// <param name="_expected">pass in expected results for the given inputs</param>
        /// <returns></returns>
        public Neuron[] Train(double[] _inputs, double[] _expected)
        {
            var output = FeedForward(_inputs);
            InitiateBackPropogation(_expected);
            return output;
        }

        /// <summary>
        ///training the network using the following formula: 
        ///new_weight = (current_weight - (learning_rate * (forward_weights * (2 * (output - expected) * previous_neuron.value)))
        /// </summary>
        /// <param name="_expected">Expected Output for the given connected Neuron</param>
        public void InitiateBackPropogation(double[] _expected)
        {
            adjustedLayers = new List<Layer>(this.layers);
            var outputLayer = adjustedLayers.Last();

            for (var i = 0; i < outputLayer.Neurons.Length; i++)
            {
                var neuron = outputLayer.Neurons[i];

                //Calculating the cost for the specific output neuron...
                var outputCost = 2 * (neuron.ActivationValue - _expected[i]);
                PerformBackPropogation(neuron, outputCost, this.layers.Length - 2);
            }
            this.layers = adjustedLayers.ToArray();
        }

        private void PerformBackPropogation(Neuron _neuron, double _cost, int _previousIndex, double? _weights = null)
        {
            for (var j = 0; j < adjustedLayers[_previousIndex].Neurons.Length; j++){
                double forwardWeights = 1;
                if (_weights.HasValue){
                    forwardWeights = _weights.Value;
                }

                var weightCostDerivative = (adjustedLayers[_previousIndex].Neurons[j].ActivationValue * forwardWeights) * _cost;
                forwardWeights *= _neuron.ConnectionWeights[j];
                _neuron.ConnectionWeights[j] = (_neuron.ConnectionWeights[j] - (LEARNING_RATE * weightCostDerivative));

                if (_previousIndex > 0){
                    //Perform the backpropogation for every connection between you and the input layer...
                    PerformBackPropogation(adjustedLayers[_previousIndex].Neurons[j], _cost, _previousIndex - 1, forwardWeights);
                }
            }
        }
    }
}
