using System;
using System.Collections.Generic;
using System.Text;

namespace Machine.Learning.Models
{
    class Neuron
    {
        //TODO: do something with the bias to make it useful...
        //If used it could enhance the performance of the network...
        public double Bias { get; set; } = 0;
        public double[] ConnectionWeights { get; set; }
        public double ActivationValue { get; set; }

        public Neuron() { }

        public Neuron(double[] _initialConnectionWeights)
        {
            this.ConnectionWeights = _initialConnectionWeights;
        }
    }
}
