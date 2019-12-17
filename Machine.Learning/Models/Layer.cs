using System;
using System.Collections.Generic;
using System.Text;

namespace Machine.Learning.Models
{
    class Layer
    {
        public Neuron[] Neurons { get; set; }

        public Layer(int _numberOfNeurons)
        {
            this.Neurons = new Neuron[_numberOfNeurons];
        }

    }
}
