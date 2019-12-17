using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Text;

namespace Machine.Learning
{
    static class NeuralNetworkFactory
    {

        public static NeuralNetwork GenrateNewNetwork(int[] _layout)
        {
            var result = new NeuralNetwork();
            result.Initialize(_layout);
            return result;
        }

        public static NeuralNetwork LoadNetwork(Stores.IStore _store)
        {
            var layers = _store.Load();
            var result = new NeuralNetwork();
            result.Initialize(layers);
            return result;
        }
    }
}
