using Machine.Learning.Models;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Machine.Learning
{
    static class Helpers
    {
        public static double LogSigmoid(this double _x)
        {
            if (_x < -45.0) return 0.0;
            else if (_x > 45.0) return 1.0;
            else return 1.0 / (1.0 + Math.Exp(-_x));
        }

        public static double HyperbolicTangtent(this double _x)
        {
            if (_x < -45.0) return -1.0;
            else if (_x > 45.0) return 1.0;
            else return Math.Tanh(_x);
        }


        public static string PrintHighestWeightedValue(this Neuron[] _input)
        {
            double maxValue = _input.Max(x => x.ActivationValue);
            int maxIndex = _input.ToList().IndexOf(_input.Where(x => x.ActivationValue == maxValue).FirstOrDefault());
            return (maxIndex + 1).ToString();
        }

        public static string PrintHighestWeightedValue(this double[] _input)
        {
            double maxValue = _input.Max();
            int maxIndex = _input.ToList().IndexOf(maxValue);
            return (maxIndex + 1).ToString();
        }

        public static string PrintNeuronValue(this Neuron[] _input)
        {
            var sb = new StringBuilder("{ ");
            for (var i = 0; i < _input.Length; i++)
            {
                var b = _input[i];
                sb.Append($"{b.ActivationValue:0.##}");
                if (i < _input.Length - 1)
                {
                    sb.Append(", ");
                }
            }
            sb.Append(" }");
            return sb.ToString();
        }


        public static string PrintDouble(this double[] _input)
        {
            var sb = new StringBuilder("{ ");
            for (var i = 0; i < _input.Length; i++)
            {
                var b = _input[i];
                sb.Append($"{b:0.##}");
                if (i < _input.Length - 1)
                {
                    sb.Append(", ");
                }
            }
            sb.Append(" }");
            return sb.ToString();
        }
    }
}
