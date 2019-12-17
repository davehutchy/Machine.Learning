using System;
using System.Collections.Generic;
using System.Text;

namespace Machine.Learning.Training
{
    class TrainingFactory
    {
        private static Random rand = new Random();

        public static List<TrainingSet> GetRandomTrainingSets(int _count, int _inputCount, int _expectedCount)
        {
            List<TrainingSet> result = new List<TrainingSet>(_count);

            for (var i = 0; i < _count; i++)
            {
                var input = new List<double>(_inputCount);
                for (var j = 0; j < _inputCount; j++)
                {
                    input.Add(rand.NextDouble());
                }

                var expected = new List<double>(_expectedCount);
                for (var j = 0; j < _expectedCount; j++)
                {
                    expected.Add(rand.NextDouble());
                }

                var set = new TrainingSet(input, expected);
                result.Add(set);
            }
            return result;
        }


        public static List<TrainingSet> GetTrainingSets(int _count, List<double> _inputs, List<double> _expected)
        {
            List<TrainingSet> result = new List<TrainingSet>(_count);
            for (var c = 0; c < _count; c++)
            {
                var set = new TrainingSet(_inputs, _expected);
                result.Add(set);
            }
            return result;

        }
    }
}
