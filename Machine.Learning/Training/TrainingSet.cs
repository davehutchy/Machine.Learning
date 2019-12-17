using System;
using System.Collections.Generic;
using System.Text;

namespace Machine.Learning.Training
{
    public class TrainingSetObj
    {
        public List<TrainingSet> TrainingSets { get; set; }
    }
    public class TrainingSet
    {
        public List<double> input { get; set; }
        public List<double> expected { get; set; }
        public TrainingSet()
        {

        }
        public TrainingSet(List<double> _input, List<double> _expected)
        {
            this.input = _input;
            this.expected = _expected;
        }
    }
}
