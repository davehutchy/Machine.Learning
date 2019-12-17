using Machine.Learning.Models;
using Machine.Learning.Training;
using Newtonsoft.Json;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;

namespace Machine.Learning
{

    //class Program
    //{
    //    static void Main(string[] args)
    //    {
    //        Random rand = new Random();
    //        Console.WriteLine("Machine Learning test");
    //        TestNetwork testNetwork = new TestNetwork();
    //        testNetwork.Initialize(new int[] { 1, 16, 1 });

    //        double rssi = -76;
    //        var value = (rssi * -1) / 100;
    //        double[] input = new double[] { value };
    //        var output = testNetwork.FeedForward(input);
    //        Console.WriteLine("Output:");
    //        for (var i = 0; i < output.Length; i++)
    //        {
    //            Console.WriteLine(output[i].Value);
    //        }
    //        Console.ReadLine();
    //    }
    //}

    class Program
    {

        //This reperesnts the Layers and how many neurons per layer [Input Neurons - (Hidden Layers) - Output Neurons]
        static int[] NETWORK_LAYOUT = new int[] { 4, 36, 3 };

        static Stores.IStore store;

        static void Main(string[] args)
        {
            store = new Stores.JsonStore();
            Random rand = new Random();

            //Number of iterations for training...
            int numberOfIterations = 10000;

            string irisTrainingSet = File.ReadAllText(Path.Combine(Environment.CurrentDirectory, @"Training\DataSets\iris_dataset.json")).Replace(Environment.NewLine, "");


            var trainingSetObj = JsonConvert.DeserializeObject<TrainingSetObj>(irisTrainingSet);
            var trainingSets = trainingSetObj.TrainingSets;
            if (trainingSets == null || trainingSets.Count <= 0)
            {
                Console.WriteLine("Incorrect Training Data..");
                return;
            }

           
            NeuralNetwork network = null;
            //------ Loading Pre-existing (trained) network...
            if (File.Exists(Path.Combine(Environment.CurrentDirectory, "saved_network.json")))
                network = NeuralNetworkFactory.LoadNetwork(store);
            else
                //---- Generating the network using the predefined NETWORK_LAYOUT...
                network = NeuralNetworkFactory.GenrateNewNetwork(NETWORK_LAYOUT);

            StringBuilder builder = new StringBuilder("Creating Network");
            for (var i = 0; i < NETWORK_LAYOUT.Length; i++)
                builder.Append($"{NETWORK_LAYOUT[i]}-");
            Console.WriteLine(builder.ToString());


            //--------------------TESTING THE NETWORK BEFORE BEING TRAINIED--------------------------


            Console.WriteLine("----BEFORE TRAINING-----");
            Console.WriteLine("A New Test Set of {0} is Tested", trainingSets.Count);
            foreach (var item in trainingSets)
            {
                var output = network.FeedForward(item.input.ToArray());
                Console.WriteLine("Expected: {0} Output: {1}", item.expected.ToArray().PrintDouble(), output.PrintNeuronValue());
            }
            Console.WriteLine();


            //------------------------- PERFORMING THE TRAINING NOW---------------------


            Console.WriteLine("Training Networking using {0} training sets", trainingSets.Count);
            Stopwatch watch;
            watch = Stopwatch.StartNew();
            for (var i = 0; i < numberOfIterations; i++)
            {
                // var index = i % trainingSets.Count;
                var item = trainingSets[rand.Next(0, trainingSets.Count)];
                //Training the network by passing in the input/s and what the "Expected Output/s should be...

                var output = network.Train(item.input.ToArray(), item.expected.ToArray());
            }
            watch.Stop();
            Console.WriteLine("Completed Training in {0}", watch.Elapsed);
            Console.WriteLine();

            store.Add(network.Layers);


            //-------------------------TESTING THE NETWORK AFTER THE TRAINING--------------------------


            Console.WriteLine("----AFTER TRAINING-----");
            Console.WriteLine("A New Test Set of {0} is Tested", trainingSets.Count);
            foreach (var item in trainingSets)
            {
                var output = network.FeedForward(item.input.ToArray());
                Console.WriteLine("Expected: {0} Output: {1}", item.expected.ToArray().PrintHighestWeightedValue(), output.PrintHighestWeightedValue());
            }

            Console.ReadLine();
        }

    }
}
