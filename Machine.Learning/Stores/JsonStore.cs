using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using Machine.Learning.Models;
using Newtonsoft.Json;

namespace Machine.Learning.Stores
{
    class JsonStore : IStore
    {
        private static string filePath = Path.Combine(Environment.CurrentDirectory, "saved_network.json");
        public void Add(List<Layer> _layers)
        {
            try
            {
                string json = JsonConvert.SerializeObject(_layers);
                File.WriteAllText(filePath, json);
            }catch(Exception ex)
            {
                //Directory probably doesnt exist..
                throw;
            }
        }

        public List<Layer> Load()
        {
            if (!File.Exists(filePath))
                return null;
            try
            {
                string json = File.ReadAllText(filePath);
                return JsonConvert.DeserializeObject<List<Layer>>(json);
            }catch(Exception)
            {
                return null;
            }
            
        }
    }
}
