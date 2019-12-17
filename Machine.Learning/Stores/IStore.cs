using Machine.Learning.Models;
using System;
using System.Collections.Generic;
using System.Text;

namespace Machine.Learning.Stores
{
    interface IStore
    {
        void Add(List<Layer> _layers);
        List<Layer> Load();
    }
}
