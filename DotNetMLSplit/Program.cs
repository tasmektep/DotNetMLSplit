using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DotNetMLSplit
{
    class Program
    {
        static void Main(string[] args)
        {
            var mlContext = new MLContext();

            var file = @"MainTestSamplesRev - 500.csv";

            IDataView data = mlContext.Data.LoadFromTextFile<ModelInput>(file, hasHeader: true, separatorChar: ';', allowQuoting: true, allowSparse: false);
            var dpreview = data.Preview();

            var testTrainSplit = mlContext.Data.TrainTestSplit(data, testFraction: 0.2);
            var testTrainSplitLabel = mlContext.Data.TrainTestSplit(data, testFraction: 0.2, "Label");

            // without label it can easily split data (expected 100 test and 400 train samples)
            var TestNolabel = testTrainSplit.TestSet.Preview();
            var TrainNolabel = testTrainSplit.TrainSet.Preview();

            // You can see in "Testlabel" has 0 rows
            var Testlabel = testTrainSplitLabel.TestSet.Preview();
            var Trainlabel = testTrainSplitLabel.TrainSet.Preview();

            Console.Read();
        }
    }
    public class ModelInput
    {
        [ColumnName("x0"), LoadColumn(0)]
        public float x0 { get; set; }


        [ColumnName("y0"), LoadColumn(1)]
        public float y0 { get; set; }


        [ColumnName("x1"), LoadColumn(2)]
        public float x1 { get; set; }


        [ColumnName("y1"), LoadColumn(3)]
        public float y1 { get; set; }


        [ColumnName("a"), LoadColumn(4)]
        public float a { get; set; }


        [ColumnName("b"), LoadColumn(5)]
        public float b { get; set; }


        [ColumnName("z0"), LoadColumn(6)]
        public float z0 { get; set; }


        [ColumnName("z1"), LoadColumn(7)]
        public float z1 { get; set; }


        [ColumnName("z2"), LoadColumn(8)]
        public float z2 { get; set; }


        [ColumnName("z3"), LoadColumn(9)]
        public float z3 { get; set; }


        [ColumnName("Label"), LoadColumn(10)]
        public bool Plausibility { get; set; }


    }

}
