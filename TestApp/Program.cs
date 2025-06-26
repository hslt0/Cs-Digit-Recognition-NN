using DigitRecognitionNN.Data;
using DigitRecognitionNN.Models;
using DigitRecognitionNN.Utils;

namespace TestApp;

internal static class Program
{
    private static void Main()
    {
        Console.WriteLine("Training nn for digit recognition");
    
        // 1. Loading mnist
        Console.WriteLine("Loading data... ");
        var allData = DataLoader.LoadMnist("mnist_train.csv");
        var (trainData, testData) = DataLoader.SplitData(allData);
    
        Console.WriteLine($"Training data: {trainData.Count}");
        Console.WriteLine($"Test data: {testData.Count}");
    
        // 2. Creating network
        var network = new NeuralNetwork(
            inputSize: 784,   // 28x28 pixels
            hiddenSize: 16,  
            outputSize: 10,   // 10 digits (0-9)
            learningRate: 0.01f
        );
        
        Console.WriteLine("Do you want load model from JSON? (true/false):");
        bool loadModel = bool.TryParse(Console.ReadLine(), out bool result) && result;

        if (loadModel) // 3. Load, dont peak if first run
        {
            network.LoadModel("trained_model.json");
            Console.WriteLine("Model is loaded");
        }
        else // 3. Train
        {
            int epochs = 10;
            Console.WriteLine("Start training");
            network.TrainBatch(trainData, epochs);
        }
        
        // 4. Accuracy after training
        float accuracy = network.TestAccuracy(testData);
        Console.WriteLine($"Accuracy on test data: {accuracy * 100:F2}%");
    
        // 5. Saving model
        network.SaveModel("trained_model.json");
        Console.WriteLine("Model is saved");
    
        // 6. Test
        Console.WriteLine("\nTest with random data:");
        int testCount = Math.Min(5, testData.Count);
        for (int i = 0; i < testCount; i++)
        {
            var testPoint = testData[i];
            float[] prediction = network.Predict(testPoint.Input);
            int predictedDigit = MathUtils.ArgMax(prediction);
        
            Console.WriteLine($"Real digit: {testPoint.Label}, " +
                              $"Predict digit: {predictedDigit}, " +
                              $"Confidence: {prediction[predictedDigit]:F2}");
        }
    }

}