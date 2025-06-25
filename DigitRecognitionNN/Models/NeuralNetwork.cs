using DigitRecognitionNN.Data;
using DigitRecognitionNN.Utils;
using System.Text.Json;

namespace DigitRecognitionNN.Models;

public class NeuralNetwork
{
    private Matrix weightsInputHidden;
    private Matrix weightsHiddenHidden;
    private Matrix weightsHiddenOutput;
    private Matrix biasHidden;
    private Matrix biasHidden2;
    private Matrix biasOutput;
    
    private readonly double learningRate;

    public NeuralNetwork(int inputSize, int hiddenSize, int outputSize, double learningRate)
    {
        this.learningRate = learningRate;
        weightsInputHidden = new Matrix(hiddenSize, inputSize);         // 16 x 784
        weightsHiddenHidden = new Matrix(hiddenSize, hiddenSize);       // 16 x 16
        weightsHiddenOutput = new Matrix(outputSize, hiddenSize);       // 10 x 16

        biasHidden = new Matrix(hiddenSize, 1);                          // 16 x 1
        biasHidden2 = new Matrix(hiddenSize, 1);                         // 16 x 1
        biasOutput = new Matrix(outputSize, 1);                          // 10 x 1

        // Ініціалізація ваг
        weightsInputHidden.RandomizeWeights();
        weightsHiddenHidden.RandomizeWeights();
        weightsHiddenOutput.RandomizeWeights();

        biasHidden.RandomizeWeights();
        biasHidden2.RandomizeWeights();
        biasOutput.RandomizeWeights();

    }

    public double[] Predict(double[] input)
    {
        var inputMatrix = Matrix.FromArray(input); // 784 x 1

        // Layer 1: input → hidden1
        var hidden1 = weightsInputHidden * inputMatrix + biasHidden; // (16 x 784) * (784 x 1) + (16 x 1)
        ActivationFunctions.ApplyReLu(hidden1); // in-place

        // Layer 2: hidden1 → hidden2
        var hidden2 = weightsHiddenHidden * hidden1 + biasHidden2; // (16 x 16) * (16 x 1) + (16 x 1)
        ActivationFunctions.ApplyReLu(hidden2);

        // Output layer
        var output = weightsHiddenOutput * hidden2 + biasOutput; // (10 x 16) * (16 x 1) + (10 x 1)

        // Apply Softmax to vector
        return ActivationFunctions.Softmax(output.ToArray());
    }
    
    private void Train(double[] input, double[] target)
    {
        // ==== 1. FORWARD ====
        var inputMatrix = Matrix.FromArray(input);
    
        var z1 = weightsInputHidden * inputMatrix + biasHidden;
        var a1 = z1.Copy(); ActivationFunctions.ApplyReLu(a1);

        var z2 = weightsHiddenHidden * a1 + biasHidden2;
        var a2 = z2.Copy(); ActivationFunctions.ApplyReLu(a2);

        var z3 = weightsHiddenOutput * a2 + biasOutput;
        var output = Matrix.FromArray(ActivationFunctions.Softmax(z3.ToArray()));

        // ==== 2. ERROR ====
        var targetMatrix = Matrix.FromArray(target);
        var errorOutput = output - targetMatrix;

        // ==== 3. BACKPROP ====

        // Output layer
        var gradWeightsOut = errorOutput * a2.Transpose();
        var gradBiasOut = errorOutput;

        // Hidden layer 2
        var errorHidden2 = (weightsHiddenOutput.Transpose() * errorOutput);
        ActivationFunctions.ApplyReLuDerivative(z2, errorHidden2); // δ * ReLU'(z)

        var gradWeightsHidden2 = errorHidden2 * a1.Transpose();
        var gradBiasHidden2 = errorHidden2;

        // Hidden layer 1
        var errorHidden1 = (weightsHiddenHidden.Transpose() * errorHidden2);
        ActivationFunctions.ApplyReLuDerivative(z1, errorHidden1);

        var gradWeightsHidden1 = errorHidden1 * inputMatrix.Transpose();
        var gradBiasHidden1 = errorHidden1;

        // ==== 4. UPDATE WEIGHTS ====
        weightsHiddenOutput -= gradWeightsOut * learningRate;
        biasOutput -= gradBiasOut * learningRate;

        weightsHiddenHidden -= gradWeightsHidden2 * learningRate;
        biasHidden2 -= gradBiasHidden2 * learningRate;

        weightsInputHidden -= gradWeightsHidden1 * learningRate;
        biasHidden -= gradBiasHidden1 * learningRate;
    }

    public void TrainBatch(List<DataPoint> data, int epochs)
    {
        for (int epoch = 0; epoch < epochs; epoch++)
        {
            DataLoader.Shuffle(data);
            double totalLoss = 0;

            foreach (var dp in data)
            {
                double[] prediction = Predict(dp.Input);
                totalLoss += MathUtils.CrossEntropy(prediction, dp.Target);
                Train(dp.Input, dp.Target);
            }

            double averageLoss = totalLoss / data.Count;
            Console.WriteLine($"Epoch {epoch + 1}/{epochs} completed. Avg Loss: {averageLoss:F4}");
        }
    }
    
    public double TestAccuracy(List<DataPoint> testData)
    {
        int correctCount = 0;

        foreach (var dp in testData)
        {
            double[] prediction = Predict(dp.Input);
            int predictedLabel = MathUtils.ArgMax(prediction);

            if (predictedLabel == dp.Label)
                correctCount++;
        }

        return (double)correctCount / testData.Count;
    }

    public void SaveModel(string filename)
    {
        var model = new ModelData
        {
            WeightsInputHidden = weightsInputHidden.ToJaggedArray(),
            WeightsHiddenHidden = weightsHiddenHidden.ToJaggedArray(),
            WeightsHiddenOutput = weightsHiddenOutput.ToJaggedArray(),
        
            BiasHidden = biasHidden.ToJaggedArray(),
            BiasHidden2 = biasHidden2.ToJaggedArray(),
            BiasOutput = biasOutput.ToJaggedArray()
        };

        var options = new JsonSerializerOptions { WriteIndented = true };
        string json = JsonSerializer.Serialize(model, options);
        File.WriteAllText(filename, json);
    }
    
    public void LoadModel(string filename)
    {
        string json = File.ReadAllText(filename);
        var model = JsonSerializer.Deserialize<ModelData>(json);

        if (model == null)
            throw new Exception("Не вдалося десеріалізувати модель");

        weightsInputHidden = Matrix.FromJaggedArray(model.WeightsInputHidden);
        weightsHiddenHidden = Matrix.FromJaggedArray(model.WeightsHiddenHidden);
        weightsHiddenOutput = Matrix.FromJaggedArray(model.WeightsHiddenOutput);

        biasHidden = Matrix.FromJaggedArray(model.BiasHidden);
        biasHidden2 = Matrix.FromJaggedArray(model.BiasHidden2);
        biasOutput = Matrix.FromJaggedArray(model.BiasOutput);
    }

}