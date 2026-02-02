using DigitRecognitionNN.Data;
using DigitRecognitionNN.Utils;
using System.Text.Json;

namespace DigitRecognitionNN.Models;

public class NeuralNetwork
{
    private Matrix _weightsInputHidden;
    private Matrix _weightsHiddenHidden;
    private Matrix _weightsHiddenOutput;
    private Matrix _biasHidden;
    private Matrix _biasHidden2;
    private Matrix _biasOutput;
    
    private readonly float _learningRate;
    private readonly JsonSerializerOptions _jsonSerializerOptions = new() { WriteIndented = false };

    public NeuralNetwork(int inputSize, int hiddenSize, int outputSize, float learningRate)
    {
        this._learningRate = learningRate;
        _weightsInputHidden = new Matrix(hiddenSize, inputSize);         // 16 x 784
        _weightsHiddenHidden = new Matrix(hiddenSize, hiddenSize);       // 16 x 16
        _weightsHiddenOutput = new Matrix(outputSize, hiddenSize);       // 10 x 16

        _biasHidden = new Matrix(hiddenSize, 1);                          // 16 x 1
        _biasHidden2 = new Matrix(hiddenSize, 1);                         // 16 x 1
        _biasOutput = new Matrix(outputSize, 1);                          // 10 x 1

        // Ініціалізація ваг
        _weightsInputHidden.RandomizeWeights();
        _weightsHiddenHidden.RandomizeWeights();
        _weightsHiddenOutput.RandomizeWeights();

        _biasHidden.RandomizeWeights();
        _biasHidden2.RandomizeWeights();
        _biasOutput.RandomizeWeights();

    }

    public float[] Predict(float[] input)
    {
        var inputMatrix = Matrix.FromArray(input); // 784 x 1

        // Layer 1: input → hidden1
        var hidden1 = _weightsInputHidden * inputMatrix + _biasHidden; // (16 x 784) * (784 x 1) + (16 x 1)
        ActivationFunctions.ApplyReLu(hidden1); // in-place

        // Layer 2: hidden1 → hidden2
        var hidden2 = _weightsHiddenHidden * hidden1 + _biasHidden2; // (16 x 16) * (16 x 1) + (16 x 1)
        ActivationFunctions.ApplyReLu(hidden2);

        // Output layer
        var output = _weightsHiddenOutput * hidden2 + _biasOutput; // (10 x 16) * (16 x 1) + (10 x 1)

        // Apply Softmax to vector
        return ActivationFunctions.Softmax(output.ToArray());
    }
    
    private void Train(float[] input, float[] target)
    {
        // ==== 1. FORWARD ====
        var inputMatrix = Matrix.FromArray(input);
    
        var z1 = _weightsInputHidden * inputMatrix + _biasHidden;
        var a1 = z1.Copy(); ActivationFunctions.ApplyReLu(a1);

        var z2 = _weightsHiddenHidden * a1 + _biasHidden2;
        var a2 = z2.Copy(); ActivationFunctions.ApplyReLu(a2);

        var z3 = _weightsHiddenOutput * a2 + _biasOutput;
        var output = Matrix.FromArray(ActivationFunctions.Softmax(z3.ToArray()));

        // ==== 2. ERROR ====
        var targetMatrix = Matrix.FromArray(target);
        var errorOutput = output - targetMatrix;

        // ==== 3. BACKPROP ====

        // Output layer
        var gradWeightsOut = errorOutput * a2.Transpose();
        var gradBiasOut = errorOutput;

        // Hidden layer 2
        var errorHidden2 = (_weightsHiddenOutput.Transpose() * errorOutput);
        ActivationFunctions.ApplyReLuDerivative(z2, errorHidden2); // δ * ReLU'(z)

        var gradWeightsHidden2 = errorHidden2 * a1.Transpose();
        var gradBiasHidden2 = errorHidden2;

        // Hidden layer 1
        var errorHidden1 = (_weightsHiddenHidden.Transpose() * errorHidden2);
        ActivationFunctions.ApplyReLuDerivative(z1, errorHidden1);

        var gradWeightsHidden1 = errorHidden1 * inputMatrix.Transpose();
        var gradBiasHidden1 = errorHidden1;

        // ==== 4. UPDATE WEIGHTS ====
        _weightsHiddenOutput -= gradWeightsOut * _learningRate;
        _biasOutput -= gradBiasOut * _learningRate;

        _weightsHiddenHidden -= gradWeightsHidden2 * _learningRate;
        _biasHidden2 -= gradBiasHidden2 * _learningRate;

        _weightsInputHidden -= gradWeightsHidden1 * _learningRate;
        _biasHidden -= gradBiasHidden1 * _learningRate;
    }

    public void TrainBatch(List<DataPoint> data, int epochs)
    {
        for (var epoch = 0; epoch < epochs; epoch++)
        {
            DataLoader.Shuffle(data);
            float totalLoss = 0;

            foreach (var dp in data)
            {
                var prediction = Predict(dp.Input);
                totalLoss += MathUtils.CrossEntropy(prediction, dp.Target);
                Train(dp.Input, dp.Target);
            }

            var averageLoss = totalLoss / data.Count;
            Console.WriteLine($"Epoch {epoch + 1}/{epochs} completed. Avg Loss: {averageLoss:F4}");
        }
    }
    
    public float TestAccuracy(List<DataPoint> testData)
    {
        var correctCount = 0;

        foreach (var dp in testData)
        {
            var prediction = Predict(dp.Input);
            var predictedLabel = MathUtils.ArgMax(prediction);

            if (predictedLabel == dp.Label)
                correctCount++;
        }

        return (float)correctCount / testData.Count;
    }

    public void SaveModel(string filename)
    {
        var model = new ModelData
        {
            WeightsInputHidden = _weightsInputHidden.ToJaggedArray(),
            WeightsHiddenHidden = _weightsHiddenHidden.ToJaggedArray(),
            WeightsHiddenOutput = _weightsHiddenOutput.ToJaggedArray(),
        
            BiasHidden = _biasHidden.ToJaggedArray(),
            BiasHidden2 = _biasHidden2.ToJaggedArray(),
            BiasOutput = _biasOutput.ToJaggedArray()
        };
        
        var json = JsonSerializer.Serialize(model, _jsonSerializerOptions);
        File.WriteAllText(filename, json);
    }
    
    public void LoadModel(string filename)
    {
        var json = File.ReadAllText(filename);
        var model = JsonSerializer.Deserialize<ModelData>(json);

        if (model == null)
            throw new Exception("Deserialize error");

        if (model.WeightsInputHidden != null) _weightsInputHidden = Matrix.FromJaggedArray(model.WeightsInputHidden);
        if (model.WeightsHiddenHidden != null) _weightsHiddenHidden = Matrix.FromJaggedArray(model.WeightsHiddenHidden);
        if (model.WeightsHiddenOutput != null) _weightsHiddenOutput = Matrix.FromJaggedArray(model.WeightsHiddenOutput);

        if (model.BiasHidden != null) _biasHidden = Matrix.FromJaggedArray(model.BiasHidden);
        if (model.BiasHidden2 != null) _biasHidden2 = Matrix.FromJaggedArray(model.BiasHidden2);
        if (model.BiasOutput != null) _biasOutput = Matrix.FromJaggedArray(model.BiasOutput);
    }

}