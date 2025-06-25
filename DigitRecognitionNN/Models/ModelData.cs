namespace DigitRecognitionNN.Models;

public class ModelData
{
    public double[][]? WeightsInputHidden { get; init; }
    public double[][]? WeightsHiddenHidden { get; init; }
    public double[][]? WeightsHiddenOutput { get; init; }

    public double[][]? BiasHidden { get; init; }
    public double[][]? BiasHidden2 { get; init; }
    public double[][]? BiasOutput { get; init; }
}
