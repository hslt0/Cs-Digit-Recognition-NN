namespace DigitRecognitionNN.Models;

public class ModelData
{
    public float[][]? WeightsInputHidden { get; init; }
    public float[][]? WeightsHiddenHidden { get; init; }
    public float[][]? WeightsHiddenOutput { get; init; }

    public float[][]? BiasHidden { get; init; }
    public float[][]? BiasHidden2 { get; init; }
    public float[][]? BiasOutput { get; init; }
}
