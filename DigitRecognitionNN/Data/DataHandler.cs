using DigitRecognitionNN.Models;

namespace DigitRecognitionNN.Data;

public static class DataLoader
{
    public static List<DataPoint> LoadMnist(string filename, int fallbackCount = 1000)
    {
        string fullPath = Path.Combine("Data", filename);

        Console.WriteLine($"Loading MNIST data from: {fullPath}");
        if (!File.Exists(fullPath))
        {
            Console.WriteLine("File not found, generating data...");
            return GenerateTestData(fallbackCount);
        }

        var dataPoints = new List<DataPoint>();

        using var reader = new StreamReader(fullPath);
        reader.ReadLine(); // skip header

        while (!reader.EndOfStream)
        {
            string? line = reader.ReadLine();
            if (line == null) break;

            string[] parts = line.Split(',');
            if (parts.Length != 785) continue;

            if (!int.TryParse(parts[0], out int label))
                continue;

            float[] pixelsRaw = parts.Skip(1).Select(byte.Parse).Select(b => (float)b).ToArray();
            float[] pixels = NormalizePixels(pixelsRaw);

            var dp = new DataPoint(pixels, label);
            dataPoints.Add(dp);
        }

        return dataPoints;
    }

    private static float[] NormalizePixels(float[] pixels)
    {
        return pixels.Select(p => p / 255.0f).ToArray();
    }
    
    public static (List<DataPoint> train, List<DataPoint> test) SplitData(List<DataPoint> data, float trainRatio = 0.8f)
    {
        Shuffle(data);
        int trainCount = (int)(data.Count * trainRatio);
        var trainSet = data.Take(trainCount).ToList();
        var testSet = data.Skip(trainCount).ToList();
        return (trainSet, testSet);
    }
    
    public static void Shuffle(List<DataPoint> data)
    {
        var rnd = new Random();
        for (int i = data.Count - 1; i > 0; i--)
        {
            int j = rnd.Next(i + 1);
            (data[i], data[j]) = (data[j], data[i]);
        }
    }

    private static List<DataPoint> GenerateTestData(int count)
    {
        var rnd = new Random();
        var list = new List<DataPoint>(count);
        for (int i = 0; i < count; i++)
        {
            float[] pixels = new float[784];
            for (int j = 0; j < 784; j++)
            {
                pixels[j] = (float)rnd.NextDouble();
            }
            int label = rnd.Next(10);
            list.Add(new DataPoint(pixels, label));
        }
        return list;
    }
}