using System;
using OpenCvSharp;

namespace SeamCarving
{
    class Program
    {
        static void Main(string[] args)
        {
            Mat testImage = Cv2.ImRead("F:\\Users\\Daniel\\Documents\\Visual Studio 2019\\SeamCarving\\SeamCarving\\SeamCarving\\test.jpg", ImreadModes.Color);
            var watch = System.Diagnostics.Stopwatch.StartNew();
            Mat imageGradient = CalculateImageGradient(testImage);
            watch.Stop();
            Mat imageEnergy = CalculateImageEnergy(imageGradient, 1);
            var elapsedMs = watch.ElapsedMilliseconds;
            Console.Write($"{elapsedMs}");

            Window.ShowImages(imageGradient);
            Window.ShowImages(imageEnergy);
        }

        public static Mat CalculateImageGradient(Mat input)
        {
            Mat greyScale = new Mat();
            Mat gradX = new Mat();
            Mat gradY = new Mat();
            Mat grad = new Mat();
            Mat absX = new Mat();
            Mat absY = new Mat();

            Cv2.CvtColor(input, greyScale, ColorConversionCodes.RGBA2GRAY, 0);

            Cv2.Scharr(greyScale, gradX, MatType.CV_16S, 1, 0, 1, 0, BorderTypes.Default);
            Cv2.ConvertScaleAbs(gradX, absX);

            Cv2.Scharr(greyScale, gradY, MatType.CV_16S, 0, 1, 1, 0, BorderTypes.Default);
            Cv2.ConvertScaleAbs(gradY, absY);

            Cv2.AddWeighted(absX, 0.5, absY, 0.5, 0, grad);
            grad.ConvertTo(grad, MatType.CV_64F, 1.0/255.0);
            return grad;
        }

        public static Mat CalculateImageEnergy(Mat input, int direction)
        {
            int numRows = input.Rows;
            int numCols = input.Cols;

            Mat colormap = new Mat();
            Mat energy = Mat.Zeros(new Size(numCols, numRows), MatType.CV_64F);

            if (direction ==  1)
            {
                double left, center, right;

                input.CopyTo(energy);

                for (int i = 1; i < numRows; i++)
                {
                    for (int j = 0; j < numCols; j++)
                    { 
                        if (j == 0)
                        {
                            left = energy.At<double>(i - 1, 0);
                            right = energy.At<double>(i - 1, j + 1);
                        }

                        else if (j == numCols - 1)
                        {
                            left = energy.At<double>(i - 1, j - 1);
                            right = energy.At<double>(i - 1, numCols - 1);
                        }

                        else
                        {
                            left = energy.At<double>(i - 1, j - 1);
                            right = energy.At<double>(i - 1, j + 1);
                        }

                        center = energy.At<double>(i - 1, j);
                        energy.Set(i, j, input.At<double>(i, j) + Math.Min(right, Math.Min(center, left)));
                    }
                }
            }
  
            Cv2.MinMaxLoc(energy, out double cmin, out double cmax);
            float scale =  (float)(255.0 / (cmax - cmin));
            energy.ConvertTo(colormap, MatType.CV_8UC1, scale);
            Cv2.ApplyColorMap(colormap, colormap, ColormapTypes.Jet);

            return energy;
        }
    }
}
