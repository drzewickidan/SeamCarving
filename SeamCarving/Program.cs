using System;
using OpenCvSharp;

namespace SeamCarving
{
    class Program
    {
        static void Main(string[] args)
        {
            Mat imageGradient = new Mat();
            Mat imageEnergy = new Mat();
            Mat testImage = Cv2.ImRead("F:\\Users\\Daniel\\Documents\\Visual Studio 2019\\SeamCarving\\SeamCarving\\SeamCarving\\test.jpg", ImreadModes.Color);
            Console.WriteLine($"{testImage.Width}");

            var watch = System.Diagnostics.Stopwatch.StartNew();

            for (int i = 0; i < 100; i++)
            {
                if (i % 10 == 0)
                {
                    imageGradient = CalculateImageGradient(testImage);
                    imageEnergy = CalculateImageEnergy(imageGradient, 1);
                }

                testImage = FindSeam(imageEnergy, testImage, 1, i + 1);
            }

            watch.Stop();
            var elapsedMs = watch.ElapsedMilliseconds;
            Console.WriteLine($"Elapsed: {elapsedMs}");

            Console.WriteLine($"{testImage.Width}");
            Window.ShowImages(testImage);
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
            grad.ConvertTo(grad, MatType.CV_64F, 1.0 / 255.0);
            return grad;
        }

        public static Mat CalculateImageEnergy(Mat input, int direction)
        {
            int numRows = input.Rows;
            int numCols = input.Cols;

            Mat energy = Mat.Zeros(new Size(numCols, numRows), MatType.CV_64F);

            if (direction == 1)
            {
                double left, center, right;

                input.Row[0].CopyTo(energy.Row[0]);

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

            return energy;
        }

        public static Mat FindSeam(Mat energyMap, Mat originalImage, int direction, int iteration)
        {
            int numRows = originalImage.Rows;
            int numCols = originalImage.Cols;

            if (direction == 1)
            {
                double left, center, right;
                Mat bottomRow = energyMap.Row[numRows - 1];

                Cv2.MinMaxLoc(bottomRow, out double min, out double max, out Point minLoc, out Point maxLoc);
                int currentPoint = minLoc.X;

                for (int i = numRows - 2; i >= 0; i--)
                {
                    RemoveSeam(ref originalImage, i, currentPoint, direction);

                    if (i == 0) break;

                    left = energyMap.At<double>(i - 1, currentPoint - 1);
                    min = currentPoint - 1;
                    center = energyMap.At<double>(i - 1, currentPoint);

                    if (center < left)
                        min = currentPoint;

                    if (currentPoint == numCols - 1)
                        right = 99999;

                    else
                        right = energyMap.At<double>(i - 1, currentPoint + 1);

                    if (right < center && right < left)
                        min = currentPoint + 1;

                    currentPoint = (int)min;
                }

                originalImage = originalImage.ColRange(0, numCols - 1);
            }

            return originalImage;
        }

        private static void RemoveSeam(ref Mat input, int i, int j, int direction)
        {
            int numRows = input.Rows;
            int numCols = input.Cols;

            if (direction == 1)
            {
                Mat newRow = new Mat();
                Mat dummy = Mat.Zeros(new Size(1, 1), MatType.CV_8UC3);

                Mat firstHalf = input.RowRange(i, i + 1).ColRange(0, j);
                Mat secondHalf = input.RowRange(i, i + 1).ColRange(j + 1, numCols);

                if (!firstHalf.Empty() && !secondHalf.Empty())
                {
                    Cv2.HConcat(firstHalf, secondHalf, newRow);
                    Cv2.HConcat(newRow, dummy, newRow);
                }

                else
                {
                    if (firstHalf.Empty())
                    {
                        Cv2.HConcat(firstHalf, dummy, newRow);
                    }
                    else if (secondHalf.Empty())
                    {
                        Cv2.HConcat(firstHalf, dummy, newRow);
                    }
                }

                newRow.CopyTo(input.Row[i]);
            }
        }
    }
}
