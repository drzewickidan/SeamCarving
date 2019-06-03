using System;
using OpenCvSharp;

namespace SeamCarving.App
{
    public class Program
    {
        public static byte[] Resize(byte[] file, int width)
        {
            byte[] output;
            Mat imageGradient = new Mat();
            Mat imageEnergy = new Mat();

            Mat image = Cv2.ImDecode(file, ImreadModes.AnyColor);

            for (int i = 0; i < width; i++)
            {
                if (i % 10 == 0)
                {
                    imageGradient = CalculateImageGradient(image);
                    imageEnergy = CalculateImageEnergy(imageGradient, 1);
                }

                image = FindSeam(imageEnergy, image, 1, i + 1);
            }

            Cv2.ImEncode(".jpg", image, out output);
            return output;
        }

        private static Mat CalculateImageGradient(Mat input)
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

        private static Mat CalculateImageEnergy(Mat input, int direction)
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

            else
            {
                double up, center, down;

                input.Col[0].CopyTo(energy.Col[0]);

                for (int j = 1; j < numCols; j++)
                {
                    for (int i = 0; i < numRows; i++)
                    {
                        if (i == 0)
                        {
                            up = energy.At<double>(0, j - 1);
                            down = energy.At<double>(i + 1, j - 1);
                        }

                        else if (i == numRows - 1)
                        {
                            up = energy.At<double>(i - 1, j - 1);
                            down = energy.At<double>(numRows - 1, j - 1);
                        }

                        else
                        {
                            up = energy.At<double>(i - 1, j - 1);
                            down = energy.At<double>(i + 1, j - 1);
                        }

                        center = energy.At<double>(i, j - 1);
                        energy.Set(i, i, input.At<double>(i, j) + Math.Min(up, Math.Min(center, down)));
                    }
                }
            }

            return energy;
        }

        private static Mat FindSeam(Mat energyMap, Mat originalImage, int direction, int iteration)
        {
            int numRows = originalImage.Rows;
            int numCols = originalImage.Cols;

            if (direction == 1)
            {
                double left, center, right;
                Mat bottomRow = energyMap.Row[numRows - 1];

                Cv2.MinMaxLoc(bottomRow, out double min, out double max, out Point minLoc, out Point maxLoc);

                int currentPoint = minLoc.X - iteration;
                if (currentPoint < 0)
                {
                    currentPoint = 0;
                }

                energyMap.Set<double>(numRows - 1, currentPoint, 999);

                for (int i = numRows - 2; i >= 0; i--)
                {
                    RemoveSeam(ref originalImage, i, currentPoint, direction);

                    if (i == 0) break;

                    if (currentPoint == 0)
                    {
                        left = 9999;
                        min = currentPoint;
                    }

                    else
                    {
                        left = energyMap.At<double>(i - 1, currentPoint - 1);
                        min = currentPoint - 1;
                    }

                    center = energyMap.At<double>(i - 1, currentPoint);

                    if (center < left)
                        min = currentPoint;

                    if (currentPoint >= numCols - 1)
                    {
                        right = 99999;
                    }

                    else
                    {
                        right = energyMap.At<double>(i - 1, currentPoint + 1);
                    }

                    if (right < center && right < left)
                    {
                        min = currentPoint + 1;
                    }

                    currentPoint = (int)min;
                    energyMap.Set<double>(i, currentPoint, 999);
                }

                originalImage = originalImage.ColRange(0, numCols - 1);
            }

            else
            {
                double down, center, up;
                Mat rightCol = energyMap.Row[numCols - 1];;

                Cv2.MinMaxLoc(rightCol, out double min, out double max, out Point minLoc, out Point maxLoc);

                int currentPoint = minLoc.Y - iteration;
                if (currentPoint < 0)
                {
                    currentPoint = 0;
                }

                energyMap.Set<double>(numCols - 1, currentPoint, 999);

                for (int j = numCols - 2; j >= 0; j--)
                {
                    RemoveSeam(ref originalImage, j, currentPoint, direction);

                    if (j == 0) break;

                    if (currentPoint == 0)
                    {
                        up = 9999;
                        min = currentPoint;
                    }

                    else
                    {
                        up = energyMap.At<double>(currentPoint - 1, j - 1);
                        min = currentPoint - 1;
                    }

                    center = energyMap.At<double>(currentPoint, j - 1);

                    if (center < up)
                        min = currentPoint;

                    if (currentPoint >= numCols - 1)
                    {
                        down = 99999;
                    }

                    else
                    {
                        down = energyMap.At<double>(currentPoint + 1, j - 1);
                    }

                    if (down < center && down < up)
                    {
                        min = currentPoint + 1;
                    }

                    currentPoint = (int)min;
                    energyMap.Set<double>(currentPoint, j, 999);
                }

                originalImage = originalImage.ColRange(0, numRows - 1);
            }

            return originalImage;
        }

        private static void RemoveSeam(ref Mat input, int i, int j, int direction)
        {
            int numRows = input.Rows;
            int numCols = input.Cols;
            Mat firstHalf;
            Mat secondHalf;

            Mat dummy = Mat.Zeros(new Size(1, 1), MatType.CV_8UC3);

            if (direction == 1)
            {
                Mat newRow = new Mat();

                if (j == 0)
                {
                    secondHalf = input.RowRange(i, i + 1).ColRange(j + 1, numCols);
                    Cv2.HConcat(secondHalf, dummy, newRow);
                }

                else if (j == numCols - 1)
                {
                    firstHalf = input.RowRange(i, i + 1).ColRange(0, j);
                    Cv2.HConcat(firstHalf, dummy, newRow);
                }

                else
                {
                    firstHalf = input.RowRange(i, i + 1).ColRange(0, j);
                    secondHalf = input.RowRange(i, i + 1).ColRange(j + 1, numCols);
                    Cv2.HConcat(secondHalf, dummy, secondHalf);
                    Cv2.HConcat(firstHalf, secondHalf, newRow);
                }

                newRow.CopyTo(input.Row[i]);
            }

            else 
            {
                Mat newCol = new Mat();

                if (i == 0)
                {
                    secondHalf = input.RowRange(i + 1, numRows).ColRange(j, j + 1);
                    Cv2.Transpose(secondHalf, secondHalf);
                    Cv2.HConcat(secondHalf, dummy, newCol);
                }

                else if (i == numRows - 1)
                {
                    firstHalf = input.RowRange(0, i).ColRange(j, j + 1);
                    Cv2.Transpose(firstHalf, firstHalf);
                    Cv2.HConcat(firstHalf, dummy, newCol);
                }
                else
                {
                    firstHalf = input.RowRange(0, i).ColRange(j, j + 1);
                    Cv2.Transpose(firstHalf, firstHalf);
                    secondHalf = input.RowRange(i + 1, numRows).ColRange(j, j + 1);
                    Cv2.Transpose(secondHalf, secondHalf);
                    Cv2.HConcat(secondHalf, dummy, secondHalf);
                    Cv2.HConcat(firstHalf, secondHalf, newCol);
                }

                Cv2.Transpose(newCol, newCol);
                newCol.CopyTo(input.Col[j]);
            }
        }
    }
}