#include <stdio.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include <omp.h>

using namespace cv;

int main() {
    // Load an image
    Mat inputImage = imread("test.jpg", IMREAD_COLOR);
    if (inputImage.empty()) {
        printf("Error: Could not open or find the image.\n");
        return -1;
    }

    // Define the size of the Gaussian kernel
    int kernelSize = 5;
    double sigma = 1.0;

    // Create an output image
    Mat outputImage(inputImage.size(), inputImage.type());

    // Apply Gaussian blur in parallel
    #pragma omp parallel for
    for (int y = 0; y < inputImage.rows; y++) {
        for (int x = 0; x < inputImage.cols; x++) {
            Vec3b sum(0, 0, 0);
            int count = 0;

            for (int ky = -kernelSize / 2; ky <= kernelSize / 2; ky++) {
                for (int kx = -kernelSize / 2; kx <= kernelSize / 2; kx++) {
                    int newX = x + kx;
                    int newY = y + ky;

                    if (newX >= 0 && newX < inputImage.cols && newY >= 0 && newY < inputImage.rows) {
                        Vec3b pixel = inputImage.at<Vec3b>(newY, newX);
                        double weight = exp(-(kx * kx + ky * ky) / (2.0 * sigma * sigma));
                        sum += pixel * weight;
                        count++;
                    }
                }
            }

            outputImage.at<Vec3b>(y, x) = sum / count;
        }
    }

    // Save the output image
    imwrite("output.jpg", outputImage);

    printf("Gaussian blur completed.\n");

    return 0;
}
