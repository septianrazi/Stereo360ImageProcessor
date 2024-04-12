#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/cudawarping.hpp>
#include <chrono> // for high_resolution_clock
#include <cfloat>
#include <opencv2/core/cuda/common.hpp>
#include <opencv2/core/cuda/border_interpolate.hpp>
#include <opencv2/core/cuda/vec_traits.hpp>
#include <opencv2/core/cuda/vec_math.hpp>

#include "helper_math.h"

using namespace std;

// default values for arguments
bool enableAnaglyph = false;
__device__ int anaglyphValue = 1;

bool enableGaussian = false;
__device__ int gausKernel = 4;
__device__ float gausSigma = 2.0;

bool enableDenoising = false;
__device__ int denoisingNbhoodSize = 8;
__device__ float denoisingFactorRatio = 60;

bool convertToCubeMap = false;
bool convertToEquirect = false;

__device__ float gaussian(float x, float sigma)
{
  return 1 / (sqrt(2 * M_PI) * sigma) * exp(-x * x / (2 * sigma * sigma));
}

__device__ uchar3 gaussianPixel(const cv::cuda::PtrStep<uchar3> src, int rows, int cols, int i, int j, int kernelSize, float sigma)
{
  float counter = 0;

  // set limits differently for left image or right image
  int imageStart = 0;
  int imageLimit = cols;

  uchar3 result = make_uchar3(0, 0, 0);

  for (int w = -kernelSize / 2; w <= kernelSize / 2; w++)
  {
    for (int l = -kernelSize / 2; l <= kernelSize / 2; l++)
    {
      if (i + w >= 0 && i + w < rows && j + l >= imageStart && j + l < imageLimit)
      {
        float thisGaussian = gaussian(w, sigma) * gaussian(l, sigma);

        result.x += src(i + w, j + l).x * thisGaussian;
        result.y += src(i + w, j + l).y * thisGaussian;
        result.z += src(i + w, j + l).z * thisGaussian;

        counter += thisGaussian;
      }
      // result += prev_result;
    }
  }

  // normalise results with gaussian weights
  result.x = result.x / counter;
  result.y = result.y / counter;
  result.z = result.z / counter;

  return result;
}
__device__ float determinant(float matrix[3][3])
{
  return matrix[0][0] * matrix[1][1] * matrix[2][2] + matrix[0][1] * matrix[1][2] * matrix[2][0] + matrix[0][2] * matrix[1][0] * matrix[2][1] - matrix[0][2] * matrix[1][1] * matrix[2][0] - matrix[0][1] * matrix[1][0] * matrix[2][2] - matrix[0][0] * matrix[1][2] * matrix[2][1];
}

__device__ uchar3 denoisingProcess(int i, int j, const cv::cuda::PtrStep<uchar3> src, int rows, int cols, int nbhoodSize, float factorRatio, int baseGaussianKernel)
{
  float counter = 0;

  // set limits differently for left image or right image
  int imageStart = 0;
  int imageLimit = cols;

  float covariance[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};

  // max neighbourhood number is 10 because of array size
  if (nbhoodSize > 10)
    nbhoodSize = 10;
  const int nbHoodArea = nbhoodSize * nbhoodSize;
  float redValues[100];
  float greenValues[100];
  float blueValues[100];
  float redMean = 0;
  float greenMean = 0;
  float blueMean = 0;

  int index = 0;

  for (int w = -nbhoodSize / 2; w <= nbhoodSize / 2; w++)
  {
    for (int l = -nbhoodSize / 2; l <= nbhoodSize / 2; l++)
    {
      if (i + w >= 0 && i + w < rows && j + l >= imageStart && j + l < imageLimit)
      {
        // store values of each cvolour for covariance calcualtion
        redValues[index] = src(i + w, j + l).z;
        greenValues[index] = src(i + w, j + l).y;
        blueValues[index] = src(i + w, j + l).x;
      }
      else
      {
        // mirror the main pixel if out of bounds
        redValues[index] = src(i, j).z;
        greenValues[index] = src(i, j).y;
        blueValues[index] = src(i, j).x;
      }

      // add values of each colour for mean calculation
      redMean += redValues[index];
      greenMean += greenValues[index];
      blueMean += blueValues[index];

      index++;
    }
  }

  // calculate mean
  redMean /= nbHoodArea;
  greenMean /= nbHoodArea;
  blueMean /= nbHoodArea;

  // cout << "Mean: " << redMean << " " << greenMean << " " << blueMean << endl;

  // populate covariance matrix
  for (int i = 0; i < nbHoodArea; i++)
  {
    covariance[0][0] += (redValues[i] - redMean) * (redValues[i] - redMean) / nbHoodArea;
    covariance[0][1] += (redValues[i] - redMean) * (greenValues[i] - greenMean) / nbHoodArea;
    covariance[0][2] += (redValues[i] - redMean) * (blueValues[i] - blueMean) / nbHoodArea;
    covariance[1][0] += (greenValues[i] - greenMean) * (redValues[i] - redMean) / nbHoodArea; // redundant ;
    covariance[1][1] += (greenValues[i] - greenMean) * (greenValues[i] - greenMean) / nbHoodArea;
    covariance[1][2] += (greenValues[i] - greenMean) * (blueValues[i] - blueMean) / nbHoodArea;
    covariance[2][0] += (blueValues[i] - blueMean) * (redValues[i] - redMean) / nbHoodArea;     // redundant ?
    covariance[2][1] += (blueValues[i] - blueMean) * (greenValues[i] - greenMean) / nbHoodArea; // redundant ;?
    covariance[2][2] += (blueValues[i] - blueMean) * (blueValues[i] - blueMean) / nbHoodArea;
  }
  s
      // calculate determinant
      float det = determinant(covariance);
  det = std::abs(det);

  // cout << "Covariance: " << covariance << endl;
  // cout << "Det " << det << endl;

  // calculate gaussian
  // float gaussianValue = baseGaussianKernel + exp(-det / factorRatio);
  // float gaussianValue = factorRatio / (baseGaussianKernel + det);
  // float gaussianValue = factorRatio / (baseGaussianKernel + 1000 * pow(det, 3));
  float gaussianValue = factorRatio / (baseGaussianKernel + max(0.0f, log(det)));

  // cout << "Gaussian: " << gaussianValue << endl;
  uchar3 result = gaussianPixel(src, rows, cols, i, j, gaussianValue, 2.0);

  return result;
}

__device__ float getTheta(float x, float y)
{
  float rtn = 0;
  if (y < 0)
  {
    rtn = atan2f(y, x) * -1;
  }
  else
  {
    rtn = M_PI + (M_PI - atan2f(y, x));
  }
  return rtn;
}

__device__ void EquirectToCubeMap(const cv::cuda::PtrStep<uchar3> src, cv::cuda::PtrStep<uchar3> dst, int rows, int cols)
{
  int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
  int dst_y = blockIdx.y * blockDim.y + threadIdx.y;

  int inputWidth = cols;
  int inputHeight = rows;
  float sqr = inputWidth / 4.0f;
  int outputWidth = sqr * 3;
  int outputHeight = sqr * 2;

  if (dst_x < outputWidth && dst_y < outputHeight)
  {
    float sqr = inputWidth / 4.0f;
    float tx, ty, normTheta, normPhi;
    float tempX, tempY, tempZ;

    if (dst_y < sqr + 1)
    {
      if (dst_x < sqr + 1)
      {
        tx = dst_x;
        ty = dst_y;
        tempX = tx - 0.5f * sqr;
        tempY = 0.5f * sqr;
        tempZ = ty - 0.5f * sqr;
      }
      else if (dst_x < 2 * sqr + 1)
      {
        // top middle [X+]
        tx = dst_x - sqr;
        ty = dst_y;
        tempX = 0.5f * sqr;
        tempY = (tx - 0.5f * sqr) * -1;
        tempZ = ty - 0.5f * sqr;
      }
      else
      {
        // top right [Y-]
        tx = dst_x - 2 * sqr;
        ty = dst_y;
        tempX = (tx - 0.5f * sqr) * -1;
        tempY = -0.5f * sqr;
        tempZ = ty - 0.5f * sqr;
      }
    }
    else
    {
      if (dst_x < sqr + 1)
      {
        // bottom left box [X-]
        tx = dst_x;
        ty = dst_y - sqr;
        tempX = -0.5f * sqr;
        tempY = tx - 0.5f * sqr;
        tempZ = ty - 0.5f * sqr;
      }
      else if (dst_x < 2 * sqr + 1)
      {
        // bottom middle [Z-]
        tx = dst_x - sqr;
        ty = dst_y - sqr;
        tempX = (ty - 0.5f * sqr) * -1;
        tempY = (tx - 0.5f * sqr) * -1;
        tempZ = 0.5f * sqr;
      }
      else
      {
        // bottom right [Z+]
        tx = dst_x - 2 * sqr;
        ty = dst_y - sqr;
        tempX = ty - 0.5f * sqr;
        tempY = (tx - 0.5f * sqr) * -1;
        tempZ = -0.5f * sqr;
      }
    }

    // Normalize theta and phi
    float rho = sqrtf(tempX * tempX + tempY * tempY + tempZ * tempZ);
    normTheta = getTheta(tempX, tempY) / (2 * M_PI);
    normPhi = (M_PI - acosf(tempZ / rho)) / M_PI;

    // Calculate input coordinates
    float iX = normTheta * inputWidth;
    float iY = normPhi * inputHeight;

    // Handle possible overflows
    if (iX >= inputWidth)
    {
      iX -= inputWidth;
    }
    if (iY >= inputHeight)
    {
      iY -= inputHeight;
    }

    // Copy pixel value from input to output
    dst(dst_y, dst_x) = src((int)iY, (int)iX);
  }
}

__device__ void unit3DToUnit2D(float x, float y, float z, int faceIndex, float &x2D, float &y2D)
{
  if (faceIndex == 0)
  { // X+
    x2D = y + 0.5;
    y2D = z + 0.5;
  }
  else if (faceIndex == 1)
  { // Y+
    x2D = (-x) + 0.5;
    y2D = z + 0.5;
  }
  else if (faceIndex == 2)
  { // X-
    x2D = (-y) + 0.5;
    y2D = z + 0.5;
  }
  else if (faceIndex == 3)
  { // Y-
    x2D = x + 0.5;
    y2D = z + 0.5;
  }
  else if (faceIndex == 4)
  { // Z+
    x2D = y + 0.5;
    y2D = (-x) + 0.5;
  }
  else
  { // Z-
    x2D = y + 0.5;
    y2D = x + 0.5;
  }
  y2D = 1 - y2D;
}

__device__ void project(float theta, float phi, float sign, int axis, float &x, float &y, float &z)
{
  float rho;
  if (axis == 0)
  { // X
    x = sign * 0.5;
    rho = x / (cos(theta) * sin(phi));
    y = rho * sin(theta) * sin(phi);
    z = rho * cos(phi);
  }
  else if (axis == 1)
  { // Y
    y = sign * 0.5;
    rho = y / (sin(theta) * sin(phi));
    x = rho * cos(theta) * sin(phi);
    z = rho * cos(phi);
  }
  else
  { // Z
    z = sign * 0.5;
    rho = z / cos(phi);
    x = rho * cos(theta) * sin(phi);
    y = rho * sin(theta) * sin(phi);
  }
}

__device__ void CubeMapToEquirect(cv::cuda::PtrStep<uchar3> src, cv::cuda::PtrStep<uchar3> output, int rows, int cols, int subrows, int subcols)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < cols && y < rows)
  {
    float U = float(x) / (cols - 1);
    float V = float(y) / (rows - 1);
    float theta = U * 2 * M_PI;
    float phi = V * M_PI;
    float x3D = cos(theta) * sin(phi);
    float y3D = sin(theta) * sin(phi);
    float z3D = cos(phi);
    float maximum = max(abs(x3D), max(abs(y3D), abs(z3D)));
    x3D /= maximum;
    y3D /= maximum;
    z3D /= maximum;
    int faceIndex;
    if (x3D == 1 || x3D == -1)
    {
      faceIndex = (x3D == 1) ? 0 : 2;
      project(theta, phi, x3D, 0, x3D, y3D, z3D);
    }
    else if (y3D == 1 || y3D == -1)
    {
      faceIndex = (y3D == 1) ? 1 : 3;
      project(theta, phi, y3D, 1, x3D, y3D, z3D);
    }
    else
    {
      faceIndex = (z3D == 1) ? 4 : 5;
      project(theta, phi, z3D, 2, x3D, y3D, z3D);
    }
    float x2D, y2D;
    unit3DToUnit2D(x3D, y3D, z3D, faceIndex, x2D, y2D);
    x2D *= subcols;
    y2D *= subrows;
    int xPixel = int(x2D);
    int yPixel = int(y2D);
    uchar3 color;
    switch (faceIndex)
    {
    case 0:

      // cv::cuda::GpuMat posx = src_mat(cv::Rect(0, 0, cols / 4, rows / 3));
      // cv::cuda::GpuMat negx = src_mat(cv::Rect(cols / 4, 0, cols / 4, rows / 3));
      // cv::cuda::GpuMat posy = src_mat(cv::Rect(cols / 2, 0, cols / 4, rows / 3));
      // cv::cuda::GpuMat negy = src_mat(cv::Rect(0, rows / 3, cols / 4, rows / 3));
      // cv::cuda::GpuMat posz = src_mat(cv::Rect(cols / 4, rows / 3, cols / 4, rows / 3));
      // cv::cuda::GpuMat negz = src_mat(cv::Rect(cols / 2, rows / 3, cols / 4, rows / 3));
      color = src(yPixel, xPixel);
      break;
    case 1:
      color = src(subcols * 2 + yPixel, xPixel);
      break;
    case 2:
      color = src(subcols + yPixel, xPixel);
      break;
    case 3:
      color = src(yPixel, subrows + xPixel);
      break;
    case 4:
      color = src(subcols + yPixel, subrows + xPixel);
      break;
    case 5:
      color = src(subcols * 2 + yPixel, subrows + xPixel);
      break;
    }

    output(y, x) = color;
  }
}

__global__ void process(const cv::cuda::PtrStep<uchar3> src, cv::cuda::PtrStep<uchar3> dst, int rows, int cols, bool enableGaussian, bool enableDenoising, bool convertToCubeMap, bool convertToEquirect)
{
  const int dst_x = blockDim.x * blockIdx.x + threadIdx.x;
  const int dst_y = blockDim.y * blockIdx.y + threadIdx.y;

  if (convertToCubeMap)
  {
    EquirectToCubeMap(src, dst, rows, cols);
  }

  if (convertToEquirect)
  {
    // cv::cuda::GpuMat src_mat(rows, cols, CV_8UC3, src.data, src.step);

    float subrows = rows / 3;
    float subcols = cols / 4;

    // cv::cuda::GpuMat posx = src_mat(cv::Rect(0, 0, cols / 4, rows / 3));
    // cv::cuda::GpuMat negx = src_mat(cv::Rect(cols / 4, 0, cols / 4, rows / 3));
    // cv::cuda::GpuMat posy = src_mat(cv::Rect(cols / 2, 0, cols / 4, rows / 3));
    // cv::cuda::GpuMat negy = src_mat(cv::Rect(0, rows / 3, cols / 4, rows / 3));
    // cv::cuda::GpuMat posz = src_mat(cv::Rect(cols / 4, rows / 3, cols / 4, rows / 3));
    // cv::cuda::GpuMat negz = src_mat(cv::Rect(cols / 2, rows / 3, cols / 4, rows / 3));

    CubeMapToEquirect(src, dst, rows, cols, subrows, subcols);
  }

  if (dst_x < cols && dst_y < rows)
  {
    // uchar3 left = src(dst_y, dst_x);
    // uchar3 right = src(dst_y, cols / 2 + dst_x);

    if (enableDenoising)
    {
      int nbhoodSize = denoisingNbhoodSize;
      float factorRatio = denoisingFactorRatio;
      int baseGaussianKernel = 1;

      dst(dst_y, dst_x) = denoisingProcess(dst_y, dst_x, src, rows, cols, true, nbhoodSize, factorRatio, baseGaussianKernel);

      left = dst(dst_y, dst_x);
    }
    else if (enableGaussian)
    {
      int kernelSize = gausKernel;
      float sigma = gausSigma;

      dst(dst_y, dst_x) = gaussianPixel(src, rows, cols, dst_y, dst_x, true, kernelSize, sigma);

      left = dst(dst_y, dst_x);
    }
  }
}

int divUp(int a, int b)
{
  return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

void processCUDA(cv::cuda::GpuMat &src, cv::cuda::GpuMat &dst)
{
  const dim3 block(16, 16);
  const dim3 grid(divUp(src.cols, block.x), divUp(src.rows, block.y));

  process<<<grid, block>>>(src, dst, src.rows, src.cols, enableGaussian, enableDenoising, convertToCubeMap, convertToEquirect);
}

// default values for arguments
int hostAnaglyphValue = 1;

int hostGausKernel = 4;
float hostGausSigma = 2.0;

int hostDenoisingNbhoodSize = 8;
float hostDenoisingFactorRatio = 60;

bool saveToFile = false;
bool resizeImage = false;

std::string saveFileName = "cubemap.jpg";

int main(int argc, char **argv)
{
  if (argc > 2)
  {
    for (int i = 2; i < argc; i++)
    {
      string arg = argv[i];
      if (arg == "-gaussian" || arg == "-g")
      {
        enableGaussian = true;
        if (i + 2 < argc && std::isdigit(*argv[i + 1]) && std::isdigit(*argv[i + 2])) // Ensure we have two numerical values after the argument
        {
          hostGausKernel = std::stod(argv[i + 1]); // Convert the next argument to a double
          hostGausSigma = std::stod(argv[i + 2]);  // Convert the argument after that to a double
          i += 2;                                  // Skip the next two arguments since we just processed them
        }
        cudaMemcpyToSymbol(gausKernel, &hostGausKernel, sizeof(int));
        cudaMemcpyToSymbol(gausSigma, &hostGausSigma, sizeof(float));
      }
      else if (arg == "-anaglyph" || arg == "-a")
      {
        enableAnaglyph = true;
        if (i + 1 < argc && std::isdigit(*argv[i + 1])) // Ensure we have a value after the argument
        {
          hostAnaglyphValue = std::stod(argv[i + 1]); // Convert the next argument to a double
          i++;                                        // Skip the next argument since we just processed it
        }
        cudaMemcpyToSymbol(anaglyphValue, &hostAnaglyphValue, sizeof(int));
      }
      else if (arg == "-denoising" || arg == "-d")
      {
        enableDenoising = true;
        if (i + 2 < argc && std::isdigit(*argv[i + 1]) && std::isdigit(*argv[i + 2])) // Ensure we have two numerical values after the argument
        {
          int hostDenoisingNbhoodSize = std::stod(argv[i + 1]);    // Convert the next argument to a double
          float hostDenoisingFactorRatio = std::stod(argv[i + 2]); // Convert the argument after that to a double
          i += 2;                                                  // Skip the next two arguments since we just processed them
        }
        cudaMemcpyToSymbol(denoisingNbhoodSize, &hostDenoisingNbhoodSize, sizeof(int));
        cudaMemcpyToSymbol(denoisingFactorRatio, &hostDenoisingFactorRatio, sizeof(float));
      }

      else if (arg == "-s" || arg == "--save")
      {
        saveToFile = true;
        if (i + 1 < argc) // Ensure we have a value after the argument
        {
          saveFileName = std::string(argv[i + 1]); // Convert the next argument to a string
          i++;                                     // Skip the next argument since we just processed it
        }
      }

      else if (arg == "-cm" || arg == "--toCubemap")
      {
        convertToCubeMap = true;
      }

      else if (arg == "-er" || arg == "--toEquirect")
      {
        convertToEquirect = true;
      }
      else if (arg == "-r" || arg == "--resize")
      {
        resizeImage = true;
      }
      else if (arg == "-h" || arg == "--help")
      {
        std::cout << "Usage: program imagePath [-g gaussianKernel gaussianSigma] [-a anaglyphValue] [-d denoisingNbhoodSize denoisingFactorRatio]\n";
        std::cout << "Options:\n";
        std::cout << "  -g, --gaussian     Enable Gaussian filter with specified kernel and sigma\n";
        std::cout << "                      kernel: size of the Gaussian kernel (int, default: 4)\n";
        std::cout << "                      sigma: standard deviation of the Gaussian distribution (double, default: 2.0)\n";
        std::cout << "  -a, --anaglyph     Enable Anaglyph with specified value\n";
        std::cout << "                      anaglyphValue: value for the Anaglyph effect (int, default: 1)\n";
        std::cout << "                        1: True Anaglyph\n";
        std::cout << "                        2: Grey Anaglyph\n";
        std::cout << "                        3: Color Anaglyph\n";
        std::cout << "                        4: Half-Color Anaglyph\n";
        std::cout << "                        5: Optimised Anaglyph\n";
        std::cout << "  -d, --denoising    Enable Denoising with specified neighborhood size and factor ratio\n";
        std::cout << "                      neighbourhood size: size of the neighborhood for denoising (int, default: 8)\n";
        std::cout << "                      factor ratio: factor ratio for denoising (double, default: 60)\n";
        std::cout << "  -h, --help         Display this help message and exit\n";
        return 0;
      }
    }
  }

  cv::namedWindow("Original Image", cv::WINDOW_OPENGL | cv::WINDOW_AUTOSIZE);
  cv::namedWindow("Processed Image", cv::WINDOW_OPENGL | cv::WINDOW_AUTOSIZE);

  cv::Mat h_img = cv::imread(argv[1]);
  cv::cuda::GpuMat d_img;
  d_img.upload(h_img);

  cv::imshow("Original Image", h_img);

  int outputWidth;
  int outputHeight;

  if (convertToCubeMap)
  {
    int inputWidth = h_img.cols;
    int inputHeight = h_img.rows;
    float sqr = inputWidth / 4.0f;
    outputWidth = sqr * 3;
    outputHeight = sqr * 2;
  }
  else if (convertToEquirect)
  {
    float sqr = h_img.rows / 2;
    outputHeight = h_img.rows;
    outputWidth = sqr * 4;
  }

  cv::Mat h_result(outputHeight, outputWidth, CV_8UC3);
  cv::cuda::GpuMat d_result(outputHeight, outputWidth, CV_8UC3);

  auto begin = chrono::high_resolution_clock::now();

  processCUDA(d_img, d_result);

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = end - begin;

  d_result.download(h_result);

  if (saveToFile)
    cv::imwrite(saveFileName, h_result);

  // resize just for the visualisation
  if (resizeImage)
  {
    double scale = 0.1;
    // cv::resize(h_img, h_img, cv::Size(), scale, scale);
    cv::resize(h_result, h_result, cv::Size(), scale, scale);
  }

  cv::imshow("Processed Image", h_result);
  cout << "Time: " << diff.count() << endl;

  cv::waitKey();

  return 0;
}
