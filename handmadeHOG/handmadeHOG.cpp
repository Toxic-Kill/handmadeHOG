#include <iostream>
#include<opencv2/opencv.hpp>
#include<cmath>

using namespace std;
using namespace cv;

int main()
{


	//函数声明
	void calHOG(cv::Mat Mat1, float *hist, int dim, int size);
	float calDis(float *hist1, float *hist2, int size);



	//读取图像
	cv::Mat srcMat = cv::imread("D:\\Files\\hogtemplate.jpg",0);
	cv::Mat fstMat = cv::imread("D:\\Files\\img1.jpg",0);
	cv::Mat sedMat = cv::imread("D:\\Files\\img2.jpg",0);

	//判断读取成功性
	if (srcMat.empty() || fstMat.empty() || sedMat.empty())
	{
		std::cout << "Can't open the image" << endl;
		return -1;
	}

	//设置参数
	int cellSize = 16;
	int aglDim = 8;
	int nx = srcMat.cols / cellSize;
	int ny = srcMat.rows / cellSize;

	int bins = nx * ny * aglDim;
	
	float * src_hist = new float[bins];
	memset(src_hist, 0, sizeof(float)*bins);
	float * fst_hist = new float[bins];
	memset(fst_hist, 0, sizeof(float)*bins);
	float * sed_hist = new float[bins];
	memset(sed_hist, 0, sizeof(float)*bins);

	//计算图像HOG
	calHOG(srcMat, src_hist, aglDim, cellSize);
	calHOG(fstMat, fst_hist, aglDim, cellSize);
	calHOG(sedMat, sed_hist, aglDim, cellSize);

	//比较两张图片
	float dis1 = calDis(src_hist, fst_hist, bins);
	float dis2 = calDis(src_hist, sed_hist, bins);

	//输出比较结果
	if (dis1 <= dis2)
	{
		std::cout << "img1 is more similar" << endl;
	}
	else
	{
		std::cout << "img2 is more similar" << endl;
	}

	delete[] src_hist;
	delete[] fst_hist;
	delete[] sed_hist;

	return 0;
}

//定义计算图像HOG的函数
void calHOG(cv::Mat Mat1, float *hist, int dim, int size)
{
	int nx = Mat1.cols / size;
	int ny = Mat1.rows / size;
	
	int sinAngle = 360 / dim;

	//计算梯度与角度
	cv::Mat gx, gy;
	cv::Mat mag, angle;
	cv::Sobel(Mat1, gx, CV_32F, 1, 0, 1);
	cv::Sobel(Mat1, gy, CV_32F, 0, 1, 1);
	cv::cartToPolar(gx, gy, mag, angle, true);

	//遍历赋值
	int cellNum = 0;
	for (int i = 0; i < nx; i++)
	{
		for (int j = 0; j < ny; j++)
		{
			for (int m = 0; m < size; m++)
			{
				for (int n = 0; n < size; n++)
				{
					int num = (angle.at<float>(j*size + n, i*size + m) )/ sinAngle;
					hist[cellNum + num] += mag.at<float>(j*size + m, i*size + n);
				}
			}
			cellNum++;
		}
	}
}


//定义比较图像的函数
float calDis(float *hist1, float *hist2, int size)
{
	float sum = 0;
	for (int i = 0; i < size; i++)
	{
		sum += (hist1[i] - hist2[i])*(hist1[i] - hist2[i]);
	}
	sum = sqrt(sum);
	return sum;
}