#include<iostream>
#include<opencv2/opencv.hpp>
 
using namespace std;
using namespace cv;   //声明使用OpenCV 4.1.1的命名空间
 
int main(int agrc,char** agrv){
	Mat img=imread("./test.jpg");
	imshow("test",img);
	waitKey(0);
	return 0;
}