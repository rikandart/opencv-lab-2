#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
using namespace cv;

int main(int argc, char* argv[])
{
	Mat kadr1 = imread("E:/кадр1.png");
	Mat kadr2 = imread("E:/кадр2.png");
	//разность кадров
	Mat diff;
	absdiff(kadr1, kadr2, diff);
	imshow("Разностное изображение", diff);
	//маска кенни
	Mat mask;
	Mat b_i;
	Canny(kadr2, mask, 50, 200);
	imshow("Маска Кэнни", mask);
	Mat diff_gray;
	cvtColor(diff, diff_gray, COLOR_RGB2GRAY);
	bitwise_and(mask, diff_gray, b_i);
	imshow("Конъюнкция", b_i);
	//морфологическая фильтрация
	threshold(b_i, b_i, 5, 255, THRESH_BINARY_INV);
	Mat el = getStructuringElement(MORPH_RECT, Size(3, 3), Point(2, 2));
	morphologyEx(b_i, b_i, MORPH_ERODE, el);
	imshow("Морфологическая фильтрация", b_i);
	waitKey(0);
}