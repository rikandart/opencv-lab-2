// #define MISHA
#ifndef MISHA
// opencv lib
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

// c++
#include <iostream>
#include <vector>
#include <thread>
#include <math.h>

using namespace cv;

void show(const Mat& frame_1, const Mat& frame_2, const char* name_1, const char* name_2) {
	namedWindow(name_1, WINDOW_NORMAL | WINDOW_FREERATIO | WINDOW_GUI_EXPANDED);
	resizeWindow(name_1, 1280, 720);
	moveWindow(name_1, 0, 0);
	imshow(name_1, frame_1);
	namedWindow(name_2, WINDOW_NORMAL | WINDOW_FREERATIO | WINDOW_GUI_EXPANDED);
	resizeWindow(name_2, 1280, 720);
	moveWindow(name_2, 1280, 0);
	imshow(name_2, frame_2);
	waitKey(0);
}

void editQuantizeLevel(const unsigned level, const Mat& img, Mat& dest_img) {
	if (remainder(256.0, level) != 0.0 || level == 0 || level == 1) {
		std::cout << "Incorrect quantize level" << std::endl;
		return;
	}
	// заполнение матрицы нулями
	dest_img = Mat::zeros(img.rows, img.cols, CV_8U);
	// шаг квантования
	const double step = 255.0 / (level - 1);
	// пробегаем по пикселям x и y
	for (unsigned i = 0; i < img.cols; i++) {
		for (unsigned j = 0; j < img.rows; j++) {
			unsigned br = img.at<unsigned char>(j, i);
			for (unsigned k = 0; k < level; k++) {
				// смотрим в каком интервале лежит число
				if (br >= k * step && br <= (k + 1) * step / 2) {
					// смотрим, какая разница меньше и присваиваем нужную градацию
					dest_img.at<unsigned char>(j, i) = k * step;
					break;
				}
				else if (br >= k * step / 2 && br <= (k + 1) * step) {
					dest_img.at<unsigned char>(j, i) = (k + 1) * step;
					break;
				}
			}
		}
	}
}

Mat diff(const Mat& frame_1, const Mat& frame_2) {
	Mat res;
	cvtColor(frame_1 - frame_2, res, COLOR_RGB2GRAY);
	bitwise_not(res, res);
	//show(res, res, "Grayed first", "Grayed second");
	return res;
}

/*Mat SobelMask(const Mat& frame) {
	Mat gr, sobel_x, sobel_y, g_res;
	g_res = Mat::zeros(720, 1280, CV_8U);
	cvtColor(frame, gr, COLOR_RGB2GRAY);
	Sobel(gr, sobel_x, CV_8U, 1, 0);
	Sobel(gr, sobel_y, CV_8U, 0, 1);
	for (int i = 0; i < sobel_x.rows; i++)
		for(int j = 0; j < sobel_x.cols; j++)
			g_res.at<unsigned char>(i, j) = sqrt(
				pow(sobel_x.at<unsigned char>(i, j), 2) +
				pow(sobel_y.at<unsigned char>(i, j), 2));
	//show(g_res, g_res, "First frame", "Second frame");
	bitwise_not(g_res, g_res);
	return g_res;
}*/

Mat SobelMask(const Mat& frame) {
	Mat g_res = Mat::zeros(720, 1280, CV_8U), gr;
	cvtColor(frame, gr, COLOR_RGB2GRAY);
	Mat sobel_x, sobel_y;
	Sobel(gr, sobel_x, CV_8U, 1, 0);
	Sobel(gr, sobel_y, CV_8U, 0, 1);
	std::cout << gr.rows << " " << gr.cols << std::endl;
	/*for (int i = 0; i < gr.rows; i++)
		for (int j = 0; j < gr.cols; j++) {

			unsigned char x = gr.at<unsigned char>(i + 1, j - 1)
				+ 2 * gr.at<unsigned char>(i + 1, j) + gr.at<unsigned char>(i + 1, j + 1)
				- (gr.at<unsigned char>(i - 1, j - 1) + 2 * gr.at<unsigned char>(i - 1, j)
					+ gr.at<unsigned char>(i - 1, j + 1));

			unsigned char y = gr.at<unsigned char>(i - 1, j + 1)
				+ 2 * gr.at<unsigned char>(i, j + 1) + gr.at<unsigned char>(i + 1, j + 1)
				- (gr.at<unsigned char>(i - 1, j - 1) + 2 * gr.at<unsigned char>(i, j - 1)
					+ gr.at<unsigned char>(i + 1, j - 1));
			g_res.at<unsigned char>(i, j) = sqrt(pow(x, 2) + pow(y, 2));
		}*/
	for(int i = 0; i < sobel_x.rows; i++)
		for (int j = 0; j < sobel_x.cols; j++)
		g_res.at<unsigned char>(i, j) = sqrt(
			pow(sobel_x.at<unsigned char>(i, j), 2) +
			pow(sobel_y.at<unsigned char>(i, j), 2));
	show(g_res, g_res, "sobel1", "sobel2");
	bitwise_not(g_res, g_res);
	return g_res;
}

Mat getHist(const Mat& image)
{
	// Создаем заполненный нулями Mat-контейнер размером 1 x 256
	Mat hist = Mat::zeros(1, 256, CV_64FC1);

	// последовательно считываем яркость каждого элемента изображения
	// и увеличиваем на единицу значение соответствующего элемента матрицы hist
	for (int i = 0; i < image.cols; i++)
		for (int j = 0; j < image.rows; j++) {
			int r = image.at<unsigned char>(j, i);
			hist.at<double>(0, r) = hist.at<double>(0, r) + 1.0;
		}

	double m = 0, M = 0;
	minMaxLoc(hist, &m, &M); // ищем глобальный минимум и максимум
	hist = hist / M; // используем максимум для нормировки по высоте

	Mat hist_img = Mat::zeros(100, 256, CV_8U);

	for (int i = 0; i < 256; i++)
		for (int j = 0; j < 100; j++) {
			if (hist.at<double>(0, i) * 100 > j) {
				hist_img.at<unsigned char>(99 - j, i) = 255;
			}
		}
	bitwise_not(hist_img, hist_img); // инвертируем изображение
	return hist_img;
}

int main(int argc, char* argv[])
{
	setlocale(LC_ALL, "Russian");
	Mat frame_1, frame_2, outline, outline_res;
	VideoCapture cap("E:/video.mp4");
	for(int i = 0; i<11; i++)
		cap >> frame_1;
	cap >> frame_2;
	Mat diff_frame = diff(frame_1, frame_2);
	// show(frame_1, frame_2, "First frame", "Second frame");
	Mat sobel_res = SobelMask(frame_1);
	bitwise_and(diff_frame, sobel_res, outline);
	outline_res = Mat::zeros(720, 1280, CV_8U);
	/*editQuantizeLevel(2, outline, outline_res);
	show(outline_res, outline_res, "First frame", "Second frame");*/
	return 0;
}
#endif

