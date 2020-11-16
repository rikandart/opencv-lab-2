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
	resizeWindow(name_1, frame_1.cols, frame_1.rows);
	moveWindow(name_1, 0, 0);
	imshow(name_1, frame_1);
	namedWindow(name_2, WINDOW_NORMAL | WINDOW_FREERATIO | WINDOW_GUI_EXPANDED);
	resizeWindow(name_2, frame_2.cols, frame_2.rows);
	moveWindow(name_2, frame_1.cols, 0);
	imshow(name_2, frame_2);
	waitKey(0);
}

void show(const Mat& frame_1, const char* name_1) {
	namedWindow(name_1, WINDOW_NORMAL | WINDOW_FREERATIO | WINDOW_GUI_EXPANDED);
	resizeWindow(name_1, frame_1.cols, frame_1.rows);
	moveWindow(name_1, 0, 0);
	imshow(name_1, frame_1);
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
	Mat g_res = Mat::zeros(frame.rows, frame.cols, CV_8U), gr;
	Mat g_res_2 = Mat::zeros(frame.rows, frame.cols, CV_8U);
	cvtColor(frame, gr, COLOR_RGB2GRAY);
	Mat sobel_x, sobel_y, abs_grad_x, abs_grad_y;
	Sobel(gr, sobel_x, CV_8U, 1, 0);
	Sobel(gr, sobel_y, CV_8U, 0, 1);
	addWeighted(sobel_x, 1, sobel_y, 1, 0, g_res);

	/*for (int i = 0; i < gr.rows; i++) {
		int z1 = -1, z2 = -1, z3 = -1, z4 = -1, z5 = -1, z6 = -1,
			z7 = -1, z8 = -1, z9 = -1;
		if (i == 0) { z1 = 0; z2 = 0; z3 = 0; }
		else if (i == gr.rows) { z7 = 0; z8 = 0; z9 = 0; }
		for (int j = 0; j < gr.cols; j++) {
			if (j == 0) { z1 = 0; z4 = 0; z7 = 0; }
			else if (j == gr.cols) { z3 = 0; z6 = 0; z9 = 0; }

			if (z1 != 0) z1 = gr.at<unsigned char>(i - 1, j - 1);
			if (z2 != 0) z2 = gr.at<unsigned char>(i - 1, j);
			if (z3 != 0) z3 = gr.at<unsigned char>(i - 1, j + 1);
			if (z4 != 0) z4 = gr.at<unsigned char>(i, j - 1);
			if (z6 != 0) z6 = gr.at<unsigned char>(i, j + 1);
			if (z7 != 0) z7 = gr.at<unsigned char>(i + 1, j - 1);
			if (z8 != 0) z8 = gr.at<unsigned char>(i + 1, j);
			if (z9 != 0) z9 = gr.at<unsigned char>(i + 1, j + 1);

			int x = z7 + 2 * z8 + z9 - (z1 + 2 * z2 + z3);
			int y = z3 + 2 * z6 + z9 - (z1 + 2 * z4 + z7);
			g_res.at<unsigned char>(i, j) = sqrt(pow(x, 2) + pow(y, 2));
		}
	}*/

	/*for (int i = 1; i < gr.rows-1; i++)
		for (int j = 1; j < gr.cols-1; j++) {

			int x = gr.at<unsigned char>(i + 1, j - 1)
				+ 2 * gr.at<unsigned char>(i + 1, j) + gr.at<unsigned char>(i + 1, j + 1)
				- (gr.at<unsigned char>(i - 1, j - 1) + 2 * gr.at<unsigned char>(i - 1, j)
					+ gr.at<unsigned char>(i - 1, j + 1));
			
			int y = gr.at<unsigned char>(i - 1, j + 1)
				+ 2 * gr.at<unsigned char>(i, j + 1) + gr.at<unsigned char>(i + 1, j + 1)
				- (gr.at<unsigned char>(i - 1, j - 1) + 2 * gr.at<unsigned char>(i, j - 1)
					+ gr.at<unsigned char>(i + 1, j - 1));
			g_res.at<unsigned char>(i, j) = sqrt(pow(x, 2) + pow(y, 2));
		}*/
	//show(g_res - g_res_2, g_res_2 - g_res, "sobel1", "sobel2");
	bitwise_not(g_res, g_res);
	return g_res;
}

int main(int argc, char* argv[])
{
	setlocale(LC_ALL, "Russian");
	Mat frame_1, frame_2, outline, m_erode, m_dilate;
	Mat ones = Mat::ones(2, 2, CV_8U);
	VideoCapture cap("E:/video.mp4");
	for(int i = 0; i<11; i++)
		cap >> frame_1;
	cap >> frame_2;
	Mat diff_frame = diff(frame_1, frame_2);
	/*show(frame_1, frame_2, "First frame", "Second frame");
	show(diff_frame, diff_frame, "First frame", "Second frame");*/
	Mat sobel_res = SobelMask(frame_2);
	Mat sobel_res_2, diff_frame_2;
	editQuantizeLevel(2, sobel_res, sobel_res_2);
	editQuantizeLevel(2, diff_frame, diff_frame_2);
	bitwise_and(diff_frame_2, sobel_res_2, outline);	
	show(outline, "Outline");
	erode(outline, m_erode, ones);
	dilate(m_erode, m_dilate, ones);
	ones = Mat::ones(3, 3, CV_8U);
	dilate(m_dilate, m_erode, ones);
	erode(m_erode, outline, ones);
	// morphologyEx(m_dilate, morph, MORPH_BLACKHAT, ones);
	/*erode(outline, m_erode, ones);
	dilate(m_erode, m_dilate, ones);*/
	show(outline, "Morphed image");
	return 0;
}
#endif

