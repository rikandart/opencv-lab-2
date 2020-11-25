#define MISHA
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

#define THRESHOLD 65
#define THRESHOLD_2 20

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

// квантование
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

// межразностный кадр
Mat diff(const Mat& frame_1, const Mat& frame_2) {
	Mat res;
	cvtColor(frame_1 - frame_2, res, COLOR_RGB2GRAY);
	bitwise_not(res, res);
	return res;
}

// ядро собеля
Mat SobelMask(const Mat& frame) {
	Mat g_res = Mat::zeros(frame.rows, frame.cols, CV_8U), gr;
	Mat g_res_2 = Mat::zeros(frame.rows, frame.cols, CV_8U);
	cvtColor(frame, gr, COLOR_RGB2GRAY);
	Mat sobel_x, sobel_y, abs_grad_x, abs_grad_y;
	Sobel(gr, sobel_x, CV_8U, 1, 0);
	Sobel(gr, sobel_y, CV_8U, 0, 1);
	addWeighted(sobel_x, 1, sobel_y, 1, 0, g_res);
	show(g_res, "Sobel kernel");
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
	bitwise_not(g_res, g_res);
	return g_res;
}

// гистограммы проекций
void getHist(const Mat& image, Mat* out_hist_arr)
{
	// гор. гистограмма
	out_hist_arr[0] = Mat::zeros(image.cols, image.rows, CV_8U);
	// верт. гистограмма
	out_hist_arr[1] = Mat::zeros(image.rows, image.cols, CV_8U);
	for (int i = 0; i < image.rows; i++) {
		int hist_i = -1;
		for (int j = 0; j < image.cols; j++) {
			if (image.at<unsigned char>(i, j) > 0) {
				out_hist_arr[0].at<unsigned char>(image.cols-1 - ++hist_i, i) = 255;
			}
		}
	}
		
	for (int j = 0; j < image.cols; j++) {
		int hist_i = -1;
		for (int i = 0; i < image.rows; i++) {
			if (image.at<unsigned char>(i, j) > 0) {
				out_hist_arr[1].at<unsigned char>(image.rows-1 - ++hist_i, j) = 255;
			}
		}
	}
	for(int i = 0; i < out_hist_arr[0].cols; i++)
		out_hist_arr[0].at<unsigned char>(out_hist_arr[0].rows-1-THRESHOLD, i) = 255;
	for (int i = 0; i < out_hist_arr[1].cols; i++)
		out_hist_arr[1].at<unsigned char>(out_hist_arr[1].rows - 1 - THRESHOLD_2, i) = 255;
	bitwise_not(out_hist_arr[0], out_hist_arr[0]);
	bitwise_not(out_hist_arr[1], out_hist_arr[1]);
}

// получение стробов
void proj(const Mat& image, Mat* strobes) {
	// x strobe
	strobes[0] = Mat::zeros(image.rows, image.cols, CV_8U);
	// y stobe
	strobes[1] = Mat::zeros(image.rows, image.cols, CV_8U);
	for (int i = 0; i < image.rows; i++) {
		unsigned count_i = 0;
		for (int j = 0; j < image.cols; j++) {
			static bool write = false;
			if (image.at<unsigned char>(i,j) > 0 && !write)
				count_i++;
			// сравниваем с порогом
			if (count_i > THRESHOLD && !write) {
				j = 0;
				write = true;
			}
			if (write)
				strobes[0].at<unsigned char>(i, j) = image.at<unsigned char>(i, j);
		}
	}
	for (int j = 0; j < image.cols; j++) {
		unsigned count_j = 0;
		for (int i = 0; i < image.rows; i++) {
			static bool write = false;
			if (image.at<unsigned char>(i, j) > 0)
				count_j++;
			if (count_j > THRESHOLD_2 && !write) {
				i = 0;
				write = true;
			}
			if (write)
				strobes[1].at<unsigned char>(i, j) = image.at<unsigned char>(i, j);
		}
	}
}

int main(int argc, char* argv[])
{
	setlocale(LC_ALL, "Russian");
	Mat frame_1, frame_2, outline, opened, closed;
	Mat ones = Mat::ones(2, 2, CV_8U);
	VideoCapture cap("E:/video.mp4");
	for(int i = 0; i<11; i++)
		cap >> frame_1;
	cap >> frame_2;
	Mat diff_frame = diff(frame_1, frame_2);
	show(frame_1, frame_2, "First frame", "Second frame");
	show(diff_frame, "Difference frame");
	Mat sobel_res = SobelMask(frame_2);
	Mat sobel_res_2, diff_frame_2;
	editQuantizeLevel(2, sobel_res, sobel_res_2);
	editQuantizeLevel(2, diff_frame, diff_frame_2);
	bitwise_and(diff_frame_2, sobel_res_2, outline);	
	show(outline, "Outline");
	morphologyEx(outline, opened, MORPH_OPEN, ones);
	ones = Mat::ones(3, 3, CV_8U);
	morphologyEx(opened, closed, MORPH_CLOSE, ones);
	show(closed, "Morphed image");
	bitwise_not(closed, closed);
	Mat strobes[] = { Mat(), Mat() };
	Mat proj_hist[] = { Mat(), Mat() };
	proj(closed, strobes);
	getHist(closed, proj_hist);
	bitwise_not(strobes[0], strobes[0]);
	bitwise_not(strobes[1], strobes[1]);
	show(strobes[0], strobes[1], "X Strobe", "Y Strobe");
	show(proj_hist[0], proj_hist[1], "X Strobe Hist", "Y Strobe Hist");
	return 0;
}
#endif

