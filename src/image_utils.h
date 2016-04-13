#ifndef IMAGE_UTILS_H
#define IMAGE_UTILS_H

#include <armadillo>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

arma::fmat read_image(
	const std::string & image_name
	) {
	cv::Mat image_temp = cv::imread(image_name, cv::IMREAD_GRAYSCALE);
	arma::fmat image(image_temp.cols, image_temp.rows);

	for (int y = 0; y < image_temp.rows; ++y)
	{
		for (int x = 0; x < image_temp.cols; ++x)
		{
			image(y, x) = image_temp.at<unsigned char>(y, x);
		}
	}
	return image;
}

std::vector<arma::fmat > gaussian_smoothing(
	const arma::fmat & image,
	const arma::fvec & rect,
	const std::vector<float> & sigmas
	) {

	int num_sigmas = sigmas.size();
	std::vector<arma::fmat > image_blur(num_sigmas);

	cv::Mat src(image.n_cols, image.n_rows, CV_8U);
	cv::Mat dst(image.n_cols, image.n_rows, CV_8U);

	for (int y = 0; y < src.rows; ++y)
	{
		for (int x = 0; x < src.cols; ++x)
		{
			src.at<unsigned char>(y, x) = image(y, x);
		}
	}
	for (int i = 0; i < num_sigmas; i++)
	{
		float tsig = sigmas[i] * std::sqrt((float)(rect(2)*rect(3)));
		arma::fmat image_buf(image.n_rows, image.n_cols);
		if (tsig != 0)
		{	
			int kernelSize = (int)((tsig - 0.5f) / 2 * 0.5) * 2 + 1;
			cv::GaussianBlur(src, dst, cv::Size(kernelSize, kernelSize), tsig);
			for (int y = 0; y < dst.rows; ++y)
			{
				for (int x = 0; x < dst.cols; ++x)
				{
					image_buf(y, x) = dst.at<unsigned char>(y, x);
				}
			}
		}
		else
		{
			image_buf = image;
			image_blur.push_back(image_buf);
		}
	}
	return image_blur;
}
#endif