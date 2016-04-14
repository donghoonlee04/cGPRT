#ifndef IMAGE_UTILS_H
#define IMAGE_UTILS_H

#include <armadillo>

arma::fmat read_image(
	const std::string & image_name
	) {
	arma::fmat image;
	image.load(image_name, arma::pgm_binary);
	return image;
}

arma::fvec get_filter(
	const int size,
	const float sigma
	) {
	arma::fvec filter(size, arma::fill::zeros);

	float sum = 0;
	int center = (int)(size / 2);
	for (int i = 0; i < size; i++)
	{
		filter(i) = std::exp((float)(-(i-center)*(i-center))  / (2*sigma*sigma));
		sum += filter(i);
	}
	return filter / sum;
}

std::vector<arma::fmat> gaussian_smoothing(
	const arma::fmat & image,
	const arma::fvec & rect,
	const std::vector<float> & sigmas
	) {

	int num_sigmas = sigmas.size();
	std::vector<arma::fmat > image_blur(num_sigmas);

	for (int i = 0; i < num_sigmas; i++)
	{
		float tsig = sigmas[i] * std::sqrt((float)(rect(2)*rect(3)));
		arma::fmat image_buf = image;
		if (tsig != 0)
		{	
			
			int kernel_size = std::min(9, std::max(1, (int)((tsig - 0.5f) / 2 * 0.5) * 2 + 1));
			arma::fvec filter = get_filter(kernel_size, tsig);
			for (int y = 0; y < image.n_rows; y++)
				image_buf.row(y) = arma::conv(image_buf.row(y), filter, "same");
			for (int x = 0; x < image.n_cols; x++)
				image_buf.col(x) = arma::conv(image_buf.col(x), filter, "same");
		}
		image_blur[i] = image_buf;
		
	}
	return image_blur;
}
#endif