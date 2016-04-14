#ifndef SHAPE_UTILS_H
#define SHAPE_UTILS_H

#include <armadillo>
#include <iomanip>
#include <image_utils.h>

void load_data(
	const std::string & image_list_file,
	std::vector<std::string> & image_names,
	std::vector<arma::fvec> & rects,
	std::vector<arma::fmat> & shapes
	)
{
	std::ifstream fin;
	fin.open(image_list_file);
	int num_data, num_points;
	fin >> num_data >> num_points;
		
	image_names.resize(num_data); rects.resize(num_data); shapes.resize(num_data);

	for (int i = 0; i < num_data; ++i)
	{		
		fin >> image_names[i];
		rects[i] = arma::fvec(4);
		fin >> rects[i](0) >> rects[i](1) >> rects[i](2) >> rects[i](3);
		shapes[i] = arma::fmat(2, num_points);
		for (int j = 0; j < num_points; ++j)
			fin >> shapes[i](0, j) >> shapes[i](1, j);
	}
	fin.close();
}

void save_data(
	const std::vector<std::string> & image_names,
	const std::vector<arma::fvec> & rects,
	const std::vector<arma::fmat> & shapes,
	std::string & result_file
	)
{
	std::ofstream fout;
	fout.open(result_file);
	fout << std::setprecision(6) << std::fixed;
	int num_data = (int)shapes.size();
	int num_points = (int)shapes[0].n_cols;

	fout << num_data << std::endl;
	fout << num_points << std::endl;

	for (int i = 0; i < num_data; ++i)
	{
		fout << image_names[i] << std::endl;
		fout << (int)rects[i](0) << " " << (int)rects[i](1) << " " << (int)rects[i](2) << " " << (int)rects[i](3) << std::endl;
		for (int j = 0; j < num_points; ++j)
			fout << shapes[i](0, j) << " " << shapes[i](1, j) << std::endl;
	}
	fout.close();
}

arma::fmat normalizing_shape(
	const arma::fvec & rect,
	const arma::fmat & shape_unnorm
	) {
	arma::fmat shape_norm = shape_unnorm;
	for (int i = 0; i < shape_unnorm.n_cols; i++)
	{
		shape_norm(0, i) -= rect(0);
		shape_norm(0, i) /= rect(2);
		shape_norm(1, i) -= rect(1);
		shape_norm(1, i) /= rect(3);
	}
	return shape_norm;
}

arma::fmat unnormalizing_shape(
	const arma::fvec & rect,
	const arma::fmat & shape_norm
	) {
	arma::fmat shape_unnorm = shape_norm;
	for (int i = 0; i < shape_norm.n_cols; i++)
	{
		shape_unnorm(0, i) *= rect(2);
		shape_unnorm(0, i) += rect(0);
		shape_unnorm(1, i) *= rect(3);
		shape_unnorm(1, i) += rect(1);
	}
	return shape_unnorm;
}

float inter_occular_distance(
	const arma::fmat & shape,
	const std::vector<int> & left_eye_idx,
	const std::vector<int> & right_eye_idx
	) {
	int num_left_eye_points = (int)left_eye_idx.size();
	int num_right_eye_points = (int)right_eye_idx.size();
	arma::fvec left_eye_point(2, 1, arma::fill::zeros);
	arma::fvec right_eye_point(2, 1, arma::fill::zeros);
	for (int i = 0; i < num_left_eye_points; i++)
		left_eye_point += shape.col(left_eye_idx[i]);
	for (int i = 0; i < num_right_eye_points; i++)
		right_eye_point += shape.col(right_eye_idx[i]);
	left_eye_point /= (float)num_left_eye_points;
	right_eye_point /= (float)num_right_eye_points;
	arma::fvec diff = left_eye_point - right_eye_point;
		
	return sqrt(diff(0)*diff(0) + diff(1)*diff(1));
}

std::vector<float> extract_pixel_values(
	const arma::fmat & image,
	const std::vector<arma::fmat> & image_vec,
	const arma::fvec & rect,
	const arma::fmat & shape_normed,
	const arma::fmat & inv_tform,
	const std::vector<arma::fvec> & deltas,
	const std::vector<int> sigma_idx,
	const bool enable_smoothing
	) {
		
	int num_sampling_patterns = (int)deltas.size();
	int num_points = shape_normed.n_cols;
	std::vector<float> pixel_values;
	pixel_values.resize(num_sampling_patterns*num_points);

	std::vector<arma::fvec> deltas_unnormed(num_sampling_patterns);
	for (int i = 0; i < num_sampling_patterns; i++)
		deltas_unnormed[i] = unnormalizing_shape(rect, inv_tform*deltas[i]);
	arma::fmat shape_unnormed = unnormalizing_shape(rect, shape_normed);

	int idx = 0;
	for (int i = 0; i < num_points; i++)
	{
		for (int j = 0; j < num_sampling_patterns; j++)
		{
			arma::fvec pixel_location = deltas_unnormed[j] + shape_unnormed.col(i);
			int x = (int)(pixel_location(0) + 0.5f);
			int y = (int)(pixel_location(1) + 0.5f);

			x = x > 0 ? x : 0;
			y = y > 0 ? y : 0;
			x = rect(2) - 1 > x ? x : rect(2) - 1;
			y = rect(3) - 1 > y ? y : rect(3) - 1;

			pixel_values[idx] = enable_smoothing ? image_vec[sigma_idx[j]](y, x) : image(y, x);
			idx++;
		}
	}
	return pixel_values;
}

void find_similarity_tform(
	const arma::fmat & from_points,
	const arma::fmat & to_points, 
	arma::fmat & tform,
	arma::fmat & tform_inv
	) {

	int num_points = from_points.n_cols;
	arma::fmat from_mean, to_mean, from_temp, to_temp, cov;
	float from_sigma, to_sigma = 0;
	from_temp = from_points;
	to_temp = to_points;

	from_mean = arma::mean(from_temp, 1);
	to_mean = arma::mean(to_temp, 1);

	for (int i = 0; i < num_points; i++)
	{
		from_temp.col(i) -= from_mean;
		to_temp.col(i) -= to_mean;
	}
	from_sigma = arma::accu(from_temp % from_temp) / (float)num_points;
	to_sigma = arma::accu(to_temp % to_temp) / (float)num_points;
	cov = (to_temp * from_temp.t()) / (float)num_points;

	arma::fmat U, V, S, D, r;
	arma::fvec d;
	arma::svd(U, d, V, cov);
	S = arma::fmat(2, 2, arma::fill::eye);
	float cov_det = det(cov);
	if (cov_det < 0.0f || (cov_det == 0 && arma::det(U)*arma::det(V) < 0))
	{
		if (d(1) < d(0))
			S(1, 1) = -1.0f;
		else
			S(0, 0) = -1.0f;
	}
	r = U*S*V.t();
	D = arma::fmat(2, 2, arma::fill::zeros);	
	D(0, 0) = d(0);
	D(1, 1) = d(1);
	float c = 1.0f;
	if (from_sigma != 0)
		c = 1.0f / from_sigma * arma::trace(D*S);

	tform = c*r;
	tform_inv = arma::inv(tform);
}

float compute_error(
	const arma::fmat & shape_gnd,
	const arma::fmat & shape_pred,
	const float iod
	) {
	arma::fmat err = shape_gnd - shape_pred;
	return as_scalar(arma::mean(arma::sqrt(arma::sum(err % err, 0)), 1)) / iod * 100.0f;
}
#endif
