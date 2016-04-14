#ifndef SHAPE_TRAINER_H
#define SHAPE_TRAINER_H

#include <armadillo>
#include <iomanip>
#include <regression_tree.h>
#include <shape_utils.h>
#include <feature_defs.h>
#include <gp_utils.h>

class shape_trainer
{
private:

	int num_cascades, num_forests, num_trees, tree_depth;
	int num_points, oversampling_ratio, num_split_tests;
	bool enable_smoothing;
	float nu;
	std::vector<int> left_eye_idx, right_eye_idx;
	int num_data_tr, num_data_te;
	int random_seed;

	int anchor_target_grow, anchor_target_opt, num_split_nodes, num_leaf_nodes, num_nodes, num_feats;
	std::vector<int> anchor_target_grow_shuf, anchor_target_opt_shuf;
	arma::fmat mean_shape;
	float mean_shape_iod;
	std::vector<std::vector<regression_tree> > cascades;

	std::vector<arma::fvec> deltas;
	std::vector<float> sigmas;
	std::vector<int> sigma_idx;

public:

	shape_trainer() {}

	shape_trainer(
		const std::string & config_file_name
		) {
		std::ifstream fin;
		std::string token1, token2;
		fin.open(config_file_name);
		while (!fin.eof())
		{
			fin >> token1;
			if (!token1.compare("num_cascades"))				{ fin >> num_cascades; }
			else if (!token1.compare("num_forests"))			{ fin >> num_forests; }
			else if (!token1.compare("num_trees"))				{ fin >> num_trees; }
			else if (!token1.compare("tree_depth"))				{ fin >> tree_depth; }
			else if (!token1.compare("nu"))						{ fin >> nu; }
			else if (!token1.compare("oversampling_ratio"))		{ fin >> oversampling_ratio; }
			else if (!token1.compare("num_points"))				{ fin >> num_points; }
			else if (!token1.compare("left_eye_idx"))			{ fin >> token2; fin >> token2; while (token2.compare(")")) { left_eye_idx.push_back(atoi(token2.c_str())); fin >> token2; } }
			else if (!token1.compare("right_eye_idx"))			{ fin >> token2; fin >> token2; while (token2.compare(")")) { right_eye_idx.push_back(atoi(token2.c_str())); fin >> token2; } }
			else if (!token1.compare("enable_smoothing"))		{ fin >> enable_smoothing; }
			else if (!token1.compare("num_split_tests"))		{ fin >> num_split_tests; }
			else if (!token1.compare("random_seed"))			{ fin >> random_seed; arma::arma_rng::set_seed(random_seed); }
		}
		fin.close();
	}

	shape_predictor train(
		const std::string & image_directory_train, 
		const std::vector<std::string> & image_names_train,		
		const std::vector<arma::fvec> & rects_train,
		const std::vector<arma::fmat> & shapes_train,
		const std::string & image_directory_test,
		const std::vector<std::string> & image_names_test,
		const std::vector<arma::fvec> & rects_test,
		const std::vector<arma::fmat> & shapes_test
		) {

		std::cout << std::fixed << std::setprecision(4);
		
		print_parameters();

		std::vector<arma::fvec> rects_tr_gnd, rects_te_gnd;
		std::vector<arma::fmat> shapes_tr_pred, shapes_tr_gnd, shapes_te_pred, shapes_te_gnd;
		std::vector<int> image_tr_idx, image_te_idx;
		std::vector<float> iod_tr, iod_te;
		std::vector<std::vector<arma::fmat> > images_train, images_test;

		std::cout << "[0] Initialize training" << std::endl;
		initialize_training(rects_train, shapes_train, rects_test, shapes_test,	shapes_tr_pred, shapes_tr_gnd, rects_tr_gnd, image_tr_idx, iod_tr, shapes_te_pred, shapes_te_gnd, rects_te_gnd, image_te_idx, iod_te);
		std::cout << "[0] Initialize feature extractor" << std::endl; 
		initialize_feature_extractor(deltas, sigmas, sigma_idx);

		std::vector<arma::fmat> shapes_tr_pred_tformed(num_data_tr), shapes_tr_gnd_tformed(num_data_tr), shapes_delta_tr(num_data_tr), shapes_te_pred_tformed(num_data_te), shapes_te_gnd_tformed(num_data_te), shapes_delta_te(num_data_te);
		std::vector<arma::fmat> tform_tr(num_data_tr), tform_inv_tr(num_data_tr), tform_te(num_data_te), tform_inv_te(num_data_te);
		std::vector<std::vector<float> > pixel_values_tr(num_data_tr), pixel_values_te(num_data_te);

		float tr_err = 0.0f, te_err = 0.0f;
		for (int i = 0; i < num_data_tr; i++)
			tr_err += compute_error(shapes_tr_gnd[i], shapes_tr_pred[i], iod_tr[i]);
		tr_err /= (float)num_data_tr;
		for (int i = 0; i < num_data_te; i++)
			te_err += compute_error(shapes_te_gnd[i], shapes_te_pred[i], iod_te[i]);
		te_err /= (float)num_data_te;

		std::cout << "[0/" << num_cascades << "] tr_err/te_err : " << tr_err << " " << te_err << std::endl;

		cascades.reserve(num_cascades);

		for (int cascade_idx = 0; cascade_idx < num_cascades; cascade_idx++)
		{
			std::cout << "[" << cascade_idx << "/" << num_cascades << "] Extract tr features ";
			int image_idx_buf = -1;
			arma::fmat image_buf;
			std::vector<arma::fmat> image_blur_buf;
			for (int i = 0; i < num_data_tr; i++)
			{
				if (i && !(i % (num_data_tr / 10))) { std::cout << "."; }
				if (image_idx_buf != image_tr_idx[i])
				{
					image_idx_buf = image_tr_idx[i];
					image_buf.load(image_directory_train + "/" + image_names_train[image_idx_buf], arma::pgm_binary);
					if (enable_smoothing) { image_blur_buf = gaussian_smoothing(image_buf, rects_tr_gnd[i], sigmas); }
				}
				find_similarity_tform(shapes_tr_pred[i], mean_shape, tform_tr[i], tform_inv_tr[i]);
				shapes_tr_pred_tformed[i] = tform_tr[i] * shapes_tr_pred[i];
				shapes_tr_gnd_tformed[i] = tform_tr[i] * shapes_tr_gnd[i];
				pixel_values_tr[i] = extract_pixel_values(image_buf, image_blur_buf, rects_tr_gnd[i], shapes_tr_pred[i], tform_inv_tr[i], deltas, sigma_idx, enable_smoothing);
			}
			std::cout << std::endl;
			std::cout << "[" << cascade_idx << "/" << num_cascades << "] Extract te features ";

			image_idx_buf = -1;
			for (int i = 0; i < num_data_te; i++)
			{
				if (i && !(i % (num_data_te / 10))) { std::cout << "."; }
				if (image_idx_buf != image_te_idx[i])
				{
					image_idx_buf = image_te_idx[i]; 
					image_buf.load(image_directory_test + "/" + image_names_test[image_te_idx[i]], arma::pgm_binary);
					if (enable_smoothing) { image_blur_buf = gaussian_smoothing(image_buf, rects_te_gnd[i], sigmas); }
				}
				find_similarity_tform(shapes_te_pred[i], mean_shape, tform_te[i], tform_inv_te[i]);
				shapes_te_pred_tformed[i] = tform_te[i] * shapes_te_pred[i];
				shapes_te_gnd_tformed[i] = tform_te[i] * shapes_te_gnd[i];
				pixel_values_te[i] = extract_pixel_values(image_buf, image_blur_buf, rects_te_gnd[i], shapes_te_pred[i], tform_inv_te[i], deltas, sigma_idx, enable_smoothing);
			}
			std::cout << std::endl;
			
			for (int forest_idx = 0; forest_idx < num_forests; forest_idx++)
			{
				std::vector<regression_tree> forest;
				for (int i = 0; i < num_data_tr; i++)
					shapes_delta_tr[i] = shapes_tr_gnd_tformed[i] - shapes_tr_pred_tformed[i];

				float shape_std_val = make_unit_variance(shapes_delta_tr);

				grow_forest(shapes_delta_tr, pixel_values_tr, forest);
				
				gprt_optimize(shapes_delta_tr, pixel_values_tr, forest);

				compensate_scale(shape_std_val, nu, forest);

				for (int tree_idx = 0; tree_idx < num_trees; tree_idx++)
					cascades[cascade_idx].push_back(forest[tree_idx]);

				for (int tree_idx = 0; tree_idx < num_trees; tree_idx++)
				{
					for (int i = 0; i < num_data_tr; i++)
						shapes_tr_pred_tformed[i] += forest[tree_idx].leaf_values[forest[tree_idx](pixel_values_tr[i])];
					for (int i = 0; i < num_data_te; i++)
						shapes_te_pred_tformed[i] += forest[tree_idx].leaf_values[forest[tree_idx](pixel_values_te[i])];
				}

				for (int i = 0; i < num_data_tr; i++)
					shapes_tr_pred[i] = tform_inv_tr[i] * shapes_tr_pred_tformed[i];
				for (int i = 0; i < num_data_te; i++)
					shapes_te_pred[i] = tform_inv_te[i] * shapes_te_pred_tformed[i];

				tr_err = 0.0f; te_err = 0.0f;
				for (int i = 0; i < num_data_tr; i++)
					tr_err += compute_error(shapes_tr_gnd[i], shapes_tr_pred[i], iod_tr[i]);
				tr_err /= (float)num_data_tr;
				for (int i = 0; i < num_data_te; i++)
					te_err += compute_error(shapes_te_gnd[i], shapes_te_pred[i], iod_te[i]);
				te_err /= (float)num_data_te;
				std::cout << "---[" << forest_idx + 1 << "/" << num_forests << "] tr_err/te_err : " << tr_err << " " << te_err <<  std::endl;
			}
			
			std::cout << "[" << cascade_idx + 1 << "/" << num_cascades << "] tr_err/te_err : " << tr_err << " " << te_err << std::endl;
		}
		std::cout << "[" << num_cascades << "/" << num_cascades << "] Training completed" << std::endl;
		
		return shape_predictor(cascades, mean_shape, deltas, sigmas, sigma_idx, enable_smoothing, left_eye_idx, right_eye_idx);
	}

	void gprt_optimize(
		const std::vector<arma::fmat> & delta_normed,
		const std::vector<std::vector<float> > & pixel_values,
		std::vector<regression_tree> & forest
		) {
		
		int n = num_data_tr;
		int q = num_trees * num_leaf_nodes;
		arma::fmat Q(n, q, arma::fill::zeros);
		arma::fmat y(n, num_points * 2, arma::fill::zeros);
		
		float sn = 10.0f, sf = 1.0f;
		int length = -10;
		for (int i = 0; i<n; ++i)
		{
			for (int j = 0; j < num_points; ++j)
			{
				y(i, 2 * j) = delta_normed[i](0, j);
				y(i, 2 * j + 1) = delta_normed[i](1, j);
			}
				
			for (int j = 0; j<num_trees; ++j)
			{
				int leaf_idx = forest[j](pixel_values[i]);				
				Q(i, leaf_idx + j*num_leaf_nodes) = 1.0f * (std::sqrt(1.0f / (float)num_trees));
			}
		}
		
		float sf0 = std::log(sf) / 2.0f;
		float sn0 = std::log(sn) / 2.0f;
		
		arma::fmat yy(n, 2, arma::fill::zeros);
		for (int i = 0; i < n; i++ )
			yy.row(i) = delta_normed[i].col(anchor_target_opt_shuf[anchor_target_opt]).t();

		gp_optimize(yy, Q, sn0, length, sf, sn);

		sn = std::exp(2.0f*sn);
		sf = std::exp(2.0f*sf);

		arma::fmat Qt = Q.t();
		arma::fmat qe(q, q, arma::fill::eye);
		arma::fmat alphas = arma::solve(qe + Qt*Q / sn, Qt * y) / sn;
		arma::fmat betas = arma::inv(qe + Qt*Q / sn);
		
		int q_idx = 0;
		for (int tree_idx = 0; tree_idx < num_trees; tree_idx++)
		{
			for (int i = 0; i < num_leaf_nodes; i++)
			{
				forest[tree_idx].leaf_values[i].zeros();
				float sig_val = betas(q_idx, q_idx);
				arma::fmat temp = alphas.row(q_idx) * (std::sqrt(1.0f / (float)num_trees)) * (1.0f / (sig_val / sn + 1.0f));;
				for (int p = 0; p < num_points; p++)
				{
					forest[tree_idx].leaf_values[i](0, p) = temp(2 * p);
					forest[tree_idx].leaf_values[i](1, p) = temp(2 * p+1);
				}				
				q_idx++;
			}
		}		
		anchor_target_opt++;
		anchor_target_opt = anchor_target_opt% num_points;

	}

	void grow_forest(
		const std::vector<arma::fmat> & delta_tr_normed,
		const std::vector<std::vector<float> > & pixel_values,
		std::vector<regression_tree> & forest
		) {
		arma::fmat sums_ini(2, num_points, arma::fill::zeros);
		for (int i = 0; i < num_data_tr; i++)
			sums_ini += delta_tr_normed[i];
		
		for (int tree_idx = 0; tree_idx < num_trees; tree_idx++)
		{			
			regression_tree tree;
			bool success = false;
			while (!success)
			{
				tree.splits.clear();
				tree.leaf_values.clear();
				arma::fmat sums(2, num_nodes, arma::fill::zeros);
				sums.col(0) = sums_ini.col(anchor_target_grow_shuf[anchor_target_grow]);
				std::vector<std::vector<int> > data_indices(num_nodes);
				
				for (int i = 0; i < num_data_tr; i++)
					data_indices[0].push_back(i);
				for (int node_idx = 0; node_idx < num_split_nodes; node_idx++)
				{
					split_node split_node = generate_split(delta_tr_normed, pixel_values, node_idx, data_indices, sums);
					tree.splits.push_back(split_node);
				}		
				success = true;
				for (int node_idx = num_split_nodes; node_idx < num_nodes; node_idx++)
					if (data_indices[node_idx].size() == 0)
						success = false; 
			}

			std::vector<int> num_data_in_leaf_node;
			for (int node_idx = 0; node_idx < num_leaf_nodes; node_idx++)
			{
				tree.leaf_values.push_back(arma::fmat(2, num_points, arma::fill::zeros));
				num_data_in_leaf_node.push_back(0);
			}
			for (int i = 0; i < num_data_tr; i++)
			{
				int leaf_idx = tree(pixel_values[i]);
				tree.leaf_values[leaf_idx] += delta_tr_normed[i];
				num_data_in_leaf_node[leaf_idx]++;
			}
			for (int node_idx = 0; node_idx < num_leaf_nodes; node_idx++)
				tree.leaf_values[node_idx] /= ((float)num_data_in_leaf_node[node_idx] * (float) num_trees);
			
			forest.push_back(tree);
			anchor_target_grow++;
			anchor_target_grow %= num_points;
		}	
	}

	split_node generate_split(
		const std::vector<arma::fmat> & delta,
		const std::vector<std::vector<float> > & pixel_values,
		const int node_idx,
		std::vector<std::vector<int> > & data_indices,
		arma::fmat & sums
		) {

		arma::fmat left_sum(2, 1, arma::fill::zeros);
		std::vector<int> left_indices;
		split_node split;
		float score = -1.0f;
		std::vector<int> data_idx = data_indices[node_idx];
		int num_data_in_node = (int)data_idx.size();
		float split_thres_range = 0.3f;

		if (num_data_in_node)
		{
			for (int split_idx = 0; split_idx < num_split_tests; split_idx++)
			{
				split_node split_temp;
				arma::fmat left_sum_temp(2, 1, arma::fill::zeros);
				arma::fmat right_sum_temp(2, 1, arma::fill::zeros);
				std::vector<int> left_indices_temp, right_indices_temp;
				int left_count = 0, right_count = 0;

				int pattern_idx1 = arma::as_scalar(arma::randi(1)) % NUM_PATTERNS;
				int pattern_idx2 = arma::as_scalar(arma::randi(1)) % (NUM_PATTERNS - 1);
				if (pattern_idx2 >= pattern_idx1)
					pattern_idx2++;
				split_temp.idx1 = pattern_idx1 + NUM_PATTERNS * anchor_target_grow_shuf[anchor_target_grow];
				split_temp.idx2 = pattern_idx2 + NUM_PATTERNS * anchor_target_grow_shuf[anchor_target_grow];

				arma::fvec pixel_diff(num_data_in_node, arma::fill::zeros);
				for (int i = 0; i < num_data_in_node; i++)
					pixel_diff(i) = pixel_values[data_idx[i]][split_temp.idx1] - pixel_values[data_idx[i]][split_temp.idx2];
				float max_val = arma::max(pixel_diff);
				float min_val = arma::min(pixel_diff);

				split_temp.thres = (arma::as_scalar(arma::randu(1)) * (max_val - min_val) + min_val) * split_thres_range;
				
				for (int i = 0; i < num_data_in_node; i++)
				{
					if (pixel_diff(i) > split_temp.thres)
					{
						left_sum_temp += delta[data_idx[i]].col(anchor_target_grow_shuf[anchor_target_grow]);
						left_indices_temp.push_back(data_idx[i]);
						left_count++;
					}
					else
						right_indices_temp.push_back(data_idx[i]);
				}
				right_sum_temp = sums.col(node_idx) - left_sum_temp;
				right_count = num_data_in_node - left_count;


				if (left_count && right_count)
				{

					
					float score_temp = arma::as_scalar(arma::accu(left_sum_temp % left_sum_temp)) / (float)left_count +
						arma::as_scalar(arma::accu(right_sum_temp % right_sum_temp)) / (float)right_count;

					if (score_temp > score)
					{
						score = score_temp;
						sums.col(left_child(node_idx)) = left_sum_temp;
						sums.col(right_child(node_idx)) = right_sum_temp;
						data_indices[left_child(node_idx)] = left_indices_temp;
						data_indices[right_child(node_idx)] = right_indices_temp;
						split = split_temp;
					}
				}
			}
		}
		return split;
	}

	void initialize_training(
		const std::vector<arma::fvec> & rects_train,
		const std::vector<arma::fmat> & shapes_train,
		const std::vector<arma::fvec> & rects_test,
		const std::vector<arma::fmat> & shapes_test,
		std::vector<arma::fmat> & shapes_tr_pred,
		std::vector<arma::fmat> & shapes_tr_gnd,
		std::vector<arma::fvec> & rects_tr_gnd,
		std::vector<int> & image_tr_idx,
		std::vector<float> & iod_tr,
		std::vector<arma::fmat> & shapes_te_pred,
		std::vector<arma::fmat> & shapes_te_gnd,
		std::vector<arma::fvec> & rects_te_gnd,
		std::vector<int> & image_te_idx,
		std::vector<float> & iod_te
		) {

		anchor_target_grow = 0;
		anchor_target_opt = 0;
		num_split_nodes = (int)std::pow(2, tree_depth) - 1;
		num_leaf_nodes = (int)std::pow(2, tree_depth);
		num_nodes = num_split_nodes + num_leaf_nodes;
		cascades.resize(num_cascades);
		int num_samples_tr = (int)shapes_train.size();
		int num_samples_te = (int)shapes_test.size();
		num_data_tr = num_samples_tr * oversampling_ratio;
		num_data_te = num_samples_te;

		for (int i = 0; i < num_points; i++) 
		{
			anchor_target_grow_shuf.push_back(i);
			anchor_target_opt_shuf.push_back(i);
		}
		for (int i = 0; i < num_points; i++)
		{
			int rand_idx = arma::as_scalar(arma::randi(1)) % num_points;
			std::swap(anchor_target_grow_shuf[i], anchor_target_grow_shuf[rand_idx]);
			rand_idx = arma::as_scalar(arma::randi(1)) % num_points;
			std::swap(anchor_target_opt_shuf[i], anchor_target_opt_shuf[rand_idx]);
		}

		mean_shape.zeros(2, num_points);
		shapes_tr_pred.resize(num_data_tr); shapes_tr_gnd.resize(num_data_tr); rects_tr_gnd.resize(num_data_tr), image_tr_idx.resize(num_data_tr); iod_tr.resize(num_data_tr);
		shapes_te_pred.resize(num_data_te); shapes_te_gnd.resize(num_data_te); rects_te_gnd.resize(num_data_tr), image_te_idx.resize(num_data_te); iod_te.resize(num_data_te);
		std::vector<arma::fmat> shapes_train_temp(num_samples_tr), shapes_test_temp(num_samples_te);
		std::vector<float> iod_tr_temp(num_samples_tr), iod_te_temp(num_samples_te);

		for (int i = 0; i < num_samples_tr; i++)
		{
			shapes_train_temp[i] = normalizing_shape(rects_train[i], shapes_train[i]);
			iod_tr_temp[i] = inter_occular_distance(shapes_train_temp[i], left_eye_idx, right_eye_idx);
			mean_shape += shapes_train_temp[i];
		}
		mean_shape /= (float)num_samples_tr;
		for (int i = 0; i < num_samples_te; i++)
		{
			shapes_test_temp[i] = normalizing_shape(rects_test[i], shapes_test[i]);
			iod_te_temp[i] = inter_occular_distance(shapes_test_temp[i], left_eye_idx, right_eye_idx);
		}
		mean_shape_iod = inter_occular_distance(mean_shape, left_eye_idx, right_eye_idx);

		int data_idx = 0;
		for (int i = 0; i < num_samples_tr; i++)
		{
			for (int j = 0; j < oversampling_ratio; j++)
			{
				shapes_tr_gnd[data_idx] = shapes_train_temp[i];
				rects_tr_gnd[data_idx] = rects_train[i];
				image_tr_idx[data_idx] = i;
				iod_tr[data_idx] = iod_tr_temp[i];
				
				std::vector<int> rand_indices;
				image_tr_idx[data_idx] = i;
				rand_indices.push_back(i);
				if (!j)
					shapes_tr_pred[data_idx] = mean_shape;
				else
				{
					bool success = 0;
					int rand_idx;
					do
					{
						success = 1;
						rand_idx = arma::as_scalar(arma::randi(1)) % num_samples_tr;
						for (int k = 0; k < (int)rand_indices.size(); k++)
						{
							if (rand_idx == rand_indices[k])
								success = 0;
						}
					} while (!success);
					rand_indices.push_back(rand_idx);
					shapes_tr_pred[data_idx] = shapes_train_temp[rand_idx];
				}
				data_idx++;
			}
		}

		for (int i = 0; i < num_samples_te; i++)
		{
			shapes_te_gnd[i] = shapes_test_temp[i];
			rects_te_gnd[i] = rects_test[i];
			image_te_idx[i] = i;
			iod_te[i] = iod_te_temp[i];
			image_te_idx[i] = i;
			shapes_te_pred[i] = mean_shape;
		}
	}

	void initialize_feature_extractor(
		std::vector<arma::fvec> & deltas_,
		std::vector<float> & sigmas_,
		std::vector<int> & sigma_idx_
		) {
		float feature_pattern_scale = 0.4f;
		float scale = feature_pattern_scale * mean_shape_iod;
		num_feats = NUM_PATTERNS*num_points;
		deltas_.resize(NUM_PATTERNS);
		sigmas_.resize(NUM_SIGMAS);
		sigma_idx_.resize(NUM_PATTERNS);

		for (int i = 0; i < NUM_PATTERNS; i++)
		{
			deltas_[i] = arma::fvec(2, 1);
			deltas_[i](0, 0) = DELTA_X[i] * scale;
			deltas_[i](1, 0) = DELTA_Y[i] * scale;
		}
		if (enable_smoothing)
		{
			for (int i = 0; i < NUM_SIGMAS; i++)
				sigmas_[i] = SIGMAS[i] * scale;
			for (int i = 0; i < NUM_PATTERNS; i++)
				sigma_idx_[i] = SIGMA_IDX[i];
		}
	}

	float make_unit_variance(
		std::vector<arma::fmat> & shapes
		) {
		float shape_std_val = 0.0f;
		for (int i = 0; i < (int)shapes.size(); i++)
			shape_std_val += arma::as_scalar(arma::accu(shapes[i] % shapes[i]));

		shape_std_val /= (float)(shapes.size()*num_points * 2 - 1);
		shape_std_val = std::sqrt(shape_std_val);

		for (int i = 0; i < (int)shapes.size(); i++)
			shapes[i] /= shape_std_val;
		return shape_std_val;
	}

	void compensate_scale(
		const float shape_std_val,
		const float nu,
		std::vector<regression_tree> & forest
		) {
		for (int i = 0; i < (int)forest.size(); i++)
		{
			for (int j = 0; j < (int)forest[i].leaf_values.size(); j++)
				forest[i].leaf_values[j] *= (shape_std_val*nu);
		}
	}

	void print_parameters()
	{	
		std::cout << "num cascades : " << num_cascades << std::endl;
		std::cout << "num forests : " << num_forests << std::endl;
		std::cout << "num trees : " << num_trees << std::endl;
		std::cout << "tree depth : " << tree_depth << std::endl;
		std::cout << "nu : " << nu << std::endl;
		std::cout << "num points : " << num_points << std::endl;
		std::cout << "num split tests : " << num_split_tests << std::endl;
		std::cout << "oversampling ratio : " << oversampling_ratio << std::endl;
		std::cout << "enable_smoothing : " << enable_smoothing << std::endl;
		std::cout << "-----------------------------------------" << std::endl;
	}
};
#endif
