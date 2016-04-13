#ifndef SHAPE_PREDICTOR_H
#define SHAPE_PREDICTOR_H

#define ARMA_NO_DEBUG

#include <armadillo>
#include <regression_tree.h>
#include <shape_utils.h>

class shape_predictor
{
private:

	std::vector<std::vector<regression_tree> > cascades;
	arma::fmat mean_shape;
	std::vector<arma::fvec> deltas;

public:

	std::vector<int> left_eye_idx;
	std::vector<int> right_eye_idx;
 		
	shape_predictor () {}

	shape_predictor(
		const std::vector<std::vector<regression_tree> > & cascades_,
		const arma::fmat & mean_shape_,
		const std::vector<arma::fvec> & deltas_,
		const std::vector<int> & left_eye_idx_,
		const std::vector<int> & right_eye_idx_
		) : cascades(cascades_), mean_shape(mean_shape_), deltas(deltas_), left_eye_idx(left_eye_idx_), right_eye_idx(right_eye_idx_) {}

	shape_predictor(const std::string & model_file_name) { load(model_file_name); }

	arma::fmat operator() (
		const arma::fmat & image,
		const arma::fvec & rect
		) {

		arma::fmat shape_pred = mean_shape;
		arma::fmat tform, tform_inv;
		std::vector<float> pixel_values;
		std::vector<arma::fmat> image_vec;
		
		for (int i = 0; i < (int)cascades.size(); i++)
		{
			find_similarity_tform(shape_pred, mean_shape, tform, tform_inv);
			pixel_values = extract_pixel_values(image, rect, shape_pred, tform_inv, deltas);
			shape_pred = tform*shape_pred;			
			for (int j = 0; j < (int)cascades[i].size(); j++)
				shape_pred += cascades[i][j].leaf_values[cascades[i][j](pixel_values)];
			shape_pred = tform_inv*shape_pred;
		}
		return unnormalizing_shape(rect, shape_pred);
	}

	void load(
		const std::string & model_file_name
		) {
		std::ifstream fin;
		fin.open(model_file_name, std::ios::binary);

		mean_shape.load(fin, arma::arma_binary);
		int num_patterns;
		fin.read(reinterpret_cast<char*>(&num_patterns), sizeof(num_patterns));
		deltas.resize(num_patterns);
		for (int i = 0; i < num_patterns; i++)
			deltas[i].load(fin, arma::arma_binary);

		int num_cascades;
		fin.read(reinterpret_cast<char*>(&num_cascades), sizeof(num_cascades));
		cascades.resize(num_cascades);
		for (int i = 0; i < num_cascades; i++)
		{
			int num_trees;
			fin.read(reinterpret_cast<char*>(&num_trees), sizeof(num_trees));
			cascades[i].resize(num_trees);
			for (int j = 0; j < num_trees; j++)
			{
				int num_split_nodes;
				fin.read(reinterpret_cast<char*>(&num_split_nodes), sizeof(num_split_nodes));
				cascades[i][j].splits.resize(num_split_nodes);
				for (int k = 0; k < num_split_nodes; k++)
				{
					fin.read(reinterpret_cast<char*>(&cascades[i][j].splits[k].idx1), sizeof(cascades[i][j].splits[k].idx1));
					fin.read(reinterpret_cast<char*>(&cascades[i][j].splits[k].idx2), sizeof(cascades[i][j].splits[k].idx2));
					fin.read(reinterpret_cast<char*>(&cascades[i][j].splits[k].thres), sizeof(cascades[i][j].splits[k].thres));
				}
				int num_leaf_nodes;
				fin.read(reinterpret_cast<char*>(&num_leaf_nodes), sizeof(num_leaf_nodes));
				cascades[i][j].leaf_values.resize(num_leaf_nodes);
				for (int k = 0; k < num_leaf_nodes; k++)
					cascades[i][j].leaf_values[k].load(fin, arma::arma_binary);
			}
		}

		int num_left_eye_idx;
		fin.read(reinterpret_cast<char*>(&num_left_eye_idx), sizeof(num_left_eye_idx));
		left_eye_idx.resize(num_left_eye_idx);
		for (int i = 0; i < num_left_eye_idx; i++)
			fin.read(reinterpret_cast<char*>(&left_eye_idx[i]), sizeof(left_eye_idx[i]));
		int num_right_idx;
		fin.read(reinterpret_cast<char*>(&num_right_idx), sizeof(num_right_idx));
		right_eye_idx.resize(num_right_idx);
		for (int i = 0; i < num_right_idx; i++)
			fin.read(reinterpret_cast<char*>(&right_eye_idx[i]), sizeof(right_eye_idx[i]));
		fin.close();
	}

	void save(
		const std::string & model_file_name
		) {
		std::ofstream fout;
		fout.open(model_file_name, std::ios::binary);
		
		mean_shape.save(fout, arma::arma_binary); 
		int num_patterns = deltas.size();
		fout.write(reinterpret_cast<char*>(&num_patterns), sizeof(num_patterns));
		for (int i = 0; i < num_patterns; i++)
			deltas[i].save(fout, arma::arma_binary);
				
		int num_cascades = (int)cascades.size();
		fout.write(reinterpret_cast<char*>(&num_cascades), sizeof(num_cascades));
		
		for (int i = 0; i < num_cascades; i++)
		{
			int num_trees = (int)cascades[i].size();
			fout.write(reinterpret_cast<char*>(&num_trees), sizeof(num_trees));
			for (int j = 0; j < num_trees; j++)
			{
				int num_split_nodes = (int)cascades[i][j].splits.size();
				fout.write(reinterpret_cast<char*>(&num_split_nodes), sizeof(num_split_nodes));
				for (int k = 0; k < num_split_nodes; k++)
				{
					fout.write(reinterpret_cast<char*>(&cascades[i][j].splits[k].idx1), sizeof(cascades[i][j].splits[k].idx1));
					fout.write(reinterpret_cast<char*>(&cascades[i][j].splits[k].idx2), sizeof(cascades[i][j].splits[k].idx2));
					fout.write(reinterpret_cast<char*>(&cascades[i][j].splits[k].thres), sizeof(cascades[i][j].splits[k].thres));
				}					
				int num_leaf_nodes = (int)cascades[i][j].leaf_values.size();
				fout.write(reinterpret_cast<char*>(&num_leaf_nodes), sizeof(num_leaf_nodes));
				for (int k = 0; k < num_leaf_nodes; k++)
					cascades[i][j].leaf_values[k].save(fout, arma::arma_binary);
			}
		}

		int num_left_eye_idx = (int)left_eye_idx.size();
		fout.write(reinterpret_cast<char*>(&num_left_eye_idx), sizeof(num_left_eye_idx));
		for (int i = 0; i < num_left_eye_idx; i++)
			fout.write(reinterpret_cast<char*>(&left_eye_idx[i]), sizeof(left_eye_idx[i]));
		int num_right_eye_idx = (int)right_eye_idx.size();
		fout.write(reinterpret_cast<char*>(&num_right_eye_idx), sizeof(num_right_eye_idx));
		for (int i = 0; i < num_right_eye_idx; i++)
			fout.write(reinterpret_cast<char*>(&right_eye_idx[i]), sizeof(right_eye_idx[i]));

		fout.close();
	}
};
#endif
