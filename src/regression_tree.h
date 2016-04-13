#ifndef REGRESSION_TREE_H
#define REGRESSION_TREE_H

#include <armadillo>

struct split_node
{
	int idx1;
	int idx2;

	float thres;
};

inline int left_child(int idx) { return 2 * idx + 1; };
inline int right_child(int idx) { return 2 * idx + 2; };

struct regression_tree
{
	std::vector<split_node> splits;
	std::vector<arma::fmat> leaf_values;

	inline int operator() (
		const std::vector<float> & pixel_values
		) {
		int i = 0;
		int num_split_nodes = splits.size();
		while (i < num_split_nodes)
		{
			if (pixel_values[splits[i].idx1] - pixel_values[splits[i].idx2] > splits[i].thres)
				i = left_child(i);
			else
				i = right_child(i);
		}
		return i - num_split_nodes;
	}
};
#endif
