#include <iostream>
#include <iomanip>
#include <time.h>
#include <shape_predictor.h>
#include <shape_trainer.h>

using namespace std;
using namespace arma;

int main(int argc, char** argv)
{
	if (argc != 5)
	{
		std::cout << "Usage: " << std::endl;
		std::cout << ">> cGPRT_predict model_file image_directory image_list_file result_file" << std::endl;
		return -1;
	}

	string model_file = argv[1];		
	string image_directory = argv[2];
	string image_list_file = argv[3];
	string result_file = argv[4];

	std::cout << "model file : " << model_file << std::endl;
	std::cout << "image directory : " << image_directory << std::endl;
	std::cout << "image list file : " << image_list_file << std::endl;
	std::cout << "result file : " << result_file << std::endl;
	std::cout << "-----------------------------------------" << std::endl;

	std::vector<std::string> image_names;
	std::vector<arma::fvec> rects;
	std::vector<arma::fmat> shapes;
		
	load_data(image_list_file, image_names, rects, shapes);

	shape_predictor sp(model_file);
	
	int num_data = (int)shapes.size();
	std::vector<arma::fmat> shapes_pred(num_data);
	float pred_times = 0.0f;
	float errs = 0.0f;
	for (int i = 0; i < num_data; i++)
	{	
		arma::fmat image;
		image.load(image_directory + "/" + image_names[i], arma::pgm_binary);
		arma::fvec rect = rects[i];
		arma::fmat shape_pred;

		clock_t start = clock();
		shape_pred = sp(image, rect);
		clock_t end = clock();
		
		float iod = inter_occular_distance(shapes[i], sp.left_eye_idx, sp.right_eye_idx);
		float err = compute_error(shapes[i], shape_pred, iod);
		errs += err;
		float pred_time = (double)(end - start) / CLOCKS_PER_SEC;
		pred_times += pred_time;
		shapes_pred[i] = shape_pred;

		if (i && !((i)%10))
			std::cout << "[" << i << "/" << num_data << "] err/fps : " << errs / (float)(i + 1) << " " << 1.0f / (pred_times / (float)(i + 1)) << std::endl;
	}
	std::cout << "[" << num_data << "/" << num_data << "] err/fps : " << errs / (float)num_data << " " << 1.0f / (pred_times / (float)num_data) << std::endl;

	save_data(image_names, rects, shapes_pred, result_file);

	return 0;
}
