#include <iostream>
#include <shape_predictor.h>
#include <shape_trainer.h>

using namespace std;
using namespace arma;

int main(int argc, char** argv)
{
	if (argc != 7)
	{
		std::cout << "Usage: " << std::endl;
		std::cout << ">> cGPRT_training model_config_file image_directory_train image_list_file_train image_directory_test image_list_file_test model_file_name" << std::endl;
		return -1;
	}

	string model_config_file = argv[1];
	string image_directory_train = argv[2];
	string image_list_file_train = argv[3];
	string image_directory_test = argv[4];
	string image_list_file_test = argv[5];
	string model_file_name = argv[6];

	std::cout << "model config file : " << model_config_file << std::endl;
	std::cout << "train image directory : " << image_directory_train << std::endl;
	std::cout << "train image list file : " << image_list_file_train << std::endl;
	std::cout << "test image directory : " << image_directory_test << std::endl;
	std::cout << "test image list file : " << image_list_file_test << std::endl;
	std::cout << "output model file : " << model_file_name << std::endl;
	std::cout << "-----------------------------------------" << std::endl;

	std::vector<std::string> image_names_train, image_names_test;
	std::vector<arma::fvec> rects_train, rects_test;
	std::vector<arma::fmat> shapes_train, shapes_test;
		
	load_data(image_list_file_train, image_names_train, rects_train, shapes_train);
	load_data(image_list_file_test, image_names_test, rects_test, shapes_test);

	shape_trainer trainer(model_config_file);

	shape_predictor sp = trainer.train(image_directory_train, image_names_train, rects_train, shapes_train, image_directory_test, image_names_test, rects_test, shapes_test);
	
	sp.save(model_file_name);

	return 0;
}
