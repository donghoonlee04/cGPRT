#cGPRT
The cGPRT is an implementation of the CVPR 2015 paper, [Face Alignment Using Cascade Gaussian Process Regression Trees](http://slsp.kaist.ac.kr/paperdata/Face_Alignment_Using.pdf). It is developed by the KAIST [SLSP](http://slsp.kaist.ac.kr/xe/) Lab.

#Installation
####Dependencies
- [armadillo](http://arma.sourceforge.net/)
- [LAPACK](http://www.netlib.org/lapack/)

####Cropped data and pre-learned models
- [cropped data](http://143.248.157.13:8080/sharing/ltn8yZbU5) - original data can be found [here](http://ibug.doc.ic.ac.uk/resources/300-W/)
- [configuration files]()
- [pre-learned models]()
- [predicted landmarks]()

#Usage
####Data format
- data_list example
```
689                                     // # data
68                                      // # landmark points
ibug-cropped/ibug_crop_test/im0001.pgm  // image file path
0 0 600 600                             // x_start y_start height width
212.138741 272.980541                   // x_position y_position (landmark 0)
207.482508 312.712860                   // x_position y_position (landmark 1)
...
```
- config_file example
```
num_cascades 10                     // # cascades
num_forests 10                      // # forests
num_trees 10                        // # trees
tree_depth 5                        // tree depth
nu 0.3                              // shrinkage parameter
oversampling_ratio 20               // training data oversampling ratio
num_points 68                       // # landmark points
left_eye_idx ( 36 37 38 39 40 41 )  // left_eye_center = average of left_eye_idx landmarks
right_eye_idx ( 42 43 44 45 46 47 ) // left_eye_center = average of left_eye_idx landmarks
num_split_tests 200                 // # split tests for each split nodes
random_seed 1                       // random seed for random number generation
```
####Training
```
> cGPRT_training config_file tr_img_dir tr_data_list te_img_dir te_data_list model_file
```
####Prediction
```
> cGPRT_predict model_file te_img_dir te_data_list result_file
```
#Citation
Please cite the following [paper](http://slsp.kaist.ac.kr/paperdata/Face_Alignment_Using.pdf) in your publications if it helps your research:
```
@inproceedings{lee2015face,
  title={Face alignment using cascade gaussian process regression trees},
  author={Lee, Donghoon and Park, Hyunsin and Yoo, Chang D},
  booktitle={Computer Vision and Pattern Recognition (CVPR), 2015 IEEE Conference on},
  pages={4204--4212},
  year={2015},
  organization={IEEE}
}
```


