# cGPRT
The cGPRT is an implementation of the CVPR 2015 paper, [Face Alignment Using Cascade Gaussian Process Regression Trees](http://slsp.kaist.ac.kr/paperdata/Face_Alignment_Using.pdf). It is developed by the KAIST [SLSP](http://slsp.kaist.ac.kr/xe/) Lab.

# Installation
####Dependencies
- [armadillo](http://arma.sourceforge.net/)
- [LAPACK](http://www.netlib.org/lapack/)

# Usage
####Training
```
> cGPRT_training config_file tr_img_dir tr_data_list te_img_dir te_data_list model_file
```
####Prediction
```
> cGPRT_predict model_file te_img_dir te_data_list result_file
```
# Citation
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

