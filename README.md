# Sparse Adversarial Video Attack with Spatial Transformation


The code is tested on the tensorfow >= 1.3.0
## Prepare data
UCF101 can be downloaded and extracted following the instructions in https://github.com/harvitronix/five-video-classification-methods
HMDB51 can be downloaded as RGB images in https://github.com/feichtenhofer/twostreamfusion
UCF101 data can be storded under UCF101/video_data/test 
HMDB51 data can be storded under HMDB51/video_data/test

python test_gen.py -i video_data -o output/I3D --model i3d_inception --dataset UCF101 --file_list /video_data/batch_test/test_saved.csv

The generated adversarial videos will be stored in the folder "output/I3D"


