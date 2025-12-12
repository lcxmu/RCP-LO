

# RCP-LO: A Relative Coordinate Prediction Framework for Generalizable Deep LiDAR Odometry  
**(AAAI 2026)**


## Environment
- Python 3.8.10  
- PyTorch 1.12.1  
- CUDA 11.3  

Create the environment:
```
conda env create -f environment.yml
conda activate RCP_LO
```
##  Install PointNet2 library

Compile the furthest point sampling, grouping and gathering operation for PyTorch with following commands.
```
cd pointnet2
python setup.py install
```
## Datasets

Datasets are available at [KITTI Odometry Dataset](https://www.cvlibs.net/datasets/kitti/eval_odometry.php), [KITTI-360](https://www.cvlibs.net/datasets/kitti-360/), [Ford Campus Vision and Lidar Data Set](https://robots.engin.umich.edu/SoftwareData/InfoFord), [Oxford Radar RobotCar](https://robotcar-dataset.robots.ox.ac.uk/).

## Run

### train  
```
python train.py configs/config.yaml
```
### test  
```
python test.py configs/config.yaml
```
Configure parameters such as the GPU device, pretrained checkpoints, and dataset paths in the config.yaml file.

# Acknowledgments

We gratefully acknowledge the open-source implementations of [PointNet++](https://github.com/charlesq34/pointnet2), [PointConv](https://github.com/DylanWusee/pointconv), [DifFlow3D](https://github.com/IRMVLab/DifFlow3D) and the [KITTI Odometry Evaluation Tool](https://github.com/LeoQLi/KITTI_odometry_evaluation_tool).

# Citation
......
