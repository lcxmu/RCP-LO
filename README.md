# RCP-LO
RCP-LO: A Relative Coordinate Prediction Framework for Generalizable Deep LiDAR Odometry (AAAI 2026)

**Installation**

**Install the pointnet2 library**
Compile the furthest point sampling, grouping and gathering operation for PyTorch with following commands.

  cd pointnet2
  python setup.py install

**Datasets**

**Training**
Train the network by running :
python  train.py configs/config.yaml
Please reminder to specify the GPU, data_root,log_dir, test_list(sequences for testing) in the scripts.
**Testing**

**Acknowledgments**

**Citation**

