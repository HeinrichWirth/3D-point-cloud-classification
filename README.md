# Point Cloud Classification with a Siamese 3D Convolutional Neural Network

## Description
This project focuses on processing point clouds in LAS format to convert them into voxel representations and subsequent classification using a Siamese neural network with 3D convolutions. The challenge of this problem lies in the small amount of data available for each class. The project allows for the automatic determination of the class to which each point cloud file belongs, simplifying work with large volumes of data.

The training dataset had 3 classes: bridge, wire, tree. There were 8 files for training, 2 files for validation, and 3 files for testing. The results show that the model makes no mistakes on the test set and operates without errors during use. 

## Getting Started
### Prerequisites
1. Python == 3.9
2. LasPy == 2.5
3. PyTorch == 2.2
4. CUDA (если необхоидмо использование GPU)
5. Numpy == 1.24
6. Open3D == 0.18
7. PyYaml == 6.0

### Installation
1. Clone the repository using Git:

```git clone https://github.com/HeinrichWirth/Classification-3D-point-cloud```

2. Install the necessary dependencies:

    2.1. ```pip install -r requirements.txt```

    2.2. Install PyTorch from the official site and a compatible version of CUDA (if necessary)

## Usage
- save_las_as_voxels.py - saving point clouds in voxel form and ply format

    grid_size: The grid size for voxelization.
    files_path: Directory containing the LAS files to be processed.
    output_path: Directory where the voxelized outputs will be saved.
    train_model: Settings for training the Siamese network.

- train_model.py - training the Siamese neural network
  
    batch_size: The size of batches for training.
    TripletMarginLoss_margin: The margin parameter for the TripletMarginLoss.
    TripletMarginLoss_p: The p parameter for the TripletMarginLoss.
    lr: Learning rate for the optimizer.
    num_epochs: Number of epochs for training.
    train_path: Path to the training dataset.
    log_path: File path for saving training logs.
    grid_size: The grid size used for voxelization in training.

- classify.py - classification using the trained neural network
  
    input_file_path: Directory containing the reference vectors.
    grid_size: The grid size used for voxelization in classification.
    model_weights_path: Path to the saved model weights.
    test_las_path: Path to the LAS file for classification.

- config.yaml - configuration file

## Screenshots and Videos

### Example bridge point cloud
![images](https://github.com/HeinrichWirth/Classification-3D-point-cloud/blob/main/images/bridge_point.png "Bridge point cloud")

### Transformed point cloud to voxels
![images](https://github.com/HeinrichWirth/Classification-3D-point-cloud/blob/main/images/bridge_voxel.png "Bridge voxels")


