import yaml
import laspy
import numpy as np
import open3d as o3d
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import random

def read_las_file(file_path):
    """
    Reads a LAS file and returns the coordinates of points.

    Args:
        file_path (str): The path to the LAS file.

    Returns:
        numpy.ndarray: An array of point coordinates, where each row represents [x, y, z].
    """

    with laspy.open(file_path) as file:
        las = file.read()
        points = np.vstack((las.x, las.y, las.z)).transpose()
    return points

def normalize_points(points):
    """
    Normalizes point coordinates to a cubic space of 1x1x1.

    Args:
        points (numpy.ndarray): An array of point coordinates to be normalized.

    Returns:
        numpy.ndarray: Normalized point coordinates within a unit cube.
    """

    min_point = np.min(points, axis=0)
    max_point = np.max(points, axis=0)
    normalized_points = (points - min_point) / (max_point - min_point)
    return normalized_points

def voxelize(points, grid_size):
    """
    Converts points into a voxel representation based on the specified grid size.

    Args:
        points (numpy.ndarray): An array of point coordinates.
        grid_size (int): The size of the grid in each dimension.

    Returns:
        numpy.ndarray: A unique set of voxel indices representing the occupied voxels.
    """

    normalized_points = normalize_points(points)


    indices = np.floor(normalized_points * grid_size).astype(int)
    indices = np.clip(indices, 0, grid_size - 1)  

    unique_voxels = np.unique(indices, axis=0)
    return unique_voxels

def create_voxel_cubes(voxels, voxel_size=1):
    """
    Creates small cubes for each voxel.

    Args:
        voxels (numpy.ndarray): An array of voxel indices.
        voxel_size (float, optional): The size of each voxel cube. Defaults to 1.

    Returns:
        list: A list of open3d.geometry.TriangleMesh objects representing voxel cubes.
    """

    cubes = []
    for voxel in voxels:
        center = voxel * voxel_size
        cube = o3d.geometry.TriangleMesh.create_box(width=voxel_size,
                                                    height=voxel_size,
                                                    depth=voxel_size)
        cube.translate(center - np.array([voxel_size/2, voxel_size/2, voxel_size/2]))
        cubes.append(cube)
    return cubes

def save_as_ply(cubes, file_name):
    """
    Saves the voxel cubes as a PLY file.

    Args:
        cubes (list): A list of open3d.geometry.TriangleMesh objects representing voxel cubes.
        file_name (str): The output PLY file name.

    Returns:
        None
    """

    mesh = o3d.geometry.TriangleMesh()
    for cube in cubes:
        mesh += cube
    o3d.io.write_triangle_mesh(file_name, mesh)

def convert_to_voxel_grid(voxels, grid_size):
    """
    Converts a list of voxels into a 3D voxel grid.

    Args:
        voxels (numpy.ndarray): An array of voxel indices.
        grid_size (int): The size of the grid in each dimension.

    Returns:
        numpy.ndarray: A 3D array representing the voxel grid, where 1 indicates an occupied voxel.
    """

    voxel_grid = np.zeros((grid_size, grid_size, grid_size))
    for x, y, z in voxels:
        voxel_grid[x, y, z] = 1 
    return voxel_grid

def load_config(config_path):
    """
    Loads a configuration file.

    Args:
        config_path (str): The path to the configuration file.

    Returns:
        dict: A dictionary containing the configuration settings.
    """

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

class VoxelTripletDataset(Dataset):
    """
    A dataset class for creating triplets from a LAS file dataset for training a Siamese network.

    Attributes:
        root_dir (str): The root directory containing class folders with LAS files.
        grid_size (int, optional): The size of the grid for voxelization. Defaults to 100.
        transform (callable, optional): Optional transform to be applied on a sample.

    Methods:
        read_las_to_voxel(file_path): Reads a LAS file and converts it into a voxel grid tensor.
        __getitem__(idx): Returns a triplet of anchor, positive, and negative samples.
        __len__(): Returns the total number of samples in the dataset.
    """

    def __init__(self, root_dir, grid_size=100, transform=None):
        """
        Initializes the dataset with the directory of LAS files, grid size, and an optional transform.
        """

        self.root_dir = root_dir
        self.transform = transform
        self.grid_size = grid_size
        self.classes = os.listdir(root_dir)
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.files = []
        for cls in self.classes:
            cls_path = os.path.join(root_dir, cls)
            for file in os.listdir(cls_path):
                file_path = os.path.join(cls_path, file)
                self.files.append((file_path, self.class_to_idx[cls]))

    def read_las_to_voxel(self, file_path):
        """
        Reads a LAS file, converts it to voxels, and then into a 3D voxel grid tensor.

        Args:
            file_path (str): Path to the LAS file.

        Returns:
            torch.Tensor: A float tensor representation of the voxel grid.
        """

        with laspy.open(file_path) as file:
            las = file.read()
            points = np.vstack((las.x, las.y, las.z)).transpose()
            voxels = voxelize(points, grid_size=self.grid_size)
            voxel_grid = convert_to_voxel_grid(voxels, self.grid_size)
            voxel_grid = np.expand_dims(voxel_grid,axis=0)
            return torch.from_numpy(voxel_grid).float()

    def __getitem__(self, idx):
        """
        Returns a triplet (anchor, positive, negative) for training the Siamese network.

        Args:
            idx (int): The index of the sample.

        Returns:
            tuple: A tuple containing tensors for the anchor, positive, and negative samples.
        """

        anchor_path, class_idx = self.files[idx]
        positive_list = [file for file, cls_idx in self.files if cls_idx == class_idx and file != anchor_path]
        negative_list = [file for file, cls_idx in self.files if cls_idx != class_idx]
        positive_path = random.choice(positive_list)
        negative_path = random.choice(negative_list)

        anchor_voxel = self.read_las_to_voxel(anchor_path)
        positive_voxel = self.read_las_to_voxel(positive_path)
        negative_voxel = self.read_las_to_voxel(negative_path)

        if self.transform:
            anchor_voxel = self.transform(anchor_voxel)
            positive_voxel = self.transform(positive_voxel)
            negative_voxel = self.transform(negative_voxel)

        return anchor_voxel, positive_voxel, negative_voxel

    def __len__(self):
        """
        Returns the total number of samples available in the dataset.

        Returns:
            int: The total number of samples.
        """

        return len(self.files)
    
class Siamese3DNetwork(nn.Module):
    """
    A Siamese neural network model for 3D data using convolutional layers.

    Attributes:
        grid_size (int): The grid size used for voxelization which impacts the input size of the network.

    Methods:
        forward(x): Defines the forward pass of the model.
    """

    def __init__(self, grid_size):
        """
        Initializes the network with a specified grid size for voxel inputs.
        """

        super(Siamese3DNetwork, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            
            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            
            nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )

        self.flattened_size = 128 * (grid_size // 8) ** 3

        self.fc_layers = nn.Sequential(
            nn.Linear(self.flattened_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )

    def forward(self, x):
        """
        Performs the forward pass of the model.

        Args:
            x (torch.Tensor): The input tensor containing voxelized data.

        Returns:
            torch.Tensor: The output tensor after passing through the network.
        """

        x = self.conv_layers(x)
        x = x.view(-1, self.flattened_size) 
        x = self.fc_layers(x)
        return x
    
def get_reference_vectors(model, root_dir, grid_size, device):
    """
    Computes reference vectors for each class in the dataset by averaging feature vectors.

    Args:
        model (torch.nn.Module): The trained Siamese network model.
        root_dir (str): The root directory containing class folders.
        grid_size (int): The size of the grid used for voxelization.
        device (str or torch.device): The device on which to perform the computation.

    Returns:
        dict: A dictionary with class names as keys and averaged feature vectors as values.
    """

    model.eval() 
    classes = os.listdir(root_dir)
    reference_vectors = {}

    for cls in classes:
        cls_path = os.path.join(root_dir, cls)
        vectors = []

        for file in os.listdir(cls_path):
            file_path = os.path.join(cls_path, file)
            voxel = read_las_to_voxel(file_path, grid_size).to(device)

            with torch.no_grad():
                vector = model(voxel).cpu().numpy()
                vectors.append(vector)

        reference_vectors[cls] = np.mean(vectors, axis=0)

    return reference_vectors

def read_las_to_voxel(file_path, grid_size):
    """
    Reads a LAS file and converts it into a voxel grid tensor.

    Args:
        file_path (str): The path to the LAS file.
        grid_size (int): The size of the grid for voxelization.

    Returns:
        torch.Tensor: A float tensor representation of the voxel grid.
    """

    with laspy.open(file_path) as file:
        las = file.read()
        points = np.vstack((las.x, las.y, las.z)).transpose()
        voxels = voxelize(points, grid_size=grid_size)
        voxel_grid = convert_to_voxel_grid(voxels, grid_size)
        voxel_grid = np.expand_dims(voxel_grid, axis=0)
        return torch.from_numpy(voxel_grid).float()
    
def train_epoch(model, dataloader, loss_fn, optimizer, device):
    """
    Trains the model for one epoch.

    Args:
        model (torch.nn.Module): The model to be trained.
        dataloader (DataLoader): DataLoader for the training data.
        loss_fn (callable): The loss function.
        optimizer (torch.optim.Optimizer): The optimizer for the model.
        device (torch.device): The device to train on.

    Returns:
        float: The average loss for the epoch.
    """

    model.train()
    running_loss = 0.0

    for data in dataloader:
        anchor, positive, negative = [d.to(device) for d in data]

        optimizer.zero_grad()

        anchor_output = model(anchor)
        positive_output = model(positive)
        negative_output = model(negative)

        loss = loss_fn(anchor_output, positive_output, negative_output)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(dataloader)

def validate_epoch(model, dataloader, loss_fn, device):
    """
    Validates the model on the validation set.

    Args:
        model (torch.nn.Module): The model to be validated.
        dataloader (DataLoader): DataLoader for the validation data.
        loss_fn (callable): The loss function.
        device (torch.device): The device to validate on.

    Returns:
        tuple: The average loss and accuracy for the validation set.
    """

    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data in dataloader:
            anchor, positive, negative = [d.to(device) for d in data]
            anchor_output = model(anchor)
            positive_output = model(positive)
            negative_output = model(negative)

            loss = loss_fn(anchor_output, positive_output, negative_output)
            running_loss += loss.item()

            total += anchor.size(0)
            correct += (torch.norm(anchor_output - positive_output, dim=1) < 
                        torch.norm(anchor_output - negative_output, dim=1)).sum().item()

    accuracy = 100 * correct / total
    return running_loss / len(dataloader), accuracy

def save_model(model, epoch, save_path="model_checkpoint"):
    """
    Saves the model state to a file.

    Args:
        model (torch.nn.Module): The model to save.
        epoch (int): The current epoch number.
        save_path (str, optional): Directory to save the model. Defaults to "model_checkpoint".

    Returns:
        None
    """

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save(model.state_dict(), os.path.join(save_path, f"model_epoch_{epoch}.pth"))

def log_to_file(log_path, epoch, train_loss, val_loss, val_accuracy):
    """
    Logs training and validation results to a file.

    Args:
        log_path (str): Path to the log file.
        epoch (int): The current epoch number.
        train_loss (float): Training loss for the epoch.
        val_loss (float): Validation loss for the epoch.
        val_accuracy (float): Validation accuracy for the epoch.

    Returns:
        None
    """

    with open(log_path, "a") as log_file:
        log_file.write(f"Epoch {epoch}, Train Loss: {train_loss}, Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}%\n")

def predict_class(model, test_las_path, reference_vectors, grid_size, device):
    """
    Predicts the class of a given LAS file using the trained model and reference vectors.

    Args:
        model (torch.nn.Module): The trained model.
        test_las_path (str): Path to the LAS file to classify.
        reference_vectors (dict): Dictionary of class names to reference vectors.
        grid_size (int): The grid size used for voxelization.
        device (torch.device): The device to perform prediction on.

    Returns:
        str: The predicted class name.
    """
    
    model.eval()
    test_voxel = read_las_to_voxel(test_las_path, grid_size).to(device)

    with torch.no_grad():
        test_vector = model(test_voxel).cpu().numpy()

    closest_class = None
    closest_dist = float('inf')

    for cls, ref_vector in reference_vectors.items():
        dist = np.linalg.norm(test_vector - ref_vector)
        if dist < closest_dist:
            closest_dist = dist
            closest_class = cls

    return closest_class