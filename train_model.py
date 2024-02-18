from functions import save_model, log_to_file ,load_config, VoxelTripletDataset, Siamese3DNetwork, train_epoch, validate_epoch
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import random_split

def main():
    config_path = 'config.yml'
    config = load_config(config_path)

    input_file_path = config['train_model']['train_path']
    grid_size = config['train_model']['grid_size']
    log_path = config['train_model']['log_path']
    batch_size = config['train_model']['batch_size']
    num_epochs = config['train_model']['num_epochs']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = VoxelTripletDataset(root_dir=input_file_path, grid_size=grid_size)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    model = Siamese3DNetwork(grid_size)

    triplet_loss = nn.TripletMarginLoss(margin=9.0, p=2)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.to(device)

    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, triplet_loss, optimizer, device)
        val_loss, val_accuracy = validate_epoch(model, val_loader, triplet_loss, device)
        
        print(f"Epoch {epoch+1}, Train Loss: {train_loss}, Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}%")
        log_to_file(log_path, epoch+1, train_loss, val_loss, val_accuracy)
        
        save_model(model, epoch+1)

if __name__ == "__main__":
    main()