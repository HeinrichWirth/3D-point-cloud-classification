from functions import predict_class, load_config, Siamese3DNetwork, get_reference_vectors
import torch

def main():
    config_path = 'config.yml'
    config = load_config(config_path)

    input_file_path = config['classify']['input_file_path']
    grid_size = config['classify']['grid_size']
    model_weights_path = config['classify']['model_weights_path']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = Siamese3DNetwork(grid_size)
    model.load_state_dict(torch.load(model_weights_path))
    model.to(device)

    reference_vectors = get_reference_vectors(model, input_file_path, grid_size, device)

    test_las_path = config['classify']['test_las_path']
    predicted_class = predict_class(model, test_las_path, reference_vectors, grid_size, device)
    print(f"Predicted class for the test LAS file: {predicted_class}")

if __name__ == "__main__":
    main()