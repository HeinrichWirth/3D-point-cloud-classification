from functions import read_las_file, voxelize, create_voxel_cubes, save_as_ply, load_config
import os

def main():
    config_path = 'config.yml'
    config = load_config(config_path)

    input_file_path = config['save_las_as_voxels']['files_path']
    output_file_path = config['save_las_as_voxels']['output_path']
    grid_size = config['save_las_as_voxels']['grid_size']

    files = [f for f in os.listdir(input_file_path) if f.endswith('.las')]

    for i in files:
        points = read_las_file(os.path.join(input_file_path, i))
        voxels = voxelize(points, grid_size=grid_size)
        cubes = create_voxel_cubes(voxels)

        base = os.path.splitext(i)[0]+'.ply'
        output_path = os.path.join(output_file_path, base)

        save_as_ply(cubes, output_path)

if __name__ == "__main__":
    main()