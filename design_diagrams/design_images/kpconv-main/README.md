# KPConv Implementation Guide

Welcome to the KPConv implementation guide. This document provides detailed instructions on how to set up and use the KPConv (Kernel Point Convolution) neural network for point cloud semantic segmentation, specifically configured for the S3DIS dataset. The KPConv framework is designed for high performance and accuracy in 3D point cloud processing.

## Configuration

Before diving into the training and testing processes, it's crucial to set up your environment and dataset correctly. The configuration for dataset paths and other important parameters is handled through a YAML file. Below is an example structure of the required YAML configuration:

```yaml
dataset_path_train: ../../data/s3dis_las_train_test_val/train
dataset_path_validation: ../../data/s3dis_las_train_test_val/val
dataset_path_test: ../../data/s3dis_las_train_test_val/test

path_output: ../../data/s3dis_las_results_kpconv
saving_path: ../../data/s3dis_las_saved_kpconv

ignored_labels: [] 
label_to_names: 
  0: 'ceiling'
  1: 'floor'
  2: 'wall'
  3: 'beam'
  4: 'column'
  5: 'window'
  6: 'door'
  7: 'chair'
  8: 'table'
  9: 'bookcase'
  10: 'sofa'
  11: 'board'
  12: 'clutter'
```

Ensure this file is correctly filled out and located in an accessible path for the KPConv scripts.

## Environment Setup

1. **Conda Environment**: It's recommended to use a Conda environment for KPConv. Create your environment using the provided `env.yml` file:

   ```bash
   conda env create -f env.yml
   ```

   Activate the new environment:

   ```bash
   conda activate kpconv_env
   ```

2. **C++ Extensions**: KPConv requires the compilation of C++ extensions for efficient neighbor search and subsampling. Execute the following commands to compile these extensions:

   ```bash
   cd cpp_wrappers/cpp_neighbors/
   python setup.py build_ext --inplace
   cd ../../cpp_wrappers/cpp_subsampling/
   python setup.py build_ext --inplace
   ```

   Ensure you are in the root directory of the KPConv implementation before running these commands.

## Training

To train the KPConv model on the S3DIS dataset, use the `train_Standard.py` script. Make sure the YAML configuration file is correctly pointed to the dataset and output paths. Run the training process by executing:

```bash
python train_Standard.py  --config_path path/to/your/config.yaml
```

Replace `path/to/your/config.yaml` with the actual path to your configuration file.

## Testing

After training, you can test or infer using the `test_models_standard.py` script, which also relies on the same YAML configuration file for dataset paths. Execute the testing process with:

```bash
python test_models_standard.py  --config_path path/to/your/config.yaml
```

Ensure the `path_output` and `saving_path` in the YAML file are set to the directories where you want to save the results and trained models, respectively.

## Results

The results, including semantic segmentation predictions and model checkpoints, will be saved to the paths specified in the YAML configuration under `path_output` and `saving_path`.

## Conclusion

Following this guide will help you set up and use KPConv for semantic segmentation on point cloud data effectively. Remember to adjust paths in the YAML configuration and command lines according to your directory structure and requirements. Happy computing!