# Reproduce Record

## Environment Configuration

- use `uv` to manage the environment

    ```bash
    uv venv --python=3.10
    uv init
    ```

    ```bash
    uv add torch torchvision torchaudio
    uv add rdkit-pypi
    uv add numpy pandas scikit-learn networkx matplotlib
    uv add torch-geometric
    ```

## Download the pre-trained model

```bash
cd DeepBindGCN_BC_example
wget https://github.com/haiping1010/DeepBindGCN/releases/download/v1.0.0/full_model_out2000_BC.model
cd ..
```

```bash
cd DeepBindGCN_RG_example
wget https://github.com/haiping1010/DeepBindGCN/releases/download/v1.0.0/full_model_out2000_RG.model
cd ..
```

## Pre-process the data

### BC

```bash
cd DeepBindGCN_BC_example
```

```bash
cd all_file
bash run_all_dic.bash
cd ..
```

```bash
cd pocket
uv run extract_pocket.py 4q9v
cd ..
```

```bash
cd all_file
bash run_all_n_add.bash
cd ..
```

While running `run_all_n_add.bash`, there was some issue in the original bash script.

