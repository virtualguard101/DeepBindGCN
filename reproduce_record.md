# Reproduce Record

## 2025-10-20

### Environment Configuration

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

### Download the pre-trained model

```bash
wget https://github.com/haiping1010/DeepBindGCN/releases/download/v1.0.0/full_model_out2000_BC.model
wget https://github.com/haiping1010/DeepBindGCN/releases/download/v1.0.0/full_model_out2000_RG.model
```

### Pre-process the data

#### BC

```bash
cd all_file
uv run bash run_all_dic.bash
cd ..
```
