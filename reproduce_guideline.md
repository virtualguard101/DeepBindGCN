# DeepBindGCN 完整复现指南

## 项目概述

**DeepBindGCN** 是一个基于图卷积网络（Graph Convolutional Network, GCN）的药物-蛋白质结合亲和力预测模型。该项目主要用于大规模药物虚拟筛选，能够高效准确地从大量小分子化合物中筛选出与特定蛋白质具有高亲和力的结合子。

### 核心特点

1. **不依赖分子对接**：相比传统方法，无需预先进行分子对接，大大提高了计算效率
2. **双分支架构**：
   - **DeepBindGCN_BC**：二元分类模型（结合/不结合）
   - **DeepBindGCN_RG**：回归模型（预测结合亲和力值）
3. **图神经网络**：使用GCN、GAT等图神经网络处理分子图和蛋白质口袋
4. **预训练分子向量**：使用预训练的分子向量表示，性能优于one-hot编码

### 应用场景

- 大规模药物虚拟筛选
- 药物重定位研究
- 蛋白质-配体结合亲和力预测
- 新药发现和优化

## 项目架构分析

### 数据流处理流程

```
输入数据 → 数据预处理 → 图构建 → 模型预测 → 结果输出
    ↓           ↓          ↓        ↓         ↓
SMILES字符串  分子图构建   GCN/GAT  神经网络   结合亲和力
PDB文件      口袋提取     特征提取  训练/预测   预测分数
```

### 模型架构

项目包含多种图神经网络模型：

1. **GCNNet**: 基础图卷积网络
2. **GATNet**: 图注意力网络
3. **GAT_GCN**: 混合图注意力-图卷积网络
4. **GINConvNet**: 图同构网络

每个模型都采用双分支架构：
- **分子分支**：处理SMILES字符串转换的分子图
- **蛋白质分支**：处理蛋白质口袋的残基信息
- **融合层**：将两个分支的特征融合进行最终预测

## 环境配置

### 方法一：使用 Conda

#### 1. 创建环境
```bash
# 创建conda环境
conda create -n DeepBindGCN python=3.7 -y
conda activate DeepBindGCN
```

#### 2. 安装依赖包
```bash
# 安装基础依赖
conda install -y -c conda-forge rdkit
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch

# 安装PyTorch Geometric相关包
pip install torch-scatter==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.4.0.html
pip install torch-sparse==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.4.0.html
pip install torch-cluster==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.4.0.html
pip install torch-spline-conv==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.4.0.html
pip install torch-geometric

# 安装其他依赖
pip install numpy pandas scikit-learn networkx
```

#### 3. 验证安装
```bash
python -c "import torch; import torch_geometric; import rdkit; print('环境配置成功!')"
```

### 方法二：使用 UV

#### 1. 安装UV（如果未安装）
```bash
pip install uv
```

#### 2. 创建项目环境
```bash
# 创建虚拟环境
uv venv deepbindgcn_env
source deepbindgcn_env/bin/activate  # Linux/Mac
# 或
deepbindgcn_env\Scripts\activate     # Windows
```

#### 3. 安装依赖
```bash
# 使用uv安装包
uv pip install torch torchvision torchaudio
uv pip install rdkit-pypi
uv pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric
uv pip install numpy pandas scikit-learn networkx
```

#### 4. 验证安装
```bash
uv run python -c "import torch; import torch_geometric; import rdkit; print('UV环境配置成功!')"
```

## 数据准备

### 1. 下载预训练模型

从GitHub releases页面下载预训练模型：

- **DeepBindGCN_BC模型**: `full_model_out2000_BC.model`
- **DeepBindGCN_RG模型**: `full_model_out2000_RG.model`

下载地址：https://github.com/haiping1010/DeepBindGCN/releases/tag/v1.0.0

将模型文件放置到对应的示例文件夹中：
```
DeepBindGCN_BC_example/full_model_out2000_BC.model
DeepBindGCN_RG_example/full_model_out2000_RG.model
```

### 2. 准备输入数据

项目需要两种主要输入：

#### 配体分子数据
- 格式：`.smi` 文件（SMILES格式）
- 示例文件：`BA_n.smi`
- 格式：每行包含SMILES字符串和分子ID，用制表符分隔

#### 蛋白质数据
- 蛋白质结构文件：`4q9v_w.pdb`
- 配体分子文件：`4q9v_ligand_n.mol2`
- 口袋文件：通过`extract_pocket.py`生成

## 完整复现流程

### DeepBindGCN_BC（二元分类）复现

#### 使用 Conda 环境
```bash
# 激活环境
conda activate DeepBindGCN
cd DeepBindGCN_BC_example

# 1. 准备配体数据
cd all_file/
bash run_all_dic.bash
cd ..

# 2. 准备蛋白质口袋
cd pocket/
python extract_pocket.py 4q9v  # 注意：使用正确的文件名
cd ..

# 3. 准备输入数据框
bash run_all_n_add.bash

# 4. 运行预测
bash run_all_predict_add.bash

# 5. 排序结果
bash score_sort_add.bash
```

#### 使用 UV 环境
```bash
# 激活环境
source deepbindgcn_env/bin/activate
cd DeepBindGCN_BC_example

# 1. 准备配体数据
cd all_file/
uv run bash run_all_dic.bash
cd ..

# 2. 准备蛋白质口袋
cd pocket/
uv run python extract_pocket.py 4q9v
cd ..

# 3. 准备输入数据框
uv run bash run_all_n_add.bash

# 4. 运行预测
uv run bash run_all_predict_add.bash

# 5. 排序结果
uv run bash score_sort_add.bash
```

### DeepBindGCN_RG（回归预测）复现

#### 使用 Conda 环境
```bash
# 激活环境
conda activate DeepBindGCN
cd DeepBindGCN_RG_example

# 执行相同的步骤1-4，最后使用不同的排序脚本
bash score_sort8.6_BA.bash
```

#### 使用 UV 环境
```bash
# 激活环境
source deepbindgcn_env/bin/activate
cd DeepBindGCN_RG_example

# 执行相同的步骤1-4，最后使用不同的排序脚本
uv run bash score_sort8.6_BA.bash
```

## 关键脚本说明

### 1. `extract_pocket.py`
**功能**：从蛋白质结构中提取结合口袋
- 计算蛋白质原子与配体原子的距离
- 提取距离小于6Å的蛋白质残基作为结合口袋
- 生成口袋文件（`4q9v_poc.pdb`）

**使用方法**：
```bash
python extract_pocket.py <filebase>
# 例如：python extract_pocket.py 4q9v
```

### 2. `read_smi_protein_nnn.py`
**功能**：处理SMILES字符串，构建分子图
- 将SMILES转换为分子图
- 提取原子特征和键信息
- 保存为numpy字典格式

### 3. `training_nn3_load_name.py`
**功能**：模型训练和预测的主要脚本
- 加载预训练模型
- 进行预测推理
- 输出预测结果

### 4. 批处理脚本
- `run_all_dic.bash`：批量处理配体分子
- `run_all_n_add.bash`：准备输入数据
- `run_all_predict_add.bash`：批量预测
- `score_sort_add.bash`：结果排序

## 输出结果解读

### 预测结果文件
- `output_*.txt`：原始预测结果
- `all_out_*.sort`：排序后的结果
- `all_out_select_*.sort`：高置信度预测结果

### 结果格式
```
预测分数,分子ID,其他信息
0.95,000L-0018,...
0.89,000L-0172,...
```

## 常见问题及解决方案

### 1. 环境配置问题

**问题**：`ModuleNotFoundError: No module named 'numpy'`
**解决方案**：
```bash
# 对于conda
conda install numpy
# 对于uv
uv pip install numpy
```

**问题**：`AttributeError: module 'numpy' has no attribute 'float'`
**解决方案**：这是numpy版本兼容性问题，需要修改代码：
```python
# 将 np.float 替换为 np.float64
a1 = a.astype(np.float64)
b1 = b.astype(np.float64)
```

### 2. 文件路径问题

**问题**：`FileNotFoundError: [Errno 2] No such file or directory`
**解决方案**：
- 确保使用正确的文件名（如`4q9v`而不是`4qv9`）
- 检查文件是否存在于正确的目录中

### 3. PyTorch Geometric问题

**问题**：`ImportError: torch_scatter not found`
**解决方案**：
```bash
# 重新安装PyTorch Geometric相关包
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric
```

### 4. 模型文件问题

**问题**：预训练模型文件缺失
**解决方案**：
- 从GitHub releases下载对应的模型文件
- 确保模型文件放置在正确的目录中

## 性能优化建议

### 1. GPU加速
```bash
# 检查CUDA是否可用
python -c "import torch; print(torch.cuda.is_available())"

# 如果可用，模型会自动使用GPU加速
```

### 2. 批处理优化
- 对于大规模虚拟筛选，建议使用批处理脚本
- 可以并行处理多个分子文件以提高效率

### 3. 内存管理
- 大规模数据集可能需要较大的内存
- 可以考虑分批处理或使用数据流式处理

## 扩展使用

### 1. 自定义数据集
- 准备自己的SMILES文件和蛋白质结构文件
- 按照相同的数据格式组织文件
- 修改脚本中的文件路径和参数

### 2. 模型调优
- 可以修改模型架构参数
- 调整训练超参数
- 使用自己的数据集进行微调

### 3. 集成到工作流
- 将DeepBindGCN集成到现有的药物发现工作流中
- 结合其他预测工具进行综合分析

## 参考文献

- 原始论文：[DeepBindGCN: Predicting Drug-Target Binding Affinity with Graph Convolutional Networks](https://github.com/haiping1010/DeepBindGCN)
- PyTorch Geometric文档：https://pytorch-geometric.readthedocs.io/
- RDKit文档：https://www.rdkit.org/docs/

## 联系信息

如有问题，请联系作者：hp.zhang@siat.ac.cn

---

**注意**：本指南基于项目原始代码和文档编写，在使用过程中如遇到问题，请参考原始GitHub仓库或联系作者。
