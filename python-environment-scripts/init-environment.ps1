conda create -n antifraud python=3.10 -y
conda activate antifraud

pip install torch==2.3.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

pip install dgl -f https://data.dgl.ai/wheels/cu121/repo.html

pip install dglgo -f https://data.dgl.ai/wheels-test/repo.html

pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
    -f https://data.pyg.org/whl/torch-2.3.0+cu121.html
pip install torch-geometric

pip install numpy pandas scipy scikit-learn matplotlib seaborn

pip install xgboost

pip install comet-ml

pip install imbalanced-learn

python - <<EOF
import torch
print("Torch CUDA available:", torch.cuda.is_available())
print("Torch CUDA version:", torch.version.cuda)
print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
EOF

python - <<EOF
import dgl
import torch
g = dgl.graph(([0,1],[1,2])).to('cuda' if torch.cuda.is_available() else 'cpu')
print("DGL backend:", dgl.backend.backend_name)
print("Graph device:", g.device)
EOF
