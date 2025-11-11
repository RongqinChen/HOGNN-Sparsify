# Code for Connectivity-Guided Sparsification of 2-FWL GNNs: Preserving Full Expressivity with Improved Efficiency

## Create environment

```bash
mamba create --name gnn222 python=3.11 numpy=1.26 pytorch=2.2.2  pytorch-cuda=12.1 pyg ogb torchmetrics yacs lightning sage rdkit=2024.03.5 -c conda-forge -c pytorch -c nvidia -c pyg 

mamba activate gnn222

pip install torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-2.2.2+cu121.html
pip install comet_ml
```


## Scripts for running experiments



### For TUD

```bash
for poly_dim in 8 12 14 16
do
for num_layers in 4
do
for dname in MUTAG PROTEINS_full ENZYMES  
do

python run_tud.py --cfg configs/sppgn/tud.sppgn1.poly.yaml \
    --poly_method rrwp --poly_dim $poly_dim  --dataname $dname

done
done
done
```


### For ZINC

```bash
for poly_dim in 6 8 10 12 14
do

python run_zinc.py --cfg configs/sppgn/zinc.sppgn1.poly.yaml --poly_dim $poly_dim 

done
```

### For ZINC-Full

```bash
for poly_dim in 6 8 10 12 14
do

python run_zinc.py --cfg configs/sppgn/zincfull.sppgn1.poly.yaml --poly_dim $poly_dim 

done
```


### For QM9

```bash
for poly_dim in 6 8 10 12 14
do

python run_zinc.py --cfg configs/sppgn/nogeo_qm9.sppgn1.poly.yaml --poly_dim $poly_dim 

done
```
