# Motif-based Graph Self-Supervised Learning for Molecular Property Prediction
Official Pytorch implementation of NeurIPS'21 paper "Motif-based Graph Self-Supervised Learning for Molecular Property Prediction"
(https://arxiv.org/abs/2110.00987). 
## environment setting
```bash
conda env create -f env.yml
conda activate mgssl
pip install -r requirements.txt
```

* `motif_based_pretrain/` contains codes for motif-based graph self-supervised pretraining.
* `finetune/` contains codes for finetuning on MoleculeNet benchmarks for evaluation.

## Dataset
For the MoleculeNet dataset for finetuning, we have uploaded them to [data](https://github.com/zaixizhang/MGSSL/tree/main/finetune/dataset.zip).

## Training
You can pretrain the model by
```
cd motif_based_pretrain
python pretrain_motif.py
```

## Evaluation
You can evaluate the pretrained model by finetuning on downstream tasks
```
cd finetune
python finetune.py
```

## Cite

If you find this repo to be useful, please cite our paper. Thank you.

```
@article{zhang2021motif,
  title={Motif-based Graph Self-Supervised Learning for Molecular Property Prediction},
  author={Zhang, Zaixi and Liu, Qi and Wang, Hao and Lu, Chengqiang and Lee, Chee-Kong},
  journal={arXiv preprint arXiv:2110.00987},
  year={2021}
}
```
