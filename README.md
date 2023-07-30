# [Towards Debiasing Temporal Knowledge Graph Representation Learning (submitted to ICDE 2024)]()


## Installation
Create a conda environment with pytorch and scikit-learn :
```
conda create --name booster_env python=3.6
source activate booster_env
conda install --file requirements.txt -c pytorch
```

## Datasets


Once the datasets are downloaded, running the following commend to pre-processing datasets:
```
python process_icews.py
python process_timegran.py --tr 100 --dataset yago11k
python process_timegran.py --tr 1 --dataset wikidata12k
# For wikidata11k and yago12k, change the tr for changing time granularity
```

This will create the files required to compute the filtered metrics.

## Reproducing results of Booster

In order to reproduce the results of Booster on the four datasets in the paper, you can run the following commands

```
python learner.py --dataset ICEWS14 --model TNT --rank 2000 --emb_reg 0.0075  --time_reg 0.01 --batch_size 1000 --max_epochs 2000  -- valid_freq 100

python learner.py --dataset ICEWS14 --model HyTE --rank 200 --emb_reg 0.1 --time_reg 0 --batch_size 500 --valid_freq 20 --max_epochs 100

python learner.py --dataset ICEWS14 --model TA --rank 200 --emb_reg 0.025 --time_reg 0.0025 --batch_size 1000 --valid_freq 20 --max_epochs 200

python learner.py --dataset ICEWS05-15 --model TNT --rank 2000 --emb_reg 0.0025 --time_reg 0.1 --batch_size 1000

```

## Officical implementation of baseline models
| Baselines   | Code                                                                      |
|-------------|---------------------------------------------------------------------------|
| TEMP ([Wu et al., 2018](https://arxiv.org/pdf/2010.03526.pdf))    | [Link](https://github.com/JiapengWu/TeMP) |
| TA-TransE ([Alberto et al., 2018](https://www.aclweb.org/anthology/D18-1516.pdf))      | [Link](https://github.com/INK-USC/RE-Net/tree/master/baselines)     |
| HyTE ([Dasgupta et al., 2018](http://talukdar.net/papers/emnlp2018_HyTE.pdf))        | [Link](https://github.com/malllabiisc/HyTE)                               |
| DE-DistMult ([Goel et al., 2020](https://arxiv.org/pdf/1907.03143.pdf))        | [Link](https://github.com/BorealisAI/de-simple)                               |
| Timeplex ([Timothee et al., 2020](https://openreview.net/pdf?id=rke2P1BFwS))        | [Link](https://github.com/facebookresearch/tkbc)                               |
| TeRo ([Chenjin et al., 2020](https://arxiv.org/pdf/2010.01029.pdf))        | [Link](https://github.com/soledad921/ATISE)                               |


## Acknowledgement
We refer to the code of TNTComplEx. Thanks for their great contributions!
