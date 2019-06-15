# Prototypical Random Walk Networks(PRWN)
Code for paper
*Semi-Supervised Few-Shot Learning with Prototypical Random Walks* [[Arxiv](https://github.com/AhmedAyad89/Consitent-Prototypical-Networks-Semi-Supervised-Few-Shot-Learning)][[Poster](https://docdro.id/Hgw3KWJ)][[Slides](https://docdro.id/qLZBLm2)]

## Overview

In this paper we design a semi-supervised loss to leverage unlabelled data in the few-shot setting. Our base model is a prototypical network, and we add the Prototypical Random Walk loss in order to leverage the unlabeled data during the episodic meta-training. Our loss is designed to train Prototypical Networks to produce embeddings where points of the each class form a tight cluster around the class prototype. 
We find that PRWN outperform the prior state-of-the-art on all 9 benchamrak tests we run, in a variety of semi-supervised few-shot learning settings. We often see dramatic improvements over prior SOTA, for example, PRWN obtains 69.65% in one test comapred to 64.59 for the prior SOTA. Remarkably, PRWN also outperforms the __fully supervised__ prototypical network in one test, obtaining 50.89% to 49.4% for the baseline. 


## Dependencies
* cv2
* numpy
* pandas
* python 2.7 / 3.5+
* tensorflow 1.3+
* tqdm

Our code is tested on Ubuntu 16.04.

## Setup
First, designate a folder to be your data root:
```
export DATA_ROOT={DATA_ROOT}
```

Then, set up the datasets following the instructions in the subsections.

### Omniglot
[[Google Drive](https://drive.google.com/open?id=1INlOTyPtnCJgm0hBVvtRLu5a0itk8bjs)]  (9.3 MB)
```
# Download and place "omniglot.tar.gz" in "$DATA_ROOT/omniglot".
mkdir -p $DATA_ROOT/omniglot
cd $DATA_ROOT/omniglot
mv ~/Downloads/omniglot.tar.gz .
tar -xzvf omniglot.tar.gz
rm -f omniglot.tar.gz
```

### miniImageNet
[[Google Drive](https://drive.google.com/open?id=16V_ZlkW4SsnNDtnGmaBRq2OoPmUOc5mY)]  (1.1 GB)
```
# Download and place "mini-imagenet.tar.gz" in "$DATA_ROOT/mini-imagenet".
mkdir -p $DATA_ROOT/mini-imagenet
cd $DATA_ROOT/mini-imagenet
mv ~/Downloads/mini-imagenet.tar.gz .
tar -xzvf mini-imagenet.tar.gz
rm -f mini-imagenet.tar.gz
```

## Core Experiments
Please run the following scripts to reproduce the core experiments.
```
#First place the data_root folder inside the provided code folder. 

# To train a model.
python run_exp.py --data_root $DATA_ROOT             \
                  --dataset {DATASET}                \
                  --label_ratio {LABEL_RATIO}        \
                  --model {MODEL}                    \
                  --results {SAVE_CKPT_FOLDER}       \
                  [--disable_distractor]			 \
				  [--nshot]                          \
				  [--nclasses_train]				 \
				 
				  

# To test a model.
python run_exp.py --data_root $DATA_ROOT             \
                  --dataset {DATASET}                \
                  --label_ratio {LABEL_RATIO}        \
                  --model {MODEL}                    \
                  --results {SAVE_CKPT_FOLDER}       \
                  --eval --pretrain {MODEL_ID}       \
                  [--num_unlabel {NUM_UNLABEL}]      \
                  [--num_test {NUM_TEST}]            \
                  [--disable_distractor]             \
                  [--use_test]
```
* Relevant `{MODEL}` options are `basic`, `basic-RW`(PRWN), `kmeans-refine`(semi-supervised inference), 'kmeans-filter'.
* Relevant `{DATASET}` options are `omniglot`, `mini-imagenet`.
* Use `{LABEL_RATIO}` 0.1 for `omniglot` and and 0.4 for `mini-imagenet`. 
* Replace `{MODEL_ID}` with the model ID obtained from the training program.
* Replace `{SAVE_CKPT_FOLDER}` with the folder where you save your checkpoints.
* Add additional flags `--num_unlabel 20 --num_test 20` for testing `mini-imagenet` models, so that each episode contains 20 unlabeled images per class and 20 query images per class.
* Add an additional flag `--disable_distractor` to remove all distractor classes in the unlabeled images.
* Add an additional flag `--use_test` to evaluate on the test set instead of the validation set.
* More commandline details see `run_exp.py`.
* Hyperparams internal to the SSL methods(RW) are set as flags, for info see `ssl_utils.py`
* Model architercture and all other hyperparams are set from the config files, contained in the `configs` folder.
*Flags for episode construction and training setting can be found in run_exp.py

## Simple Baselines for Few-Shot Classification
Please run the following script to reproduce a suite of baseline results.
```
python run_baseline_exp.py --data_root $DATA_ROOT    \
                           --dataset {DATASET}
```
* Possible `DATASET` options are `omniglot`, `mini-imagenet`.

## Run SOTA PRWN models
To train/test the state of the art PRWN, and reproduce the results in the paper, set hyperparams as specified in the paper, and run the `basic-RW` model.

For example, to train a PRWN on 5-shot mini-imagenet:
 
```
python run_exp.py --data_root $DATA_ROOT 			\
                        --dataset mini-imagenet     \
                        --label_ratio 0.4  			\
                        --model basic-RW            \
						--nshot 5					\
						--num_unlabel 10            \
                        [--disable_distractor]      \
```

To test:
```
python run_exp.py --data_root $DATA_ROOT           	\
                  --dataset mini-imagenet           \
                  --model basic-RW                  \
                  --results {SAVE_CKPT_FOLDER}      \
                  --eval --pretrain {MODEL_ID}      \
                  [--num_unlabel {NUM_UNLABEL}]     \
                  [--num_test {NUM_TEST}]           \
                  [--disable_distractor]            \
                  [--use_test]

```

To test PRWN+semi-supervised inference:
```
python run_exp.py --data_root $DATA_ROOT             \
                  --dataset mini-imagenet            \
                  --model kmeans-refine              \
                  --results {SAVE_CKPT_FOLDER}       \
                  --eval --pretrain {MODEL_ID}       \
                  [--num_unlabel {NUM_UNLABEL}]      \
                  [--num_test {NUM_TEST}]            \
                  [--disable_distractor]             \
                  [--use_test]
```			  

To test PRWN+semi-supervised inference with the distractor filtering:
```
python run_exp.py --data_root $DATA_ROOT             \
                  --dataset mini-imagenet            \
                  --model kmeans-filter              \
                  --results {SAVE_CKPT_FOLDER}       \
                  --eval --pretrain {MODEL_ID}       \
                  [--num_unlabel {NUM_UNLABEL}]      \
                  [--num_test {NUM_TEST}]            \
                  [--disable_distractor]             \
                  [--use_test]
```	

# Acknowledgements
This code is based on [https://github.com/renmengye/few-shot-ssl-public].
Based on the paper:
* Mengye Ren, Eleni Triantafillou, Sachin Ravi, Jake Snell, Kevin Swersky, Joshua B. Tenenbaum, Hugo Larochelle and Richard S. Zemel.
Meta-Learning for Semi-Supervised Few-Shot Classification. 
In *Proceedings of 6th International Conference on Learning Representations (ICLR)*, 2018.