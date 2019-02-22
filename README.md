# Consistent Prototypical Networks(CPN)
Code for paper
*Semi-Supervised Few-Shot Learning with Local and Global Consistency.* [[arxiv](LINK)]

This code is based on [https://github.com/renmengye/few-shot-ssl-public].

Based on the paper:
* Mengye Ren, Eleni Triantafillou, Sachin Ravi, Jake Snell, Kevin Swersky, Joshua B. Tenenbaum, Hugo Larochelle and Richard S. Zemel.
Meta-Learning for Semi-Supervised Few-Shot Classification. 
In *Proceedings of 6th International Conference on Learning Representations (ICLR)*, 2018.


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
# Clone the repository.
git clone https://github.com/AhmedAyad89/Consitent-Prototypical-Networks-Semi-Supervised-Few-Shot-Learning
cd few-shot-ssl-public

# To train a model.
python run_exp.py --data_root $DATA_ROOT             \
                  --dataset {DATASET}                \
                  --label_ratio {LABEL_RATIO}        \
                  --model {MODEL}                    \
                  --results {SAVE_CKPT_FOLDER}       \
                  [--disable_distractor]

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
* Relevant `{MODEL}` options are `basic`, `basic-VAT`, `basic-VAT-ENT`, `kmeans-refine`.
* Relevant `{DATASET}` options are `omniglot`, `mini-imagenet`.
* Use `{LABEL_RATIO}` 0.1 for `omniglot` and and 0.4 for `mini-imagenet`. 
* Replace `{MODEL_ID}` with the model ID obtained from the training program.
* Replace `{SAVE_CKPT_FOLDER}` with the folder where you save your checkpoints.
* Add additional flags `--num_unlabel 20 --num_test 20` for testing `mini-imagenet` models, so that each episode contains 20 unlabeled images per class and 20 query images per class.
* Add an additional flag `--disable_distractor` to remove all distractor classes in the unlabeled images.
* Add an additional flag `--use_test` to evaluate on the test set instead of the validation set.
* More commandline details see `run_exp.py`.
* Hyperparams internal to the SSL methods(VAT/RW) are set as flags, for info see `VAT_utils.py`
* Model architercture and all other hyperparams are set from the config files, contained in the `configs` folder.

## Simple Baselines for Few-Shot Classification
Please run the following script to reproduce a suite of baseline results.
```
python run_baseline_exp.py --data_root $DATA_ROOT    \
                           --dataset {DATASET}
```
* Possible `DATASET` options are `omniglot`, `mini-imagenet`.

## Run SOTA CPN models
To train/test the state of the art CPN, and reproduce the results in the paper, set hyperparams as specified in the paper, and run the `basic-VAT-ENT` model.

For example, to train a CPN on 5-shot mini-imagenet:
 
```
python run_exp.py --data_root $DATA_ROOT 		\
                        --dataset mini-imagenet		\
                        --label_ratio 0.4  		\
                        --model bas-VAT-ENT		\
			--nshot 5			\
                        [--disable_distractor]		\
```

To test:
```
python run_exp.py --data_root $DATA_ROOT		\
                  --dataset mini-imagenet		\
                  --model basic-VAT-ENT 		\
                  --results {SAVE_CKPT_FOLDER}		\
                  --eval --pretrain {MODEL_ID}		\
                  [--num_unlabel {NUM_UNLABEL}]		\
                  [--num_test {NUM_TEST}]		\
                  [--disable_distractor]		\
                  [--use_test]

```

To test CPN+semi-supervised inference:
```
python run_exp.py --data_root $DATA_ROOT		\
                  --dataset mini-imagenet		\
                  --model kmeans-refine			\
                  --results {SAVE_CKPT_FOLDER}		\
                  --eval --pretrain {MODEL_ID}		\
                  [--num_unlabel {NUM_UNLABEL}]		\
                  [--num_test {NUM_TEST}]		\
                  [--disable_distractor]		\
                  [--use_test]
```			  


## Citation
If you use our code, please consider cite the following:


```
@inproceedings{ayad19cpn,
  author   = {Ahmed Ayyad and
			Nassir Navab and
			Mohamed Elhoseiny and
			Shadi  Albarqouni},
  title    = {Semi-Supervised Few-Shot Learning with Local and Global Consistency},
  booktitle= {},
  year     = {2019},
}
```
