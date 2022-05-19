# Semantic Autoencoder (SemAE)

This repository presents the implementation of the ACL 2022 paper:

> [**Unsupervised Extractive Opinion Summarization Using Sparse Coding**](https://arxiv.org/abs/2203.07921),<br/>
[Somnath Basu Roy Chowdhury](https://www.cs.unc.edu/~somnath/), [Chao Zhao](https://zhaochaocs.github.io/) and [Snigdha Chaturvedi](https://sites.google.com/site/snigdhac/) 
>

<!-- <p align="left">
  <img src="https://www.cs.unc.edu/~somnath/SemAE/data/SemAE.png" alt="" width="400"/>
</p> -->

The implementation of SemAE is based on the open-source framework of [Quantized Transformer](https://arxiv.org/pdf/2012.04443.pdf).



## Data

Download the SPACE corpus from this [link](https://github.com/stangelid/qt).
Amazon dataset is publicly available [here](https://github.com/abrazinskas/Copycat-abstractive-opinion-summarizer/tree/master/gold_summs).

For Amazon dataset, the data was processed using instruction from [here](https://github.com/stangelid/qt/blob/main/custom.md).

To directly access the data used in our experiments, use the files in this [link](https://www.cs.unc.edu/~somnath/SemAE/data/) as the `data/` folder. Please cite the respective papers if you are using the above datasets.


## Using our model

### Setting up the environment

* __Python version:__ `python3.6`

* __Dependencies:__ Use the `requirements.txt` file and conda/pip to install all necessary dependencies. E.g., for pip:

		pip install -U pip
		pip install -U setuptools
		pip install -r requirements.txt 



### Training SemAE

To train SemAE on a subset of the training set using a GPU, go to the `./src`
directory and run the following:

    python3 train.py --max_num_entities 500 --run_id space_run --gpu 0

This will train a SemAE model with default hyperparameters (for general
summarization), store tensorboard logs under `./logs` and save a
model snapshot after every epoch under `./models` (filename:
`space_run_<epoch>_model.pt`). 

For training the full model on SPACE, run the following:
```
cd scripts/
chmod +x train_space.sh
./train_space.sh
```
For training the model on full Amazon dataset, please run `scripts/train_amazon.sh` bash script in a similar manner.



### Summarization with SemAE

To perform general opinion summarization with a trained SemAE model, go to the `./src` directory and run the following:

	python3 inference.py \
			--model ../models/space_run_10_model.pt \
			--run_id space_run \
			--gpu 0

This will store the summaries under `./outputs/space_run` and also the output of ROUGE evaluation in `./outputs/eval_space_run.json`. 
For aspect opinion summarization, run:

	python3 aspect_inference.py \
			--model ../models/space_run_10_model.pt \
			--sample_sentences --run_id aspects_run \
			--gpu 0

The summarization scripts for SPACE and Amazon are: `scripts/evaluate_*.sh`


### Citation

```
@inproceedings{chowdhury2022unsupervised,
    title = {Unsupervised Extractive Opinion Summarization Using Sparse Coding},
    author = {Basu Roy Chowdhury, Somnath  and
      Zhao, Chao  and
      Chaturvedi, Snigdha},
    booktitle = {ACL},
    year = {2022},
}
```
