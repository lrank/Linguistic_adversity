# Linuistic Adversity

## Introduction
This repository	is an implementation of	the following work:

Li, Yitong , Trevor Cohn and Timothy Baldwin (2017) Robust Training under Linguistic Adversity, In Proceedings of the 15th Conference of the European Chapter of the Association for Computational Linguistics (EACL 2017), Valencia, Spain.

## Respository Structure
In this repository, files are separated into three fold:
- ```data```
- ```noise generators```
- ```text_cnn``` : the convolutional neural network model for sentence-level classification tasks.

Details are following:
## data/
This fold contains the original dataset,
- movie review dataset (Pang and Lee, 2008)
- customer review dataset (Hu and Liu, 2004)
- subjectivity dataset (Pang and Lee, 2005)
- Stanford Sentiment Treenbank (Socher et al., 2013)

For more sentiment analysis dataset can be found at [HarvardNLP](https://github.com/harvardnlp/sent-conv-torch/tree/master/data).

Please cite the original paper when you use data.

## noise_generator/
This fold contains four different linguistic noise generators within four sub-folds each.

Please refer to the ```README```file for each noise generator methods to run them.

### ./WN/
The fold contains the code of semantic noise generator based on Wordnet.

For running the wordnet noise genereator code, you need the following dependencies:
- NLTK, with following packages downloaded:
-- averaged_perceptron_tagger
-- punkt
-- stopwords
-- universal_tagset
-- wordnet
- Numpy
- kenLM, with pre-built n-gram language model
- Stanford-ner


### ./CFit/
Based on idea of [Counter-fitting](https://arxiv.org/abs/1603.00892)

Dependencies:
- NLTK
- Numpy
- kenLM
- Stanford-ner
- pre-trained counter-fitting dictionary


### ./ERG/
Based on English Resource Grammar (ERG) system, and ACE.
Dependencies:
- ERG ACE

### ./Comp/
Based on sentence compression method.

## text_cnn/
A convolutional neural network model for text classification tasks.
The model is based on YoonKim's [Convolutional Neural Networks for Sentence Classification](http://arxiv.org/abs/1408.5882) and Denny Britz's implementation (https://github.com/dennybritz/cnn-text-classification-tf).
Notice that the code has been implemented and tested with [Tensorflow r1.0](https://www.tensorflow.org) and python 2.7, which may not be able to run on other version.

### requirements
For run the cnn code, you need the following dependencies:
- Python 2.7
- Numpy
- Tensorflow r1.0

### Running the code
```bash
python train.py [parameters]
```
```
parameters:
    --dataset
        The training dataset (default:"mr")
    --noise_type
        Type of noise (default:"raw")
    --is_noise_train
        To train on the noisy data (default:False)
    --is_noise_test
        To test on the noisy data (default:False)
    --l2_reg_lambda
        Model L2 regularizaion lambda (default: 0)
    --dropout_keep_prob=1.0
        Model dropout rate (default:0.5)
```
For example, to train the model with settings:
```bash
nice python  train.py --dataset="mr" --is_noise_train=False --is_noise_test=False --noise_type="cf=0.5" --dropout_keep_prob=1.0
```

Also, refer to ```run_script_sample_on_subj.sh``` to get a better sense of training with different noise.

## Contact us
Please email us if anything. All comments are welcome.