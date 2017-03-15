# Noise Generation

For generating noise text, please do refer to ```README```s of each methods.

Here are some general things to follow, which can make you get easier if you do so:

- make sure you to split your text into ```pos``` and ```neg``` respectively. (for cross-validation datasets)
- name different noisy text with different file extensions, such as ```.wn``` for wordnet noise.
- your noisified text file should consist of the sentence instances in the original order of the original text file.
Each of the instance looks like the following:
```
L1: n(Number of the total noisy sentences)
L2: strings(The original sentence)
L3~L(2+n): strings(the noisy sentences)
```
- some of the methods could generate thousands of noisy sentences for one instance, therefore
please use ```preprocess_noisydata.py``` to cut off overloaded sentences to save your memory during the training.
(or, of course, you can write your own stretegy and codes to do this)

Usage:
```bash
python preprocess_noisydata.py filename
```


For more details of the algorithm, please refer to our paper:

Li, Yitong , Trevor Cohn and Timothy Baldwin (2017) Robust Training under Linguistic Adversity, In Proceedings of the 15th Conference of the European Chapter of the Association for Computational Linguistics (EACL 2017), Valencia, Spain.

or contact us by email.