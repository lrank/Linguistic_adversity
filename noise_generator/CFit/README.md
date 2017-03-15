Usage:

```
python add_CFit_noise.py input_filename lm_threshold
```

For example:

```
python add_CFit_noise.py "sample.txt" 0.5
```

For running add_CFit_noise.py, the following packages are needed:
- ```numpy```
- ```nltk```
- StanfordNERTagger
- ```kenlm```


And, please specify your model path in python code:
- StanfordNERTagger
- Model
- Overfitting dictionary ('overfitting.dict')
