Usage:
```
python dec.py filename
```

For example:
```
python dev.py sample.txt
```

For running python you need the following dependency:
- ```nltk```
- StanfordParser and its pre-trained model
- ```numpy```
- pre-trained sentence compression model ('./written.model')

Please special the path of these model file in ```dec.py```.

And note that the sentence compression model we provided here is a simple one,
and you might want to train your own fancy model for better performance.
