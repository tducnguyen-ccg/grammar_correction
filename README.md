# Grammar Correction
## Training new model

### Pre-processing data
Pre-process and binarize IWSLT dataset and write binarized data that can be used for model training to path: ```data-bin/iwslt14.tokenized.de-en```

```
> cd examples/translation/
> bash prepare-iwslt14.sh
> cd ../..
> TEXT=examples/translation/iwslt14.tokenized.de-en
> fairseq-preprocess --source-lang de --target-lang en \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir data-bin/iwslt14.tokenized.de-en
```
