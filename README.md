# Grammar Correction
## Training new model

### Pre-processing data
Pre-process IWSLT dataset and write binarized data that can be used for model training to path: <br>
_data-bin/iwslt14.tokenized.de-en_

```
> cd examples/translation/
> bash prepare-iwslt14.sh
> cd ../..
> TEXT=examples/translation/iwslt14.tokenized.de-en
> fairseq-preprocess --source-lang de --target-lang en \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir data-bin/iwslt14.tokenized.de-en
```

### Training the IWSLT 2014 dataset:
```
> mkdir -p checkpoints/fconv
> CUDA_VISIBLE_DEVICES=0 fairseq-train data-bin/iwslt14.tokenized.de-en \
    --lr 0.25 --clip-norm 0.1 --dropout 0.2 --max-tokens 4000 \
    --arch fconv_iwslt_de_en --save-dir checkpoints/fconv
```
