# Grammar Correction
## 1. Training new model

### a) Pre-processing data
Fairseq contains example pre-processing scripts for several translation datasets: IWSLT 2014 (German-English), WMT 2014 (English-French) and WMT 2014 (English-German), etc.

To pre-process and binarize the IWSLT dataset:

```
> cd examples/translation/
> bash prepare-iwslt14.sh
> cd ../..
> TEXT=examples/translation/iwslt14.tokenized.de-en
> fairseq-preprocess --source-lang de --target-lang en \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir data-bin/iwslt14.tokenized.de-en
```

Note that the binarized data that can be used for model training is written to path: <br>
_./data-bin/iwslt14.tokenized.de-en_


### b) Training the IWSLT 2014 dataset:
```
> mkdir -p checkpoints/fconv
> CUDA_VISIBLE_DEVICES=0 fairseq-train data-bin/iwslt14.tokenized.de-en \
    --lr 0.25 --clip-norm 0.1 --dropout 0.2 --max-tokens 4000 \
    --arch fconv_iwslt_de_en --save-dir checkpoints/fconv
```

Note that checkpoint files are stored in ./checkpoints/fconv directory, which is newly created by the mkdir command

### c) Generation:
Generate translations using fairseq-generate (for binarized data) or fairseq-interactive (for raw text) with trained model:

```
> fairseq-generate data-bin/iwslt14.tokenized.de-en \
    --path checkpoints/fconv/checkpoint_best.pt \
    --batch-size 128 --beam 5
```

Note that this task uses the best checkpoint from the training process, which is the _checkpoint_best.pt_ file
