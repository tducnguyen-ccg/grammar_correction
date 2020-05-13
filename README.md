# Grammar Correction
Referent: Improving Grammatical Error Correction via Pre-Training a Copy-Augmented Architecture with Unlabeled Data 
https://github.com/zhawe01/fairseq-gec

## Dependecies
PyTorch version >= 1.0.0

Python version >= 3.6

## Downloads
### Download CoNLL-2014 evaluation scripts
    cd gec_scripts/
    sh download.sh
### Download pre-processed data & pre-trained models

pre-trained model: (Google Drive/Baidu Pan)

url1: https://drive.google.com/file/d/1zewifHUUwvqc2F-MfDRsZFio6PlSzx2c/view?usp=sharing

url2: https://pan.baidu.com/s/1hCwQeNFjng_0_NiViJq6fg (code: mxrf)

pre-processed data: (Google Drive)(train/valid/test)

url: https://drive.google.com/open?id=17s-TZiM6ilQ-SHklxTUun2Jdgg8B9zS3

## Train with the pre-trained model
    cd fairseq-gec
    pip install --editable
    sh train.sh \${device_id} \${experiment_name}
    
 or: Move all of files in ./out/data_bin into ./out/data_raw
 
 Check file checkpoint_last.pt in folder ./models_pretrain. If you don't have, please change filename of checkpoint9.pt from download above into checkpoint_last.pt.
 
```
!python train.py out/data_raw --save-dir=models_pretrain --arch=transformer --seed=4321 --max-epoch=25 \
--batch-size=64 --max-tokens=3000 --train-subset=train --valid-subset=valid \
--lr-scheduler triangular --max-lr=0.004 --lr-period-updates=73328 \
--clip-norm=2 --lr=0.001 --lr-shrink=0.95 --shrink-min \
--dropout 0.2 --relu-dropout 0.2 --attention-dropout 0.2 --copy-attention-dropout 0.2 \
--encoder-embed-dim 512 --decoder-embed-dim 512 \
--max-target-positions 1024 --max-source-positions 1024 \
--encoder-ffn-embed-dim 4096 --decoder-ffn-embed-dim 4096 \
--encoder-attention-heads 8 --decoder-attention-heads 8 \
--copy-attention-heads 1 \
--share-all-embeddings \
--no-progress-bar \
--log-interval 1000 \
--positive-label-weight 1.2 \
--copy-attention --copy-attention-heads 1
```

## Testing
### generative
Have test.src-tgt.src and test.src-tgt.tgt in data_raw path folder.

    python generate.py out/data_raw --path=models_pretrain/checkpoint_last.pt --source-lang=src --target-lang=tgt --raw-text


### interactive
Type the input sentence and press return to evaluate the model:
    
    python interactive.py out/data_raw --path=models_pretrain/checkpoint_last.pt --source-lang=src --target-lang=tgt --raw-text
