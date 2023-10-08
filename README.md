# ISDN-main
## train：
##### First download the datasets DIV2K and FLICK2K
##### Configure the location of the dataset in the option.py file

```
python main.py
```

## test：
##### Download datasets Set5, Set14, BSD100，Urban100, Manga109
##### Configure the location of the dataset in the option.py file

```
 python main.py --pre_train ./experiment/test/model/model_best.pt --save_results --test_only --chop
```