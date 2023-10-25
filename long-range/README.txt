We provide the pretrained weight of the model used in Table 2 from the paper.


##########################################################

To reproduce results on peptide-struct, run:
```
python downstream.py --dataset peptide_reg --save_path pretrain/GIN_CF_05_LR.pth --runseed <seed>
```

To reproduce results peptide-func, run:
```
python downstream.py --dataset peptide_cls --save_path pretrain/GIN_CF_05_LR.pth --runseed <seed>
```
##########################################################

`<seed>` should be from [0,9]




 
