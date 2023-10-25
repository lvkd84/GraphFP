We provide the pretrained weights of the 3 models used in Table 1 from the paper.

##########################################################

To reproduce GIN_C, run:
```
python downstream_old.py --dataset <dataset name> --pretrain_path pretrain/GIN_C.pth --drop <drop rate> --runseed <seed>
```

To reproduce GIN_CP, run:
```
python downstream_old.py --dataset <dataset name> --pretrain_path pretrain/GIN_CP_03.pth --drop <drop rate> --runseed <seed>
```

To reproduce GIN_CPF, run:
```
python downstream_frag_old.py --dataset <dataset name> --pretrain_path pretrain/GIN_CPF_01.pth --drop <drop rate> --runseed <seed>
```
##########################################################

The ranges of values from the paper:
* `<drop rate>` should be from {0.0,0.5}
* `<seed>` should be from [0,9]
 
