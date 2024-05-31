
## Installation
```
cd MLHospital;
conda env create -f environment.yml;
conda activate ml-hospital;
python setup.py install;
```


## Membership inference attacks
### Step 0: train target/shadow models
```
cd MLHospital/mlh/examples;
python train_models.py --mode target --training_type Normal --loss_type ccel --alpha 0.5 --beta 0.05 --gpu 0 --optimizer sgd --scheduler multi_step --epoch 300 --learning_rate 0.1;
python train_models.py --mode shadow --training_type Normal --loss_type ccel --alpha 0.5 --beta 0.05 --gpu 0 --optimizer sgd --scheduler multi_step --epoch 300 --learning_rate 0.1;
``` 
Note that you can also specify the `--loss_type` with different loss function, e.g., `ce`, `focal` and `ccql`.

### Step 1: perform membership inference attacks
```
python mia.py  --training_type Normal --loss_type ccel --alpha 0.5 --scheduler multi_step --epoch 300 --learning_rate 0.1;
```

## Citation

```
@inproceedings{liu2024mitigating,
  title={Mitigating Privacy Risk in Membership Inference by Convex-Concave Loss},
  author={Liu, Zhenlong and Feng, Lei and Zhuang, Huiping and Cao, Xiaofeng and Wei, Hongxin},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2024}
}
```

## Acknowledgements
Our implementation uses the source code from the following repositories:
[MLHospital](https://github.com/TrustAIResearch/MLHospita)