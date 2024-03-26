
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
python train_models.py --mode target --training_type Normal --loss_type ccel --alpha 0.5 --gpu 0 --optimizer sgd --scheduler multi_step --epoch 300;
python train_models.py --mode shadow --training_type Normal --loss_type ccel --alpha 0.5 --gpu 0;
```
Note that you can also specify the `--loss_type` with different loss function, e.g., `ce`, `focal` and `ccql`.

### Step 1: perform membership inference attacks
```
python mia.py  --training_type Normal --loss_type ccel --alpha 0.5
```
