
## Installation
```
cd MLHospital;
conda activate ml-hospital;
pip install -r requirements.txt
```


## Membership inference attacks
### Step 0: train target/shadow models
```
cd MLHospital/mlh/examples;
python train_models.py --mode target --training_type Normal --loss_type ccel --alpha 0.5 --gpu 0 --optimizer sgd --scheduler multi_step --epoch 300 --learning_rate 0.1;
python train_models.py --mode shadow --training_type Normal --loss_type ccel --alpha 0.5 --gpu 0 --optimizer sgd --scheduler multi_step --epoch 300 --learning_rate 0.1;
``` 
Note that you can also specify the `--loss_type` with different loss function, e.g., `ce`, `focal` and `ccql`.

### Step 1: perform membership inference attacks
```
python attack.py  --training_type Normal --loss_type ccel --alpha 0.5 --scheduler multi_step --epoch 300 --learning_rate 0.1;
```
