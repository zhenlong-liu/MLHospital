# MLHospital

MLHospital is a repo to evaluate inference attacks and the corresponding defenses against machine learning models.

Currently we support membership inference attacks and attribute inference attacks.

## Installation
```
git clone https://github.com/TrustAIResearch/MLHospital.git;
cd MLHospital;
conda env create -f environment.yml;
conda activate ml-hospital;
python setup.py install;
```


## Membership inference attacks
### Step 0: train target/shadow models

```
cd MLHospital/mlh/examples;
python train_models.py --mode target --training_type Normal --loss_type ccel --alpha 0.5;
python train_models.py --mode shadow --training_type Normal --loss_type ccel --alpha 0.5;
```
Note that you can also specify the `--loss_type` with different loss function, e.g., `ce`, `focal` and `ccql`.

### Step 1: perform membership inference attacks
```
python mia.py  --training_type Normal --loss_type ccel --alpha 0.5
```
Note that you can also specify the `--attack_type` with different attacks, e.g., `black-box`, `black-box-sorted`, `black-box-top3`, `metric-based`, and `label-only`.

## Attribute inference attacks

```
cd MLHospital/mlh/examples;
python3 aia_example.py --task aia --dataset CelebA --defense AdvTrain --alpha 1.0;
```
The aia_example.py first trains target models (with or without defense), then trains and evaluates the attack model.

In this example, CelebA dataset is used, and the defense method and alpha (the hyperparameter to balance utility and privacy) are set to be AdvTrain and 1.0, respectively.
Note that you can also specify the `--defense` with different defense mechanisms, e.g., `Normal`, `AdvTrain`, `Olympus`, and `AttriGuard`.


# Authors
The tool is designed and developed by Xinlei He (CISPA), Zheng Li (CISPA), Yukun Jiang (CISPA), Yun Shen (NetApp), and Yang Zhang (CISPA).
