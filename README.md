# Dynamic Effective Connectivity Learning based on non- Parametric State Estimation and GAN (nPSE-GAN)
# programming language：matlab and python
# Step 1:non-Parametric State Estimation (matlab)
Parameters：TC = load('\dataname.txt'),n = 5,subj = 4,sublen = 200
# Step 2: Dynamic Effective Connectivity Learning (python)
Parameters：lr=0.01, dlr=0.01, l1=0.1, nh=100, dnh=100, train_epochs=400, test_epochs=1600
# Program entry:run  main.py

## Requirments
* pyhton 3.7
* numpy 1.19.2
* pandas 1.2.2
* torch 1.4.0
