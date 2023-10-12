# Dynamic Effective Connectivity Learning based on non- Parametric State Estimation and GAN (nPSE-GAN)
# programming language：matlab and python
# run  main.py
# Step 1:non-Parametric State Estimation (matlab)
    Parameters：TC = load('\dataname.txt'),n = 5,subj = 4,sublen = 200
# Step 2: Dynamic Effective Connectivity Learning (python)
    eng = matlab.engine.start_matlab()
    [NC, TP] = eng.nPSE # Output of Step 1
    p_txt = "/dataname.txt"
    data = np.zeros((sub*TP[0], n))
    for i in range(0, sub):
        data[i*TP[0]:(i+1)*TP[0], :] = sim[i*sublen+start:i*sublen+TP[0], :]
    GAN.Test_RUN(data)

## Requirments
* pyhton 3.7
* numpy 1.19.2
* pandas 1.2.2
* torch 1.4.0
