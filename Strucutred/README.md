# The code
You have made it to my code!

I tried to tidy everything up. But my experiments stopped working so for laziness sake it is still a mess.
This code is created in Python 3.7.3. and with modlues `NumPy`,`scipy`,`matplotlib` and `tqdm`.

---	
To create the plots in chapter 5.2 the normal shell was used. To runn this please runn the file `OuterShell.py`. It runs the Newton-Rapsons method described in te method. 
To do this it calls the following files:
- `Model_Numerical_Jacobian_total_model.py`
- `Model_parameters.py`
- `Model_functions.py`

# The experiments
The variation of parmaters equilibria are created via `OuterShell_nonlinear_scipy_multiple_phi_test.py`,`OuterShell_nonlinear_scipy_multiple_L_test.py`, `OuterShell_nonlinear_scipy_multiple_H_test.py` and `OuterShell_nonlinear_scipy_multiple_A_test.py`.
These are analysed via `eigenplots.py`,`eigenplots_vary_A.py`,`eigenplots_vary_H.py` and `eigenplots_vary_L.py`. The data fore these experiments are stored in the `DATA` folder. 


