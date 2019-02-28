All code found on my github at: https://github.com/joel99/4641-work

I used mlrose to run the experiments, generously provided by 4641 alum Genevieve Hayes - documentation here: https://mlrose.readthedocs.io/en/stable/
  - Some changes were made for the assignment
    - For more consistent RHC results, restarts were incorporated into neural.py. Local version of neural.py on github.


  
In a standard conda env (with sklearn, numpy, matplotlib):
  pip install mlrose
  pip install DEAP

DEAP was used because mlrose GA were running slower.
Some notes: Since mlrose doesn't support per run seeding - the reported results correspond to complete runs of the notebook (repeated runs of intermediate cells will cause numpy.random to return different results)
