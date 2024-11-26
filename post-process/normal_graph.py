import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

mean_result = pd.read_csv("log/cma/pattern1mean_result.csv", header=None)
var_result = pd.read_csv("log/cma/pattern1var_result.csv", header=None)

m_params = np.array(mean_result.values.flatten())
v_params = np.array(var_result.values.flatten())  

fig = plt.figure( figsize = (32,16))

x = np.arange(-50,50,0.1)
for n in range(31):
    y = norm.pdf( x,m_params[n], v_params[n] )
    plt.subplot(31,1,n+1)
    plt.title(n)
    plt.plot(x,y)

plt.tight_layout()
plt.show()


        