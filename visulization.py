import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl
import matplotlib.ticker as ticker

# score=[[0.31,0.29,0.28,0.28,0.26,0.70,0.65,0.82,0.29],[0.01,0.01,0.89,0.08,0.00,0.02,0.00,0.00,0.00],[0.65,0.51,0.97,0.26,0.29,1.00,0.95,1.00,0.26]]
# score=[[0.31,0.29,0.28,0.28,0.26,0.70,0.65,0.82,0.29],
#        [0.00,0.01,0.89,0.08,0.00,0.2,0.10,0.25,0.00],
#        [0.65,1.00,0.26,0.51,0.29,1.00,0.90,1.00,0.26]]
score=[[0.26,0.33,0.31,0.36,0.34,0.37,0.35,0.31,0.31,0.33,0.30,0.29,0.36,0.34],
       [0.20,0.8,0.16,0.16,0.24,0.4,0.16,0.48,0.24,0.08,0.4,0.16,0.08,0.08],
       [0.30,0.35,0.99,0.34,1.00,0.89,0.37,0.33,0.30,0.28,0.28,0.27,0.93,0.41]]

d=np.array(score)

# variables = ['The','staff','should','be','a','bit','more','friendly','.']
# variables = ['The','staff','should','be','a','bit','more','friendly','.']
variables = ['I','never','had','an','orange','donut','before','so','I','gave','it','a','shot','.']

labels = ['Contextual','Syntactic','Knowledge']

df = pd.DataFrame(d, columns=variables, index=labels)

fig = plt.figure()

ax = fig.add_subplot(111)

cax = ax.matshow(df, interpolation='nearest', cmap='Greens')
fig.colorbar(cax, shrink=0.3)
fontdict = {'rotation': 45}
tick_spacing = 1
ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

ax.set_xticklabels([''] + list(df.columns),fontdict=fontdict)
ax.set_yticklabels([''] + list(df.index))

plt.title('Aspect terms: orange donut     Label: Neu.       Predict: Neu.',y=-0.5)
plt.savefig('case2.pdf', bbox_inches='tight')
plt.show()

