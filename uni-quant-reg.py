#-----------------------------------------
# This script sets out to produce a
# quantile regression on some student
# load data
#-----------------------------------------

#-----------------------------------------
# Author: Trent Henderson, 7 November 2020
#-----------------------------------------

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns; sns.set_style("darkgrid")

# Read in data

d = pd.read_csv("/Users/trenthenderson/Documents/Git/uni-quant-reg/data/eftsl-and-pop.csv")

#-------------------- PRE PROCESSING FOR MODELLING -----------------

# Remove NAs

d = d.dropna()

#-------------------- GRAPH THE DATA -------------------------------

fig, ax = plt.subplots(figsize = (8, 6))
fig.suptitle('Resident population and student load by postcode', fontsize = 20)
ax.scatter(d.usual_resident_population, d.eftsl, alpha = .2, color = '#0292B7')
legend = ax.legend()
ax.xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
ax.set_xlabel('Usual Resident Population', fontsize = 16)
ax.set_ylabel('Domestic Online EFTSL', fontsize = 16)

# Save the plot

plt.savefig('/Users/trenthenderson/Documents/Git/uni-quant-reg/output/pop-eftsl-scatter.png', dpi = 1000)

#-------------------- MODEL SPECIFICATION --------------------------

# Fit quantile regression

mod = smf.quantreg('eftsl ~ usual_resident_population', d)
res = mod.fit(q = .5)

quantiles = np.arange(.05, .96, .1)

# Collect relevant quantile regression metrics for plotting

def fit_model(q):
    res = mod.fit(q = q)
    return [q, res.params['Intercept'], res.params['usual_resident_population']] + \
            res.conf_int().loc['usual_resident_population'].tolist()

models = [fit_model(x) for x in quantiles]
models = pd.DataFrame(models, columns = ['q', 'a', 'b', 'lb', 'ub'])

# Get regular OLS regression for comparison

ols = smf.ols('eftsl ~ usual_resident_population', d).fit()
ols_ci = ols.conf_int().loc['usual_resident_population'].tolist()
ols = dict(a = ols.params['Intercept'],
           b = ols.params['usual_resident_population'],
           lb = ols_ci[0],
           ub = ols_ci[1])

print(models)
print(ols)

#-------------------- PLOT THE OUTPUT ------------------------------

x = np.arange(d.usual_resident_population.min(), d.usual_resident_population.max(), 50)
get_y = lambda a, b: a + b * x

fig, ax = plt.subplots(figsize = (8, 6))

for i in range(models.shape[0]):
    y = get_y(models.a[i], models.b[i])
    ax.plot(x, y, linestyle = 'dotted', color = '#5CACEE')

y = get_y(ols['a'], ols['b'])

fig.suptitle('Quantile regression of population and student load by postcode', fontsize = 20)
ax.plot(x, y, color = '#FD62AD', label = 'OLS')
ax.scatter(d.usual_resident_population, d.eftsl, alpha = .2, color = '#0292B7')
legend = ax.legend()
ax.xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
ax.set_xlabel('Usual Resident Population', fontsize = 16)
ax.set_ylabel('Domestic Online EFTSL', fontsize = 16)

# Save the plot

plt.savefig('/Users/trenthenderson/Documents/Git/uni-quant-reg/output/quant-reg.png', dpi = 1000)
