import pandas as pd
import numpy as np
from exponential_familes import gaussian_dist
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = 'png'
import matplotlib.pyplot as plt

#%%
# Let's define the population is in normal distribution with 10000 data points
x_min = 0
x_max = 10000
x= np.arange(x_min, x_max, 1)
y = gaussian_dist(x=x, mu=5000, std=500)
population_noraml = pd.DataFrame({'x':x, 'y':y})
plt.plot(population_noraml['x'], population_noraml['y'])
plt.show()
# fig = go.Figure()
# fig.add_trace(go.Scatter(x = population_noraml['x'],y= population_noraml['y']))
# fig.show()

#%%
# Take 500 samples from the population
sample = population_noraml.sample(300).reset_index(drop=True)
plt.scatter(sample['x'], sample['y'])
plt.show()



