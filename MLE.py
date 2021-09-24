import pandas as pd
import numpy as np
from exponential_familes import gaussian_dist
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = 'png'
import matplotlib.pyplot as plt
import random
from functools import reduce

#%%
'''
Let's define the population in normal distribution with 5,000 data points with mu at 500 and std with 50
'''
x_min = 0
x_max = 1000
x = np.arange(x_min, x_max, 1)
y = gaussian_dist(x = x,
                  mu = (x_min + x_max)/2,
                  std = 50)
population_dist = pd.DataFrame({'x':x, 'y':y})
plt.plot(population_dist['x'], population_dist['y'])
plt.show()

n = 0
population = []
while n < 5000:
    population.append(np.random.choice(population_dist['x'], p=population_dist['y']))
    n += 1


#%%
'''
Take 1,000 samples from the population
'''
sample = random.sample(population, 100)
plt.hist(sample)
plt.show()

#%%
'''
Assuming the std is known and we are estimating θ
likelihood = ∏(p(D|θ))
if θ = 100 ...
'''
prob_sample = gaussian_dist(np.array(sample), mu=100, std=50)
likelihood_100 = reduce(lambda x,y: x*y, prob_sample)
print(likelihood_100)

'''
Product of probabilities becomes very small, so let's take log of the values
log likelihood = ∑(log(p(D|θ)))
'''

thetas = [100, 300, 500, 600, 700]
likelihoods = []
for theta in thetas:
    likelihoods.append(np.sum(gaussian_dist(np.array(sample), mu=theta, std=50)))

plt.bar(x=[str(t) for t in thetas], height=likelihoods)
plt.xlabel('Theta')
plt.ylabel('Likelihood')
plt.show()