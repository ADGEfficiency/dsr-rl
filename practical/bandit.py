from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


alpha = 0.0001
q = 10
omega = 0

stats = defaultdict(list)

for step in range(50):

    stats['q'].append(q)
    stats['omega'].append(omega)

    omega = omega + alpha * (1 - omega)
    beta = alpha / omega
    stats['beta'].append(beta)

    reward = np.random.normal(loc=5, scale=1)
    stats['reward'].append(reward)

    q += beta * (reward - q)

result = pd.DataFrame().from_dict(stats)

f, a = plt.subplots(nrows=4)

result.plot(y='reward', ax=a[0])
result.plot(y='q', ax=a[1])
result.plot(y='omega', ax=a[2])
result.plot(y='beta', ax=a[3])

print('final estimate {}'.format(stats['q'][-1]))

f.show()
