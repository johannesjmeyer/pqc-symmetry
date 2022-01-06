# %%
import numpy as np
from matplotlib import pyplot as plt

symm = []
for i in range(2):
    symm.append(np.load('output/symm{}.npy'.format(i), allow_pickle=True))

asymm = []
for i in range(2):
    asymm.append(np.load('output/asymm{}.npy'.format(i), allow_pickle=True))

# %%
print('Symmetric runs:')
for i in symm:
    print(str(i.item()['accuracy']) + ' n={}'.format(3*i.item()['sample size']) + ' s={}'.format(i.item()['steps']+1))
    y = [j.detach().numpy() for j in i.item()['cost function']]
    plt.plot(y, label='symmetric', color='C0')

print('Asymmetric runs:')
for i in asymm:
    print(str(i.item()['accuracy']) + ' n={}'.format(3*i.item()['sample size']) + ' s={}'.format(i.item()['steps']+1))
    y = [j.detach().numpy() for j in i.item()['cost function']]
    plt.plot(y, label='asymmetric', color='C1')

plt.ylabel('cost function')
plt.xlabel('steps')
plt.legend()
plt.show()
# %%
