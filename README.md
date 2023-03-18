# tf2-lr-schedulers
 Modern LearningRateSchedulers in Tensorflow2
![LRS_comparison](images/Different_LR_schedulers.png)

## Example
https://arxiv.org/abs/1706.02677

![Goyal et al paper](images/Goyal_et_al_image.png)

```python
from tf2_lr_schedulers import Goyal_LR
import matplotlib.pyplot as plt
x = np.array(range(300))
plt.plot(x, Goyal_LR(x, 1))
plt.yscale('log')
```
![Goyal et al LR](images/Goyal_et_al_LR.png)
