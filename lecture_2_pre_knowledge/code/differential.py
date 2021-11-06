import numpy as np
import d2l.torch as d2l


def f(x):
    return x * x - x - 1


x = np.arange(0, 3, 0.1)
d2l.plot(x, [f(x), 2 * x - 3], 'x', 'f(x)', legend=['f(x)', 'Tangent line (x=1)'])
d2l.plt.show()
