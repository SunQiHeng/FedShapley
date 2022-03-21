import matplotlib.pyplot as plt
import numpy as np


def draw(x, y, title):
    plt.axis([1, x, 0, 1])
    plt.title(title)
    # x = np.array([1, x])
    # y = np.array([0,1])
    plt.plot(y)
    plt.show()
    plt.savefig('res.png', dpi=300)


if __name__ == '__main__':
    draw()
