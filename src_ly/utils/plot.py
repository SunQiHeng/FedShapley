import matplotlib.pyplot as plt
import numpy as np


def draw(y, y1, title):
    plt.title(title)
    # x = np.array([1, x])
    # y = np.array([0,1])
    x = np.arange(len(y1))

    plt.scatter(x,y)
    plt.scatter(x,y1)

    plt.show()
    plt.savefig('res.png', dpi=300)


if __name__ == '__main__':
    draw([1,2,3],[2,3,4],'test')
