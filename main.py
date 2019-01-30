"""
author:taurus
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats
def plot_hist(x):
    print("plot samples:{}".format(len(x)))
    plt.hist(X, bins=100, normed=True, facecolor='green', alpha=0.5)

    
def uni_sampling(n_sample=100):
    return np.random.rand(n_sample)

def binom_sampling_self(n_sample=100, p=0.8):
    orig = np.random.rand(n_sample)
    samples = np.zeros(n_sample)
    for i,x in enumerate(orig):
        if x < 1 - p:
            samples[i] = 0
        else:
            samples[i] = 1
    return samples

def binom_sampling(n_sample=100, p=0.8):
    return stats.binom.rvs(n=1, p=p, size=n_sample)

def multinom_sampling(n_sample=100, p=[0.1,0.3,0.6]):
    orig = stats.multinomial.rvs(n=1, p=p, size=n_sample)
    samples = np.argwhere(orig == 1)[:, 1]
    return samples

def multinom_sampling_self(n_sample=100, p=[0.1,0.3,0.6]):
    orig = np.random.rand(n_sample)
    samples = np.zeros(n_sample)
    for i,x in enumerate(orig):
        
        start_p = 0
        for j in range(len(p)):
            end_p = start_p + p[j]
            if start_p <= x and x < end_p:
                samples[i] = j
                break
            start_p += p[j]

    return samples

def reject_sampling(n_sample):
    orig = stats.norm.rvs(0, 1, size=n_sample * 2)
    samples = []
    for xi in orig:
        u = np.random.rand(1)[0]
        if u < stats.norm.pdf(xi) / (stats.norm.pdf(xi) * 1.51):
            samples.append(xi)
    return np.array(samples)
def draw_pdf(funcs, points=np.arange(-10, 10, 0.1)):
#     f = stats.norm.pdf(x, 0, 1)
#     gc = stats.norm.pdf(x, 0, 1) * 1.51
    for func in funcs:
        
        if len(points) == 1:
            val = func.pdf(points)
            plt.plot(points, val)
        elif len(points) == 2:
            val = func.pdf(np.dstack(points))
            fig1 = plt.figure()
            ax = fig1.add_subplot(111)
            ax = Axes3D(fig1)
            ax.plot_surface(points[0], points[1], val, rstride=1, cstride=1, cmap=plt.cm.coolwarm)
            
            fig2 = plt.figure()
            ax2 = fig2.add_subplot(111)
            ax2.contourf(points[0], points[1], val)

    
def metropolis_hastings_samling(n_sample, distribute_func):
    for i in range(n_sample):
        u = np.random.rand(1)[0]
        
    X = np.random.multivariate_normal(mean=[0,0], cov=[[1, 0], [0,1]], size = n_sample)
    plt.scatter(X[:,0], X[:,1])

def get_distribute(name="multivariate_normal"):
    func = stats.multivariate_normal
    return func



def test_matrix():
#     x = np.array([[0.65, 0.28, 0.07],
#                   [0.15, 0.67, 0.18],
#                   [0.12, 0.36, 0.52]])
    x = np.array([[0.9, 0.02],
                  [0.1, 0.98]])
    a, b = np.linalg.eig(x)
    inv_b = np.linalg.inv(b)
    print(a)
    print(b)
    print(inv_b)
#     l = np.array([[1, 0.0, 0.0],
#                   [0.0, 0.51848858, 0.0],
#                   [0.0, 0.0, 0.3215142]])
    l = np.array([[0.88, 0],
                  [0, 1]])
    c = inv_b * x * b
    print(c)
if __name__ == "__main__":
    func = stats.multivariate_normal([0.5, -0.2], [[2.0, 0.3], [0.3, 0.5]])
    funcs = [func]
    x, y = np.mgrid[-1:1:.01, -1:1:.01]
    points = [x, y]
    draw_pdf(funcs, points)
     
    X = metropolis_hastings_samling(10000, funcs[0])
#     plot_hist(X)
     
    plt.show()
#     test_matrix()
    
    