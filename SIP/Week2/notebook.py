#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import scipy.ndimage as sn
import scipy.signal as ss
import matplotlib.pyplot as plt
from skimage.io import imsave
import math
from skimage.color import rgb2hsv, hsv2rgb
from skimage.util import random_noise as rn
from PIL.Image import open as imread
from PIL import ImageOps as Iops
import time


# In[ ]:


im1 = imread('movie_flicker1.tif')
im2 = imread('movie_flicker2.tif')
im3 = imread('autumn.tif')
im4 = imread('pout.tif')
im5 = imread('eight.tif')
g1 = Iops.grayscale(im1)
g2 = Iops.grayscale(im2)
g4 = Iops.grayscale(im4)
g5 = Iops.grayscale(im5)
A1 = np.array(g1)
A2 = np.array(g2)
A3 = np.array(im3)
A3_cp = rgb2hsv(A3.copy())
A4 = np.array(g4)
A5 = np.array(g5)


# In[ ]:


plt.imshow(A1)


# In[ ]:


# Exercise 1
def showG(im, cmap = 'gray'):
    if np.average(im) < 1:
        im = im*255
    plt.imshow(im, cmap=cmap, vmin=0, vmax=255)
    return plt.show()

def showC(im, cmap = None):
    plt.imshow(im, cmap=cmap, vmin=0, vmax=255)
    return plt.show()

def gamma_map(im, gamma = 1.0):
    return (pow(im*(1/255), gamma)*255).astype(np.uint8)

# Exercise 2
def plot_hist(hist): 
    y, x = hist
    fig, ax = plt.subplots()
    ax.bar(x, y,width = 1)
    ax.set_xlabel('Pixel intensities')
    ax.set_ylabel('Probability')
    return plt.show()


def hist_homebrew(arr, bins):
    count = np.zeros(bins)
    hist = np.arange(bins)
    for i in hist:
        count[i] = sum(arr[arr == i])
    return count, hist
    

def hist_pre_processing(im, bins = 256):
    count, intensity = hist_homebrew(np.ravel(im), bins=bins)
    
    prob = count/sum(count)
    return (((bins-1) * np.cumsum(prob)) / (bins-1)), intensity


def hist_processing(im, bins = 256):
    c = im.copy()
    sums, intensity = hist_pre_processing(c, bins)
    for i, s in zip(intensity, sums):
        c = np.where(im == i, s*(bins-1), c)
    
    return np.round(c)


def inverse(cdf, l = 0.5):
    s, i = cdf
    return np.min(i[s >= l])


# Could be optimized probably, is quite slow rn
def hist_match(im1, im2):
    cdf1,_ = hist_pre_processing(im1)
    cdf2 = hist_pre_processing(im2)

    J = im1.copy()
    for x in range(len(im1)):
        for y in range(len(im1[0])):
            try:
                J[x,y] = inverse(cdf2, l = cdf1[im1[x,y]])
            except ValueError:  #raised if `y` is empty.
                J[x,y] = 255
    return J

def hist_match_uni(im1):
    cdf1,_ = hist_pre_processing(im1)
    uni = np.full((256), 1/256)
    unicdf = np.cumsum(uni)

    J = im1.copy()
    for x in range(len(im1)):
        for y in range(len(im1[0])):
            try:
                l = cdf1[im1[x,y]]
                I = np.min(np.where(unicdf >= l))
                J[x,y] = I
            except ValueError:  #raised if `y` is empty.
                J[x,y] = np.average(J)
    return J


# In[ ]:


showG(A1)
showG(gamma_map(A1, 0.1))
showG(gamma_map(A1, 0.4))
showG(gamma_map(A1, 2.0))


# In[ ]:


showC(A3)
showC(gamma_map(A3, 0.5))


# In[ ]:


A3_cp[:,:,2] = pow(rgb2hsv(A3)[:,:,2], 0.5)
showC(hsv2rgb(A3_cp))


# In[ ]:


# 2.1
plot_hist(hist_homebrew(A4, 256))
plot_hist(hist_pre_processing(A4))

# 2.2
showG(A4)
plt.imshow(hist_processing(A4) / 255, cmap='gray', vmin=0, vmax=1)


# In[ ]:


# 2.3
cdf = hist_pre_processing(A2)
inverse(cdf, l = 0.1)


# In[ ]:


showG(hist_match(A4,hist_processing(A4).astype(int)))


# In[ ]:


# 2.4
g = hist_match(A1, A4)
showG(A1)
showG(A4)
showG(g)
plot_hist(hist_pre_processing(A1)) # Original
plot_hist(hist_pre_processing(A4)) # Target
plot_hist(hist_pre_processing(g))  # Result


# In[ ]:


# 2.5
g = hist_match_uni(A4)
showG(g)
plot_hist(hist_pre_processing(A4)) # Original
plot_hist(hist_pre_processing(g))  # Result, should look like it has been equlized


# In[ ]:


import skimage.exposure as se
showG(se.equalize_hist(A4))
plot_hist(hist_pre_processing(np.round(se.equalize_hist(A4)*255)))


# In[ ]:





# In[ ]:


gaus = rn(A5, mode='gaussian')
showG(gaus)
sp = rn(A5, mode='s&p')
showG(sp)


# In[ ]:


def meanKernel(n):
    return np.ones((n,n)) * (1/n**2)

def runAndPlot(func, args, plot = True, times_per_arg = 1):
    res = [] # Set of results
    times = []

    for arg in args:
        t = 0
        for _ in range(times_per_arg):
            start = time.time()
            r = func(arg)
            end = time.time()
            t = t + (end - start)
        res.append(r)
        times.append(t/times_per_arg) # average time is saved

    if plot:
        plt.plot(args, times)
        plt.ylabel('Time in ms')
        plt.xlabel('Kernel size')
        plt.show()
    return res, times



# In[ ]:


# 3.1
r1, t1 = runAndPlot(lambda x: ss.convolve2d(sp, meanKernel(x)), range(1,26), times_per_arg=1)
showG(r1[4])
showG(r1[14])
showG(r1[-1])


# In[ ]:


# def grab(im, x, y, n, offset):
#     kernel = np.zeros((n,n))
#     for i in range(-offset, offset+1):
#         for j in range(-offset, offset+1):
#             kernel[i] = im[x+i,y+j]
#     return np.ravel(kernel)

def applyMedianKernel(image, n):#, pad_type='constant'):
    # offset = math.floor(n/2)
    # im = np.pad(image.copy(), offset, pad_type)
    
    
    # for x in range(image.shape[0]):
    #     for y in range(image.shape[1]):
    #         im[x+offset,y+offset] = np.median(grab(im, x+offset,y+offset, n, offset))
    # return im
    return sn.median_filter(image, size=n)


# In[ ]:


# 3.1
r, t = runAndPlot(lambda x: applyMedianKernel(sp, x), range(1, 26), times_per_arg=1)
showG(r[4])
showG(r[14])
showG(r[-1])


# In[ ]:


def gausK(sigma = 5, n = None):
    if n is None:
        n = 3*sigma
    x, y = np.mgrid[-n:n+1,-n:n+1]
    H = np.exp(-(x**2 + y**2) / (2*sigma**2))
    H *= 1 / (math.sqrt(2 * np.pi) * sigma**2) # Her?
    return H/np.sum(H)
showG(gausK(n = 1))


# In[ ]:


rg, tg = runAndPlot(lambda x: ss.convolve2d(gaus, gausK(n = x)), [2,5], plot=False, times_per_arg=1)
showG(rg[0]) # 10
showG(rg[1]) # 15
rg, tg = runAndPlot(lambda x: ss.convolve2d(sp, gausK(n = x)), [2,5], plot=False, times_per_arg=1)
showG(rg[0]) # 10
showG(rg[1]) # 15


# In[ ]:


rg2, tg2 = runAndPlot(lambda x: ss.convolve2d(gaus, gausK(sigma = x)), [2,5], plot=False, times_per_arg=1)
showG(rg2[0])
showG(rg2[1])
rg2, tg2 = runAndPlot(lambda x: ss.convolve2d(sp, gausK(sigma = x)), [2,5], plot=False, times_per_arg=1)
showG(rg2[0])
showG(rg2[1])


# In[ ]:


# 4
def bilateral_filtering(image, n, sigma = 5, tau = 5):
    l = k = math.floor(n/2)
    im = np.pad(image.copy(), l)
    
    imr = image.copy()

    def f(x,y):
        return np.exp(-(x**2 + y**2)/(2*sigma**2))

    def g(u):
        return np.exp(-(u**2)/(2*tau**2))

    def w(x,y,i,j):
        o = f(i,j)
        p = g(im[x+i,y+j]-im[x,y])
        return o*p

    def I(x, y):
        s1 = 0
        s2 = 0
        for i in range(-l, l+1):
            for j in range(-k, k+1):
                v = w(x,y,i,j)
                s1 = s1 + v
                s2 = s2 + v*im[x+i,y+j]

        return s2/s1

    
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            imr[x,y] = I(x+l,y+k) # Adding k and l removes padding from final image

    return imr


# In[ ]:


one = bilateral_filtering(gaus, 10, sigma=5, tau=5)
showG(one)
showG(gaus)
showG(one-gaus)


# In[ ]:


two = bilateral_filtering(gaus, 10, 20, 5)
showG(two)
showG(one-two)


# In[ ]:


three = bilateral_filtering(gaus, 10, 5, 20)
showG(three)
showG(two-three)


# In[ ]:


four = bilateral_filtering(gaus, 10, 20, 20)
showG(four)
showG(three-four)


# In[ ]:



showG((one-three)/(np.sum(one-three))*10000)


# In[ ]:





# In[ ]:





# <a style='text-decoration:none;line-height:16px;display:flex;color:#5B5B62;padding:10px;justify-content:end;' href='https://deepnote.com?utm_source=created-in-deepnote-cell&projectId=119cedc2-5afe-4f3b-8288-e9309c214ff0' target="_blank">
# <img style='display:inline;max-height:16px;margin:0px;margin-right:7.5px;' src='data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyB3aWR0aD0iODBweCIgaGVpZ2h0PSI4MHB4IiB2aWV3Qm94PSIwIDAgODAgODAiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bWxuczp4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8IS0tIEdlbmVyYXRvcjogU2tldGNoIDU0LjEgKDc2NDkwKSAtIGh0dHBzOi8vc2tldGNoYXBwLmNvbSAtLT4KICAgIDx0aXRsZT5Hcm91cCAzPC90aXRsZT4KICAgIDxkZXNjPkNyZWF0ZWQgd2l0aCBTa2V0Y2guPC9kZXNjPgogICAgPGcgaWQ9IkxhbmRpbmciIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJBcnRib2FyZCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoLTEyMzUuMDAwMDAwLCAtNzkuMDAwMDAwKSI+CiAgICAgICAgICAgIDxnIGlkPSJHcm91cC0zIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxMjM1LjAwMDAwMCwgNzkuMDAwMDAwKSI+CiAgICAgICAgICAgICAgICA8cG9seWdvbiBpZD0iUGF0aC0yMCIgZmlsbD0iIzAyNjVCNCIgcG9pbnRzPSIyLjM3NjIzNzYyIDgwIDM4LjA0NzY2NjcgODAgNTcuODIxNzgyMiA3My44MDU3NTkyIDU3LjgyMTc4MjIgMzIuNzU5MjczOSAzOS4xNDAyMjc4IDMxLjY4MzE2ODMiPjwvcG9seWdvbj4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik0zNS4wMDc3MTgsODAgQzQyLjkwNjIwMDcsNzYuNDU0OTM1OCA0Ny41NjQ5MTY3LDcxLjU0MjI2NzEgNDguOTgzODY2LDY1LjI2MTk5MzkgQzUxLjExMjI4OTksNTUuODQxNTg0MiA0MS42NzcxNzk1LDQ5LjIxMjIyODQgMjUuNjIzOTg0Niw0OS4yMTIyMjg0IEMyNS40ODQ5Mjg5LDQ5LjEyNjg0NDggMjkuODI2MTI5Niw0My4yODM4MjQ4IDM4LjY0NzU4NjksMzEuNjgzMTY4MyBMNzIuODcxMjg3MSwzMi41NTQ0MjUgTDY1LjI4MDk3Myw2Ny42NzYzNDIxIEw1MS4xMTIyODk5LDc3LjM3NjE0NCBMMzUuMDA3NzE4LDgwIFoiIGlkPSJQYXRoLTIyIiBmaWxsPSIjMDAyODY4Ij48L3BhdGg+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMCwzNy43MzA0NDA1IEwyNy4xMTQ1MzcsMC4yNTcxMTE0MzYgQzYyLjM3MTUxMjMsLTEuOTkwNzE3MDEgODAsMTAuNTAwMzkyNyA4MCwzNy43MzA0NDA1IEM4MCw2NC45NjA0ODgyIDY0Ljc3NjUwMzgsNzkuMDUwMzQxNCAzNC4zMjk1MTEzLDgwIEM0Ny4wNTUzNDg5LDc3LjU2NzA4MDggNTMuNDE4MjY3Nyw3MC4zMTM2MTAzIDUzLjQxODI2NzcsNTguMjM5NTg4NSBDNTMuNDE4MjY3Nyw0MC4xMjg1NTU3IDM2LjMwMzk1NDQsMzcuNzMwNDQwNSAyNS4yMjc0MTcsMzcuNzMwNDQwNSBDMTcuODQzMDU4NiwzNy43MzA0NDA1IDkuNDMzOTE5NjYsMzcuNzMwNDQwNSAwLDM3LjczMDQ0MDUgWiIgaWQ9IlBhdGgtMTkiIGZpbGw9IiMzNzkzRUYiPjwvcGF0aD4KICAgICAgICAgICAgPC9nPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+' > </img>
# Created in <span style='font-weight:600;margin-left:4px;'>Deepnote</span></a>
