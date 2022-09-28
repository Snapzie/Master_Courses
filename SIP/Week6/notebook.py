#!/usr/bin/env python3
# coding: utf-8

# In[1]:


import numpy as np
import scipy.ndimage as sn
import scipy.signal as ss
import matplotlib.pyplot as plt
from skimage.io import imsave
from skimage import morphology as mp
from skimage import transform
from skimage import feature
from skimage import filters
from scipy.ndimage import gaussian_filter
import math
from skimage.color import rgb2hsv, hsv2rgb
from skimage.util import random_noise as rn
from PIL.Image import open as imread
from PIL import ImageOps as Iops
import time
import scipy


# In[2]:


def showG(im, boundaries = False):
    if np.average(im) < 1:
        im = im*255
    if boundaries:
        plt.imshow(im, cmap='gray', vmin=0, vmax=255)
    else:    
        plt.imshow(im, cmap='gray')
    return plt.show()


def loadIm(im):
    im = imread(im)
    g = Iops.grayscale(im)
    A = np.array(g)
    showG(A)
    return A


# In[3]:


A1 = loadIm('hand.tiff')
A2 = loadIm('modelhouses.png')
A3 = loadIm('sunflower.tiff')


# In[4]:


# Part 1.1

res = feature.canny(A1,sigma=0.5, low_threshold=1, high_threshold=50.0)
showG(res)

res = feature.canny(A1,sigma=1.5, low_threshold=50.0, high_threshold=200.0)
showG(res)

res = feature.canny(A1,sigma=1.5, low_threshold=1, high_threshold=50.0)
showG(res)

res = feature.canny(A1,sigma=5, low_threshold=1, high_threshold=50.0)
showG(res)


# In[5]:


# Part 1.2
showG(A2)

res = feature.corner_harris(A2, k = 0.0001, sigma=0.5)
x,y = np.where(res > 0.5)
print(len(x))
fig, ax = plt.subplots()
ax.imshow(A2, cmap='gray')
ax.scatter(y,x, marker='x', color='red', s=.5)
plt.show()

res = feature.corner_harris(A2, k = 0.1, sigma=0.5)
x,y = np.where(res > 0.5)
print(len(x))
fig, ax = plt.subplots()
ax.imshow(A2, cmap='gray')
ax.scatter(y,x, marker='x', color='red', s=.5)
plt.show()

res = feature.corner_harris(A2, method='eps', eps=0.01, sigma=0.5)
x,y = np.where(res > 0.5)
print(len(x))
fig, ax = plt.subplots()
ax.imshow(A2, cmap='gray')
ax.scatter(y,x, marker='x', color='red', s=.5)
plt.show()

res = feature.corner_harris(A2, method='eps', eps=5, sigma=0.5)
x,y = np.where(res > 0.5)
print(len(x))
fig, ax = plt.subplots()
ax.imshow(A2, cmap='gray')
ax.scatter(y,x, marker='x', color='red', s=.5)
plt.show()

res = feature.corner_harris(A2, k = 0.0001, sigma=0.5)
x,y = np.where(res > 0.5)
print(len(x))
fig, ax = plt.subplots()
ax.imshow(A2, cmap='gray')
ax.scatter(y,x, marker='x', color='red', s=.5)
plt.show()

res = feature.corner_harris(A2, k = 0.0001, sigma=5)
x,y = np.where(res > 0.5)
print(len(x))
fig, ax = plt.subplots()
ax.imshow(A2, cmap='gray')
ax.scatter(y,x, marker='x', color='red', s=.5)
plt.show()


# In[6]:


# Part 1.3
def pad_to_square(a, pad_value=0):
  m = a.reshape((a.shape[0], -1))
  padded = pad_value * np.ones(2 * [max(m.shape)], dtype=m.dtype)
  padded[0:m.shape[0], 0:m.shape[1]] = m
  return padded


def A(im, sig, k):
    im = pad_to_square(im)
    G = filters.gaussian(im, sig*k)
    gf = gaussian_filter
    Lx = gf(im, sig, order=(0,1))
    Ly = gf(im, sig, order=(1,0))
    
    return np.array([
        [Lx**2, Lx*Ly],
        [Lx*Ly, Ly**2]
    ])

a = A(A2, .5, 0.001).T
alpha = 1
det = np.linalg.det(a)
t = alpha*np.trace(a.T)**2
res = det.T - t
print(res.shape)
corners = res


# In[7]:


corners = feature.corner_peaks(np.abs(res)/np.max(np.abs(res)), threshold_rel=0.85, indices=False)
x,y = np.where(corners != 0)
print(len(x))
fig, ax = plt.subplots()
ax.imshow(A2, cmap='gray')
ax.scatter(y,x, marker='x', color='red', s=.5)    
plt.show()


# In[8]:


# 2.1
def gaus(sigma, n = None):
    if n is None:
        n = 3 * sigma
    x, y = np.mgrid[-n:n+1, -n:n+1]
    H = np.exp(-(x**2 + y**2)/(2*sigma**2))
    H *= 1 / (2 * np.pi * sigma**2)

    return H

B = gaus(25, 100)
for tau in [1,2,4,8,16]:
    G = gaus(tau, 10)
    plt.imshow(ss.convolve2d(B, G), cmap='gray')
    plt.colorbar()
    plt.show()


# In[9]:


B = gaus(1, 5)
G = gaus(2, 5)
M = gaus(np.sqrt(1**2+2**2), 5)

plt.imshow(B, cmap='gray')
plt.colorbar()
plt.show()

plt.imshow(G, cmap='gray')
plt.colorbar()
plt.show()

plt.imshow(ss.convolve2d(B,G, mode='same'), cmap='gray')
plt.colorbar()
plt.show()

plt.imshow(M, cmap='gray')
plt.colorbar()
plt.show()

plt.imshow(M-ss.convolve2d(B,G, mode='same'), cmap='gray')
plt.colorbar()
plt.show()


# In[10]:


def laplacian(sigma):
    n = 3.8 * sigma
    x, y = np.mgrid[-n:n+1, -n:n+1]
    H = 1 / (2 * np.pi * sigma**2)
    H *= ((x**2 + y**2 - 2 * sigma**2) / sigma**4)
    H *= (np.exp((-(x**2 + y**2)) / (2 * sigma**2)))
    return H


# In[55]:


def H(x,y,t):
    a = t**2
    b = (-1/(np.pi*np.sqrt(1**2+t**2)**4))
    c = (1-((x**2+y**2)/(2*np.sqrt(1**2+t**2)**2)))
    d = np.exp(-(x**2+y**2)/(2*np.sqrt(1**2+t**2)**2))
    return a*(b*c*d)


# In[56]:


T = np.arange(-2,2,0.01)
ys = []
for tau in T:
    ys.append(H(0,0,tau))

plt.plot(T, ys)
plt.xlabel('tau')
plt.ylabel('H(0,0,tau)')
plt.show()


# In[20]:


# 2.4
taus = np.arange(1,30, 5)

res = []

for tau in taus:
    res.append(tau**2 * scipy.ndimage.gaussian_laplace(A3/255,tau))

def abs_sort(ele):
    return np.absolute(ele[3])

extrema = []
copy = np.array(res)
z,y,x = copy.shape
zero = np.zeros((y,x))
print(copy.shape)
stack = np.pad(copy, 1, mode='constant', constant_values=(np.mean(A3/255)))
print(stack.shape)
for i in range(1,z+1):
    for j in range(1,y+1):
        for k in range(1,x+1):
            vs = np.delete(stack[i-1:i+2,j-1:j+2,k-1:k+2], [13])
            if(stack[i,j,k] > np.max(vs)):
                extrema.append([i,j,k, stack[i,j,k]])
            elif(stack[i,j,k] < np.min(vs)):
                extrema.append([i,j,k, stack[i,j,k]])


# In[23]:


S = np.array(sorted(extrema, key=abs_sort)[-150:])
print(S.shape)
fig, ax = plt.subplots()
ax.imshow(A3,cmap='gray')
zipped = zip(S[:,2],S[:,1],S[:,0],S[:,3])
for x,y,r,v in zipped:
    if(v > 0):
        c = plt.Circle((x,y),r*8,color='red', fill=False)
    else:
        c = plt.Circle((x,y),r*8,color='blue', fill=False)
    ax.add_patch(c)
plt.show()


# In[25]:


# 3.1
def S(X, sigma = 1):
    xs = []
    for x in range(-X,X+1):
        val = (1/np.sqrt(2*np.pi*sigma**2))*np.exp(-((x**2)/(2*sigma**2)))
        xs.append(val)
    return np.cumsum(xs)


# In[28]:


sig = 20
size = 40
im = np.zeros((2*size+1,2*size+1))
for _ in range(size*2+1):
    im[_:,] = S(size, sig)

s = [0.01,0.1,1.0,10.0,100.0,1000.0]
for sigma in s:
    plt.imshow(scipy.ndimage.gaussian_filter(im,sigma),cmap='gray')
    plt.colorbar()
    plt.show()
    


# In[61]:


#3.2.b
def H3(x,y,t):
    return 2*t*((1/(np.sqrt(2*np.pi*np.sqrt(1**2+t**2)))**2)*np.exp(-(x**2+y**2)/(2*np.sqrt(1**2+t**2))))**2

T = np.arange(-2,2,0.01)
ys = []
for tau in T:
    #print(H(0,0,tau,1))
    ys.append(H3(0,0,tau))

plt.plot(T, ys)
plt.xlabel('tau')
plt.ylabel('||∇J(0,0,tau)||^2')
plt.show()


# In[36]:


#3.3
taus = np.arange(1,30, 1) # First change made

res = []

for tau in taus:
    res.append(tau * (scipy.ndimage.gaussian_filter(A1/255,tau,order=(0,1)))**2 + 
        tau * (scipy.ndimage.gaussian_filter(A1/255,tau,order=(1,0)))**2)

extrema = []
copy = np.array(res)
z,y,x = copy.shape
zero = np.zeros((y,x))
print(copy.shape)
stack = np.pad(copy, 1, mode='constant', constant_values=(0)) # Last change made
print(stack.shape)
for i in range(1,z+1):
    for j in range(1,y+1):
        for k in range(1,x+1):
            vs = np.delete(stack[i-1:i+2,j-1:j+2,k-1:k+2], [13])
            if(stack[i,j,k] > np.max(vs)):
                extrema.append([i,j,k, stack[i,j,k]])
            elif(stack[i,j,k] < np.min(vs)):
                extrema.append([i,j,k, stack[i,j,k]])


# In[35]:


S = np.array(sorted(extrema, key=abs_sort)[-200:])
print(S.shape)
fig, ax = plt.subplots()
ax.imshow(A1,cmap='gray')
zipped = zip(S[:,2],S[:,1],S[:,0],S[:,3])
for x,y,r,v in zipped:
    if(v > 0):
        c = plt.Circle((x,y),r*3,color='red', fill=False)
    else:
        c = plt.Circle((x,y),r*8,color='blue', fill=False)
    ax.add_patch(c)
plt.show()


# In[ ]:





# <a style='text-decoration:none;line-height:16px;display:flex;color:#5B5B62;padding:10px;justify-content:end;' href='https://deepnote.com?utm_source=created-in-deepnote-cell&projectId=c077d00b-8231-47d9-8b65-6b95f56b4d02' target="_blank">
# <img style='display:inline;max-height:16px;margin:0px;margin-right:7.5px;' src='data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyB3aWR0aD0iODBweCIgaGVpZ2h0PSI4MHB4IiB2aWV3Qm94PSIwIDAgODAgODAiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bWxuczp4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8IS0tIEdlbmVyYXRvcjogU2tldGNoIDU0LjEgKDc2NDkwKSAtIGh0dHBzOi8vc2tldGNoYXBwLmNvbSAtLT4KICAgIDx0aXRsZT5Hcm91cCAzPC90aXRsZT4KICAgIDxkZXNjPkNyZWF0ZWQgd2l0aCBTa2V0Y2guPC9kZXNjPgogICAgPGcgaWQ9IkxhbmRpbmciIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJBcnRib2FyZCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoLTEyMzUuMDAwMDAwLCAtNzkuMDAwMDAwKSI+CiAgICAgICAgICAgIDxnIGlkPSJHcm91cC0zIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxMjM1LjAwMDAwMCwgNzkuMDAwMDAwKSI+CiAgICAgICAgICAgICAgICA8cG9seWdvbiBpZD0iUGF0aC0yMCIgZmlsbD0iIzAyNjVCNCIgcG9pbnRzPSIyLjM3NjIzNzYyIDgwIDM4LjA0NzY2NjcgODAgNTcuODIxNzgyMiA3My44MDU3NTkyIDU3LjgyMTc4MjIgMzIuNzU5MjczOSAzOS4xNDAyMjc4IDMxLjY4MzE2ODMiPjwvcG9seWdvbj4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik0zNS4wMDc3MTgsODAgQzQyLjkwNjIwMDcsNzYuNDU0OTM1OCA0Ny41NjQ5MTY3LDcxLjU0MjI2NzEgNDguOTgzODY2LDY1LjI2MTk5MzkgQzUxLjExMjI4OTksNTUuODQxNTg0MiA0MS42NzcxNzk1LDQ5LjIxMjIyODQgMjUuNjIzOTg0Niw0OS4yMTIyMjg0IEMyNS40ODQ5Mjg5LDQ5LjEyNjg0NDggMjkuODI2MTI5Niw0My4yODM4MjQ4IDM4LjY0NzU4NjksMzEuNjgzMTY4MyBMNzIuODcxMjg3MSwzMi41NTQ0MjUgTDY1LjI4MDk3Myw2Ny42NzYzNDIxIEw1MS4xMTIyODk5LDc3LjM3NjE0NCBMMzUuMDA3NzE4LDgwIFoiIGlkPSJQYXRoLTIyIiBmaWxsPSIjMDAyODY4Ij48L3BhdGg+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMCwzNy43MzA0NDA1IEwyNy4xMTQ1MzcsMC4yNTcxMTE0MzYgQzYyLjM3MTUxMjMsLTEuOTkwNzE3MDEgODAsMTAuNTAwMzkyNyA4MCwzNy43MzA0NDA1IEM4MCw2NC45NjA0ODgyIDY0Ljc3NjUwMzgsNzkuMDUwMzQxNCAzNC4zMjk1MTEzLDgwIEM0Ny4wNTUzNDg5LDc3LjU2NzA4MDggNTMuNDE4MjY3Nyw3MC4zMTM2MTAzIDUzLjQxODI2NzcsNTguMjM5NTg4NSBDNTMuNDE4MjY3Nyw0MC4xMjg1NTU3IDM2LjMwMzk1NDQsMzcuNzMwNDQwNSAyNS4yMjc0MTcsMzcuNzMwNDQwNSBDMTcuODQzMDU4NiwzNy43MzA0NDA1IDkuNDMzOTE5NjYsMzcuNzMwNDQwNSAwLDM3LjczMDQ0MDUgWiIgaWQ9IlBhdGgtMTkiIGZpbGw9IiMzNzkzRUYiPjwvcGF0aD4KICAgICAgICAgICAgPC9nPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+' > </img>
# Created in <span style='font-weight:600;margin-left:4px;'>Deepnote</span></a>
