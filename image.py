""" @author: Muhammad Ali Qureshi """

# Importing Libraries
import cv2
import matplotlib.pyplot as plt
import numpy as np
import glob

# Select The Path
path = "C:/.../stinkbug.png"
for file in glob.glob(path):
    img=cv2.imread(file)[...,::-1]

# For RGB Image
b, g, r    = img[:, :, 0], img[:, :, 1], img[:, :, 2] 

# Auto adjusting frame height and width
height = int(np.size(img, 0))
width = int(np.size(img, 1))
num=height*width

# Lorenz Attractor for Generating Chaos
def lorenz(x, y, z, s=10, r=28, b=2.667):
    '''
    Given:
       x, y, z: a point of interest in three dimensional space
       s, r, b: parameters defining the lorenz attractor
    Returns:
       x_dot, y_dot, z_dot: values of the lorenz attractor's partial
           derivatives at the point x, y, z
    '''
    x_dot = s*(y - x)
    y_dot = r*x - y - x*z
    z_dot = x*y - b*z
    return x_dot, y_dot, z_dot

dt = 0.01
num_steps =num

# Need One More For The Initial Values
xs = np.empty(num_steps + 1)
ys = np.empty(num_steps + 1)
zs = np.empty(num_steps + 1)

# Set Initial Values
xs[0], ys[0], zs[0] = (0., 1., 1.05)

# Step through "time", calculating the partial derivatives at the current point
# and using them to estimate the next point
for i in range(num_steps):
    x_dot, y_dot, z_dot = lorenz(xs[i], ys[i], zs[i])
    xs[i + 1] = xs[i] + (x_dot * dt)
    ys[i + 1] = ys[i] + (y_dot * dt)
    zs[i + 1] = zs[i] + (z_dot * dt)

xr=xs;yr=ys;zr=zs

# Data Slicing and Reshaping for Encryption
xr1=np.array(xr[0:num]);yr1=np.array(yr[0:num]);zr1=np.array(zr[0:num])
rr=r;bb=b;gg=g
xr2=np.int16(xr1)*1000;yr2=np.int16(yr1)*1000;zr2=np.int16(zr1)*1000
xrr=xr2.reshape(height,width)
yrr=yr2.reshape(height,width)
zrr=zr2.reshape(height,width)

"""Encryption Portion"""
R=np.bitwise_xor(rr,xrr)
G=np.bitwise_xor(gg,yrr)
B=np.bitwise_xor(bb,zrr)
R1=R.astype(np.uint8)
G1=G.astype(np.uint8)
B1=B.astype(np.uint8)

im = cv2.merge((R1, G1, B1))

"""Decryption Portion"""
R11=np.bitwise_xor(R,xrr)
G11=np.bitwise_xor(G,yrr)
B11=np.bitwise_xor(B,zrr)
dR=R11.astype(np.uint8)
dG=G11.astype(np.uint8)
dB=B11.astype(np.uint8)

im2 = cv2.merge((dR, dG, dB))[...,::-1]

"""ploting Portion with Histogram"""
fig, axs = plt.subplots(2,3, figsize=(20, 10))
axs[0,0].set_title('Original Image')
axs[0,0].imshow(img)
axs[1,0].set_title('Original Image Histogram')
axs[1,0].hist(img.ravel(),256,[0,256])
axs[0,1].set_title('Encrypted Image')
axs[0,1].imshow(im)
axs[1,1].set_title('Encrypted Image Histogram')
axs[1,1].hist(im.ravel(),256,[0,256])
axs[0,2].set_title('Decrypted Image')
axs[0,2].imshow(im2)
axs[1,2].set_title('Decrypted Image Histogram')
axs[1,2].hist(im2.ravel(),256,[0,256])