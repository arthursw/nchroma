# #%%
# import numpy as np
# import matplotlib.pyplot as plt
# import scipy

# y = np.random.rand((10))
# y = (y*5).astype(np.int32)

# x = np.arange(len(y))
# x, y
# #%%

# # Greatly increase resolution:
# f = scipy.interpolate.interp1d(x, y, kind='previous')

# x2 = np.arange(0, len(y)-1, 0.1)
# y2 = f(x2)
# x2, y2
# #%%

# fig, ax = plt.subplots()
# ax.plot(x2, y2, label='Interpolated Data')  # Plot interpolated data
# ax.scatter(x, y, color='red', label='Original Data')  # Scatter plot of original data
# ax.legend()  # Display legend
# plt.show()  # Show the combined plot

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, UnivariateSpline, Akima1DInterpolator, KroghInterpolator, PchipInterpolator, barycentric_interpolate

# Generate a random numpy array Y of 10 integers
np.random.seed(42)  # Setting seed for reproducibility
Y = np.random.randint(0, 100, 10)

# Generate x values for plotting
x = np.linspace(0, 9, 100)

# Linear Interpolation
linear_interp = interp1d(np.arange(10), Y, kind='linear')(x)

# Nearest-neighbor Interpolation
nearest_interp = interp1d(np.arange(10), Y, kind='nearest')(x)

# Cubic Spline Interpolation
cubic_interp = interp1d(np.arange(10), Y, kind='cubic')(x)

# Univariate Spline Interpolation
spline_interp = UnivariateSpline(np.arange(10), Y)(x)

# Akima Interpolation
akima_interp = Akima1DInterpolator(np.arange(10), Y)(x)

# Krogh Interpolation
krogh_interp = KroghInterpolator(np.arange(10), Y)(x)

# Pchip Interpolation
pchip_interp = PchipInterpolator(np.arange(10), Y)(x)

# barycentric_interp = barycentric_interpolate(np.arange(10), Y, x)


# Plotting
plt.figure(figsize=(10, 8))
plt.plot(np.arange(10), Y, 'o', label='Original Points')
plt.plot(x, linear_interp, label='Linear Interpolation')
# plt.plot(x, nearest_interp, label='Nearest-neighbor Interpolation')
# plt.plot(x, cubic_interp, label='Cubic Spline Interpolation')
# plt.plot(x, spline_interp, label='Univariate Spline Interpolation')
# plt.plot(x, akima_interp, label='Akima Interpolation')
# plt.plot(x, krogh_interp, label='Krogh Interpolation')
plt.plot(x, pchip_interp, label='Pchip Interpolation')

# plt.plot(x, barycentric_interp, label='barycentric_interp')

plt.legend()
plt.title('1D Interpolation Methods in Scipy')
plt.xlabel('Index')
plt.ylabel('Value')
plt.show()
# %%

import numpy as np
from scipy.interpolate import pchip_interpolate
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, UnivariateSpline, Akima1DInterpolator, KroghInterpolator, PchipInterpolator, barycentric_interpolate
# Step1: Generate numpy array of 10 random integers
A = np.random.rand(11)

# Step2: Interpolate it using PCHIP method
x = np.arange(len(A))  # x-coordinates i.e., indices
scale = 100
n = len(A)
x_new = np.linspace(0, n-1, num=10*n)  # New x-coordinates, for better curve plotting
A_pchip = pchip_interpolate(x, A, x_new)  # Performing pchip interpolation

# Step3: Setup plots
fig, axes = plt.subplots(nrows=2, ncols=1)

# Step4: Plot the PCHIP interpolation
axes[0].plot(x_new, A_pchip, label='pchip interpolation')
axes[0].scatter(x, A, color='red')  # To mark actual points
axes[0].legend()
axes[0].set_title('PCHIP Interpolation')

# # Step5: Perform spreading along line, Here we will just normalize values
# normalized_A = np.interp(A, (A.min(), A.max()), (-1, +1))  # This spreads the values between -1 and 1 along line

# # Step6: Plot the spreading
# axes[1].scatter(np.cumsum(x_new), np.random.rand(len(x_new)), color='blue')  # To mark spread points
# axes[1].set_title('Spreading along line')

cumsum = np.cumsum(A_pchip)
# x = np.linspace(0, 99, 1000)
# linear_interp = interp1d(np.arange(len(cumsum)), cumsum, kind='linear')(x)
n_points = 60
indices = np.flatnonzero(np.diff((cumsum) % (cumsum.max()/n_points)) < 0) / scale

xs = np.array([(s/scale, s/scale) for s in indices])
zs = np.array([(0, 1) for _ in indices])
c = np.zeros((xs.shape[0] + zs.shape[0],2,))
c[0::2,:] = xs[:]
c[1::2] = zs

axes[1].plot(*c)


# axes[1].plot(*c, color='blue')

# axes[1].plot(x_new, xs, color='blue')

# Step7: Display plots
plt.tight_layout()
plt.show()

# %%
from IPython.display import SVG, display


# %%
SVG(url='http://upload.wikimedia.org/wikipedia/en/a/a4/Flag_of_France.svg')
# %%
import svgwrite

zigzag_size = (1,1)
zigzag_shape = ['M', (0, 1/2), 'S', (1/3, 0), (1/3, 0), 'S', (2/3, 1), (2/3, 1), 'S', (1, 1/2), (1, 1/2)]

def zigzag(position, size, shape):
    return [c if type(c) is str else tuple(np.array(position) + c * np.array(size)) for c in shape]

def shape_to_svg(shape):
    return ' '.join([str(p) for c in shape for p in c])

viewBox = f'0 0 10 10'
drawing = svgwrite.Drawing('output.svg', height=100, width=100, viewBox=viewBox)

scaleGroup = drawing.add(drawing.g(id='scale-group'))
strokeColor = 'black'
strokeWidth = 0.1
hlines = scaleGroup.add(drawing.g(id='hlines', stroke=(strokeColor), stroke_width=strokeWidth, opacity=1))

posx = [0] + indices
sizex = np.diff(indices, append=10)

shapes = []
for i, p in enumerate(posx):
    shapes += zigzag([p, 2.5], [sizex[i], 5], zigzag_shape if i==0 else zigzag_shape[2:])
#%%

hlines.add(drawing.path( d= shape_to_svg(shapes), fill='none' ) )
# hlines.add(drawing.rect((1, 1), (8, 8), stroke=svgwrite.rgb(10, 10, 16, '%'), fill='red'))

# hlines.add(drawing.path( d='M 0 0 S 1 2 10 5', fill='none') )
#%%
SVG(data=drawing.tostring())

# drawing.save()

# %%
