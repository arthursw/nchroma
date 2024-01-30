#%%
import numpy as np
from math import cos, acos, atan2, sin, sqrt, pi
from PIL import Image, ImageOps
from IPython.display import display
from scipy import ndimage
from cairosvg import svg2png
import svgwrite

# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=True, type=str, help="The input image path or input folder.")
# ap.add_argument("-o", "--output", default=None, type=str, help="The output image path or output folder (default is input_image_nchroma.svg).")
# ap.add_argument("-fs", "--frame_size", default='1000x650', type=str, help="The size of the frame (format is WIDTHxHEIGHT, default is 1000x650).")
# ap.add_argument("-pw", "--pixel_width", default=10, type=float, help="The pixel width (pixels are square).")
# ap.add_argument("-sw", "--stroke_width", default=1, type=float, help="The pen width.")

# args = ap.parse_args()

def normalize(v):
	return v / np.linalg.norm(v)

def rotate_v(v, theta):
	rot = np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])
	return np.dot(rot, v)

def get_angle_rad(v):
	return atan2(v[1], v[0]) if abs(v[0]) > 1e-6 or abs(v[1]) > 1e-6 else 0

def get_angle_deg(v):
	return np.rad2deg(get_angle_rad(v))

def show_small_image(image):
	return display(image.resize((200,200), Image.Resampling.NEAREST ))

def disk_mask(radius):
	xx, yy = np.mgrid[:2*radius, :2*radius]
	circle = (xx - radius) ** 2 + (yy - radius) ** 2
	return circle < radius

# image = Image.fromarray((np.random.rand(5,6) * 255).astype(np.uint8))

image = Image.open("indian.jpg")
image = image.convert(mode='L')

# Size (amount traveled forward) of a sub-curve compare to its parent of size 1
# When the parent curve travels 1 forward, the child curve travels ratio forward

stepS = cos(pi/3) # small step
stepM = sin(pi/3) # medium step
stepL = 1 # large step
curve_size = 2 + stepS
delta = np.array([curve_size, stepM]) # start to end vector
ratio = 1 / curve_size
delta_to_one = 1 / np.linalg.norm(delta)
angle_a = get_angle_rad(delta)

#%%
images = [np.asarray(image)]
n_iterations = 7

# show_small_image(image)
# image.show()
for i in range(1, n_iterations):
	# p = pow(2, i)
	# lrimage = ndimage.convolve(np.asarray(image).astype(np.float64), np.ones((p, p)), mode='reflect')  / pow(p, 2)

	lrimage = ndimage.gaussian_filter(image, sigma=pow(ratio,n_iterations-i)*image.width)
	# lrimage = ndimage.minimum_filter(image, footprint=disk_mask(pow(ratio,n_iterations-i)*image.width))

	# show_small_image(lrimage)
	# print(lrimage.min(), lrimage.max(), lrimage.mean())
	# Image.fromarray(lrimage).show()
	images.append(lrimage)

#%%

def get_intensity(image, point):
	point = point.astype(np.int64)
	if point[0] < 0 or point[0] >= image.shape[0] or point[1] < 0 or point[1] >= image.shape[1]:
		return 255
	return image[point[0], point[1]]

def gosper(images, n, p1, p2, threshold, points, invert=False):
	if n >= len(images):
		return points.append([p1,p2])

	if invert:
		p1, p2 = p2, p1

	dir = p2 - p1
	vStep = rotate_v(dir, angle_a) * delta_to_one
	
	new_points = [p1]
	new_points.append( new_points[-1] + vStep )
	new_points.append( new_points[-1] + rotate_v(vStep, -pi/3) )
	new_points.append( new_points[-1] - vStep )
	new_points.append( new_points[-1] + rotate_v(vStep, -2*pi/3) )
	new_points.append( new_points[-1] + vStep )
	new_points.append( new_points[-1] + vStep )
	new_points.append( new_points[-1] + rotate_v(vStep, pi/3) )
	inversions = [False, True, True, False, False, False, True]
	# inversions = [invert if not inv else not invert for inv in inversions]
	
	# if invert:
	# 	new_points = new_points[::-1]
	# 	inversions = [not inv for inv in inversions]
	# 	inversions = inversions[::-1]

	subdivide =  [False, True, False, True, False, False, True]
	# for i in range(1, len(new_points)):
	# 	p1, p2 = new_points[i-1], new_points[i]
	# 	intensity = get_intensity(images[-n-1], (p1 + p2) / 2)
	# 	if subdivide[i-1]: #intensity < threshold:
	# 	# if intensity < threshold:
	# 		gosper(images, n+1, p1, p2, threshold, points, inversions[i-1])
	# 	else:
	# 		points.append([p1,p2])
	
	intensity = get_intensity(images[-n-1], (p1 + p2) / 2)
	if n<2 or intensity < threshold:
		for i in range(1, len(new_points)):
			p1, p2 = new_points[i-1], new_points[i]
		# if intensity < threshold:
			gosper(images, n+1, p1, p2, threshold, points, inversions[i-1])
	else:
		for i in range(1, len(new_points)):
			p1, p2 = new_points[i-1], new_points[i]
			points.append([p1,p2])

pixels_per_unit = image.size[1] / 2
offset = (1 - 2 * ratio) / 2
p1 = np.array([0, -offset ]) * pixels_per_unit
p2 = np.array([stepM, curve_size]) * pixels_per_unit

# p1 = np.array([0, 0 ]) * pixels_per_unit
# p2 = np.array([1, 1]) * pixels_per_unit
points = []
gosper(images, 0, p1, p2, 250, points)

# points2 = []
# gosper(images, 1, p1, p2, 250, points2)

# points3 = []
# gosper(images, 2, p1, p2, 250, points3)

svgName = 'indian.svg'
frame_width, frame_height = 1000, 1000
# idealViewBox = f'0 0 {frame_width} {frame_height}'
# offset *= image.size[0]
offset = 500
viewBox = f'{-offset} {-offset} {frame_width + 2*offset} {frame_height + 2*offset}'
drawing = svgwrite.Drawing(svgName, width=frame_width, height=frame_height, viewBox=viewBox)
group = drawing.add(drawing.g(id='scale-group', stroke='black', stroke_width=0.1, opacity=1, fill='none'))
for pl in points:
	# group.add(drawing.polyline(id='space-filling-curve', stroke='black', stroke_width=0.1, opacity=1, fill='none', points=pl))
	group.add(drawing.line(pl[0], pl[1]))
# for pl in points2:
# 	group.add(drawing.line(pl[0], pl[1], stroke='red'))
# # group.add(drawing.polyline(id='space-filling-curve', stroke='green', stroke_width=0.1, opacity=1, fill='none', points=pl))
# for pl in points3:
# 	group.add(drawing.line(pl[0], pl[1], stroke='green'))
# group.add(drawing.polyline(id='space-filling-curve', stroke='green', stroke_width=0.1, opacity=1, fill='none', points=pl))
group.add(drawing.circle(center=p1, r=10, fill='green'))
group.add(drawing.circle(center=p2, r=10, fill='red'))
group.scale(frame_width / image.size[0])
drawing.save()
# print('converting svg to png...')

# svg2png(url=svgName, write_to='indian_sfc.png')

# function gosper(rasters, nIterations, i, p1, p2, invert, container) {
# 	n = nIterations - i

# 	let p1p2 = p2.subtract(p1)
# 	let p1p2Length = p1p2.length
# 	let p1p2Noramlized = p1p2.normalize()

# 	let delta = p1p2Noramlized.rotate(-30).multiply(p1p2Length / Math.sqrt(3))
# 	let center = p1.add(delta)

# 	let imageSize = rasters[n - 1].width
# 	let centerImage = center.multiply(imageSize / container.width).floor()

# 	let direction = new paper.Point(2.5, Math.sqrt(3) / 2)
# 	let step = p1p2Length / direction.length
# 	let angle = direction.angle
# 	let vStep = p1p2Noramlized.rotate(angle).multiply(step)

# 	let deltas = []
# 	deltas.push({ point: p1, invert: false })
# 	deltas.push({ point: deltas[deltas.length-1].point.add(vStep), invert: true })
# 	deltas.push({ point: deltas[deltas.length-1].point.add(vStep.rotate(-60)), invert: true })
# 	deltas.push({ point: deltas[deltas.length-1].point.subtract(vStep), invert: false })
# 	deltas.push({ point: deltas[deltas.length-1].point.add(vStep.rotate(-120)), invert: false })
# 	deltas.push({ point: deltas[deltas.length-1].point.add(vStep), invert: false })
# 	deltas.push({ point: deltas[deltas.length-1].point.add(vStep), invert: true })
# 	deltas.push({ point: deltas[deltas.length-1].point.add(vStep.rotate(60)), invert: null })

# 	if(n - 1 > 0 && n - 1 < rasters.length) {

# 		let raster = rasters[n - 1]
# 		let color = raster.getAverageColor(new paper.Path.Circle(raster.bounds.topLeft.add(centerImage), 1.5))
# 		let gray = color != null ? color.gray : -1

# 		if(1 - gray >= parameters.threshold) {

# 			for(let j=0 ; j<deltas.length-1 ; j++) {
# 				let invert = deltas[j].invert
# 				gosper(rasters, nIterations, i+1, deltas[invert ? j + 1 : j].point, deltas[invert ? j : j+1].point, invert, container)
# 			}
# 			return
# 		}
# 	}

# 	let path = new paper.Path()
# 	path.strokeWidth = 0.5;
# 	path.strokeColor = 'black';

# 	compoundPath.addChild(path)

# 	for(let d of deltas) {
# 		path.add(d.point)
# 	}
# }

# %%
