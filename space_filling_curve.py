import numpy as np
from math import cos, acos, atan2, sin, sqrt

def rotate(v, angle):
	theta = np.deg2rad(angle)
	rot = np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])
	return np.dot(rot, v)

def get_angle_deg(v):
	return np.rad2deg(atan2(v[1], v[0]) if abs(v[0]) > 1e-6 or abs(v[1]) > 1e-6 else 0)

def gosper(images, nIterations, i, p1, p2, invert):
	p1p2 = p2 - p1
	p1p2Length = np.linalg.norm(p1p2)
	p1p2Noramlized = p1p2 / p1p2Length

	direction = np.array([2.5, sqrt(3) / 2])
	step = p1p2Length / np.linalg.norm(direction)
	angle = get_angle_deg(direction)
	vStep = rotate(p1p2Noramlized, angle) * step

	deltas = []
	deltas.append(dict(point=p1, invert=False))
	deltas.append(dict(point=deltas[-1]['point'] + vStep, invert=True))
	deltas.append(dict(point=deltas[-1]['point'] + rotate(vStep, -60), invert=True))
	deltas.append(dict(point=deltas[-1]['point'] - vStep, invert=False))
	deltas.append(dict(point=deltas[-1]['point'] + rotate(vStep, -120), invert=False))
	deltas.append(dict(point=deltas[-1]['point'] + vStep, invert=False))
	deltas.append(dict(point=deltas[-1]['point'] + vStep, invert=True))
	deltas.append(dict(point=deltas[-1]['point'] + rotate(vStep, 60), invert=None))

	if n - 1 > 0 and n - 1 < nIterations:
		color = images[n-1][center]
		if color > threshold:
			for i, delta in enumerate(deltas):
				gosper(images, nIterations, i-1, delta.point, delta.point, invert)
	return

images = [image]
for i in range(1, nIterations):
	p = pow(2, i)
	images.append(np.convolve(image, np.ones((p, p)), 'same') / pow(p, 2) )

function gosper(rasters, nIterations, i, p1, p2, invert, container) {
	n = nIterations - i

	let p1p2 = p2.subtract(p1)
	let p1p2Length = p1p2.length
	let p1p2Noramlized = p1p2.normalize()

	let delta = p1p2Noramlized.rotate(-30).multiply(p1p2Length / Math.sqrt(3))
	let center = p1.add(delta)

	let imageSize = rasters[n - 1].width
	let centerImage = center.multiply(imageSize / container.width).floor()

	let direction = new paper.Point(2.5, Math.sqrt(3) / 2)
	let step = p1p2Length / direction.length
	let angle = direction.angle
	let vStep = p1p2Noramlized.rotate(angle).multiply(step)

	let deltas = []
	deltas.push({ point: p1, invert: false })
	deltas.push({ point: deltas[deltas.length-1].point.add(vStep), invert: true })
	deltas.push({ point: deltas[deltas.length-1].point.add(vStep.rotate(-60)), invert: true })
	deltas.push({ point: deltas[deltas.length-1].point.subtract(vStep), invert: false })
	deltas.push({ point: deltas[deltas.length-1].point.add(vStep.rotate(-120)), invert: false })
	deltas.push({ point: deltas[deltas.length-1].point.add(vStep), invert: false })
	deltas.push({ point: deltas[deltas.length-1].point.add(vStep), invert: true })
	deltas.push({ point: deltas[deltas.length-1].point.add(vStep.rotate(60)), invert: null })

	if(n - 1 > 0 && n - 1 < rasters.length) {

		let raster = rasters[n - 1]
		let color = raster.getAverageColor(new paper.Path.Circle(raster.bounds.topLeft.add(centerImage), 1.5))
		let gray = color != null ? color.gray : -1

		if(1 - gray >= parameters.threshold) {

			for(let j=0 ; j<deltas.length-1 ; j++) {
				let invert = deltas[j].invert
				gosper(rasters, nIterations, i+1, deltas[invert ? j + 1 : j].point, deltas[invert ? j : j+1].point, invert, container)
			}
			return
		}
	}

	let path = new paper.Path()
	path.strokeWidth = 0.5;
	path.strokeColor = 'black';

	compoundPath.addChild(path)

	for(let d of deltas) {
		path.add(d.point)
	}
}
