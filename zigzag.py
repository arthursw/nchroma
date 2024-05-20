import os
import sys
import numpy as np
import torch
from torchvision import transforms
# import colour
from PIL import Image
import svgwrite
from pathlib import Path
Path.ls = lambda self: list(self.iterdir())

# Line version:
# - CYMK: 4 levels: all lines, 3/4 lines, 1/2 lines, 1/4 lines.
# - Other space, find their component values (bruteforce ? Or read https://www.ryanjuckett.com/rgb-color-space-conversion/)
# ZigZag version: from low res image, draw each pixel with a zigzag whose frequence depends on the intensity, with interpolation


# from svglib.svglib import svg2rlg
# from reportlab.graphics import renderPM

# from cairosvg import svg2png
import argparse
from scipy.interpolate import pchip_interpolate


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, type=str, help="The input image path or input folder.")
ap.add_argument("-o", "--output", default=None, type=str, help="The output image path or output folder (default is input_image_nchroma.svg).")
ap.add_argument("-fs", "--frame_size", default='1000x650', type=str, help="The size of the frame (format is WIDTHxHEIGHT, default is 1000x650).")
ap.add_argument("-pw", "--pixel_width", default=5, type=float, help="The pixel width (pixels are square).")
ap.add_argument("-sw", "--stroke_width", default=0.8, type=float, help="The pen width.")

args = ap.parse_args()

image_path = Path(args.image)
if not image_path.exists():
    sys.exit('Unable to find image', image_path)

angles = [i*45 for i in range(4)]
cmyk_colors = ['cyan', 'magenta', 'yellow', 'black']

pixel_width = args.pixel_width
stroke_width = args.stroke_width
try:
    frame_width, frame_height = args.frame_size.split('x')
    frame_width, frame_height = int(frame_width), int(frame_height)
except Exception:
    sys.exit('Error: arg frame_width must follow the format WIDTHxHEIGHT, for example 1000x650.')

shape_width = 4 * stroke_width
n_shapes_per_pixel = pixel_width / shape_width

width, height = (int(frame_width // pixel_width), int(frame_height // pixel_width))

output_arg = Path(args.output) if args.output else image_path if image_path.is_dir() else image_path.parent

zigzag_size = (1,1)
# zigzag_shape = ['M', (0, 1/2), 'S', (1/3, 0), (1/3, 0), 'S', (2/3, 1), (2/3, 1), 'S', (1, 1/2), (1, 1/2)]
zigzag_shape = ['M', (0, 1/2), 'L', (1/3, 0), 'L', (2/3, 1), 'L', (1, 1/2)]

def zigzag(position, size, shape):
    return [c if type(c) is str else tuple(np.array(position) + np.array(c) * np.array(size)) for c in shape]

def shape_to_svg(shape):
    return ' '.join([str(p) for c in shape for p in c])

if image_path.is_dir():
    output_arg.mkdir(exist_ok=True, parents=True)
images = image_path.ls() if image_path.is_dir() else [image_path]

rgb_scale = 255
cmyk_scale = 255

def rgb_to_cmyk(r,g,b):
    # rgb [0,255] -> cmy [0,1]
    c = 1 - r / float(rgb_scale)
    m = 1 - g / float(rgb_scale)
    y = 1 - b / float(rgb_scale)

    # extract out k [0,1]
    min_cmy = np.minimum(np.minimum(c, m), y)
    c = (c - min_cmy) 
    m = (m - min_cmy) 
    y = (y - min_cmy) 
    k = min_cmy

    # rescale to the range [0,cmyk_scale]
    return c*cmyk_scale, m*cmyk_scale, y*cmyk_scale, k*cmyk_scale

def cmyk_to_rgb(c,m,y,k):
    r = rgb_scale*(1.0-(c+k)/float(cmyk_scale))
    g = rgb_scale*(1.0-(m+k)/float(cmyk_scale))
    b = rgb_scale*(1.0-(y+k)/float(cmyk_scale))
    return r,g,b

for image_path in images:
    
    print(image_path)
    
    if image_path.suffix not in ['.jpg', '.jpeg', '.png']: continue

    image = Image.open(str(image_path))
    # a = np.zeros((250,250,3))
    # a[:,:,0] = 255
    # a[:,:,1] = 0
    # a[:,:,2] = 255
    # image = Image.fromarray(a.astype(np.uint8))

    image_name = os.path.splitext(image_path.name)[0] + '_nchroma' if output_arg.is_dir() else output_arg
    output_path = str(output_arg / image_name if output_arg.is_dir() else image_path.parent / image_name)

    image = image.convert('RGB')
    # r, g, b = image.split()
    # image.show()
    rgb = np.asarray(image)
    
    # cmy = colour.RGB_to_CMY(rgb)
    # cmyk = colour.CMY_to_CMYK(cmy)
    # image = Image.fromarray(cmyk.astype(np.uint8), mode='CMYK')
    # image.show()

    # c, m, y, k = rgb_to_cmyk(rgb[:,:,2], rgb[:,:,1], rgb[:,:,0])
    c, m, y, k = rgb_to_cmyk(rgb[:,:,0], rgb[:,:,1], rgb[:,:,2])
    # image = Image.merge('CMYK', (k, y, m, c))
    image = Image.fromarray(np.stack([c, m, y, k], axis=2).astype(np.uint8), mode='CMYK')
    
    # tfs = transforms.Compose([transforms.Grayscale(), transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])
    tfs = transforms.Compose([transforms.ToTensor()])
    # tfs = transforms.Compose([transforms.ToTensor()])

    image = tfs(image)
    # torchvision.transforms.functional.affine(img: torch.Tensor, angle: float, translate: List[int], scale: float, shear: List[float], interpolation: torchvision.transforms.functional.InterpolationMode = <InterpolationMode.NEAREST: 'nearest'>, fill: Union[List[float], NoneType] = None, resample: Union[int, NoneType] = None, fillcolor: Union[List[float], NoneType] = None) → torch.Tensor

    
    toPIL = transforms.ToPILImage()

    def showTensor(t, title=None, mode='CMYK'):
        # if isinstance(t, torch.DoubleTensor):
        # elif isinstance(t, torch.IntTensor):
        #     tImage = toPIL(t[0].numpy())
        # tImage = toPIL((t[0]*255).int().numpy())
        # tImage = toPIL((t*255).numpy().astype(np.uint8).transpose(1, 2, 0))
        # import ipdb ; ipdb.set_trace()
        # tImage = tImage.convert('CMYK')
        # tImage.show(title)
        if mode == 'L':
            Image.fromarray((t.squeeze()*255).numpy().astype(np.uint8), mode=mode).show()
        else:    
            Image.fromarray((t*255).numpy().astype(np.uint8).transpose(1,2,0), mode=mode).show()
    # image = torch.flip(image, [0])
    # showTensor(image, 'Initial')

    # scaled = transforms.functional.affine(image, angle=0, translate=[0,0], shear=0, scale=scale)
    scaled = transforms.Resize([height, width], antialias=True)(image)
    # showTensor(scaled, 'Scaled')

    def countValues(t):
        unique, counts = torch.unique(t, return_counts=True)
        cv = dict(zip(unique.tolist(), counts.tolist()))
        print(cv)
        return unique, counts, cv

    # print('VALUES:')
    # unique, _, _ = countValues(image)
    
    # scaled = pad2d(scaled)
    # showTensor(scaled, 'Padded')

    # Rotated images will have larger size, so coordinates will be differents from one image to another
    # to solve the problem, just pretend that images are in bigger frame of size 2*width, 2*height (or any size containing all rotated frames)
    # so coordinates of all images will be put in this bigger frame before being processed, meaning:
    # x_in_bigger_frame = x + (bigger_frame_width - rotated_image_width) / 2
    # that way all x_in_bigger_frame and y_in_bigger_frame will be in the same coord system

    svgName = output_path + '.svg'
    viewBox = f'0 0 {frame_width} {frame_height}'
    drawing = svgwrite.Drawing(svgName, width=frame_width, height=frame_height, viewBox=viewBox)


    # def projectPoint(xi, yi, frameWidth, frameHeight, rotatedWidth, rotatedHeight, angleRad, pixelSize):
    #     x = xi + (frameWidth - rotatedWidth) / 2
    #     y = yi + (frameHeight - rotatedHeight) / 2
    #     x += math.cos(-angleRad)
    #     y += math.sin(-angleRad)
    #     x *= pixelSize
    #     y *= pixelSize
    #     return x, y

    # a = np.array([[1, 0],[1, 0]])
    # conv1 = torch.nn.Conv2d(1, 1, kernel_size=2)
    # conv1.weight = torch.nn.Parameter(torch.from_numpy(a).float().unsqueeze(0).unsqueeze(0))

    # rotated = transforms.functional.rotate(scaled, 45, expand=True, fill=0)
    # showTensor(rotated>0.1)
    # rotated = transforms.functional.rotate(scaled, 90, expand=True, fill=0)
    # showTensor(rotated>0.1)

    # scaleGroup = drawing.add(drawing.g(id='scale-group'))

    # scaled_channels = scaled.split()
    cmykColors = ['cyan', 'magenta', 'yellow', 'black']#[::-1]
    

    for i, angle in enumerate(angles):
        # if i!=0: continue
        channel = scaled[i].unsqueeze(0)
        # channel = torch.from_numpy(np.indices(scaled[i].shape)[0]/width).unsqueeze(0) # test gradient
        # showTensor(channel, f'Channel {i}', mode='L')
        rotated = transforms.functional.rotate(channel, angle, expand=True, fill=torch.nan).squeeze()
        # showTensor(rotated, f'Rotated {i}', mode='L')

        lineStartPoint = None
        lineEndPoint = None
        strokeColor = cmykColors[i]
        hlines = drawing.add(drawing.g(id='hlines-'+str(i), stroke=(strokeColor), stroke_width=stroke_width, opacity=1))
        rheight, rwidth = rotated.shape
        # transposed = rotated.transpose(1,0) # Numpy and Pytorch are row major
        # for ny, row in enumerate(transposed):

        # Prepare pixel values interpolation (along the width)
        x = np.arange(rwidth)
        n_points = int(n_shapes_per_pixel * rwidth)
        x_new = np.linspace(0, rwidth-1, num=int(n_shapes_per_pixel * rwidth))

        for ny, row in enumerate(rotated):
            py = ( ny - (rheight - height) / 2 ) * pixel_width
            
            # Interpolate the pixel values
            row_pchip = pchip_interpolate(x, row.nan_to_num(0), x_new)

            # Split this interpolation in n chunks, each chunk must have the same area of this interpolation (their spacing must be proportional to the pixel intensities), 
            # find the x positions matching this criteria: compute the cumulated sum, and split it at regular interval
            # use modulo (to create a saw tooth), when the diff of the modulo is negative get the index
            # scale this index to [0, width] = n points / (2 * n_shapes_per_pixel)
            cumsum = np.cumsum(row_pchip)
            # indices = np.flatnonzero(np.diff( cumsum % (cumsum.max() / n_points) ) < 0) / (2 * n_shapes_per_pixel)
            # v = cumsum // (cumsum.max() / n_points)
            v = cumsum // 1
            
            # v = np.where(v[:-1] != v[1:])[0] + 1
            v = np.flatnonzero(np.diff(v))
            indices = v / n_shapes_per_pixel
            
            # np.where(v[:-1] != v[1:])[0] + 1
            
            defined_row_indices = np.flatnonzero(~row.isnan())
            pxmin, pxmax = (np.min(defined_row_indices), np.max(defined_row_indices)) if len(defined_row_indices) > 0 else (0, 0)

            posx = np.concatenate(([0], indices))
            sizex = np.diff(posx, append=pxmax)
            # mask = torch.logical_and(posx > pxmin, posx < pxmax)
            # posx = np.take(posx, mask)
            # sizex = np.take(sizex, mask)

            # if ny==28:
            #     import ipdb ; ipdb.set_trace()
            shapes = []

            for nx, px in enumerate(posx):
                if px < pxmin or px > pxmax: continue
                shapes += zigzag([(px - (rwidth - width) / 2) * pixel_width, py], [sizex[nx] * pixel_width, pixel_width], zigzag_shape if len(shapes)==0 else zigzag_shape[2:])

            if len(shapes)>0:
                hlines.add(drawing.path( d= shape_to_svg(shapes), fill='none' ) )

        hlines.rotate(angle, (frame_width / 2, frame_height / 2))

    drawing.save()
    # print('converting svg to png...')

    # svg2png(url=svgName, write_to=output_path + '.png')
    # drawing = svg2rlg(svgName)


    # renderPM.drawToFile(drawing, svgName.replace('.svg', '.png'), fmt="PNG")



