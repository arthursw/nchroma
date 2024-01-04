import spectra
import functools
import itertools
import math
import os
import sys
import numpy as np
import torch
from torchvision import transforms
from PIL import Image, ImageOps
import svgwrite
from pathlib import Path
Path.ls = lambda self: list(self.iterdir())
import colour

# from svglib.svglib import svg2rlg
# from reportlab.graphics import renderPM

import cairosvg
import argparse

angles = [i*45 for i in range(4)]
pixelSize = 5
invertColor = False
cmyk_colors = ['cyan', 'magenta', 'yellow', 'black']

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, type=str, help="The input image path or input folder.")
ap.add_argument("-o", "--output", default=None, type=str, help="The output image path or output folder (default is input_image_nchroma.svg).")
ap.add_argument("-ps", "--pixel_size", default=pixelSize, type=int, help="The pixel size.")
ap.add_argument("-a", "--angles", nargs='+', default=angles, help="The angles to use (if not in CMYK mode).")
ap.add_argument("-ic", "--invert_color", action='store_true', help="Invert colors before vectorizing.")
ap.add_argument("-co", "--colors", nargs='+', default=cmyk_colors, help="The colors to use to render the image.")
ap.add_argument("-c", "--cmyk", action='store_true', help="CMYK mode (this implies 4 angles), defaultl is grayscale.")
ap.add_argument("-r", "--resize", default=None, type=int, help="Resize images to given size before processing.")
ap.add_argument("-e", "--equalize", action='store_true', help="Equalize image before processing.")
ap.add_argument("-sw", "--stroke_width", default=1, type=int, help="Stroke width.")
ap.add_argument("-si", "--show_images", action='store_true', help="Show intermediate images.")

args = ap.parse_args()

image_path = Path(args.image)
if not image_path.exists():
    sys.exit('Unable to find image', image_path)

pixelSize = args.pixel_size
angles = args.angles if not args.cmyk else angles
invertColor = args.invert_color
resize = args.resize
equalize = args.equalize
strokeWidth = args.stroke_width
output_arg = Path(args.output) if args.output else image_path if image_path.is_dir() else image_path.parent
showIntermediateImages = args.show_images

# Blend the input colors
colors = []

def blend(color_tuple):
    return functools.reduce(lambda x, y: x.blend(y, ratio=0.5), color_tuple)

cmyk_colors = [spectra.html(c).to('cmyk') for c in args.colors]

for i in range(1, len(cmyk_colors)):
    colors.append(blend(itertools.combinations(cmyk_colors, i)))

# colors = [c.to('rgb') for c in colors]

if image_path.is_dir():
    output_arg.mkdir(exist_ok=True, parents=True)

nColors = len(angles) + 1

# image_path = "/Users/Arthur/Projects/IDLV/ArtInSitu/images/faune/renard/2.jpg"
# image_path = "/Users/Arthur/Projects/IDLV/ArtInSitu/images/flore/Astragale à feuille de réglisse/1.jpg"
# image_path = "/Users/Arthur/Projects/IDLV/ArtInSitu/images/flore/cardière/0.jpg"
# image_path = "/Users/Arthur/Projects/IDLV/ArtInSitu/images/flore/Ophrys apifera/1.jpg"
# image_path = "/Users/Arthur/Projects/IDLV/Tipibot/PenPlotter/PenPlotter/data/grace.jpg"

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

    if resize:
        maxSize = resize
        newsize = (maxSize, image.height * maxSize // image.imwidth) if image.width > maxSize else (image.width * maxSize // image.height, maxSize)
        image = image.resize(newsize)

    if args.cmyk:
        # image = image.quantize(colors=nColors, kmeans=nColors)
        image = image.convert('RGB')
        # r, g, b = image.split()
        rgb = np.asarray(image)
        
        cmy = colour.RGB_to_CMY(rgb)
        cmyk = colour.CMY_to_CMYK(cmy)
        image = Image.fromarray(cmyk.astype(np.uint8), mode='CMYK')

        # c, m, y, k = rgb_to_cmyk(rgb[:,:,2], rgb[:,:,1], rgb[:,:,0])
        c, m, y, k = rgb_to_cmyk(rgb[:,:,0], rgb[:,:,1], rgb[:,:,2])
        # image = Image.merge('CMYK', (k, y, m, c))
        # import ipdb ; ipdb.set_trace()
        image = Image.fromarray(np.stack([c, m, y, k], axis=2).astype(np.uint8), mode='CMYK')
        # image = Image.fromarray(np.stack([k, y, m, c], axis=2).astype(np.uint8), mode='CMYK')
        # image = image.convert('CMYK')im
    else:
        image = image.convert(mode='L')
        if equalize:
            image = ImageOps.equalize(image)
            if showIntermediateImages:
                image.show()
        image = image.quantize(colors=nColors, kmeans=nColors)
    if showIntermediateImages:
        image.show()
    # Convert a pil image to a tensor:
    # image = torch.tensor(image)

    # tfs = transforms.Compose([transforms.Grayscale(), transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])
    tfs = transforms.Compose([transforms.ToTensor()]) if args.cmyk else transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])
    # tfs = transforms.Compose([transforms.ToTensor()])
    height, width = image.size
    print(height, width)
    image = tfs(image)
    # torchvision.transforms.functional.affine(img: torch.Tensor, angle: float, translate: List[int], scale: float, shear: List[float], interpolation: torchvision.transforms.functional.InterpolationMode = <InterpolationMode.NEAREST: 'nearest'>, fill: Union[List[float], NoneType] = None, resample: Union[int, NoneType] = None, fillcolor: Union[List[float], NoneType] = None) → torch.Tensor

    channels, height, width = image.shape
    print(height, width, channels)
    
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
    scaled = transforms.Resize([height//pixelSize, width//pixelSize], antialias=True)(image)

    # showTensor(scaled, 'Scaled')

    def countValues(t):
        unique, counts = torch.unique(t, return_counts=True)
        cv = dict(zip(unique.tolist(), counts.tolist()))
        print(cv)
        return unique, counts, cv

    # print('VALUES:')
    # unique, _, _ = countValues(image)
    
    pad2d = torch.nn.ZeroPad2d((1,0,0,0))
    # scaled = pad2d(scaled)
    # showTensor(scaled, 'Padded')

    channels, smallHeight, smallWidth = scaled.shape

    # Rotated images will have larger size, so coordinates will be differents from one image to another
    # to solve the problem, just pretend that images are in bigger frame of size 2*width, 2*height (or any size containing all rotated frames)
    # so coordinates of all images will be put in this bigger frame before being processed, meaning:
    # x_in_bigger_frame = x + (bigger_frame_width - rotated_image_width) / 2
    # that way all x_in_bigger_frame and y_in_bigger_frame will be in the same coord system

    frameWidth = 2 * smallWidth
    frameHeight = 2 * smallHeight

    svgName = output_path + '.svg'
    viewBox = f'{pixelSize * smallWidth // 2} {pixelSize * smallHeight // 2} {pixelSize * smallWidth} {pixelSize * smallHeight}'
    drawing = svgwrite.Drawing(svgName, height=100, width=100, viewBox=viewBox)


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

    if invertColor:
        background = drawing.add(drawing.rect((0, 0), (pixelSize * frameWidth, pixelSize * frameHeight), fill='black'))
    scaleGroup = drawing.add(drawing.g(id='scale-group'))

    contour = 0 if invertColor else 1
    scaled[:, :, 0] = contour
    scaled[:, :, -1] = contour
    scaled[:, 0, :] = contour
    scaled[:, -1, :] = contour

    # scaled_channels = scaled.split()
    cmykColors = ['cyan', 'magenta', 'yellow', 'black']#[::-1]
    
    for i, angle in enumerate(angles):
        channel = scaled[i].unsqueeze(0) if args.cmyk else scaled
        # showTensor(channel, f'Channel {i}', mode='L')
        rotated = transforms.functional.rotate(channel.clone(), angle, expand=True, fill=0 if invertColor else 1)
        # showTensor(rotated, f'Rotated {i}', mode='L')
        channels, rotatedHeight, rotatedWidth = rotated.shape
        # offsetY = rotatedHeight - smallHeight
        # offsetX = rotatedWidth - smallWidth

        # thresholded = (rotated < (i+1) / nColors).int()
        # thresholded = ( rotated > unique[i] + 1e-6 if invertColor else rotated < unique[i] + 1e-6 ).int() 
        thresholded = (rotated > 0.5).char()

        # thresholded[:, :, 0] = 1

        # showTensor(thresholded)

        thresholdedxp1 = pad2d(thresholded[:, :, :-1])
        # diff2 = conv1(torch.autograd.Variable(thresholdedxp1)).data.view(1, thresholdedxp1.shape[2], thresholdedxp1.shape[3])
        # showTensor(diff2)
        
        # showTensor(thresholdedxp1)

        diff = thresholdedxp1 - thresholded
        
        # import ipdb ; ipdb.set_trace()
        # di = diff != 0

        indices = torch.nonzero(diff)
        lastYi = 0

        # angleRad = 2.0 * math.pi * angle / 360

        lineStartPoint = None
        lineEndPoint = None
        strokeColor = cmykColors[i] if args.cmyk else ('white' if invertColor else 'black')
        hlines = scaleGroup.add(drawing.g(id='hlines-'+str(i), stroke=(strokeColor), stroke_width=strokeWidth, opacity=1))

        # image_lab = colour.XYZ_to_Lab(colour.RGB_to_XYZ(image_rgb))
        # delta_E = colour.delta_E(image1_lab, image2_lab)

        for (n, yt, xt) in indices:
            xi = xt.item()
            yi = yt.item()
            x = xi + (frameWidth - rotatedWidth) / 2
            y = yi + (frameHeight - rotatedHeight) / 2
            x *= pixelSize
            y *= pixelSize
            # x, y = projectPoint(xi, yi, frameWidth, frameHeight, rotatedWidth, rotatedHeight, angleRad, pixelSize)
            if yi != lastYi:
                if lineStartPoint is not None:
                    # print('add end line: ', lineStartPoint, '-', lineEndPoint)
                    hlines.add(drawing.line(start=lineStartPoint, end=lineEndPoint))
                    lineStartPoint = None
                    lineEndPoint = None
            if lineStartPoint is None:
                lineStartPoint = (x, y)
                # lineEndPoint = (pixelSize * (smallWidth - 1 + (frameWidth - rotatedWidth) / 2), y)
                lineEndPoint = (pixelSize * (rotatedWidth + (frameWidth - rotatedWidth) / 2), y)
                # lineEndPoint = projectPoint(smallWidth-1, yi, frameWidth, frameHeight, rotatedWidth, rotatedHeight, angleRad, pixelSize)
            else:
                # print('add on same line: ', lineStartPoint, '-', (x, y))
                hlines.add(drawing.line(start=lineStartPoint, end=(x, y)))
                lineStartPoint = None
            lastYi = yi
        
        hlines.rotate(angle, (pixelSize * frameWidth / 2, pixelSize * frameHeight / 2))

    drawing.save()
    print('converting svg to png...')

    cairosvg.svg2png(url=svgName, write_to=output_path + '.png')
    # drawing = svg2rlg(svgName)


    # renderPM.drawToFile(drawing, svgName.replace('.svg', '.png'), fmt="PNG")



