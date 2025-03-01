# import spectra
# import functools
# import itertools
# import math
# import colour
import os
import sys
import numpy as np
import torch
from torchvision import transforms
from PIL import Image, ImageOps
import svgwrite
# import cairosvg
import argparse
from pathlib import Path
Path.ls = lambda self: list(self.iterdir())
import pillow_avif

# Line version:
# - CYMK: 4 levels: all lines, 3/4 lines, 1/2 lines, 1/4 lines.
# - Other space, find their component values (bruteforce ? Or read https://www.ryanjuckett.com/rgb-color-space-conversion/)
# ZigZag version: from low res image, draw each pixel with a zigzag whose frequence depends on the intensity, with interpolation


# from svglib.svglib import svg2rlg
# from reportlab.graphics import renderPM


angles = [i*45 for i in range(4)]
# pixelSize = 1
cmyk_colors = ['cyan', 'magenta', 'yellow', 'black']

ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
ap.add_argument("-i", "--image", required=True, type=str, help="The input image path or input folder.")
ap.add_argument("-o", "--output", default=None, type=str, help="The output image path or output folder (default is input_image_nchroma.svg).")
# ap.add_argument("-ps", "--pixel_size", default=pixelSize, type=int, help="The pixel size.")
ap.add_argument("-ls", "--line_spacing", default=0.4, type=float, help="The spacing between two lines in mm.")
ap.add_argument("-a", "--angles", nargs='+', default=angles, help="The angles to use (if not in CMYK mode).")
ap.add_argument("-co", "--colors", nargs='+', default=cmyk_colors, help="The colors to use to render the image.")
ap.add_argument("-g", "--grayscale", action='store_true', help="Grayscale mode, default is CMYK (this implies 4 angles).")
# ap.add_argument("-r", "--resize", default=None, type=int, help="Resize images to given size before processing.")
ap.add_argument("-e", "--equalize", action='store_true', help="Equalize image before processing.")
ap.add_argument('-pw', '--paperWidth', help='Paper width in mm.', default=500, type=float)
ap.add_argument('-ph', '--paperHeight', help='Paper height in mm.', default=650, type=float)
ap.add_argument('-m', '--margin', type=float, help='Margin in mm', default=30)
ap.add_argument('-ox', '--offset_x', type=float, help='X offset in mm (-20 is good for 650 x 500 on silhouette cameo pro 4)', default=0)
ap.add_argument('-oy', '--offset_y', type=float, help='Y offset in mm (-5 is good for 650 x 500 on silhouette cameo pro 4 to center rollers)', default=0)
ap.add_argument("-sw", "--stroke_width", default=0.4, type=float, help="Stroke width in mm.")
# ap.add_argument("-bbc", "--bounding_box_color", default=None, type=str, help="Bounding box color (CSS color).")
ap.add_argument('-df', '--drawFrames', action='store_true', help='Draw the drawing bounding box and the paper rectangle')
ap.add_argument("-si", "--show_images", action='store_true', help="Show intermediate images.")

args = ap.parse_args()

image_path = Path(args.image)
if not image_path.exists():
    sys.exit(f'Unable to find image {image_path}')

# pixelSize = args.pixel_size
angles = args.angles if args.grayscale else angles
# resize = args.resize
equalize = args.equalize

output_arg = Path(args.output) if args.output else image_path if image_path.is_dir() else image_path.parent
showIntermediateImages = args.show_images



# Blend the input colors
# colors = []

# def blend(color_tuple):
#     return color_tuple[0] if len(color_tuple) == 1 else functools.reduce(lambda x, y: x.blend(y, ratio=0.5), color_tuple)

# cmyk_colors = [spectra.html(c).to('cmyk') for c in args.colors]

# for i in range(1, len(cmyk_colors)):
#     colors.append(blend(list(itertools.combinations(cmyk_colors, i))))

# colors = [c.to('rgb') for c in colors]

shape = ['M', (0, 1/2), 'S', (1/3, 0), 'S', (2/3, 1), 'S', (1, 1/2)]


def zipzag(position, size, shape):
    return [c if type(c) is str else tuple(np.array(position) + c * np.array(size)) for c in shape]

def shape_to_svg(shape):
    return ' '.join([p for c in shape for p in c])

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

    # if image_path.suffix not in ['.jpg', '.jpeg', '.png', '.avif']: 
    #     print(f'Warning, the file {image_path} extension is not recognized ; it will be ignored.')
    #     continue

    image = Image.open(str(image_path))
    # a = np.zeros((250,250,3))
    # a[:,:,0] = 255
    # a[:,:,1] = 0
    # a[:,:,2] = 255
    # image = Image.fromarray(a.astype(np.uint8))

    # If image has transparency, make it white
    if image.mode == 'RGBA':
        background = Image.new('RGBA', image.size, (255, 255, 255))
        image = Image.alpha_composite(background, image)

    paperRatio = (args.paperWidth - 2 * args.margin) / (args.paperHeight - 2 * args.margin)
    width, height = image.size

    imageRatio = width / height
    if paperRatio > imageRatio:
        newImageHeight = (args.paperHeight - 2 * args.margin) / args.line_spacing
        newImageWidth = newImageHeight * imageRatio
    else:
        newImageWidth = (args.paperWidth - 2 * args.margin) / args.line_spacing
        newImageHeight = newImageWidth / imageRatio

    image_name = os.path.splitext(image_path.name)[0] + '_nchroma' if output_arg.is_dir() else output_arg
    output_path = str(output_arg / image_name if output_arg.is_dir() else image_path.parent / image_name)

    # if resize:
    #     maxSize = resize
    #     newsize = (maxSize, image.height * maxSize // image.imwidth) if image.width > maxSize else (image.width * maxSize // image.height, maxSize)
    #     image = image.resize((newsize))
    image = image.resize((int(newImageWidth), int(newImageHeight)))
    width, height = image.size
    imageRatio = width / height
    if not args.grayscale:
        # image = image.quantize(colors=nColors, kmeans=nColors)
        image = image.convert('RGB')
        # r, g, b = image.split()
        rgb = np.asarray(image)
        
        # cmy = colour.RGB_to_CMY(rgb)
        # cmyk = colour.CMY_to_CMYK(cmy)
        # image = Image.fromarray(cmyk.astype(np.uint8), mode='CMYK')

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
    tfs = transforms.Compose([transforms.ToTensor()]) if not args.grayscale else transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])
    # tfs = transforms.Compose([transforms.ToTensor()])
    
    # print(height, width)
    image = tfs(image)
    # torchvision.transforms.functional.affine(img: torch.Tensor, angle: float, translate: List[int], scale: float, shear: List[float], interpolation: torchvision.transforms.functional.InterpolationMode = <InterpolationMode.NEAREST: 'nearest'>, fill: Union[List[float], NoneType] = None, resample: Union[int, NoneType] = None, fillcolor: Union[List[float], NoneType] = None) → torch.Tensor

    # channels, height, width = image.shape
    # print(height, width, channels)
    
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
    # scaled = transforms.Resize([height//pixelSize, width//pixelSize], antialias=True)(image)

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

    # channels, smallHeight, smallWidth = scaled.shape

    # Rotated images will have larger size, so coordinates will be differents from one image to another
    # to solve the problem, just pretend that images are in bigger frame of size 2*width, 2*height (or any size containing all rotated frames)
    # so coordinates of all images will be put in this bigger frame before being processed, meaning:
    # x_in_bigger_frame = x + (bigger_frame_width - rotated_image_width) / 2
    # that way all x_in_bigger_frame and y_in_bigger_frame will be in the same coord system

    # def fitSourceInReference(source, reference):
    #     sourceRatio = source['width'] / source['height']
    #     referenceRatio = reference['width'] / reference['height']
    #     if sourceRatio > referenceRatio:
    #         source['width'] = reference['width']
    #         source['height'] = source['width'] / sourceRatio
    #         source['x'] = reference['x']
    #         source['y'] = reference['y'] + (reference['height'] - source['height']) / 2
    #     else:
    #         source['height'] = reference['height']
    #         source['width'] = source['height'] * sourceRatio
    #         source['y'] = reference['y']
    #         source['x'] = reference['x'] + (reference['width'] - source['width']) / 2
    #     return source

    frameWidth = 2 * width
    frameHeight = 2 * height



    # finalWidth = pixelSize * smallWidth
    # finalHeight = pixelSize * smallHeight
    # drawingRatio = finalWidth / finalHeight

    if paperRatio > imageRatio: # paper is wider compared to drawing: margin are defined vertically
        mmToUnit = height / (args.paperHeight - 2 * args.margin)
        marginV = args.margin * mmToUnit
        wp = height * paperRatio
        marginH = marginV + (wp - width) / 2
        totalWidth = args.paperWidth * mmToUnit
        marginH2 = (totalWidth - width) / 2
        assert(marginH > marginV)
        assert(abs(marginH - marginH2) < 1e-6)
    else: # paper is taller compared to drawing: margin are defined horizontally
        mmToUnit = width / (args.paperWidth - 2 * args.margin)
        marginH = args.margin * mmToUnit
        hp = width / paperRatio
        marginV = marginH + (hp - height) / 2 
        totalHeight = args.paperHeight * mmToUnit
        marginV2 = (totalHeight - height) / 2
        assert(marginV > marginH)
        assert(abs(marginV - marginV2) < 1e-6)

    offsetX = args.offset_x * mmToUnit
    offsetY = args.offset_y * mmToUnit

    svgName = output_path + '.svg'
    minX = width // 2
    minY = height // 2

    # frame = dict(x=minX, y=minY, width=finalWidth, height=finalHeight)
    # fitSourceInReference(frame, dict(x=0, y=0, ))

    viewBox = dict(x=minX - marginH - offsetX, y=minY - marginV - offsetY, width=width + 2 * marginH, height=height + 2 * marginV)
    viewBoxString = f'{viewBox["x"]:.5} {viewBox["y"]:.5} {viewBox["width"]:.5} {viewBox["height"]:.5}'
    drawing = svgwrite.Drawing(svgName, size=(f'{args.paperWidth}mm', f'{args.paperHeight}mm'), viewBox=viewBoxString)

    # if args.bounding_box_color:
    #     drawing.add(drawing.rect(x=marginH+offsetX, y=marginV+offsetY, width=finalWidth, height=finalHeight, stroke=args.bounding_box_color, stroke_width=1))

    assert(abs(viewBox["width"] / viewBox["height"] - args.paperWidth / args.paperHeight) < 1e-6)
    if args.drawFrames:
        drawing.add(drawing.rect(insert=(viewBox["x"], viewBox["y"]), size=(viewBox["width"], viewBox["height"]), fill='none', stroke='red', stroke_width=1))
        # drawing.add(drawing.rect(insert=(minX, minY-verticalCorrection), size=(width, height * (viewBox["height"] + hc) / viewBox["height"] ), fill='none', stroke='green', stroke_width=1))
        drawing.add(drawing.rect(insert=(viewBox["x"] + args.margin * mmToUnit + offsetX, viewBox["y"] + args.margin * mmToUnit + offsetY), size=(viewBox["width"] - 2 * args.margin * mmToUnit, viewBox["height"] - 2 * args.margin * mmToUnit), fill='none', stroke='green', stroke_width=1))
        drawing.add(drawing.rect(insert=(minX, minY), size=(width, height), fill='none', stroke='blue', stroke_width=1))

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

    scaleGroup = drawing.add(drawing.g(id='scale-group'))

    # contour = 1
    # scaled[:, :, 0] = contour
    # scaled[:, :, -1] = contour
    # scaled[:, 0, :] = contour
    # scaled[:, -1, :] = contour

    # scaled_channels = scaled.split()
    cmykColors = ['cyan', 'magenta', 'yellow', 'black']#[::-1]
    

    for i, angle in enumerate(angles):
        channel = image[i].unsqueeze(0) if not args.grayscale else image
        # showTensor(channel, f'Channel {i}', mode='L')
        rotated = transforms.functional.rotate(channel.clone(), angle, expand=True, fill=torch.nan)
            
        masks = torch.zeros((5,) + rotated.shape[1:])
        
        masks[1,::4,:] = 1
        masks[2,::2,:] = 1
        masks[3,::2,:] = 1
        masks[3,::3,:] = 1
        masks[4,::1,:] = 1

        shape_indices = torch.from_numpy(np.indices(rotated.shape[1:]))

        # showTensor(rotated, f'Rotated {i}', mode='L')
        channels, rotatedHeight, rotatedWidth = rotated.shape
        # offsetY = rotatedHeight - smallHeight
        # offsetX = rotatedWidth - smallWidth

        # thresholded = (rotated < (i+1) / nColors).int()
        # thresholded = ( rotated > unique[i] + 1e-6 if invertColor else rotated < unique[i] + 1e-6 ).int() 

        indices = (rotated * 4).int()
        thresholded = masks[indices.squeeze(), shape_indices[0], shape_indices[1]].unsqueeze(0)
        # thresholded = masks[indices, *shape_indices]


        # thresholded[:, :, 0] = 1

        # showTensor(thresholded)

        thresholdedxp1 = pad2d(thresholded[:, :, :-1])
        # diff2 = conv1(torch.autograd.Variable(thresholdedxp1)).data.view(1, thresholdedxp1.shape[2], thresholdedxp1.shape[3])
        # showTensor(diff2)
        
        # showTensor(thresholdedxp1)

        diff = thresholdedxp1 - thresholded
        
        # import ipdb ; ipdb.set_trace()
        # di = diff != 0

        diff_indices = torch.nonzero(torch.nan_to_num(diff, nan=0))
        lastYi = 0

        # angleRad = 2.0 * math.pi * angle / 360

        lineStartPoint = None
        lineEndPoint = None
        strokeColor = cmykColors[i] if not args.grayscale else 'black'
        hlines = scaleGroup.add(drawing.g(id='hlines-'+str(i), stroke=(strokeColor), stroke_width=args.stroke_width*mmToUnit, opacity=1))

        # image_lab = colour.XYZ_to_Lab(colour.RGB_to_XYZ(image_rgb))
        # delta_E = colour.delta_E(image1_lab, image2_lab)

        for (n, yt, xt) in diff_indices:
            xi = xt.item()
            yi = yt.item()
            x = xi + (frameWidth - rotatedWidth) / 2
            y = yi + (frameHeight - rotatedHeight) / 2
            # x *= pixelSize
            # y *= pixelSize
            
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
                # lineEndPoint = (pixelSize * (rotatedWidth + (frameWidth - rotatedWidth) / 2), y)
                lineEndPoint = ( rotatedWidth + (frameWidth - rotatedWidth) / 2, y )
                # lineEndPoint = projectPoint(smallWidth-1, yi, frameWidth, frameHeight, rotatedWidth, rotatedHeight, angleRad, pixelSize)
            else:
                # print('add on same line: ', lineStartPoint, '-', (x, y))
                hlines.add(drawing.line(start=lineStartPoint, end=(x, y)))
                lineStartPoint = None
            lastYi = yi
        
        # hlines.rotate(angle, (pixelSize * frameWidth / 2, pixelSize * frameHeight / 2))
        hlines.rotate(angle, (frameWidth / 2, frameHeight / 2))
        hlines.update({'style':'mix-blend-mode: multiply;'})
    
    print('saving', svgName)
    drawing.save()
    # print('converting svg to png...')

    # cairosvg.svg2png(url=svgName, write_to=output_path + '.png')
    # drawing = svg2rlg(svgName)


    # renderPM.drawToFile(drawing, svgName.replace('.svg', '.png'), fmt="PNG")



