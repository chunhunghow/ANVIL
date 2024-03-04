

# chunhung 3/2/2024 
# draw the mask in contour , overlay two contours (pred in red, gt in blue) for each test image for each  model


from PIL import Image, ImageDraw
from skimage.measure import find_contours


def get_contour(mask):
    return find_contours(mask)


def create_mask_from_polygon(image, mask, outline=0):
    """
    Creates a binary mask with the dimensions of the image and
    converts the list of polygon-contours to binary masks and merges them together
    Args:
    image: the image that the contours refer to
    contours: list of contours

    Returns:

    """ 
    colors = ['red','blue','green']
    colors = [(255,0,0,255), (0,0,255,255), (0,255,0,255)]
    colors_trans = [(255,0,0,50), (0,0,255,50), (0,255,0,50)]
    contours = get_contour(mask)
    image = Image.fromarray(image).convert('RGBA')
    img = Image.new('RGBA',image.size, (255,255,255,0))
    #lung_mask = np.array(Image.new('L', image.shape, 0))
    #img = Image.new('L', image.shape, 0)
    for contour in contours:
        x = contour[:, 0]
        y = contour[:, 1]
        polygon_tuple = list(zip(y, x))
        ImageDraw.Draw(img).polygon(polygon_tuple, outline=colors[outline], fill=colors_trans[outline])
        #mask = np.array(img)
        #lung_mask += mask

    #lung_mask[lung_mask > 1] = 1  # sanity check to make 100% sure that the mask is binary
    return Image.alpha_composite(image,img)  # transpose it to be aligned with the image dims




