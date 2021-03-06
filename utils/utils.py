import random
import numpy as np
def get_square(img, pos):
    """Extract a left or a right square from PILimg shape : (H, W, C))"""
    img = np.array(img)
    h = img.shape[0]
    if pos == 0:
        return img[:, :h]
    else:
        return img[:, -h:]


def resize_and_crop(pilimg, scale=0.5, final_height=None):
    w = pilimg.size[0]
    h = pilimg.size[1]
    # newW = int(w * scale)
    # newH = int(h * scale)
    newW = int(320)
    newH = int(320)

    if not final_height:
        diff = 0
    else:
        diff = newH - final_height

    img = pilimg.resize((newW, newH))
    img = img.crop((0, diff // 2, newW, newH - diff // 2))
    return img


def batch(iterable, batch_size):
    """Yields lists by batch"""
    b = []
    for i, t in enumerate(iterable):
        b.append(t)
        if (i + 1) % batch_size == 0:
            yield b
            b = []

    if len(b) > 0:
        yield b


def split_train_val(dataset, val_percent=0.05):
    dataset = list(dataset)
    length = len(dataset)
    n = int(length * val_percent)
    random.shuffle(dataset)
    return {'train': dataset[:-n], 'val': dataset[-n:]}


def normalize(x):
    return x / 255


def merge_masks(img1, img2, full_w):
    h = img1.shape[0]

    new = np.zeros((h, full_w), np.float32)

    new[:, :full_w // 2 + 1] = img1[:, :full_w // 2 + 1]
    new[:, full_w // 2 + 1:] = img2[:, -(full_w // 2 - 1):]

    return new


def rle_encode(mask_image):
    pixels = mask_image.flatten()
    # We avoid issues with '1' at the start or end (at the corners of
    # the original image) by setting those pixels to '0' explicitly.
    # We do not expect these to be non-zero for an accurate mask,
    # so this should not harm the score.
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]
    return runs

if __name__ == '__main__':
    img = '../data/train_masks/000000.png'
    img = Image.open(img)
    A = DataAugmentation()
    i_Crop = A.randomCrop(img, 0.05)
    i_rotation = A.randomRotation(img, 5)
    i_Color = A.randomColor(img)
    img = '../data/train/000000.jpg'
    img = Image.open(img)
    A = DataAugmentation()
    ai_Crop = A.randomCrop(img, 0.05)
    ai_rotation = A.randomRotation(img, 5)
    ai_Color = A.randomColor(img)
