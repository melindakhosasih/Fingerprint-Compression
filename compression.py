import math
import sys
import os
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm


def preprocess_image(image: Image) -> Image:
    """
    Create a square matrix with side length that's a power of 2.
    """
    image = ImageOps.grayscale(image)  # convert to grayscale first so padding is less expensive
    width, height = image.size

    dim = max(width, height)  # Find the largest dimension
    new_dim = 2 ** int(math.ceil(math.log(dim, 2)))  # Find the next power of 2

    output = np.zeros((new_dim, new_dim))
    output[0:height, 0:width] = np.array(image)
    return output
    # return ImageOps.pad(image, (new_dim, new_dim), method=Image.Resampling.NEAREST, centering=(pad_width, pad_height))
    # return ImageOps.pad(image, (new_dim, new_dim))

def crop_to_original_size(image: Image, ori_size: tuple) -> Image:
    """
    Crop the image back to its original size.
    """
    # print(image.size)
    width, height = ori_size
    return image.crop((0, 0, width, height))

def get_haar_step(i: int, k: int) -> np.ndarray:
    transform = np.zeros((2 ** k, 2 ** k))
    # Averages
    for j in range(2 ** (k - i - 1)):
        transform[2 * j, j] = 1 / 2
        transform[2 * j + 1, j] = 1 / 2
    # Details
    offset = 2 ** (k - i - 1)
    for j in range(2 ** (k - i - 1)):
        transform[2 * j, offset + j] = 1 / 2
        transform[2 * j + 1, offset + j] = -1 / 2
    # Identity
    for j in range(2 ** (k - i), 2 ** k):
        transform[j, j] = 1
    return transform


def get_haar_transform(k: int) -> np.ndarray:
    transform = np.eye(2 ** k)
    for i in range(k):
        transform = transform @ get_haar_step(i, k)
    return transform


def haar_encode(a: np.ndarray) -> np.ndarray:
    k = int(math.log2(len(a)))
    row_encoder = get_haar_transform(k)
    return row_encoder.T @ a @ row_encoder


def haar_decode(a: np.ndarray) -> np.ndarray:
    k = int(math.log2(len(a)))
    row_decoder = np.linalg.inv(get_haar_transform(k))
    return row_decoder.T @ a @ row_decoder


def truncate_values(a: np.ndarray, tolerance: float) -> np.ndarray:
    return np.where(np.abs(a) < tolerance, 0, a)

def calculate_compression_ratio(original, compressed) -> float:
    return (compressed != 0).sum() / (original != 0).sum()

def plot_image(original: np.array, encoded: np.array, tolerance: float) -> None:
    encoded = truncate_values(encoded,
                              tolerance)  # Reuse E acrossS iterations because tolerance increases
    decoded = haar_decode(encoded)
    # axes_subplot.imshow(decoded, cmap='gray')
    ratio = calculate_compression_ratio(original, encoded)
    # axes_subplot.set_title(f'({tolerance}) 1:{ratio :.1f}')
    # axes_subplot.tick_params(which='both', bottom=False, top=False, left=False, right=False,
    #                          labelbottom=False, labeltop=False, labelleft=False, labelright=False)
    return decoded, ratio

def save_image(result, ori, name, ratio, output_dir):
    result = Image.fromarray(result)
    result = result.convert("RGBA")
    result = crop_to_original_size(result, ori.size)
    # result.save(f"../dataset/result/test/tolerance1/{name}_{ratio :.1f}.BMP")
    # print(len(result.mode))
    result.save(f"{output_dir}/{name}.BMP", format="BMP")

if __name__ == '__main__':
    ratios = []
    tolerance = 0.5
    output_dir = f"./tolerance{tolerance}/"
    file_name = os.path.join(output_dir, "compression_ratio.txt")

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    for root, dirs, images in os.walk(sys.argv[1]):
        for image in tqdm(images):
            image_name = os.path.splitext(image)[0]
            image_dir = os.path.join(root, image)
            with Image.open(image_dir) as img:
                
                im = preprocess_image(img)
                A = np.array(im)
                E = haar_encode(A)
                result, ratio = plot_image(A, E, tolerance)
                
                save_image(result, img, image_name, ratio, output_dir)
                ratios.append(ratio)
                cropped = crop_to_original_size(Image.fromarray(result), img.size)
                # print(img.size)
                # print(result.shape)
                # print(cropped.size)
                # plt.figure()
                # plt.subplot(1, 3, 1)
                # plt.imshow(img)
                # plt.title('Original Image')
                
                # # Display the preprocessed image
                # plt.subplot(1, 3, 2)
                # plt.imshow(im, cmap='gray')
                # plt.title('Preprocessed Image')

                # Display the cropped image
                # plt.subplot(1, 3, 3)
                # plt.imshow(cropped, cmap='gray')
                # plt.title('Cropped Image')
                
                # plt.show()
                    
        #     break
        # break
        mean_ratio = np.mean(ratio)
        median_ratio = np.median(ratio)

        mean_string = f"Mean \t: {mean_ratio : .2f}"
        median_string = f"Median \t: {median_ratio : .2f}"
        output_string = mean_string + "\n" + median_string
        f = open(file_name,"w+")
        sys.stdout = f
        print(output_string)
