import math
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from collections import Counter


def histogram_equalization(img, rang):
    # 將圖像像素值轉換為整數
    img = img.astype(int)

    # 獲取圖像的高度和寬度
    height, width = img.shape

    # 初始化直方圖
    hist = [0] * rang

    # 計算像素值的頻率並填充到直方圖中
    for i in range(height):
        for j in range(width):
            hist[img[i, j]] += 1

    # 計算累積分佈函數 (CDF)
    cdf = [sum(hist[:i + 1]) for i in range(rang)]

    # 找到 CDF 的最小非零值
    cdf_min = next(value for value in cdf if value > 0)

    # 正規化 CDF
    cdf_normalized = [
        (value - cdf_min) * (rang - 1) / (cdf[-1] - cdf_min) for value in cdf
    ]

    # 創建對照表 (Lookup Table, LUT)
    lut = np.interp(np.arange(rang), np.arange(rang), cdf_normalized)

    # 應用 LUT 並返回結果圖像
    return lut[img].astype(np.uint8)



def equalize_hist(hsi_image):
    # Split the image into H, S, and I channels
    h, s, i = hsi_image[:, :, 0], hsi_image[:, :, 1], hsi_image[:, :, 2]

    # Equalize the histogram of the S and I channels
    s_eq = histogram_equalization((s * 255).astype(np.uint8), 256) / 255.0
    i_eq = histogram_equalization((i * 255).astype(np.uint8), 256) / 255.0

    # Combine the equalized S and I channels with the original H channel
    hsi_eq = np.dstack((h, s_eq, i_eq))

    return hsi_eq


def equalize_hist_rgb(rgb_image):
    # Split the image into R, G, and B channels
    r, g, b = rgb_image[:, :, 0], rgb_image[:, :, 1], rgb_image[:, :, 2]

    # Equalize the histogram of the R, G, and B channels
    r_eq = histogram_equalization(r, 256)
    g_eq = histogram_equalization(g, 256)
    b_eq = histogram_equalization(b, 256)

    # Combine the equalized R, G, and B channels
    rgb_eq = np.dstack((r_eq, g_eq, b_eq))

    return rgb_eq


def equalize_hist_lab(lab_image):
    # Split the image into L, a, and b channels
    l, a, b = lab_image[:, :, 0], lab_image[:, :, 1], lab_image[:, :, 2]

    # Equalize the histogram of the L channel
    l_eq = histogram_equalization(l, 101)

    # Combine the equalized L channel with the original a and b channels
    lab_eq = np.dstack((l_eq, a, b))

    return lab_eq


def rgb_to_hsi(rgb_image):
    # Normalize RGB values to the range 0-1
    rgb_image = rgb_image / 255.0

    # Split the image into R, G, and B channels
    r, g, b = rgb_image[:, :, 0], rgb_image[:, :, 1], rgb_image[:, :, 2]

    # Calculate the intensity
    i = np.mean(rgb_image, axis=2)

    # Calculate the saturation
    min_color = np.min(rgb_image, axis=2)
    s = 1 - 3 * min_color / (
        r + g + b + 1e-6
    )  # Add a small value to avoid division by zero

    # Calculate the hue
    sqrt_val = np.sqrt((r - g) ** 2 + (r - b) * (g - b))
    acos_val = np.arccos(
        0.5 * ((r - g) + (r - b)) / (sqrt_val + 1e-6)
    )  # Add a small value to avoid division by zero
    h = acos_val * 180 / np.pi
    h[b > g] = 360 - h[b > g]

    # Create the HSI image
    hsi_image = np.dstack((h, s, i))

    return hsi_image


def hsi_to_rgb(hsi_image):
    # Split the image into H, S, and I channels
    h, s, i = hsi_image[:, :, 0], hsi_image[:, :, 1], hsi_image[:, :, 2]

    # Initialize the RGB image
    rgb_image = np.zeros(hsi_image.shape)

    # Calculate the RGB values
    h_rad = h * np.pi / 180  # Convert hue from degrees to radians

    # Case 1: 0 <= H < 2*pi/3
    cond1 = np.logical_and(0 <= h_rad, h_rad < 2 * np.pi / 3)
    rgb_image[cond1, 2] = i[cond1] * (1 - s[cond1])
    rgb_image[cond1, 0] = i[cond1] * (
        1 + s[cond1] * np.cos(h_rad[cond1]) / np.cos(np.pi / 3 - h_rad[cond1])
    )
    rgb_image[cond1, 1] = 3 * i[cond1] - (rgb_image[cond1, 0] + rgb_image[cond1, 2])

    # Case 2: 2*pi/3 <= H < 4*pi/3
    cond2 = np.logical_and(2 * np.pi / 3 <= h_rad, h_rad < 4 * np.pi / 3)
    h_rad[cond2] = h_rad[cond2] - 2 * np.pi / 3
    rgb_image[cond2, 0] = i[cond2] * (1 - s[cond2])
    rgb_image[cond2, 1] = i[cond2] * (
        1 + s[cond2] * np.cos(h_rad[cond2]) / np.cos(np.pi / 3 - h_rad[cond2])
    )
    rgb_image[cond2, 2] = 3 * i[cond2] - (rgb_image[cond2, 0] + rgb_image[cond2, 1])

    # Case 3: 4*pi/3 <= H < 2*pi
    cond3 = 4 * np.pi / 3 <= h_rad
    h_rad[cond3] = h_rad[cond3] - 4 * np.pi / 3
    rgb_image[cond3, 1] = i[cond3] * (1 - s[cond3])
    rgb_image[cond3, 2] = i[cond3] * (
        1 + s[cond3] * np.cos(h_rad[cond3]) / np.cos(np.pi / 3 - h_rad[cond3])
    )
    rgb_image[cond3, 0] = 3 * i[cond3] - (rgb_image[cond3, 1] + rgb_image[cond3, 2])

    # Ensure RGB values are in the range 0-255
    rgb_image = np.clip(rgb_image, 0, 1) * 255

    return rgb_image.astype("uint8")


def lab_to_rgb(lab_image):
    # Split the image into L, a, and b channels
    l, a, b = lab_image[:, :, 0], lab_image[:, :, 1], lab_image[:, :, 2]

    # Define the function for transforming Lab to XYZ
    def f_inv(t):
        return np.where(t > 0.206893, t**3, (t - 16 / 116) / 7.787)

    # Transform Lab to XYZ
    y = (l + 16) / 116
    x = a / 500 + y
    z = y - b / 200
    xyz_image = np.dstack((f_inv(x), f_inv(y), f_inv(z)))

    # Normalize XYZ values
    xyz_image *= np.array([95.047, 100.000, 108.883])

    # Define the transformation matrix from XYZ to RGB
    M = np.array(
        [
            [3.2406, -1.5372, -0.4986],
            [-0.9689, 1.8758, 0.0415],
            [0.0557, -0.2040, 1.0570],
        ]
    )

    # Transform XYZ to RGB
    rgb_image = np.dot(xyz_image, M.T)

    # Clip the values to the range 0-1
    rgb_image = np.clip(rgb_image, 0, 1)

    # Convert the values to the range 0-255
    rgb_image = (rgb_image * 255).astype(np.uint8)

    return rgb_image


def rgb_to_lab(rgb_image):
    # Normalize RGB values to the range 0-1
    rgb_image = rgb_image / 255.0

    # Define the transformation matrix from RGB to XYZ
    M = np.array(
        [[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]]
    )

    # Transform RGB to XYZ
    xyz_image = np.dot(rgb_image, M.T)

    # Normalize XYZ values
    xyz_image /= np.array([95.047, 100.000, 108.883])

    # Define the function for transforming XYZ to Lab
    def f(t):
        return np.where(t > 0.008856, t ** (1 / 3), 7.787 * t + 16 / 116)

    # Transform XYZ to Lab
    lab_image = np.zeros(rgb_image.shape)
    lab_image[:, :, 0] = 116 * f(xyz_image[:, :, 1]) - 16  # L
    lab_image[:, :, 1] = 500 * (f(xyz_image[:, :, 0]) - f(xyz_image[:, :, 1]))  # a
    lab_image[:, :, 2] = 200 * (f(xyz_image[:, :, 1]) - f(xyz_image[:, :, 2]))  # b

    return lab_image


def main():
    # List of image paths
    image_paths = [
        "HW3_test_image/aloe.jpg",
        "HW3_test_image/church.jpg",
        "HW3_test_image/house.jpg",
        "HW3_test_image/kitchen.jpg",
    ]

    # Loop through the image paths
    for image_path in image_paths:
        # Open the image
        rgb_image = np.array(Image.open(image_path))

        # Equalize the histogram of the RGB image
        rgb_eq = equalize_hist_rgb(rgb_image)

        # Convert the image to HSI
        hsi_image = rgb_to_hsi(rgb_image)

        # Equalize the histogram of the S and I channels
        hsi_eq = equalize_hist(hsi_image)

        # Convert the equalized HSI image back to RGB
        rgb_image_back_from_hsi = hsi_to_rgb(hsi_eq)

        # Convert the image to Lab
        lab_image = rgb_to_lab(rgb_image)

        # Equalize the histogram of the L channel
        lab_eq = equalize_hist_lab(lab_image)

        # Convert the equalized Lab image back to RGB
        rgb_image_back_from_lab = lab_to_rgb(lab_eq)

        # Display the original image, the equalized RGB image, the RGB image back from HSI, and the RGB image back from Lab
        plt.figure(figsize=(20, 5))

        plt.subplot(1, 4, 1)
        plt.imshow(rgb_image)
        plt.title("Original Image")

        plt.subplot(1, 4, 2)
        plt.imshow(rgb_eq)
        plt.title("Equalized RGB Image")

        plt.subplot(1, 4, 3)
        plt.imshow(rgb_image_back_from_hsi)
        plt.title("RGB Image Back from HSI")

        plt.subplot(1, 4, 4)
        plt.imshow(rgb_image_back_from_lab)
        plt.title("RGB Image Back from Lab")

        plt.show()


# Call the main function
if __name__ == "__main__":
    main()
