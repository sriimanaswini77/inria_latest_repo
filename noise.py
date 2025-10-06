import numpy as np
import os
import cv2


def noise(image, noise_type="none", mean=0, var=0.1, lam=50):
    img = image.copy()
    row, col, ch = img.shape
    if noise_type == "gauss_n" or noise_type == "gauss":  # random distribution
        std = int(var ** 0.5 * 255)
        gauss = np.zeros(img.shape, dtype=np.uint8)
        gauss = cv2.randn(gauss, mean, std)
        return img + gauss

    elif noise_type == "gauss_u":  # uniform distribution
        std = int(var ** 0.5 * 255)
        gauss = np.zeros(img.shape, dtype=np.uint8)
        gauss = cv2.randu(gauss, mean, std)
        return img + gauss
    elif noise_type == "sp":  # salt and pepper noise
        s_vs_p = 0.5
        amount = 0.004
        out = img
        num_of_noise_points = int(s_vs_p * amount * row * col * ch)
        # salt mode
        for i in range(num_of_noise_points):
            y_coord = np.random.randint(0, row - 1)
            x_coord = np.random.randint(0, col - 1)
            out[y_coord, x_coord, :] = 255

        # pepper mode
        for i in range(num_of_noise_points):
            y_coord = np.random.randint(0, row - 1)
            x_coord = np.random.randint(0, col - 1)
            out[y_coord, x_coord, :] = 0
        return out
    elif noise_type == "poisson":
        out = np.random.poisson(lam, img.shape).astype(np.uint8)
        return out + img
    elif noise_type == "random":
        dst = np.empty_like(img)
        dst = cv2.randn(dst,(0,0,0),(20,20,20))
        out = cv2.addWeighted(img,0.7,dst,0.3,30)
        return out
    else:
        return img


def blur(image, blur_type="none", kernel=(5, 5)):
    img = image.copy()
    if blur_type == "normal":
        img = cv2.blur(img, kernel)
    elif blur_type == "median":
        img = cv2.medianBlur(img, kernel[0])
    elif blur_type == "gauss":
        img = cv2.GaussianBlur(img, kernel, 1)
    elif blur_type == "bilateral":
        img = cv2.bilateralFilter(img, 5, kernel[0], kernel[1])
    return img
