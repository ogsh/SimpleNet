import numpy as np
import cv2


def make_gaussian_map(map_size, center, gauss_size):
    x, y = np.meshgrid(range(map_size[1]), range(map_size[0]))
    x = x - center[0]
    y = y - center[1]
    sigma = 0.5 * gauss_size
    pos_map = x*x/(sigma[1]**2) + y*y/(sigma[0]**2)

    gauss_map = np.exp(-pos_map)

    return gauss_map


if __name__ == "__main__":
    center = np.array([150, 100])
    map_size = np.array([300, 200])
    gauss_size = np.array([50, 100])

    gmap = make_gaussian_map(map_size, center, gauss_size)

    lt = center - gauss_size[::-1]//2
    rb = center + gauss_size[::-1]//2

    cv2.rectangle(gmap, tuple(lt.astype(int)), tuple(rb.astype(int)), 1, )

    cv2.imshow("gmap", (gmap*255).astype(np.uint8))
    cv2.waitKey()