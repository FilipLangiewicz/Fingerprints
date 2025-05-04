import numpy as np
from PIL import Image
import cv2


def morphology_thinning(farray: np.ndarray, show: bool = False) -> np.ndarray:
    img = cv2.bitwise_not(farray)

    skeleton = np.zeros_like(img)

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    iteration = 0

    while True:
        iteration += 1
        print(f"Iteration: {iteration}")
        eroded = cv2.erode(img, kernel)
        opened = cv2.morphologyEx(eroded, cv2.MORPH_OPEN, kernel)
        temp = cv2.subtract(eroded, opened)
        skeleton = cv2.bitwise_or(skeleton, temp)
        img = eroded.copy()

        if show:
            temp_img = cv2.bitwise_not(skeleton)
            temp_img_display = (temp_img > 0).astype(np.uint8) * 255
            Image.fromarray(temp_img_display).show()

        if cv2.countNonZero(img) == 0:
            break

    final_skeleton = (skeleton > 0).astype(np.uint8)

    return final_skeleton
        
    
    
    