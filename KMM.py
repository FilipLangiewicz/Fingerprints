import numpy as np
from PIL import Image
import cv2
from collections import deque


def KMM(farray: np.ndarray, show: bool = True) -> np.ndarray:
    img = farray.copy()
    
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)
    
    if np.max(img) > 1:
        _, img = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY_INV)
    else:
        img = (img == 0).astype(np.uint8)
    
    iteration = 0
    while True:
        prev_img = img.copy()
        img = run_single_kmm_iteration(img)
        
        iteration += 1
        if show:
            print(f"Iteracja {iteration}")
            print_from_arr(img)
        
        if np.array_equal(prev_img, img):
            break
            
        if iteration > 50: 
            break
    
    return img


def run_single_kmm_iteration(img: np.ndarray) -> np.ndarray:
    result = img.copy()

    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            if img[x, y] == 1:
                is_contour = False
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = x + dx, y + dy
                    if (0 <= nx < img.shape[0] and 0 <= ny < img.shape[1] and 
                        img[nx, ny] == 0):
                        is_contour = True
                        break
                
                if is_contour:
                    result[x, y] = 2
    
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            if result[x, y] in [1, 2]:
                neighbors = get_active_neighbors(result, x, y)
                if 2 <= len(neighbors) <= 4 and are_neighbors_4_connected(result, neighbors):
                    result[x, y] = 4
    
    result[result == 4] = 0
    
    for phase in [2, 3]:
        for x in range(img.shape[0]):
            for y in range(img.shape[1]):
                if result[x, y] == phase:
                    weight = calculate_pixel_weight(result, x, y)
                    if should_be_deleted(weight):
                        result[x, y] = 0
                    else:
                        result[x, y] = 1
        
        if phase == 2:
            for x in range(img.shape[0]):
                for y in range(img.shape[1]):
                    if result[x, y] == 2:
                        all_4_neighbors_active = True
                        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            nx, ny = x + dx, y + dy
                            if (0 <= nx < img.shape[0] and 0 <= ny < img.shape[1]):
                                if result[nx, ny] == 0:
                                    all_4_neighbors_active = False
                                    break
                        
                        if all_4_neighbors_active:
                            result[x, y] = 3
    
    result[result > 1] = 1
    
    return result



def should_be_deleted(pixel_weight: int) -> bool:
    weights_to_delete = {
        3, 5, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31, 48,
        52, 53, 54, 55, 56, 60, 61, 62, 63, 65, 67, 69, 71, 77, 79, 80,
        81, 83, 84, 85, 86, 87, 88, 89, 91, 92, 93, 94, 95, 97, 99, 101,
        103, 109, 111, 112, 113, 115, 116, 117, 118, 119, 120, 121, 123, 
        124, 125, 126, 127, 131, 133, 135, 141, 143, 149, 151, 157, 159, 
        181, 183, 189, 191, 192, 193, 195, 197, 199, 205, 207, 208, 209, 
        211, 212, 213, 214, 215, 216, 217, 219, 220, 221, 222, 223, 224, 
        225, 227, 229, 231, 237, 239, 240, 241, 243, 244, 245, 246, 247, 
        248, 249, 251, 252, 253, 254, 255
    }
    
    return pixel_weight in weights_to_delete
    

def calculate_pixel_weight(array: np.ndarray, x: int, y: int) -> int:

    mask = np.array([
        [128, 1, 2],
        [64, 0, 4],
        [32, 16, 8]
    ])
    
    weight = 0
    rows, cols = array.shape
    
    for i in range(-1, 2):
        for j in range(-1, 2):
            if i == 0 and j == 0:
                continue
                
            nx = x + i
            ny = y + j
            
            if 0 <= nx < rows and 0 <= ny < cols:
                if array[nx, ny] != 0:
                    weight += mask[i + 1, j + 1]
    
    return weight



def remove_fours(array:np.ndarray) -> np.ndarray:
    t = array.copy()
    t[t == 4] = 0
    return t

def mark_black_pixels(array: np.ndarray) -> np.ndarray:
    """
    Przygotuj obraz do przetwarzania - zamień czarne piksele na 1, białe na 0.
    """
    assert np.all(np.isin(array, [0, 255])), "Array should contain only 0 and 255 values"
    
    result = array.copy()
    result[array == 0] = 1 
    result[array == 255] = 0 
    
    return result

def change_ones_to_twos(array:np.ndarray) -> np.ndarray:
    assert np.all(np.isin(array, [0, 1])), "Array should be with 0 and 1 values only"
    
    t = array.copy()
    padded = np.pad(t, pad_width=1, mode='constant', constant_values=10)
    rows, cols = padded.shape
    for i in range(1, rows-1):
        for j in range(1, cols-1):
            if padded[i, j] == 1:
                neighbors = padded[i-1:i+2, j-1:j+2].flatten()
                if 0 in neighbors:
                    t[i-1, j-1] = 2
    return t

def change_twos_to_threes(array:np.ndarray) -> np.ndarray:
    assert np.all(np.isin(array, [0, 1, 2])), "Array should be with 0, 1 and 2 values only"
    
    t = array.copy()
    padded = np.pad(t, pad_width=1, mode='constant', constant_values=10)
    rows, cols = padded.shape
    for i in range(1, rows-1):
        for j in range(1, cols-1):
            if padded[i, j] == 2:
                neighbors = padded[i-1:i+2, j-1:j+2]
                if neighbors[0, 1] != 0 and neighbors[1, 0] != 0 and neighbors[1, 2] != 0 and neighbors[2, 1] != 0:
                    t[i-1, j-1] = 3
    return t

def change_to_fours(array: np.ndarray) -> np.ndarray:
    assert np.all(np.isin(array, [0, 1, 2, 3])), "Array should be with 0, 1, 2 and 3 values only"
    
    rows, cols = array.shape
    t = array.copy()

    def get_neighbors(x, y):
        neighbors = []
        for (i, j) in [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]:
            if i == 0 and j == 0:
                continue  
            xi, yj = x + i, y + j
            if 0 <= xi < rows and 0 <= yj < cols:
                neighbors.append(array[xi, yj])
            else:
                neighbors.append(0)  
        return neighbors

    def are_neighbors_connected(neighbors):
        binary = np.array([1 if n != 0 else 0 for n in neighbors])
        
        ones_count = np.sum(binary)
        if ones_count not in [2, 3, 4]:
            return False
        
        binary = np.concatenate([binary, binary])

        max_consecutive = 0
        current = 0
        for val in binary:
            if val == 1:
                current += 1
                max_consecutive = max(max_consecutive, current)
            else:
                current = 0

        return max_consecutive == ones_count

    for x in range(rows):
        for y in range(cols):
            if t[x, y] == 0:
                continue
            neighbors = get_neighbors(x, y)
            if are_neighbors_connected(neighbors):
                t[x, y] = 4

    return t

def one_pixel_width_skeleton(array: np.ndarray) -> bool:
    assert np.all(np.isin(array, [0, 1, 2, 3, 4])), "Array should be with 0, 1, 2, 3 and 4 values only"
    rows, cols = array.shape
    
    for i in range(rows - 1):
        for j in range(cols - 1):
            window = array[i:i+2, j:j+2]
            if np.all(window != 0):
                return False
    return True    


def print_from_arr(array: np.ndarray) -> None:
    display = np.zeros_like(array, dtype=np.uint8)
    display[array == 0] = 255  
    display[array == 1] = 0   
    
    img = Image.fromarray(display)
    img.show()
    
def map_to_ones(array:np.ndarray) -> np.ndarray:
    assert np.all(np.isin(array, [0, 1, 2, 3, 4])), "Array should be with 0, 1, 2, 3 and 4 values only"
    
    t = array.copy()
    t[array == 1] = 1
    t[array == 0] = 0
    t[array == 2] = 1
    t[array == 3] = 1
    t[array == 4] = 1
    return t

def get_active_neighbors(img: np.ndarray, x: int, y: int) -> list:
    neighbors = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx == 0 and dy == 0:
                continue
            nx, ny = x + dx, y + dy
            if (0 <= nx < img.shape[0] and 0 <= ny < img.shape[1] and 
                img[nx, ny] != 0):
                neighbors.append((nx, ny))
    return neighbors

def are_neighbors_4_connected(img: np.ndarray, neighbors: list) -> bool:
    if not neighbors:
        return False
    
    visited = set()
    queue = deque([neighbors[0]])
    visited.add(neighbors[0])
    
    while queue:
        cx, cy = queue.popleft()
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = cx + dx, cy + dy
            if (nx, ny) in neighbors and (nx, ny) not in visited:
                visited.add((nx, ny))
                queue.append((nx, ny))
    
    return len(visited) == len(neighbors)

