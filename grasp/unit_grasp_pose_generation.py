import numpy as np
import os
import torch
from torch import nn
from grasp.model import get_model
import cv2
from torchvision import transforms
import torch.nn.functional as F
from PIL import Image
from grasp.grasp_detect_multibox import *
from shapely.geometry import Polygon
from shapely.geometry import LineString, Point
from shapely.affinity import rotate
from skimage.measure import find_contours

def grasp_quality(grasp, convex_boundary):
    _, x, y, h, w, angle = grasp
    # print(x, y, h, w)

    # Define the line segment and rotate the line segment by theta degrees around the given point
    line = LineString([(x-h/2, y), (x+h/2, y)])
    rotated_line = rotate(line, angle, origin=(x, y))

    # Get intersection points
    intersection_points = convex_boundary.intersection(rotated_line)
    
    quality = 0
    found = 0
    points = convex_boundary.exterior.coords

    for point in intersection_points.coords:
        edge_line = None
        for i,j in zip(points, points[1:]):
            if LineString((i,j)).distance(Point(point)) < 1e-8:
                edge_line = LineString((i,j))
                break

        if edge_line is not None:
            found += 1
            vector1 = np.array([intersection_points.coords[1][0] - intersection_points.coords[0][0], intersection_points.coords[1][1] - intersection_points.coords[0][1]])
            vector2 = np.array([edge_line.coords[1][0] - edge_line.coords[0][0], edge_line.coords[1][1] - edge_line.coords[0][1]])

            # Normalize the vectors
            vector1 /= np.linalg.norm(vector1)
            vector2 /= np.linalg.norm(vector2)

            # Calculate the dot product of the vectors
            dot_product = np.dot(vector1, vector2)

            # Calculate the angle in radians
            angle_rad = np.arccos(dot_product)
            quality += np.absolute(np.sin(angle_rad))

    if found == 1:
        quality = 0
    elif found == 0:
        quality = -1

    # Calculate the distance between the center point
    center = convex_boundary.centroid
    center_box = Point(x, y)
    distance = center.distance(center_box)
    quality_dis = quality / (distance + 1e-5)

    return [quality, distance]

def get_best_grasp(mask, boxes):
    contours = find_contours(mask, level=0.5)
    boundary_coords = contours[0]  # Use the first set of boundary points
    polygon = Polygon([(x,y) for y, x in boundary_coords])  # Convert to (x, y)
    convex_boundary = polygon.convex_hull

    scores = []
    for grasp in boxes: 
        score = grasp_quality(grasp.detach().cpu().numpy(), convex_boundary)
        scores.append(score)
        
    smallest_distance_indices = sorted(range(len(scores)), key=lambda i: scores[i][1])[:5]
    highest_quality_index = max(smallest_distance_indices, key=lambda i: scores[i][0])

    return boxes[highest_quality_index]

def load_grasp_model(device, ragt_weights_path='weights/RAGT-3-3.pth'):
    inference_multi_image = DetectMultiImage(device=device, weights_path=ragt_weights_path)
    print("Successfully loaded grasp detection model!")
    return inference_multi_image

def detect_grasp(grasp_model, image, mask, device):
    result = cv2.bitwise_and(image, image, mask=mask)
    img = torch.from_numpy(result).permute(2, 0, 1).float().unsqueeze(0).to(device)
    boxes = grasp_model(img, 0.99)
    best_grasp = get_best_grasp(mask, boxes)
    return best_grasp


def detect_grasp_directional(grasp_model, image, mask, device, direction="center"):
    """Like detect_grasp but selects the candidate closest to the requested side.

    direction: "top" | "bottom" | "left" | "right" | "center"
    Picks from all RAGT candidates (lower threshold) the one whose center
    is nearest to a target point 30% in from the requested edge of the mask.
    """
    if direction == "center":
        return detect_grasp(grasp_model, image, mask, device)

    result = cv2.bitwise_and(image, image, mask=mask)
    img_t = torch.from_numpy(result).permute(2, 0, 1).float().unsqueeze(0).to(device)

    # Get as many candidates as possible by lowering the confidence threshold
    boxes = None
    for thresh in [0.5, 0.2, 0.05]:
        boxes = grasp_model(img_t, thresh)
        if len(boxes) >= 3:
            break

    if boxes is None or len(boxes) == 0:
        return detect_grasp(grasp_model, image, mask, device)

    # Compute mask extents to locate the target point for the requested direction
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return get_best_grasp(mask, boxes)

    cx_mid = (xs.min() + xs.max()) / 2.0
    cy_mid = (ys.min() + ys.max()) / 2.0
    f = 0.3   # how far from edge toward center the target sits

    target = {
        "top":    (cx_mid, ys.min() + (cy_mid - ys.min()) * f),
        "bottom": (cx_mid, ys.max() - (ys.max() - cy_mid) * f),
        "left":   (xs.min() + (cx_mid - xs.min()) * f, cy_mid),
        "right":  (xs.max() - (xs.max() - cx_mid) * f, cy_mid),
    }.get(direction, (cx_mid, cy_mid))

    tx, ty = target
    boxes_cpu = boxes.detach().cpu()
    dists = (boxes_cpu[:, 1] - tx) ** 2 + (boxes_cpu[:, 2] - ty) ** 2
    idx = int(dists.argmin().item())
    return boxes[idx]

if __name__ == '__main__':
    image_fn = "data/grasp-anything/image/ccd42137b3646367bad45436a70454c33b819399d0d92e72f7e0119d0e43ffc2.jpg"
    image = Image.open(image_fn)

    mask_fn = "data/graspL1M/part_mask/ccd42137b3646367bad45436a70454c33b819399d0d92e72f7e0119d0e43ffc2_0_3.npy"
    masks = np.load(mask_fn)


    inference_multi_image = load_grasp_model(ragt_weights_path='pretrained_weights/RAGT-3-3.pth')
    detect_grasp(inference_multi_image, image, masks)
