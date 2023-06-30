import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

import matplotlib
matplotlib.use('agg')

from wheat_dataset import WheatDataset

from utils_dataset import get_transform_albumentation, get_transform_imgaug_img, get_transform_torch_img
from detection_transforms import Compose, PILToTensor, ConvertImageDtype, RandomHorizontalFlip
from detection_utils import collate_fn

import torch
from torchvision import transforms as T
from torchvision.utils import draw_segmentation_masks
from torchvision.utils import draw_bounding_boxes
import torchvision.ops as ops

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

import numpy as np
import math
import copy
import os
import time

# prediction of model   ->    returns dictionary
def predict_image_window(model, image_window, score_threshold):
    # prediction
    image_window = Image.fromarray(image_window)
    image_window = get_transform_torch_img(image_window).cuda() / 255
    prediction = model([image_window])

    # save as tensor
    prediction[0]["boxes"] = torch.Tensor(prediction[0]["boxes"].cpu().detach().numpy())
    prediction[0]["scores"] = torch.Tensor(prediction[0]["scores"].cpu().detach().numpy())
    prediction[0]["labels"] = torch.Tensor(prediction[0]["labels"].cpu().detach().numpy())
    prediction[0]["masks"] = torch.Tensor(prediction[0]["masks"].cpu().detach().numpy())

    # remove predictions with small probability (score_threshold)
    prediction[0]["boxes"] = prediction[0]["boxes"][prediction[0]["scores"] > score_threshold]
    prediction[0]["labels"] = prediction[0]["labels"][:len(prediction[0]["boxes"])]
    prediction[0]["masks"] = prediction[0]["masks"][prediction[0]["scores"] > score_threshold]
    prediction[0]["scores"] = prediction[0]["scores"][:len(prediction[0]["boxes"])]

    return prediction[0]

# recalculate predicted boxes on 280x280 images onto position of original 521x1721 image    ->  returns predictions dictionary with updated boxes
def recalculate_bbs(predictions, num_columns):
    for i, pred in enumerate(predictions):
        scores = copy.deepcopy(pred["scores"])
        for j, box in enumerate(pred["boxes"]):
            x1, y1, x2, y2 = copy.deepcopy(box)
            # Calculate the row and column index of the current prediction
            row_idx = int(i % (len(predictions) // num_columns))
            col_idx = i // (len(predictions) // num_columns)
            # Calculate the new coordinates based on the row and column index
            new_x1 = x1 + col_idx * 120
            new_y1 = y1 + row_idx * 240
            new_x2 = x2 + col_idx * 120
            new_y2 = y2 + row_idx * 240
            pred["boxes"][j] = torch.stack([new_x1, new_y1, new_x2, new_y2])

    return predictions

# calculate center and pixel area of bounding boxes   ->  returns three lists: x, y and pixel area for bounding boxes
def get_box_center(boxes):
    x_center, y_center, pixel_areas = [], [], []
    for box in boxes:
        x1, y1, x2, y2 = box
        # calculate values
        x_mid = x1 + ((x2-x1) / 2)
        y_mid = y1 + ((y2-y1) / 2)
        pixel_area = (x2-x1) * (y2-y1)
        # append values to lists
        x_center.append(x_mid)
        y_center.append(y_mid)
        pixel_areas.append(pixel_area)
    return x_center, y_center, pixel_areas

# extract overlapping boxes for intersection removal
def categorize_boxes(boxes):
    # x1, y1, x2, y2
    check_overlap = [(0,235,520,285), (0,475,520,525), (0,735,520,765),
                    (0,955,520,1005), (0,1195,520,1245), (0,1435,520,1485),
                    (120,0,400,1720)]

    # check_overlap = [(0,240,520,280), (0,480,520,520), (0,720,520,760),
    #                 (0,960,520,1000), (0,1200,520,1240), (0,1440,520,1480),
    #                 (240,0,280,1720)]
    # (120,0,400,1720)
    # (240,0,280,1720)

    overlap_boxes, normal_boxes = [], []
    for box in boxes:
        x1, y1, x2, y2 = box
        is_overlap = False

        for tup in check_overlap:
            if x1 >= tup[0] and x2 <= tup[2] and y1 >= tup[1] and y2 <= tup[3]:
                overlap_boxes.append(box)
                is_overlap = True
                break

        if not is_overlap:
            normal_boxes.append(box)

    overlap_boxes = torch.stack(overlap_boxes)
    normal_boxes = torch.stack(normal_boxes)
    return overlap_boxes, normal_boxes

# returns intersection area of two boxes
def intersection_area(box1, box2):
    # Calculate the coordinates of the intersection rectangle
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # Calculate the width and height of the intersection rectangle
    width = max(0, x2 - x1)
    height = max(0, y2 - y1)
    
    # Calculate the intersection area
    area = width * height
    
    return torch.as_tensor(area)

# returns area of box
def area(box):
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1

    return width * height

# removes boxes based on intersection
def intersection_filter(both_border_boxes):
    
    filtered_intersection_boxes_ids_deleted = set()
    filtered_intersection_boxes_ids = [i for i in range(both_border_boxes.shape[0])]

    copies = copy.deepcopy(both_border_boxes.tolist())

    my_count = 0

    for i in range(len(filtered_intersection_boxes_ids)-1, 0, -1):
        box1 = both_border_boxes[i]
        for j, box2 in enumerate(copies):
            if i == j:
                continue
        
            area_i = area(box1)
            intersection_ij = intersection_area(box1, box2)

            res = intersection_ij / area_i

            if res.item() > 0.5:
                filtered_intersection_boxes_ids_deleted.add(i)
                del copies[i]
                my_count += 1
                break

    filtered_intersection_boxes_ids = [i for i in range(len(filtered_intersection_boxes_ids)) if i not in filtered_intersection_boxes_ids_deleted]
    filtered_intersection_boxes = both_border_boxes[filtered_intersection_boxes_ids]

    both_border_boxes = filtered_intersection_boxes

    # print("Deleted Boxes:", my_count)

    return both_border_boxes

# removes overlapping boxes across all boxes and associates sub-image to box    ->     returns dict with boxes, scores and boxes with sub-image association
def remove_overlapping_boxes(predictions, iou_threshold):

    boxes = [torch.clone(prediction['boxes']) for prediction in predictions]
    scores = [torch.clone(prediction['scores']) for prediction in predictions]

    # Concat all values
    boxes = torch.cat(boxes)
    scores = torch.cat(scores)

    # Apply NMS to the boxes
    keep = ops.nms(boxes, scores, iou_threshold)

    # Filter out boxes and scores that didn't meet the score threshold
    boxes = boxes[keep]
    scores = scores[keep]

    overlap_boxes, normal_boxes = categorize_boxes(boxes)

    # print("Overlap Area Boxes:", len(overlap_boxes))
    filtered_boxes = intersection_filter(overlap_boxes)

    updated_boxes, all_intersection_unfiltered_boxes = [], []
    for box in normal_boxes:
        updated_boxes.append(box)
        all_intersection_unfiltered_boxes.append(box)
    for box in filtered_boxes:
        updated_boxes.append(box)
    for box in overlap_boxes:
        all_intersection_unfiltered_boxes.append(box)
    updated_boxes = torch.stack(updated_boxes)
    all_intersection_unfiltered_boxes = torch.stack(all_intersection_unfiltered_boxes)

    # Return the filtered boxes, scores, and masks as a dictionary
    predictions = {
        'scores': scores,
        'boxes': boxes,
        'normal_boxes': normal_boxes,
        'overlap_boxes': overlap_boxes,
        'all_inter_unf_boxes': all_intersection_unfiltered_boxes,
        'filtered_boxes': filtered_boxes,
        'updated_boxes': updated_boxes
    }

    return predictions

# get count and paths for all 510 images
def get_files(root_path):
    dir_path = r'/data/departments/schoen/roessle/HSWT_Aehrenzaehlen/images_full_plots/'
    all_paths = []
    count = 0
    # Iterate directory
    for path in os.listdir(dir_path):
        all_paths.append(root_path + path)
        if os.path.isfile(os.path.join(dir_path, path)):
            count += 1
    print('File count:', count)
    return(all_paths)

# "DataLoader" -> Load images based on random idx; number of images defines number of images loaded for evaluation
def get_plot_images(root_path):
    all_paths = get_files(root_path)
    images_list = []

    indices = torch.randperm(len(all_paths)).tolist()
    number_of_images = len(all_paths)

    # preprocess images
    for index in indices[:3]:
        image = all_paths[index]
        im = Image.open(image)
        im = im.rotate(85, expand=True)

        # (left,top), (right,bottom)
        im = im.crop((350, 325, 2600, 1025))
        im = im.resize(size=(1721, 521))

        image = np.array(im)
        image = image.transpose(1,0,2)
        images_list.append(image)
    return images_list

# computation of plots
def main(model, image_list, score_threshold, iou_threshold, mask_threshold):
    COLORS = [(255,0,0), (0,255,0), (0,0,255)]
    counter = 0
    image_counter = 0

    # sliding window 
    results = []
    for image in image_list:
        images = []
        # tmp = image
        step_size_width = 120
        step_size_heigth = 240
        num_columns = math.floor(image.shape[1] / (280-120))
        (w_width, w_height) = (280, 280)
        for x in range(0, image.shape[1] - w_width , step_size_width):       # width
            for y in range(0, image.shape[0] - w_height, step_size_heigth):    # heigth
                image_window = image[y:y + w_height, x:x + w_width, :]
                images.append(image_window)
                # cv2.rectangle(tmp, (x, y), (x + w_width, y + w_height), COLORS[counter], 2)   # show sub-image crops
            counter += 1
        sub_image_count = len(images)

        # predictions for sliding window     
        predictions = []
        parcel_count = []
        for i, image in enumerate(images, start=1):
            prediction = predict_image_window(model, image, score_threshold)
            predictions.append(prediction)
            parcel_count.append(len(prediction["boxes"][prediction["scores"] > score_threshold]))

        print(f"Image Number: {image_counter}")

        predictions = recalculate_bbs(predictions, num_columns)

        sub_image_boxes = [pred["boxes"] for pred in predictions]

        predictions = remove_overlapping_boxes(predictions, iou_threshold)

        x_filtered, y_filtered, pixel_areas = get_box_center(predictions["updated_boxes"])
        
        results.append([images, predictions, x_filtered, y_filtered, pixel_areas, sub_image_boxes])
        image_counter += 1
        
    return results

# Plot image prediction for index idx
def image_plot(results, images_list, idx):
    images, predictions, x_filtered, y_filtered, pixel_areas, sub_image_boxes = results[idx]
    image = images_list[idx]
    image = image.transpose(1,0,2)

    transform = T.Compose([T.ToTensor(), T.ConvertImageDtype(torch.uint8)])
    torch_img = transform(image)

    torch_img_boxes = draw_bounding_boxes(torch_img.permute(0,2,1), predictions["normal_boxes"], colors=(255,0,0))
    torch_img_boxes_b = draw_bounding_boxes(torch_img.permute(0,2,1), predictions["filtered_boxes"], colors=(255,0,0))
    torch_img_boxes_c = draw_bounding_boxes(torch_img.permute(0,2,1), predictions["updated_boxes"], colors=(255,0,0))
    torch_img_boxes_d = draw_bounding_boxes(torch_img.permute(0,2,1), predictions["all_inter_unf_boxes"], colors=(255,0,0))
    # torch_img_masks = draw_segmentation_masks(torch_img_boxes, filtered_mask, colors=(0,0,255), alpha=0.5).permute(2,1,0)
    
    a = len(predictions["all_inter_unf_boxes"])
    b = len(predictions["normal_boxes"])
    d = len(predictions["updated_boxes"])

    # Plots normal image with boxes
    f0 = plt.figure(figsize=(30,16))
    plt.title(f"Image of non-filtered spikes - Wheat count: {a}", fontsize=20)
    implot1 = plt.imshow(torch_img_boxes_d.permute(2,1,0))

    f1 = plt.figure(figsize=(30,16))
    plt.title(f"Image of most spikes - Wheat count: {b}", fontsize=20)
    implot1 = plt.imshow(torch_img_boxes.permute(2,1,0))

    f3 = plt.figure(figsize=(30,16))
    plt.title(f"Image of updated spikes - Wheat count: {d}", fontsize=20)
    implot1 = plt.imshow(torch_img_boxes_c.permute(2,1,0))

# Defines grid and cell counts for gridmap
def get_grid_counts(x_filtered, y_filtered, gridsize_heatmap, cells_w, cells_h):
    # Convert x and y coordinates to NumPy arrays
    x_filtered = np.array(x_filtered)
    y_filtered = np.array(y_filtered)
    
    # Calculate the indices of the grid cells for each point
    x_indices = np.floor(x_filtered / gridsize_heatmap).astype(int)
    y_indices = np.floor(y_filtered / gridsize_heatmap).astype(int)
    
    # Create a 2D array of zeros to store the counts for each grid cell
    grid_counts = np.zeros((cells_w, cells_h), dtype=int)

    # Increment the count for each grid cell that contains a point
    for x, y in zip(x_indices, y_indices):
        grid_counts[y][x] += 1
    
    # Transpose the array so that the rows correspond to the y axis and the columns correspond to the x axis
    counts = grid_counts.T
    
    return counts

"""
returns average count heatmap for all plots
if "plot = True" and idx is given -> show grid and heatmap for plot of given idx [idx ranges between 0 and 509 (number of images loaded)]
"""
def heatmaps(results, images_list, all_grids, gridsize_heatmap, plot, idx):
    
    images, predictions, x_filtered, y_filtered, pixel_areas, sub_image_boxes = results[idx]
    
    image = images_list[idx]
    np_grid = all_grids[idx]
    row, column = all_grids[0].shape

    average_counts = np.zeros(shape=(row,column))

    for grid in all_grids:
        average_counts += grid

    average_counts = np.round((average_counts / len(all_grids)), 1)

    print(f"Average counts per cell: {np.round(np.sum(average_counts) / (row*column), 1)}")

    # Plot heatmap for average of 510 images
    f = plt.figure(figsize=(30,16))
    f.add_subplot(1,1,1)
    heatmap_big = plt.imshow(average_counts, cmap='YlOrRd')
    for i in range(row):
        for j in range(column):
            plt.text(j, i, average_counts[i][j], ha="center", va="center", color="black", fontsize=15)
    plt.title("Average count heatmap for all plot images", fontsize=20)
    plt.yticks(ticks=np.arange(row), labels=["y{}".format(i+1) for i in range(row)])
    plt.xticks(ticks=np.arange(column), labels=["x{}".format(i+1) for i in range(column)])
    f.colorbar(heatmap_big, shrink=0.5)
    # plt.show(block=True)
    plt.savefig("heatmap_all_images.png")
    
    # Plot heatmap and grid for single image
    if plot:
        f1 = plt.figure(figsize=(30,16))
        f1.add_subplot(1,1,1)
        heatmap_big = plt.imshow(np_grid, cmap='YlOrRd')
        for i in range(row):
            for j in range(column):
                plt.text(j, i, np_grid[i][j], ha="center", va="center", color="black", fontsize=15)
        plt.title("Count heatmap for image of plot", fontsize=20)
        plt.yticks(ticks=np.arange(row), labels=["y{}".format(i+1) for i in range(row)])
        plt.xticks(ticks=np.arange(column), labels=["x{}".format(i+1) for i in range(column)])
        f1.colorbar(heatmap_big, shrink=0.5)
        # plt.show(block=True)
        plt.savefig("heatmap_single_image.png")

        f2 = plt.figure(figsize=(30,16))
        f2.add_subplot(1,1,1)
        plt.title("Grid with wheat head center points for image of plot", fontsize=20)
        plt.plot(y_filtered, x_filtered, 'ro', markersize=3)
        implot3 = plt.imshow(image.transpose(1,0,2), extent=[0, 1721, 521, 0])
        for x in range(0, 1721, gridsize_heatmap):
            plt.plot([x, x], [0, 521], c='b', linewidth=1)
        for y in range(0, 521, gridsize_heatmap):
            plt.plot([0, 1721], [y, y], c='b', linewidth=1)
        # plt.show(block=True)
        plt.savefig("grid_single_image.png")


if __name__ == "__main__":
    model_path = "/home/emj6571/model/model_5_epochs_2022-12-14_v2.pt"
    root_path = "/data/departments/schoen/roessle/HSWT_Aehrenzaehlen/images_full_plots/"
    model = torch.load(model_path)

    score_threshold = 0.7
    iou_threshold = 0.2
    mask_threshold = 0.5

    images_list = get_plot_images(root_path)        

    start = time.time()
    results = main(model, images_list, score_threshold, iou_threshold, mask_threshold)
    end = time.time()
    print("")
    print(f"Time elapsed: {np.round(end - start, 2)} seconds")

    # calculates grid size
    gridsize_heatmap = 40
    cells_w = int(1720 / gridsize_heatmap)
    cells_h = int(520 / gridsize_heatmap)

        # get grids for all images
    all_grids = []
    for parameters in results:
        np_grids = get_grid_counts(parameters[2], parameters[3], gridsize_heatmap, cells_w, cells_h)
        all_grids.append(np_grids)

    plt.rcParams.update({'font.size': 10})
    heatmaps(results, images_list, all_grids, gridsize_heatmap, True, 0)