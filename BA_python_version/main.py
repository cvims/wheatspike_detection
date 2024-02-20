import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from PIL import Image, ImageDraw

import matplotlib
matplotlib.use('agg')

import cv2

from plot_dataset import PlotDataset

import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T
import torchvision.transforms.functional as F
from torchvision.utils import draw_bounding_boxes
import torchvision.ops as ops

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

import numpy as np
import math
import copy
import os
import random
import csv
import time

from tqdm import tqdm


# set seed
def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    print("Seed set!")


# recalculate predicted boxes on 280x280 images onto position of original 521x1721 image    ->  returns predictions dictionary with updated boxes
def recalculate_bbs(window_predictions, image_size, num_columns: int, overlapping: int):
    """
    This is an in-built operation on the window predictions. It is used to update the bounding boxes.
    """

    # get image size
    image_h, image_w = image_size

    boxes = []

    for i, window in enumerate(window_predictions):
        for j in range(window['boxes'].shape[0]):
            x1, y1, x2, y2 = window['boxes'][j]
            # update box to original image size
            x1 = x1 + (i % num_columns) * image_w - ((i % num_columns) * overlapping)
            y1 = y1 + (i // num_columns) * image_h - ((i // num_columns) * overlapping)
            x2 = x2 + (i % num_columns) * image_w - ((i % num_columns) * overlapping)
            y2 = y2 + (i // num_columns) * image_h - ((i // num_columns) * overlapping)

            # update box in window
            boxes.append(torch.stack([x1, y1, x2, y2]))
    
    return torch.stack(boxes)


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


def apply_score_threshold(boxes, scores, threshold):
    # Apply score threshold
    keep = scores >= threshold

    # Keep only the boxes and scores that were kept by the threshold
    remaining_boxes = boxes[keep]
    remaining_scores = scores[keep]

    # Get the removed boxes and scores
    removed_boxes = boxes[~keep]
    removed_scores = scores[~keep]
    
    return remaining_boxes, remaining_scores, removed_boxes, removed_scores


def apply_nms(boxes, scores, iou_threshold):
    """
    Adds bounding boxes to the predictions based on the IoU threshold.
    The dictionary gets updated in place.
    """

    # shape of predictions
    # [windows, predictions]

    # Apply NMS
    keep = ops.nms(boxes, scores, iou_threshold)

    # Keep only the boxes and scores that were kept by NMS
    remaining_boxes = boxes[keep]
    remaining_scores = scores[keep]

    # Get the removed boxes and scores
    removed_boxes = boxes[~keep]
    removed_scores = scores[~keep]
    
    return remaining_boxes, remaining_scores, removed_boxes, removed_scores


@torch.no_grad()
def run_sliding_window_approach(model, data_loader, device, window_size, window_overlapping,
                                iou_threshold, score_threshold,
                                plot_single_image_heatmap=False, plot_raw_single_image_boxes=False,
                                plot_filtered_image_boxes=False,
                                plot_all_images_heatmap=False, cell_size_heatmap=40,
                                write_boxes_centers=False,
                                plot_segmentation_masks=False,
                                subplot_upsacling=False):
    """
    This function runs the sliding window approach on the model and the data loader.
    The output are the bounding boxes across the original image and
    not the bounding boxes of the sub-images.
    """
    model.eval()
    model.to(device)

    images_dir = 'images'

    single_image_boxes_dir = os.path.join(images_dir, 'single_image_boxes')
    if plot_raw_single_image_boxes:
        os.makedirs(single_image_boxes_dir, exist_ok=True)
    
    filtered_image_boxes_dir = os.path.join(images_dir, 'filtered_image_boxes')
    if plot_filtered_image_boxes:
        os.makedirs(filtered_image_boxes_dir, exist_ok=True)
    
    single_image_heatmap_dir = os.path.join(images_dir, 'single_image_heatmap')
    if plot_single_image_heatmap:
        os.makedirs(single_image_heatmap_dir, exist_ok=True)
    
    all_images_heatmap_dir = os.path.join(images_dir, 'all_images_heatmap')
    if plot_all_images_heatmap:
        os.makedirs(all_images_heatmap_dir, exist_ok=True)

    boxes_centers_dir = os.path.join(images_dir, 'box_centers')
    if plot_all_images_heatmap:
        os.makedirs(boxes_centers_dir, exist_ok=True)

    segmentation_masks_dir = os.path.join(images_dir, 'segmentation_masks')
    if plot_segmentation_masks:
        os.makedirs(segmentation_masks_dir, exist_ok=True)

    # For the full batch heatmap
    all_grids = []

    step_size_wh = window_size - window_overlapping

    image_counter = 0
    # Iterate through the data loader
    all_results = []
    for i, batch in enumerate(tqdm(data_loader, desc='Running sliding window approach')):
    # if i == 122:
        # if i == 1:
        #     break
        # To same device as model
        batch = batch.to(device)
        # Cut the image into windows
        batch_size = batch.shape[0]
        num_columns = math.floor(batch.shape[3] / step_size_wh)
        num_rows = math.floor(batch.shape[2] / step_size_wh)

        # print(f"Index: {i+1}")
        for image in batch:
            image_counter += 1
            windows = []
            for x in range(num_rows):
                for y in range(num_columns):
                    current_height = x * step_size_wh
                    current_width = y * step_size_wh
                    window = image[:, current_height:current_height + window_size, current_width:current_width + window_size]
                    windows.append(window)

            # Split the windows into batches
            windows = torch.stack(windows)
            # Shape: Tuple[batch_size, window_output]
            windows = windows.split(batch_size)

            # Run the model on all the batched window splits
            window_predictions = []
            for window in windows:
                predictions = model(window)
                # Move predictions to cpu before appending to list
                predictions = [{k: v.to('cpu') for k, v in t.items() if k != 'masks'} for t in predictions]
                window_predictions.extend(predictions)

            # stack window predictions
            boxes = torch.cat([prediction['boxes'] for prediction in window_predictions])
            scores = torch.cat([prediction['scores'] for prediction in window_predictions])
            # labels = torch.cat([prediction['labels'] for prediction in window_predictions])
            # masks = torch.cat([prediction['masks'] for prediction in window_predictions])

            # in_built bounding boxes recalculation
            boxes = recalculate_bbs(window_predictions, (window_size, window_size), num_columns, window_overlapping)       

            if plot_raw_single_image_boxes:
                single_boxes_image_path = os.path.join(single_image_boxes_dir, f"image_boxes_{image_counter}.png")
                plot_bounding_boxes_to_image(image, boxes, save_path=single_boxes_image_path)
                # write text file with box count
                single_boxes_text_path = os.path.join(single_image_boxes_dir, f"image_boxes_{image_counter}.txt")
                write_box_count_to_file(boxes, single_boxes_text_path)
            
            # Filter the boxes for complete parcel (IoU)
            boxes, scores, removed_boxes, removed_scores = apply_nms(boxes, scores, iou_threshold)
            boxes, scores, removed_boxes, removed_scores = apply_score_threshold(boxes, scores, score_threshold)

            if plot_filtered_image_boxes:
                filtered_boxes_image_path = os.path.join(filtered_image_boxes_dir, f"image_boxes_{image_counter}.png")
                plot_bounding_boxes_to_image(image, boxes, save_path=filtered_boxes_image_path)
                # write text file with box count
                filtered_boxes_text_path = os.path.join(filtered_image_boxes_dir, f"image_boxes_{image_counter}.txt")
                write_box_count_to_file(boxes, filtered_boxes_text_path)
            
            # Create bounding box centers
            boxes_centers = calculate_bounding_boxes_centers(boxes)

            if write_boxes_centers:
                boxes_centers_text_path = os.path.join(boxes_centers_dir, f"image_boxes_center_{image_counter}.csv")
                write_box_centers_to_file(boxes_centers, boxes_centers_text_path)     

            if plot_single_image_heatmap:
                grid = map_bounding_boxes_to_grid(boxes_centers, image.shape[1:], cell_size_heatmap)

                single_image_heatmap_path = os.path.join(single_image_heatmap_dir, f"image_heatmap_{image_counter}.png")
                plot_heatmap_image(grid, single_image_heatmap_path, add_numbers=True)

                # write grid to text file
                single_image_heatmap_text_path = os.path.join(single_image_heatmap_dir, f"image_heatmap_{image_counter}.txt")
                write_grid_to_file(grid, single_image_heatmap_text_path)

                # Add to all heatmaps
                all_grids.append(grid)

            if plot_all_images_heatmap and not plot_single_image_heatmap:
                # calculate grid anyway
                grid = map_bounding_boxes_to_grid(boxes_centers, image.shape[1:], cell_size_heatmap)

                # Add to all heatmaps
                all_grids.append(grid)

            if subplot_upsacling:
                sub_imageboxes_list = calculate_subplot_upsacling(image, boxes, windows, window_predictions, 
                                                                idx=0, details=False, save_plot=False)
            
            all_results.append([windows, window_predictions, boxes, sub_imageboxes_list])

    # plot mean heatmap
    if plot_all_images_heatmap:
        # Calculate the mean heatmap
        all_grids = torch.stack(all_grids)
        mean_grid = torch.mean(all_grids, axis=0)

        # Average Count
        w, h = grid.shape
        average_heatmap_counts = torch.sum(mean_grid) / (w*h)

        all_images_heatmap_path = os.path.join(all_images_heatmap_dir, f"all_images_heatmap.png")
        plot_heatmap_image(mean_grid, all_images_heatmap_path, add_numbers=True)

        # write grid to text file
        all_images_heatmap_text_path = os.path.join(all_images_heatmap_dir, f"all_images_heatmap.txt")
        write_grid_to_file(mean_grid, all_images_heatmap_text_path)

    if subplot_upsacling:
        full_subplot_estimation(all_results, subplot_idx=2, plot_all=True)

    print(f"Average Heatmap Count: {average_heatmap_counts}\n")

    return predictions


# estimate single full plot using a single 280x280 plot
def calculate_subplot_upsacling(image, boxes, windows, window_predictions, idx, details=False, save_plot=False):
    
    total_wheatheads = len(boxes)
    # print(f"Total Wheat Heads: {total_wheatheads}\n")

    estimations = []
    row = []
    array = []
    sub_image_boxes_list = []

    _, w, h = image.shape
    full_plot_size = w * h
    small_plot_size = 280*280
    inverse_proportion = full_plot_size / small_plot_size

    idx = 0
    for i, batch in enumerate(windows):
        for j, window in enumerate(batch):
            
            print("batch nms:", len(batch))
            print(idx+j)

            # Filter single sub-images
            sub_image_boxes, sub_image_scores, _, _ = apply_nms(window_predictions[idx+j]["boxes"], window_predictions[idx+j]["scores"], iou_threshold)
            sub_image_boxes, _, _, _ = apply_score_threshold(sub_image_boxes, sub_image_scores, score_threshold)
            sub_image_boxes_list.append(sub_image_boxes)

            # Count-based estimation
            count_small_plot = len(sub_image_boxes)
            count_estimation = math.floor(inverse_proportion * count_small_plot)

            # print(f"Sub-Image Wheat Heads: {count_small_plot}")

            if details:
                print(f"Sub-image {idx+j}")
                print(f"Part Wheat Heads: {count_small_plot}")
                print(f"Count-based Estimated Wheat Heads: {count_estimation}")
                print(f"Deviation (Count): {count_estimation-total_wheatheads}")
                print(f"Deviation (Percent): {np.round(((count_estimation/total_wheatheads)-1)*100, 2)}%\n")

            estimations.append(count_estimation)
            row.append(count_estimation)

            if (idx+1) % 8 == 0:
                array.append(row)
                row = []
        idx += len(batch)

    # print(f"Min estimation: {min(estimations)} | Index: {estimations.index(min(estimations))}")
    # print(f"Max estimation: {max(estimations)} | Index: {estimations.index(max(estimations))}\n")

    # print(f"{torch.as_tensor(array)}\n")

    if save_plot:
        # Sub-Image generation
        image_pil = F.to_pil_image(windows[idx])
        image_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        for box in sub_image_boxes_list[idx]:
            box = box.int()
            x1, y1, x2, y2 = box.tolist()
            image_cv = cv2.rectangle(image_cv, (x1, y1), (x2, y2), color=(0,0,255), thickness=1)
        image_pil_result = Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
        # image_pil_result.save("/home/jacobowsky/Bachelorarbeit_Emanuel_J/BA_python_version/example_subplot_upscale_subimage.png")

        # Plot-Image generation
        image_pil = F.to_pil_image(image)
        image_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        for box in boxes:
            box = box.int()
            x1, y1, x2, y2 = box.tolist()
            image_cv = cv2.rectangle(image_cv, (x1, y1), (x2, y2), color=(0,0,255), thickness=1)
        image_pil_result = Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
        # image_pil_result.save("/home/jacobowsky/Bachelorarbeit_Emanuel_J/BA_python_version/example_subplot_upscale_plot.png")

    return sub_image_boxes_list


def scatter_plot(sub_plot_estimation, full_estimation, subplot_idx):

    save_folder = r"/home/jacobowsky/Bachelorarbeit_Emanuel_J/BA_python_version/figures/scatterplots"

    r2_list, rmse_list = [], []
    x = np.arange(len(full_estimation))
    all_subplots = True
    row_counter = 0

    row = ["A", "B"]

    if all_subplots:
        for i, single_estimation in enumerate(sub_plot_estimation):
            single_full_estimate = full_estimation
            r2 = r2_score(single_full_estimate, single_estimation)
            rmse = np.sqrt(mean_squared_error(single_full_estimate, single_estimation))
            r2_index = i % 8

            # get correct row and col
            if i == 8:
                row_counter += 1

            f = plt.figure(figsize=(16, 16))
            plt.title(f'Sub-Image: {r2_index+1}{row[row_counter]} - R²: {r2:.2f} - RMSE: {rmse:.2f}', fontsize=26)
            plt.scatter(single_full_estimate, single_estimation, color='red', label=f'Data Points: {len(single_estimation)}', s=15)
            plt.plot([min(single_full_estimate), max(single_full_estimate)], [min(single_full_estimate), max(single_full_estimate)], linestyle='--', linewidth=2, color='grey', label=f'R²: {r2:.2f}')
            plt.xlabel('Sliding Window Prediction', fontsize=24)
            plt.ylabel('Subplot Upscaling', fontsize=24)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            plt.grid(True)
            plt.legend(fontsize=20)

            save_path = os.path.join(save_folder, f'scatterplots_{i+1}.png')
            plt.savefig(save_path)
            plt.close(f)

    for sb_est in sub_plot_estimation:
        r2_list.append(r2_score(full_estimation, sb_est))
        rmse_list.append(np.sqrt(mean_squared_error(full_estimation, sb_est)))
    r2_best = max(r2_list)
    r2_worst = min(r2_list)
    r2_best_index = r2_list.index(r2_best)
    r2_worst_index = r2_list.index(r2_worst)
    rmse_best = min(rmse_list)
    rmse_best_index = rmse_list.index(rmse_best)
    rmse_worst = max(rmse_list)
    rmse_worst_index = rmse_list.index(rmse_worst)
    sub_plot_estimation_best = sub_plot_estimation[r2_best_index]
    sub_plot_estimation_worst = sub_plot_estimation[r2_worst_index]

    print(f"Min R²: {min(r2_list)} - Index: {r2_list.index(min(r2_list))}")
    print(f"Max R²: {max(r2_list)} - Index: {r2_list.index(max(r2_list))}")
    print(f"Min RMSE: {min(rmse_list)} - Index: {rmse_list.index(min(rmse_list))}")
    print(f"Max RMSE: {max(rmse_list)} - Index: {rmse_list.index(max(rmse_list))}\n")
    print(f"All R² Values: {np.round(r2_list, 2)}\n")
    print(f"All RMSE Values: {np.round(rmse_list, 2)}\n")

    f = plt.figure(figsize=(24,10))

    f.add_subplot(1,2,1)
    plt.title(f'Best Sub-Image: {r2_best_index+1} - R²: {r2_best:.2f} - RMSE: {rmse_best:.2f}')
    plt.scatter(full_estimation, sub_plot_estimation_best, color='red', label=f'Data Points: {len(sub_plot_estimation)}', s=15)
    plt.plot([min(full_estimation), max(full_estimation)], [min(full_estimation), max(full_estimation)], linestyle='--', linewidth=2, color='grey', label=f'R²: {r2_best:.2f}')
    plt.xlabel('Sliding Window Prediction')
    plt.ylabel('Subplot Upscaling')
    plt.grid(True)
    plt.legend()

    f.add_subplot(1,2,2)
    plt.title(f'Worst Sub-Image: {r2_worst_index+1} - R²: {r2_worst:.2f} - RMSE: {rmse_worst:.2f}')
    plt.scatter(full_estimation, sub_plot_estimation_worst, color='red', label=f'Data Points: {len(sub_plot_estimation)}', s=15)
    plt.plot([min(full_estimation), max(full_estimation)], [min(full_estimation), max(full_estimation)], linestyle='--', linewidth=2, color='grey', label=f'R²: {r2_worst:.2f}')
    plt.xlabel('Sliding Window Prediction')
    plt.ylabel('Subplot Upscaling')
    plt.grid(True)
    plt.legend()

    save_path = os.path.join(save_folder, f'scatterplots_combined.png')
    plt.savefig(save_path)
    plt.close(f)


def full_subplot_estimation(all_results, subplot_idx, plot_all=False):

    average_prediction = np.zeros(shape=(2,8))
    average_total_wheatheads = 0
    full_plot_size = 547 * 2149
    small_plot_size = 280*280
    inverse_proportion = full_plot_size / small_plot_size
    total_wheatheads_list = []
    subplot_count_dict = {}

    # get counts for each sub-plot and add them across all plots to get the average upscaling values
    for i, full_images in enumerate(all_results):
        tmp = np.zeros(shape=(16))
        windows = full_images[0]
        total_wheatheads = len(full_images[2])
        total_wheatheads_list.append(total_wheatheads)
        idx = 0
        for j, batch in enumerate(windows):
            for k, window in enumerate(batch):
                print("batch upscale:", len(batch))
                print(idx+j)
                boxes = full_images[3][idx+k]
                count_small_plot = len(boxes)
                count_estimation = math.floor(inverse_proportion * count_small_plot)
                if idx+k not in subplot_count_dict:
                    subplot_count_dict[idx+k] = [count_estimation]
                else:
                    subplot_count_dict[idx+k].append(count_estimation)
                tmp[idx+k] += count_estimation
            idx += len(batch)
        tmp = tmp.reshape((2,8))
        average_prediction += tmp
        average_total_wheatheads += total_wheatheads
            
    average_prediction = average_prediction / len(all_results)
    average_total_wheatheads = average_total_wheatheads / len(all_results)

    difference_prediction_total = average_prediction - average_total_wheatheads

    deviation_matrix = ((average_prediction/average_total_wheatheads)-1)*100

    print(f"Average Total Wheatheads (Sliding Window): \n {int(average_total_wheatheads)}\n")
    print(f"Average Prediction: \n {average_prediction}\n")
    print(f"Difference Prediction: \n {difference_prediction_total}\n")
    print(f"Average Difference Prediction: {np.round(np.sum(difference_prediction_total)/21,2)}\n")
    print(f"Deviation (Percent): \n {np.round(deviation_matrix, 2)}\n")
    print(f"Average Deviation: {np.round(np.sum(deviation_matrix) / 21, 2)}%\n")

    # get items of dict as list for subplot creation
    subplot_values = []
    for key, value in subplot_count_dict.items():
        subplot_values.append(value)

    # by index
    if plot_all:
        scatter_plot(subplot_values, total_wheatheads_list, subplot_idx)

    return average_prediction, average_total_wheatheads


def map_bounding_boxes_to_grid(boxes, image_size, grid_cell_size):
    # create grid
    cells_h = math.ceil(image_size[0] / grid_cell_size)
    cells_w = math.ceil(image_size[1] / grid_cell_size)

    # create grid
    grid = torch.zeros((cells_h, cells_w))

    # add center points to grid
    for center in boxes:
        # calculate cell position
        cell_x = math.floor(center[0] / grid_cell_size)
        cell_y = math.floor(center[1] / grid_cell_size)

        # add center point to grid
        grid[cell_y, cell_x] += 1
    
    return grid


def add_bounding_boxes_center(predictions):
    # calculate the center poisiton of each bounding box
    for iteration in predictions:
        for batch in iteration:
            for window in batch:
                # get bounding boxes
                boxes = window['boxes']

                # calculate center width and height
                center = []
                for i, box in enumerate(boxes):
                    center.append((box[0] + box[2]) / 2)
                    center.append((box[1] + box[3]) / 2)

                # add center to dictionary
                window['center'] = center

    return predictions


def calculate_bounding_boxes_centers(boxes):
    # calculate the center poisiton of each bounding box
    centers = []
    for box in boxes:
        center_x = (box[0] + box[2]) / 2
        center_y = (box[1] + box[3]) / 2
        centers.append(torch.stack([center_x, center_y]))
    
    return torch.stack(centers)


def plot_heatmap_image(grid, save_path, add_numbers=True):
    # Plotting the heatmap

    # start color NOT at 0
    f = plt.figure(figsize=(30,16))
    f.add_subplot(1,1,1)
    # heatmap = plt.imshow(grid, cmap='YlOrRd', interpolation='nearest')
    heatmap = plt.imshow(grid, cmap='YlOrRd', interpolation='nearest')

    if add_numbers:
        # Add the numbers to the heatmap
        row, column = grid.shape
        for i in range(row):
            for j in range(column):
                # plot plt text rounded to 1 decimal
                plt.text(j, i, f"{grid[i, j].item():.1f}", ha="center", va="center", color="black", fontsize=15)

        plt.yticks(ticks=torch.arange(row), labels=["y{}".format(i+1) for i in range(row)])
        plt.xticks(ticks=torch.arange(column), labels=["x{}".format(i+1) for i in range(column)])

    f.colorbar(heatmap, shrink=0.4)
    # Save the image
    plt.savefig(save_path, dpi=300)
    plt.close()

    # start color at 0
    f = plt.figure(figsize=(30,16))
    f.add_subplot(1,1,1)
    # heatmap = plt.imshow(grid, cmap='YlOrRd', interpolation='nearest')
    heatmap = plt.imshow(grid, cmap='YlOrRd', interpolation='nearest', clim=(0, grid.max()))

    if add_numbers:
        # Add the numbers to the heatmap
        row, column = grid.shape
        for i in range(row):
            for j in range(column):
                # plot plt text rounded to 1 decimal
                plt.text(j, i, f"{grid[i, j].item():.1f}", ha="center", va="center", color="black", fontsize=15)

        plt.yticks(ticks=torch.arange(row), labels=["y{}".format(i+1) for i in range(row)])
        plt.xticks(ticks=torch.arange(column), labels=["x{}".format(i+1) for i in range(column)])

    f.colorbar(heatmap, shrink=0.4)
    # Save the image
    save_path = save_path[:-4] + "_colorbar0.png"
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_bounding_boxes_to_image(image: torch.Tensor, boxes: torch.Tensor, color=(255, 0, 0), thickness=2, save_path=None):
    """
    boxes shape [N, 4]
    """

    # Convert image tensor to PIL image
    image_pil = F.to_pil_image(image)

    # Convert PIL image to OpenCV format
    image_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

    for box in boxes:
        box = box.int()
        x1, y1, x2, y2 = box.tolist()  # Extract the coordinates
        image_cv = cv2.rectangle(image_cv, (x1, y1), (x2, y2), color, thickness)

    # Convert OpenCV image back to PIL image
    image_pil_result = Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))

    if save_path is not None:
        image_pil_result.save(save_path)


def plot_sliding_window_approach(image: torch.Tensor, window_size: int, window_overlapping: int, num_cols: int, num_rows: int, save_path=None, color=(255, 0, 0)):
    """
    Plot the sliding window approach on the image.
    """

    # Convert image tensor to PIL image
    image_pil = F.to_pil_image(image)

    # Convert PIL image to OpenCV format
    image_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

    # Draw the sliding window with overlappings on the image
    for i in range(num_rows):
        for j in range(num_cols):
            x1 = j * window_size - j * window_overlapping
            y1 = i * window_size - i * window_overlapping
            x2 = x1 + window_size
            y2 = y1 + window_size
            image_cv = cv2.rectangle(image_cv, (x1, y1), (x2, y2), color, 2)

    # Convert OpenCV image back to PIL image
    image_pil_result = Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))

    if save_path is not None:
        image_pil_result.save(save_path)


def write_box_count_to_file(boxes, path):
    with open(path, 'w') as f:
        f.write(f"Number of boxes: {len(boxes)}")


def write_box_centers_to_file(boxes, path):
    with open(path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        header = ['x', 'y']
        csv_writer.writerow(header)
        for coord in boxes:
            csv_writer.writerow(coord.tolist())
    csvfile.close()


def write_grid_to_file(grid, path):
    # Convert torch tensor to NumPy array
    grid_np = grid.numpy()

    with open(path, 'w') as f:
        f.write(f"Grid shape: {grid_np.shape}\n")
        # Write the NumPy array to file
        np.savetxt(f, grid_np, fmt='%.2f')


def get_plot_transforms(center_crop: tuple, resize_hw: tuple):
    return T.Compose([
        T.ToTensor(),
        # T.Lambda(lambda x: F.rotate(x, 85, expand=True)),
        # T.CenterCrop(center_crop),
        T.Resize(resize_hw, antialias=True),
    ])


def calculate_resizing(original_size: tuple, window_size: int, window_overlapping: int):
    # resize the image to find and optimal size to slide with the window across the image
    effective_window_size = window_size - window_overlapping
    resize_h = original_size[0] // effective_window_size * effective_window_size + window_overlapping
    resize_w = original_size[1] // effective_window_size * effective_window_size + window_overlapping

    print('Original size: ', original_size, '; New size: ', (resize_h, resize_w))

    num_cols = resize_w // effective_window_size
    num_rows = resize_h // effective_window_size

    return resize_h, resize_w, num_cols, num_rows


def create_data_loader(root_path: str, center_crop: tuple, resize_hw: tuple, batch_size: int, shuffle: bool = False, num_workers: int = 1):
    # Create dataset
    transforms = get_plot_transforms(center_crop=center_crop, resize_hw=resize_hw)
    plot_dataset = PlotDataset(root_path, transform=transforms)

    # Create dataloader
    return DataLoader(plot_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


if __name__ == "__main__":
    seed_torch()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = "Bachelorarbeit_Emanuel_J/weights/mask_rcnn_weights.pt"
    # root_path = "/data/departments/schoen/roessle/HSWT_Aehrenzaehlen/images_full_plots/"
    root_path = "/home/jacobowsky/Bachelorarbeit_Emanuel_J/BA_python_version/cropped_images_without_artifacts"
    model = torch.load(model_path)

    batch_size = 2
    num_workers = 1
    shuffle = False

    score_threshold = 0.8
    iou_threshold = 0.2

    # (H, W)
    center_crop = (1025-325, 2600-350)

    # Window size to slide over the image (vertically and horizontally)
    window_size = 280  # window size for sliding window / 280 was the original training size of the mask rcnn model
    window_overlapping = 13  # overlapping between windows

    assert window_size > window_overlapping, "Window size must be larger than overlapping"

    # Calculate new size of image to resize to
    resize_h, resize_w, num_cols, num_rows = calculate_resizing(center_crop, window_size, window_overlapping)

    image_size = (resize_h, resize_w)


    plot_dataloader = create_data_loader(
        root_path, center_crop, image_size,
        batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )

    start = time.time()
    predictions = run_sliding_window_approach(
        model, plot_dataloader, device, window_size, window_overlapping,
        iou_threshold=iou_threshold, score_threshold=score_threshold,
        plot_single_image_heatmap=True, plot_raw_single_image_boxes=True,
        plot_filtered_image_boxes=True, plot_all_images_heatmap=True,
        cell_size_heatmap=50, write_boxes_centers=False,
        plot_segmentation_masks=True, subplot_upsacling=True
    )
    end = time.time()
    print(f"Time elapsed: {np.round(end - start, 3)} seconds")