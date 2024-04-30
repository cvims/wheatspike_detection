
"""
Evaluation of all Plot images.
"""
# =============================================================================
# Imports
# =============================================================================
import os
import sys
import csv
import time
import math
import numpy as np

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')

import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T
import torchvision.transforms.functional as F
import torchvision.ops as ops

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

from tqdm import tqdm
import cv2

sys.path.insert(1, '/home/jacobowsky/Bachelorarbeit_Emanuel_J/src')
from utils.utils_dataset import set_seed
from full_plot_dataset import FullPlotDataset


# =============================================================================
# Helperfunctions
# =============================================================================
def recalculate_bbs_3rows(window_predictions, image_size, num_columns, step_size_vertical, window_overlapping):
    """
    This is an in-built operation on the window predictions. It is used to update the bounding boxes.
    """
    image_h, image_w = image_size
    boxes = []
    for i, window in enumerate(window_predictions):
        for j in range(window['boxes'].shape[0]):
            x1, y1, x2, y2 = window['boxes'][j]
            # update box to original image size
            x1 = x1 + (i % num_columns) * image_w - ((i % num_columns) * window_overlapping)
            y1 = y1 + (i // num_columns) * (image_h - step_size_vertical) - ((i // num_columns) * window_overlapping)
            x2 = x2 + (i % num_columns) * image_w - ((i % num_columns) * window_overlapping)
            y2 = y2 + (i // num_columns) * (image_h - step_size_vertical) - ((i // num_columns) * window_overlapping)
            # update box in window
            boxes.append(torch.stack([x1, y1, x2, y2]))
    return torch.stack(boxes)


def recalculate_bbs_2rows(window_predictions, image_size, num_columns, step_size_vertical, window_overlapping):
    """
    This is an in-built operation on the window predictions. It is used to update the bounding boxes.
    """
    image_h, image_w = image_size
    boxes = []
    for i, window in enumerate(window_predictions):
        for j in range(window['boxes'].shape[0]):
            x1, y1, x2, y2 = window['boxes'][j]
            # update box to original image size
            x1 = x1 + (i % num_columns) * image_w - ((i % num_columns) * window_overlapping)
            y1 = y1 + (i // num_columns) * image_h - ((i // num_columns) * window_overlapping)
            x2 = x2 + (i % num_columns) * image_w - ((i % num_columns) * window_overlapping)
            y2 = y2 + (i // num_columns) * image_h - ((i // num_columns) * window_overlapping)
            # update box in window
            boxes.append(torch.stack([x1, y1, x2, y2]))
    return torch.stack(boxes)


def area(box):
    """
    Calculates box area
    """
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1

    return width * height


def apply_score_threshold(boxes, scores, threshold):
    """
    Filters boxes and scores based on threshold value
    """
    keep = scores >= threshold

    remaining_boxes = boxes[keep]
    remaining_scores = scores[keep]

    removed_boxes = boxes[~keep]
    removed_scores = scores[~keep]
    
    return remaining_boxes, remaining_scores, removed_boxes, removed_scores


def apply_nms(boxes, scores, iou_threshold):
    """
    Filters boxes and scores based on IoU threshold
    """
    keep = ops.nms(boxes, scores, iou_threshold)

    remaining_boxes = boxes[keep]
    remaining_scores = scores[keep]

    removed_boxes = boxes[~keep]
    removed_scores = scores[~keep]
    
    return remaining_boxes, remaining_scores, removed_boxes, removed_scores


def calculate_subplot_upsacling(image, boxes, windows, window_predictions, idx, details=False, save_plot=False):
    """
    Calculates upscaling values for each 280x280 sub-plot
    """
    total_wheatheads = len(boxes)

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

            sub_image_boxes, sub_image_scores, _, _ = apply_nms(window_predictions[idx+j]["boxes"], window_predictions[idx+j]["scores"], iou_threshold)
            sub_image_boxes, _, _, _ = apply_score_threshold(sub_image_boxes, sub_image_scores, score_threshold)
            sub_image_boxes_list.append(sub_image_boxes)

            count_small_plot = len(sub_image_boxes)
            count_estimation = math.floor(inverse_proportion * count_small_plot)

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


def scatter_plot(sub_plot_estimation, full_estimation, save_folder_scatterplots, subplot_idx):
    """
    Creates scatterplots for all sub-plot positions
    """

    r2_list, rmse_list = [], []
    x = np.arange(len(full_estimation))
    all_subplots = True
    row_counter = 0

    row = ["A", "B", "C"]

    if all_subplots:
        for i, single_estimation in enumerate(sub_plot_estimation):
            single_full_estimate = full_estimation
            r2 = r2_score(single_full_estimate, single_estimation)
            rmse = np.sqrt(mean_squared_error(single_full_estimate, single_estimation))
            r2_index = i % 8

            # get correct row and col
            if i == 8:
                row_counter += 1
            if i == 16:
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

            save_path = os.path.join(save_folder_scatterplots, f'scatterplots_{i+1}.png')
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

    save_path = os.path.join(save_folder_scatterplots, f'scatterplots_combined.png')
    plt.savefig(save_path)
    plt.close(f)


def full_subplot_estimation(all_results, save_folder_scatterplots, subplot_idx, plot_all=False):
    """
    function to start calculating the average upscaling values for each 280x280 sub-plot
    """
    average_prediction = np.zeros(shape=(3,8))
    average_total_wheatheads = 0
    full_plot_size = 547*2149
    small_plot_size = 280*280
    inverse_proportion = full_plot_size / small_plot_size
    total_wheatheads_list = []
    subplot_count_dict = {}

    # get counts for each sub-plot and add them across all plots to get the average upscaling values
    for i, full_images in enumerate(all_results):
        tmp = np.zeros(shape=(24))
        windows = full_images[0]
        total_wheatheads = len(full_images[4])
        total_wheatheads_list.append(total_wheatheads)
        idx = 0
        for j, batch in enumerate(windows):
            for k, window in enumerate(batch):
                boxes = full_images[3][idx+k]
                count_small_plot = len(boxes)
                count_estimation = math.floor(inverse_proportion * count_small_plot)
                if idx+k not in subplot_count_dict:
                    subplot_count_dict[idx+k] = [count_estimation]
                else:
                    subplot_count_dict[idx+k].append(count_estimation)
                tmp[idx+k] += count_estimation
            idx += len(batch)
        tmp = tmp.reshape((3,8))
        average_prediction += tmp
        average_total_wheatheads += total_wheatheads
            
    average_prediction = average_prediction / len(all_results)
    average_total_wheatheads = average_total_wheatheads / len(all_results)

    difference_prediction_total = average_prediction - average_total_wheatheads

    deviation_matrix = ((average_prediction/average_total_wheatheads)-1)*100

    mean = np.mean(total_wheatheads_list)
    min_val = np.min(total_wheatheads_list)
    max_val = np.max(total_wheatheads_list)
    std_dev = np.std(total_wheatheads_list)

    print(f"Average Total Wheatheads (Sliding Window): \n {int(average_total_wheatheads)}\n")
    print(f"Average Prediction: \n {average_prediction}\n")
    print(f"Difference Prediction: \n {difference_prediction_total}\n")
    print(f"Average Difference Prediction: {np.round(np.sum(difference_prediction_total)/24,2)}\n")
    print(f"Deviation (Percent): \n {np.round(deviation_matrix, 2)}\n")
    print(f"Average Deviation: {np.round(np.sum(deviation_matrix) / 24, 2)}%\n")
    print(f"Mean:", mean)
    print(f"Min:", min_val)
    print(f"Max:", max_val)
    print(f"Standard Deviation:", std_dev)

    # get items of dict as list for subplot creation
    subplot_values = []
    for key, value in subplot_count_dict.items():
        subplot_values.append(value)

    # by index
    if plot_all:
        scatter_plot(subplot_values, total_wheatheads_list, save_folder_scatterplots, subplot_idx)

    return average_prediction, average_total_wheatheads


def map_bounding_boxes_to_grid(boxes, image_size, grid_cell_size):
    """
    Maps center of bounding boxes to heatmap grid
    """
    cells_h = math.ceil(image_size[0] / grid_cell_size)
    cells_w = math.ceil(image_size[1] / grid_cell_size)

    grid = torch.zeros((cells_h, cells_w))

    for center in boxes:
        cell_x = math.floor(center[0] / grid_cell_size)
        cell_y = math.floor(center[1] / grid_cell_size)

        # add center point to grid
        grid[cell_y, cell_x] += 1

    return grid


def calculate_bounding_boxes_centers(boxes):
    """
    Calculate center poisiton of each bounding box
    """
    centers = []
    for box in boxes:
        center_x = (box[0] + box[2]) / 2
        center_y = (box[1] + box[3]) / 2
        centers.append(torch.stack([center_x, center_y]))
    
    return torch.stack(centers)


def plot_heatmap_image(grid, save_path, add_numbers=True):
    """
    Plot heatmap for each plot image
    """
    f = plt.figure(figsize=(30,16))
    f.add_subplot(1,1,1)
    heatmap = plt.imshow(grid, cmap='YlOrRd', interpolation='nearest', clim=(0, grid.max()))

    if add_numbers:
        # Add the numbers to the heatmap
        row, column = grid.shape
        for i in range(row):
            for j in range(column):
                plt.text(j, i, f"{grid[i, j].item():.1f}", ha="center", va="center", color="black", fontsize=15)

        plt.yticks(ticks=torch.arange(row), labels=["y{}".format(i+1) for i in range(row)])
        plt.xticks(ticks=torch.arange(column), labels=["x{}".format(i+1) for i in range(column)])

    f.colorbar(heatmap, shrink=0.4)
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_bounding_boxes_to_image(image, boxes, color=(255, 0, 0), thickness=2, save_path=None):
    """
    Plot bounding boxes to an image
    """
    image_pil = F.to_pil_image(image)
    image_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

    for box in boxes:
        box = box.int()
        x1, y1, x2, y2 = box.tolist()
        image_cv = cv2.rectangle(image_cv, (x1, y1), (x2, y2), color, thickness)

    image_pil_result = Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))

    if save_path is not None:
        image_pil_result.save(save_path)


def plot_sliding_window_approach(image, window_size, window_overlapping, num_cols, num_rows, save_path=None, color=(255, 0, 0)):
    """
    Plot the sliding window approach on the image.
    """
    image_pil = F.to_pil_image(image)
    image_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

    # Draw the sliding window with overlappings on the image
    for i in range(num_rows):
        for j in range(num_cols):
            x1 = j * window_size - j * window_overlapping
            y1 = i * window_size - i * window_overlapping
            x2 = x1 + window_size
            y2 = y1 + window_size
            image_cv = cv2.rectangle(image_cv, (x1, y1), (x2, y2), color, 2)

    image_pil_result = Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))

    if save_path is not None:
        image_pil_result.save(save_path)


def write_box_count_to_file(boxes, path):
    """
    Writes number of boxes to file
    """
    with open(path, 'w') as f:
        f.write(f"Number of boxes: {len(boxes)}")


def write_box_centers_to_file(boxes, path):
    """
    Writes all box centers to file
    """
    with open(path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        header = ['x', 'y']
        csv_writer.writerow(header)
        for coord in boxes:
            csv_writer.writerow(coord.tolist())
    csvfile.close()


def write_grid_to_file(grid, path):
    """
    Writes heatmap grid to file
    """
    grid_np = grid.numpy()

    with open(path, 'w') as f:
        f.write(f"Grid shape: {grid_np.shape}\n")
        np.savetxt(f, grid_np, fmt='%.2f')


def get_plot_transforms(center_crop, resize_hw):
    """
    Transforms plot images (if needed)
        - param: center_crop :: is only needed if working with "full plots" folder images
    """
    return T.Compose([
        T.ToTensor(),
        # T.Lambda(lambda x: F.rotate(x, 85, expand=True)),
        # T.CenterCrop(center_crop),
        T.Resize(resize_hw, antialias=True),
    ])


def calculate_resizing(original_size, window_size, window_overlapping):
    """
    Resize the image to find and optimal size to slide with the window across the image
    """ 
    effective_window_size = window_size - window_overlapping
    resize_h = original_size[0] // effective_window_size * effective_window_size + window_overlapping
    resize_w = original_size[1] // effective_window_size * effective_window_size + window_overlapping

    print('Original size: ', original_size, '; New size: ', (resize_h, resize_w))

    num_cols = resize_w // effective_window_size
    num_rows = resize_h // effective_window_size

    return resize_h, resize_w, num_cols, num_rows


def create_data_loader(root_path, center_crop, resize_hw, batch_size, shuffle=False, num_workers=1):
    """
    Create data loader for the plot images
    """
    transforms = get_plot_transforms(center_crop=center_crop, resize_hw=resize_hw)
    plot_dataset = FullPlotDataset(root_path, transform=transforms)
    return DataLoader(plot_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


# =============================================================================
# Run sliding window
# =============================================================================
@torch.no_grad()
def run_sliding_window_approach(images_dir, model, data_loader, device, window_size, window_overlapping,
                                iou_threshold, score_threshold, save_folder_scatterplots,
                                plot_single_image_heatmap=False, plot_raw_single_image_boxes=False,
                                plot_filtered_image_boxes=False,
                                plot_all_images_heatmap=False, cell_size_heatmap=40,
                                write_boxes_centers=False,
                                subplot_upsacling=False):
    """
    This function runs the sliding window approach on the model and the data loader.
    The output are the bounding boxes across the original image and
    not the bounding boxes of the sub-images.
    """
    model.eval()
    model.to(device)

    # folders to save all evaluated plots
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

    # For the full batch heatmap
    all_grids = []

    # Stepsizes
    step_size_horizontal = window_size - window_overlapping
    step_size_vertical = math.floor((window_size - window_overlapping)/2)

    image_counter = 0
    all_results = []
    for i, batch in enumerate(tqdm(data_loader, desc='Running sliding window approach')):
        batch = batch.to(device)
        batch_size = batch.shape[0]

        # get current number of rows and columns
        num_columns = math.floor(batch.shape[3] / step_size_horizontal)
        num_rows = 3

        for image in batch:
            image_counter += 1
            windows = []
            for x in range(num_rows):
                for y in range(num_columns):
                    current_height = x * step_size_vertical
                    current_width = y * step_size_horizontal
                    window = image[:, current_height:current_height + window_size, current_width:current_width + window_size]
                    windows.append(window)

            # Split the windows into batches
            windows = torch.stack(windows)
            windows = windows.split(batch_size)

            # Run the model on all the batched window splits
            window_predictions = []
            for window in windows:
                predictions = model(window)
                # Move predictions to cpu before appending to list
                predictions = [{k: v.to('cpu') for k, v in t.items() if k != 'masks'} for t in predictions]
                window_predictions.extend(predictions)

            # get boxes and scores for 3 rows (upscaling method)
            boxes = torch.cat([prediction['boxes'] for prediction in window_predictions])
            scores = torch.cat([prediction['scores'] for prediction in window_predictions])

            # get boxes and scores for 2 rows (sliding window method)
            boxes_2rows = torch.cat([prediction['boxes'] for prediction in window_predictions[:8]] +
                  [prediction['boxes'] for prediction in window_predictions[16:]])
            scores_2rows = torch.cat([prediction['scores'] for prediction in window_predictions[:8]] +
                   [prediction['scores'] for prediction in window_predictions[16:]])

            window_predictions_2rows = window_predictions[:8] + window_predictions[16:]

            # in_built bounding boxes recalculation for sliding window and upscaling method
            boxes = recalculate_bbs_3rows(window_predictions, (window_size, window_size), num_columns, step_size_vertical, window_overlapping)       
            boxes_2rows = recalculate_bbs_2rows(window_predictions_2rows, (window_size, window_size), num_columns, step_size_horizontal, window_overlapping)

            if plot_raw_single_image_boxes:
                single_boxes_image_path = os.path.join(single_image_boxes_dir, f"image_boxes_{image_counter}.png")
                plot_bounding_boxes_to_image(image, boxes, save_path=single_boxes_image_path)

                single_boxes_text_path = os.path.join(single_image_boxes_dir, f"image_boxes_{image_counter}.txt")
                write_box_count_to_file(boxes, single_boxes_text_path)
            
            # Filter the boxes for complete parcel (IoU)
            boxes, scores, _, _ = apply_nms(boxes, scores, iou_threshold)
            boxes, scores, _, _ = apply_score_threshold(boxes, scores, score_threshold)

            boxes_2rows, scores_2rows, _, _ = apply_nms(boxes_2rows, scores_2rows, iou_threshold)
            boxes_2rows, scores_2rows, _, _ = apply_score_threshold(boxes_2rows, scores_2rows, score_threshold)

            if plot_filtered_image_boxes:
                filtered_boxes_image_path = os.path.join(filtered_image_boxes_dir, f"image_boxes_{image_counter}.png")
                plot_bounding_boxes_to_image(image, boxes, save_path=filtered_boxes_image_path)

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

                single_image_heatmap_text_path = os.path.join(single_image_heatmap_dir, f"image_heatmap_{image_counter}.txt")
                write_grid_to_file(grid, single_image_heatmap_text_path)

                all_grids.append(grid)

            if plot_all_images_heatmap and not plot_single_image_heatmap:
                grid = map_bounding_boxes_to_grid(boxes_centers, image.shape[1:], cell_size_heatmap)
                all_grids.append(grid)

            if subplot_upsacling:
                sub_imageboxes_list = calculate_subplot_upsacling(image, boxes, windows, window_predictions, 
                                                                idx=0, details=False, save_plot=False)
            
            all_results.append([windows, window_predictions, boxes, sub_imageboxes_list, boxes_2rows])


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

        all_images_heatmap_text_path = os.path.join(all_images_heatmap_dir, f"all_images_heatmap.txt")
        write_grid_to_file(mean_grid, all_images_heatmap_text_path)

    if subplot_upsacling:
        full_subplot_estimation(all_results, save_folder_scatterplots, subplot_idx=2, plot_all=True)

    print(f"Average Heatmap Count: {average_heatmap_counts}\n")

    return predictions



# =============================================================================
if __name__ == "__main__":
    set_seed(42)

    root = os.path.dirname(os.path.realpath(__file__))

    dataset_path = os.path.join(root, "data/dataset_plots/cropped_full_plots/without_artifacts")
    images_dir = os.path.join(root, "images")
    save_folder_scatterplots = os.path.join(root, "figures/scatterplots")
    model_path_full = os.path.join(root, "weights/model_5_epochs_2022-12-14_v2.pt")
    model = torch.load(model_path_full)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = 2
    num_workers = 1
    shuffle = False

    score_threshold = 0.8
    iou_threshold = 0.2

    center_crop = (1025-325, 2600-350) # (H, W) - only needed if working with "full plots" folder images
    window_size = 280 # Window size to slide over the image (vertically and horizontally)
    window_overlapping = 13  # overlapping between windows

    assert window_size > window_overlapping, "Window size must be larger than overlapping"

    # Calculate new size of image to resize to
    resize_h, resize_w, num_cols, num_rows = calculate_resizing(center_crop, window_size, window_overlapping)
    image_size = (resize_h, resize_w)

    # creates dataloader
    plot_dataloader = create_data_loader(
        dataset_path, center_crop, image_size,
        batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )

    start = time.time()
    # starts sliding window
    predictions = run_sliding_window_approach(
        images_dir, model, plot_dataloader, device, window_size, window_overlapping,
        iou_threshold=iou_threshold, score_threshold=score_threshold, 
        save_folder_scatterplots=save_folder_scatterplots,
        plot_single_image_heatmap=True, plot_raw_single_image_boxes=True,
        plot_filtered_image_boxes=True, plot_all_images_heatmap=True,
        cell_size_heatmap=50, write_boxes_centers=True,
        subplot_upsacling=True
    )
    end = time.time()
    print(f"Time elapsed: {np.round(end - start, 3)} seconds")