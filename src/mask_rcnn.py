"""
Mask R-CNN training and evaluation.
"""
# =============================================================================
# Imports
# =============================================================================
import os
import sys
import traceback
import numpy as np
import pandas as pd
from datetime import date
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

import torch

sys.path.insert(1, os.path.dirname(os.path.realpath(__file__)))
from wheat_dataset import WheatDataset
from utils.detection_utils import collate_fn
from utils.detection_engine import train_one_epoch, evaluate, evaluate_loss
from utils.utils_network import set_pandas_display_options, get_model_maskrcnn, prediction, get_model_maskrcnn_backbone
from utils.utils_dataset import get_transform_albumentation, set_seed, get_annotations


# =============================================================================
def main(root, model_path):
    """
    Training and evaluation of Mask R-CNN model.
    """
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')

    dataset = WheatDataset(root, get_transform_albumentation(train=True), plot=False)
    dataset_test = WheatDataset(root, get_transform_albumentation(train=False), plot=False)

    # split dataset in train, validation and test set
    indices = torch.randperm(len(dataset)).tolist()
    dataset_train = torch.utils.data.Subset(dataset, indices[:362])
    dataset_validation = torch.utils.data.Subset(dataset_test, indices[362:439])
    dataset_testing = torch.utils.data.Subset(dataset_test, indices[-77:])

    # define training and validation data loaders
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=4, shuffle=True, num_workers=0,
        collate_fn=collate_fn)

    data_loader_validation = torch.utils.data.DataLoader(
        dataset_validation, batch_size=2, shuffle=False, num_workers=0,
        collate_fn=collate_fn)
    
    # get the model using a helper function
    NUM_CLASSES = 2
    
    # for further testing different backbones
    # backbones = ['resnet18', 'resnet34', 'resnet50', 'resnet152', 'resnext50_32x4d']
    # model = get_model_maskrcnn_backbone(backbones[0], NUM_CLASSES, BOX_DETECTIONS_PER_IMG=200)

    model = get_model_maskrcnn(NUM_CLASSES, BOX_DETECTIONS_PER_IMG=200)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    num_epochs = 1

    logger_full, logger_sum, val_losses = [], [], []
    for epoch in range(1, num_epochs+1):
        batch_logger_full, batch_logger_sum = train_one_epoch(model, optimizer, data_loader_train, device, epoch, print_freq=10)
        logger_full.append(batch_logger_full)
        logger_sum.append(batch_logger_sum)

        lr_scheduler.step()

        coco_evaluator = evaluate(model, data_loader_validation, device=device)
        val_losses.append(evaluate_loss(model, data_loader_validation, device))

        if epoch % 5 == 0:
            torch.save(model, model_path+"/model_" + str(epoch) + "_epochs_" + str(date.today()) + ".pt")

    torch.save(model, model_path+"/model_" + str(num_epochs) + "_epochs_" + str(date.today()) + ".pt")

    print("Training done!")
    return dataset_testing, logger_full, logger_sum, val_losses


def plot_losses_over_epochs(logger_full, logger_sum, val_losses, save_path):
    """
    Plots losses over all epochs and save figure in specified save_path.
    """
    loss_classifier, loss_box_reg, loss_mask, loss_objectness, loss_rpn_box_reg =  [], [], [], [], []
    n_epochs = len(logger_full)
    save_path = save_path + "/losses_over_epochs.png"

    # calculate losses for each epoch
    for list in logger_full:
        loss_classifier_v, loss_box_reg_v, loss_mask_v, loss_objectness_v, loss_rpn_box_reg_v =  0, 0, 0, 0, 0
        for dict in list:
            for key in dict.keys():
                if key == "loss_classifier":
                    loss_classifier_v += dict[key].item()
                if key == "loss_box_reg":
                    loss_box_reg_v += dict[key].item()
                if key == "loss_mask":
                    loss_mask_v += dict[key].item()
                if key == "loss_objectness":
                    loss_objectness_v += dict[key].item()
                if key == "loss_rpn_box_reg":
                    loss_rpn_box_reg_v += dict[key].item()

        loss_classifier.append(loss_classifier_v / len(list))
        loss_box_reg.append(loss_box_reg_v / len(list))
        loss_mask.append(loss_mask_v / len(list))
        loss_objectness.append(loss_objectness_v / len(list))
        loss_rpn_box_reg.append(loss_rpn_box_reg_v / len(list))

    losses = []
    for list in logger_sum:
        losses.append((sum(list)/len(list)))

    val_loss = []
    val_loss = [value.cpu() for value in val_losses] 

    # plot losses
    fig = plt.figure(figsize=(30, 10))
    ax0 = fig.add_subplot(121, title="Losses")

    ax0.plot(range(n_epochs), losses, 'ko-', label='loss')
    ax0.plot(range(n_epochs), loss_classifier, 'bo-', label='classifier')
    ax0.plot(range(n_epochs), loss_box_reg, 'ro-', label='box_reg')
    ax0.plot(range(n_epochs), loss_mask, 'go-', label='mask')
    ax0.plot(range(n_epochs), loss_objectness, 'co-', label='objectness')
    ax0.plot(range(n_epochs), loss_rpn_box_reg, 'o-', label='rpn_box_reg')
    ax0.plot(range(n_epochs), val_loss, 'o-', label='validation')

    plt.grid(True)
    plt.legend(loc="upper right")
    plt.savefig(save_path, bbox_inches='tight')

    print("Losses saved!")


def dict_for_testresults(dataset_testing, annotations_path, score_threshold=0.5):
    """
    Returns a dictionary with the following keys:
        "Image no", "Ground-Truth", "Predicted", "Instance difference", "Percentual difference"
    """
    image_ids, count_annotations, count_predictions, instance_difference, percentual_difference = [], [], [], [], []

    for idx in range(len(dataset_testing)):
        dict = dataset_testing[idx]
        img = dict[0] * 255

        key_image = int(dict[1]["image_id"])
        key_anno = int(dict[1]["anno_idx"]) - 1

        pred = prediction(model, img)
        anno_wheat_heads, _, _ = get_annotations(annotations_path, key_anno)

        n_detections = pred["labels"][pred["scores"] > score_threshold]

        image_ids.append(int(key_image))
        count_annotations.append(int(anno_wheat_heads))
        count_predictions.append(int(len(n_detections)))
        instance_difference.append(int(len(n_detections) - anno_wheat_heads))
        percentual_difference.append(np.round((len(n_detections)/anno_wheat_heads)-1, 5))

    results = {"Image no" : image_ids,
            "Ground-Truth" : count_annotations, 
            "Predicted" : count_predictions, 
            "Instance difference" : instance_difference,
            "Percentual difference" : percentual_difference}

    return results


def create_dataframes(results):
    """
    Returns two dataframes:
        - df: dataframe with the results of the test set
        - df_stats: dataframe with the statistics of the test set
    """
    total, average, deviation = ["Total:"], ["Average:"], ["Standard deviation:"]
    for key in results:
        if key != "Image no":
            total.append(sum(results[key]))
            average.append(total[-1]/(len(results["Image no"])))
            deviation.append(np.std(results[key]))

    # create dataframe for image values
    df = pd.DataFrame(data=results)
    df = df.sort_values("Image no")
    df["Image no"] = df["Image no"].astype("int")
    df["Ground-Truth"] = df["Ground-Truth"].astype("int")
    df["Predicted"] = df["Predicted"].astype("int")
    df["Instance difference"] = df["Instance difference"].astype("int")

    # create dataframe for stats
    stats = {"Category" : None,
                "Ground-Truth" : None, 
                "Predicted" : None, 
                "Instance difference" : None,
                "Percentual difference" : None}
                
    df_stats = pd.DataFrame(data=stats, index=[0])
    df_stats.loc[len(df_stats)] = total
    df_stats.loc[len(df_stats)] = average
    df_stats.loc[len(df_stats)] = deviation
    df_stats = df_stats.drop(index=0)

    return df, df_stats


def create_scatterplot(ground_truth, predicted, save_path):
    """
    Creates a scatterplot of the ground truth vs. the predicted values.
    """
    save_path = save_path + "/scatterplot_testresults.png"
    r2 = r2_score(ground_truth, predicted)
    rmse = np.sqrt(mean_squared_error(ground_truth, predicted))
    print(f"R²: {r2:.2f}")
    print(f"RMSE: {rmse:.2f}")

    x = np.arange(len(ground_truth))

    f = plt.figure(figsize=(8,8))

    # Scatter plot of actual vs predicted values
    plt.scatter(ground_truth, predicted, color='red', label=f'Data Points: {len(ground_truth)}', s=15)

    # Plot the diagonal line representing perfect fit
    plt.plot([min(ground_truth), max(ground_truth)], [min(ground_truth), max(ground_truth)], linestyle='--', linewidth=2, color='grey', label=f'R²: {r2:.2f}')

    plt.xlabel('Annotated Wheat Heads')
    plt.ylabel('Predicted Wheat Heads')
    plt.title(f'Results on Test set - R²-Score: {r2:.2f} - RMSE: {rmse:.2f}')
    plt.grid(True)
    plt.legend()
    plt.savefig(save_path, bbox_inches='tight')

    print("Scatterplot saved!")



# =============================================================================
if __name__ == "__main__":
    set_seed()
    # panda display options for jupyter notebooks
    set_pandas_display_options()

    root = os.path.dirname(os.path.realpath(__file__))

    dataset_path = os.path.join(root, "data/dataset_aehren")
    annotations_path = os.path.join(root, "data/dataset_aehren/annotations.json")
    save_path = os.path.join(root, "figures")
    csv_path = os.path.join(root, "csv-data")

    model_path = os.path.join(root, "weights")
    model_path_full = os.path.join(model_path, "model_5_epochs_2022-12-14_v2.pt")
    model = torch.load(model_path_full)

    score_threshold = 0.5

    # starts training and evaluation
    try:
        dataset_testing, logger_full, logger_sum, val_losses = main(dataset_path, model_path)
    except:
        traceback.print_exc()

    # plots/saves training losses
    plot_losses_over_epochs(logger_full, logger_sum, val_losses, save_path)

    # prepare dict of test results
    results = dict_for_testresults(dataset_testing, annotations_path, score_threshold)

    # create dataframe through dict of test results
    df, df_stats = create_dataframes(results)

    # g-t and predictions for scatterplot
    ground_truth = df["Ground-Truth"]
    predicted = df["Predicted"]

    # save dataframes as csv file
    df_stats.to_csv(f'{csv_path}/data_stats_{str(date.today())}.csv', index=False)
    df.to_csv(f'{csv_path}/data_{str(date.today())}.csv', index=False)

    # creates scatterplot of test results
    create_scatterplot(ground_truth, predicted, save_path)