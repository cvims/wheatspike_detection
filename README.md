# Code of the Paper: Mapping Wheat Spike Heads and Within-Plot Spike Density Variability from UAV images: A Plot-Level Analysis


## Table of Contents

- [Dataset](#dataset)
  - [Wheat Dataset](#Wheat)
  - [Full Plots Dataset](#FullPlots)
- [Model](#model)
  - [Training](#training)
  - [Evaluation](#evaluation)
- [Citation](#citation)

## Dataset
The datasets generated during and/or analysed during the current study are available from the corresponding author on reasonable request.

### Wheat Dataset

*wheat_dataset.py* defines a class file to prepare all sub-images used for training the model trained in file *mask_rcnn.py*

Annotations are available for all sub-images

### Full Plots Dataset

*full_plot_dataset.py* defines a class file to prepare all full-plot images used for evaluation in file *main.py*

Annotations are not available for all plot-images

## Model

Before training and evaluation please use the packages listed in the *requirements.txt*.

### Training

To start the training run the file *mask_rcnn.py*.

You can manually change the config inside the file to adapt to you needs (see section *__name__ == '__main__'*).

The training process will automatically save the model weights (for evaluation purposes).

### Evaluation

To evaluate your model use the file *src/main.py*.

Manually adapt the section *__name__ == '__main__'* of the file to run the evaluation with correct parameters.

The evaluation script automatically creates all results for the metrics described in the corresponding publication.


## Citation

If you use this project in your research, please consider citing it using the following BibTeX entry:

t.b.a.
<!-- ```bibtex
@article{
	RoessleFHBClassification2023,
	author = {Dominik Rößle and Lukas Prey and Ludwig Ramgraber and Anja Hanemann and Daniel Cremers and Patrick Ole Noack and Torsten Schön },
	title = {Efficient Non-Invasive FHB Estimation using RGB Images from a Novel Multi-Year, Multi-Rater Dataset},
	journal = {Plant Phenomics},
	year = {2023},
	doi = {10.34133/plantphenomics.0068},
	URL = {https://spj.science.org/doi/abs/10.34133/plantphenomics.0068}
} -->