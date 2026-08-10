# Remote Urban Tree Identification

A multimodal machine-learning pipeline for assigning urban trees to six
remote-sensing-informed taxonomic groups.

The project combines tree-crown-centered aerial imagery, LiDAR structure,
Sentinel-2 phenology, satellite embeddings, and daily PRISM climate variables.
It includes preprocessing, model-input shard assembly, taxon-discriminator
training, centroid-based group selection, and final k=6 classification.

> Links to model weights and the training dataset are provided in
> the project links section below

## Overview

The current workflow has three modeling stages:

1. A taxon discriminator learns an embedding space from fine scientific labels.
2. Taxon centroids are clustered and evaluated across k=5 through k=12 using
   nearest-prototype Oracle scores and centroid-separability metrics.
3. The selected k=6 partition is used to train the final multimodal classifier.

The six groups are learned collections of taxa with similar remotely sensed
signatures. They should not be interpreted as formal botanical ranks.

<!-- Add a current pipeline or model-architecture figure here. -->

## Why urban trees?

Urban trees provide shade, cooling, carbon storage, stormwater interception,
and neighborhood greening. They also influence pollen exposure in ways that
vary by taxon, location, and season.

Municipal tree inventories are often incomplete or inconsistent across cities.
This project uses remote sensing and machine learning to support large-area
tree identification and downstream research on urban vegetation, meteorology,
and allergenic pollen exposure.

## Repository structure

- [`dataCollectionPreprocessing/`](dataCollectionPreprocessing/) — Data
  collection and preprocessing from crown coordinates through model-input
  shard creation.
- [`model/`](model/) — Current discriminator and k=6 classifier training,
  centroid-partition construction, and model evaluation.
- `modelImplementation/` — Model application and deployment workflows, if and
  when a maintained implementation is added.

The preprocessing directory contains focused workflows for:

- Tree-crown detection and inventory matching
- NAIP download and crown-centered crop extraction
- LiDAR download, CHM creation, and structural metrics
- Sentinel-2 download and local phenology calculation
- Satellite embeddings
- Daily PRISM climate variables
- QA screening and model-input shard assembly

## Current model inputs

The current classifier can use:

- Crown-centered NAIP RGB-NIR imagery
- Crown-centered LiDAR canopy-height products
- NAIP–CHM structural metrics
- Sentinel-2 seasonal and phenological features
- Raw Sentinel-2 sequences
- Satellite embeddings
- Date-matched daily PRISM precipitation, temperature, and vapor-pressure
  deficit

See the folder READMEs under
[`dataCollectionPreprocessing/`](dataCollectionPreprocessing/) for collection
and preprocessing instructions.

## Model training

The active entry points are:

- [`train_clean_tree_id_centered_taxon_discriminator.py`](model/train_clean_tree_id_centered_taxon_discriminator.py)
- [`build_oracle_centroid_taxon_partitions.py`](model/build_oracle_centroid_taxon_partitions.py)
- [`train_clean_tree_id_centered_k6_classifier.py`](model/train_clean_tree_id_centered_k6_classifier.py)

See the [`model` README](model/README.md) for the training order and current
commands.

## Data and model weights

[Training Data Shards](https://www.linkedin.com/in/andrew-larkin-525ba3b5/)  
[Discriminator Model Weights](https://www.linkedin.com/in/andrew-larkin-525ba3b5/)  
[Classifier Model Weights](https://www.linkedin.com/in/andrew-larkin-525ba3b5/)  

## Author

[Andrew Larkin](https://www.linkedin.com/in/andrew-larkin-525ba3b5/)  
Oregon State University, College of Health

## Funding and related resources

- [NIH/NIEHS award 1R01HL178727-01](https://www.niehs.nih.gov/)
- [Spatial Health Lab](https://health.oregonstate.edu/research/spatial-health)
- [Google Earth Engine](https://earthengine.google.com/)
- [PyTorch](https://pytorch.org/)

## Related publications

- [Sentinel-2 and LiDAR](https://www.nature.com/articles/s41598-025-10971-6)
- [UrbanTreeDetector](https://www.sciencedirect.com/science/article/pii/S1569843224002024)

## Related repositories

- [PollenModeling](https://github.com/larkinandy/PollenModeling)
- [UrbanTreeDetector](https://github.com/jonathanventura/urban-tree-detection)
