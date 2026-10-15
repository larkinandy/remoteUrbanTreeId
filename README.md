# Multimodal Remote-Sensing Classification of Urban Trees

A multimodal machine-learning pipeline for assigning urban trees to
remote-sensing-derived taxonomic groups. This repository is the reproducibility
snapshot accompanying the submitted manuscript. The final analytical dataset
contains 1,851,968 records representing 579 taxa across 36 U.S. cities.

The stage-two model predicts 11
intermediate groups and reports six prespecified groups obtained by summing
the corresponding class probabilities.

The project combines tree-crown-centered aerial imagery, LiDAR structure,
Sentinel-2 phenology, satellite embeddings, and daily PRISM climate variables.
It includes preprocessing, model-input shard assembly, taxon-discriminator
training, validation-only group construction, fixed-partition classification,
checkpoint ensembling, and independent evaluation.

**Overview**

The current workflow has four modeling stages:

1. A multimodal taxon discriminator learns a 64-dimensional embedding space
   from core taxa with at least 500 records.
2. Discriminator validation embeddings are used to select and freeze 11
   intermediate groups. Taxa with 50–499 records are subsequently assigned
   from deterministic development-only centroids; discriminator test records
   are reserved for independent evaluation.
3. A second multimodal model is trained to predict the 11 frozen groups using
   inherited training, validation, and test partitions. The five checkpoints
   with the strongest validation scores are combined by averaging their
   11-class probabilities.
4. The ensembled probabilities are summed according to a frozen 11-to-6 map
   to produce the six groups reported in the manuscript.

Both label spaces are empirical collections of taxa that are separable using
the available remotely sensed inputs. They are not formal botanical ranks or
validated ecological functional groups.

On the independent record-level test set, the final collapsed six-group model
achieved 0.793 accuracy, 0.772 macro-F1, and 0.792 weighted-F1. This split
measures performance for new records from represented cities. Separate
leave-one-city-out and single-city-trained analyses quantify transfer to unseen
geographic domains; their frozen summaries are stored under
[`analysis/geographic/provenance/`](analysis/geographic/provenance/).

**Reproducibility**

The frozen taxon partition and 11-to-6 collapse are stored in [`config/`](config/).
Model commands and the required order of operations are documented in
[`model/README.md`](model/README.md). Create the GPU training environment with:

```powershell
conda env create --file environment.yml
conda activate treeid-gpu
```

The recorded PyTorch build requires a CUDA 12.8-compatible NVIDIA driver.
ArcPy-dependent NAIP preprocessing must be run separately in an ArcGIS Pro
Python environment.

**Repository structure**

- [`dataCollectionPreprocessing/`](dataCollectionPreprocessing/) — Data
  collection and preprocessing from crown coordinates through model-input
  shard creation.
- [`model/discriminator/`](model/discriminator/) — Taxon-discriminator
  training, validation-only group construction, rare-taxon extension, and
  independent discriminator evaluation.
- [`model/stage2/`](model/stage2/) — Fixed-partition 11-class training,
  checkpoint ensembling, and probability-sum collapse to six reported groups.
- [`analysis/`](analysis/) — Manuscript ablation, geographic-generalization,
  sensitivity, gate-weight, and figure-generation analyses.
- [`config/`](config/) — Frozen manuscript group mapping and partition artifact.

**Data and model weights**

- [Training data shards](https://drive.google.com/file/d/1w_dBZZpTgRVNfBL0-zrhhU7tJkKSsjJM/view?usp=sharing)
- [Final discriminator model weights](https://drive.google.com/file/d/1yDKvUhOKYt_-wgEWzVxq-gO6Azwr9rrj/view?usp=sharing)
- [Final stage-two model weights and ensemble checkpoints](https://drive.google.com/file/d/1HoV5XiQSYIoHN0NanS3hPHoOTnQjWzU-/view?usp=sharing)

**Funding and related resources**

- [NIH/NIEHS award 1R01HL178727-01](https://www.niehs.nih.gov/)
- [Spatial Health Lab](https://health.oregonstate.edu/research/spatial-health)
- [Google Earth Engine](https://earthengine.google.com/)
- [PyTorch](https://pytorch.org/)

**Related publications**

- [Classification Using Sentinel-2 and LiDAR](https://www.nature.com/articles/s41598-025-10971-6)
- [UrbanTreeDetector](https://www.sciencedirect.com/science/article/pii/S1569843224002024)

**Related repositories**

- [PollenModeling](https://github.com/larkinandy/PollenModeling)
- [UrbanTreeDetector](https://github.com/jonathanventura/urban-tree-detection)

**Citation**

The manuscript has been submitted to *Remote Sensing*. Complete bibliographic
information and a versioned repository release will be added after acceptance.
Until then, cite the repository URL and the commit hash used for reproduction.
