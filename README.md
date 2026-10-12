## Remote Urban Tree Identification

A multimodal machine-learning pipeline for assigning urban trees to
remote-sensing-informed taxonomic groups. The manuscript model predicts 11
intermediate groups and reports six prespecified groups obtained by summing
the corresponding class probabilities.

<img src = https://github.com/larkinandy/remoteUrbanTreeId/blob/main/images/SanDiego.png width="1024">

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
   intermediate groups. Taxa with 50--499 records are subsequently assigned
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

<!-- Add a current pipeline or model-architecture figure here. -->

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
- `modelImplementation/` — Model application and deployment workflows, if and
  when a maintained implementation is added.

**Data and model weights**

[Training Data Shards](https://drive.google.com/file/d/1EQMfVH5jSy4wMFY_SdeiGh6jlVpExmff/view?usp=sharing)  
[Discriminator Model Weights](https://drive.google.com/file/d/18XbRq-QmOd4JK0uw4ois8fOJoFIqQH3P/view?usp=sharing)  
[Classifier Model Weights](https://drive.google.com/file/d/1i0VKAXUv2_UB2Y_uVvMKgp0pG83G9Cve/view?usp=sharing)  

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
