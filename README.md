## Remote Urban Tree Identification

A multimodal machine-learning pipeline for assigning urban trees to six
remote-sensing-informed taxonomic groups.

The project combines tree-crown-centered aerial imagery, LiDAR structure,
Sentinel-2 phenology, satellite embeddings, and daily PRISM climate variables.
It includes preprocessing, model-input shard assembly, taxon-discriminator
training, centroid-based group selection, and final k=6 classification.

**Overview**

The current workflow has three modeling stages:

1. A taxon discriminator learns an embedding space from fine scientific labels.
2. Taxon centroids are clustered and evaluated across k=5 through k=12 using
   nearest-prototype Oracle scores and centroid-separability metrics.
3. The selected k=6 partition is used to train the final multimodal classifier.

The six groups are learned collections of taxa with similar remotely sensed
signatures. They should not be interpreted as formal botanical ranks.

<!-- Add a current pipeline or model-architecture figure here. -->

**Repository structure**

- [`dataCollectionPreprocessing/`](dataCollectionPreprocessing/) — Data
  collection and preprocessing from crown coordinates through model-input
  shard creation.
- [`model/`](model/) — Current discriminator and k=6 classifier training,
  centroid-partition construction, and model evaluation.
- `modelImplementation/` — Model application and deployment workflows, if and
  when a maintained implementation is added.

**Data and model weights**

[Training Data Shards](https://www.linkedin.com/in/andrew-larkin-525ba3b5/)  
[Discriminator Model Weights](https://www.linkedin.com/in/andrew-larkin-525ba3b5/)  
[Classifier Model Weights](https://www.linkedin.com/in/andrew-larkin-525ba3b5/)  

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
