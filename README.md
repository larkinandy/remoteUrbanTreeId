# remoteUrbanTreeId
Identify trees in urban areas and classify tree locations as broadleaf, evergreen or mixed

**Author:** [Andrew Larkin](https://www.linkedin.com/in/andrew-larkin-525ba3b5/) <br>
**Affiliation:** [Oregon State University, College of Health](https://health.oregonstate.edu/) <br>

**Summary** <br>
This GitHub repo contains python scripts for collecting training datasets, training and evaluating a machine learning model for predicting urban tree genus from remote sensing products 


**Why Urban Trees** <br>
Urban trees provide important environmental benefits, including shade, cooling, carbon storage, stormwater interception, and neighborhood greening, but they also shape local pollen exposure in ways that vary strongly by species, location, and season. Many cities maintain incomplete or inconsistent tree inventories, which limits the ability to estimate species-specific pollen sources across large urban areas. This project uses remote sensing products and machine learning to improve urban tree identification, supporting downstream work that links tree composition, meteorology, and airborne pollen measurements to better understand spatial patterns of allergenic exposure.

**Repository Structure** <br>
The repository is made up of 3 folders, corresponding to various stages of the project
- **DataCollection** - sources, datasets, and methods for collecdting tree and environmental dat4
- **ClassifyModel** - model training and evaluation
- **ModelImplementation** - implemennting the mmodel (extent TBD).

<img src="https://github.com/larkinandy/remoteUrbanTreeId/blob/main/figures/multimodal_tree_taxon_model_architecture.png" alt="Descriptive alt text" width="800">

**External Links**
- **[Funding: NIH/NIEHS, 1R01HL178727-01](https://www.niehs.nih.gov/)**
- **[Spatial Health Lab](https://health.oregonstate.edu/research/spatial-health)**

- **[Google Earth Engine](https://earthengine.google.com/)**

- **[PyTorch](https://pytorch.org/get-started/locally/)**
  
**Related Publications**
- **[Sentinel-2 and LiDAR](https://www.nature.com/articles/s41598-025-10971-6)**
- **[UrbanTreeDetector](https://www.sciencedirect.com/science/article/pii/S1569843224002024)**

**Related Repositories**
- **[PollenModeling](https://github.com/larkinandy/PollenModeling)**
- **[UrbanTreeDetector](https://github.com/jonathanventura/urban-tree-detection)**

