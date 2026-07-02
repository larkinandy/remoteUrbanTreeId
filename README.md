# remoteUrbanTreeId
Identify trees in urban areas and classify tree locations as broadleaf, evergreen or mixed

**Author:** [Andrew Larkin](https://www.linkedin.com/in/andrew-larkin-525ba3b5/) <br>
**Affiliation:** [Oregon State University, College of Health](https://health.oregonstate.edu/) <br>

**Summary** <br>
This GitHub repo contains python scripts for collecting training datasets, training and evaluating a machine learning model for predicting urban tree locations and broad classifications (broadleaf, evergreen, or mixed). 

**Why Urban Trees** <br>
For the pollen study (see GitHub in related repositories links below). More details coming later...

**Repository Structure** <br>
The repository is made up of 4 folders, corresponding to various parts of the project
- **DataCollection** - sources, datasets, and methods for collecdting tree and environmental dat4
- **ModelTrainingEval** - model training and evaluation
- **ModelImplementation** - implemennting the mmodel (extent TBD).

**External Links**
- **[Funding: NIH/NIEHS, 1R01HL178727-01](https://www.niehs.nih.gov/)**
- **[Spatial Health Lab](https://health.oregonstate.edu/research/spatial-health)**
- **[Sentinel-2](https://dataspace.copernicus.eu/data-collections/copernicus-sentinel-missions/sentinel-2)**
- **[Google Earth Engine](https://earthengine.google.com/)**
- **[PyTorch](https://pytorch.org/get-started/locally/)**
  
**Related Publications**
- **[Sentinel-2 and LiDAR](https://www.nature.com/articles/s41598-025-10971-6)**
- **[UrbanTreeDetector](https://www.sciencedirect.com/science/article/pii/S1569843224002024)**

**Related Repositories**
- **[PollenModeling](https://github.com/larkinandy/PollenModeling)**
- **[UrbanTreeDetector](https://github.com/jonathanventura/urban-tree-detection)**

## Multimodal taxon model

`classifyModel/scripts/train_multimodal_taxa.py` implements the multimodal
NAIP + Sentinel-2/ERA5 + Google Satellite Embedding model. It predicts seven
named taxa (`Quercus`, `Acer`, `Betula`, `Ulmus`, `Fraxinus`, `Populus`, and
`Pinaceae`) behind a separately calibrated NamedTaxa-vs-Other gate.

Activate the environment in `environment.yml`, then run an Albuquerque-only
workflow smoke test:

```powershell
python -m classifyModel.scripts.train_multimodal_taxa --dry-run --dry-run-city Albuquerque
```

While Sentinel-2 and ERA5 are still being collected, train the same architecture
with only NAIP chips and Google Satellite Embeddings:

```powershell
python -m classifyModel.scripts.train_multimodal_taxa `
  --workflow naip-embedding-pretrain `
  --dry-run `
  --dry-run-city Albuquerque `
  --rebuild-cache
```

Then launch the full NAIP + embedding pretraining run:

```powershell
python -m classifyModel.scripts.train_multimodal_taxa `
  --workflow naip-embedding-pretrain `
  --rebuild-cache `
  --epochs 40 `
  --batch-size 256 `
  --num-workers 8 `
  --prefetch-factor 3
```

The dry run uses only `Albuquerque_Final_2022-06-18.csv`, caps the inventory at
20,000 trees, trains for two epochs, and writes separate dry-run artifacts. A
full run is:

```powershell
python -m classifyModel.scripts.train_multimodal_taxa
```

Pretraining artifacts are kept separate from the later full four-modality run,
for example `E:/TreeID/ModelInputs/naip_embedding_pretrain` and
`E:/TreeID/ModelOutputs/naip_embedding_pretrain_best_model.pt`.

The optimized two-drive layout is:

- McCoy inventory: `C:/Users/larki/Desktop/PollenSense/training/McCoy`
- Sentinel-2 tables: `E:/TreeID/Sentinel2`
- ERA5 tables: `E:/TreeID/ERA5`
- Satellite Embedding tables: `E:/TreeID/SatelliteEmbedding`
- NAIP chips and indexes: `E:/TreeID/NAIP_Chips`
- Sequential training shards: `E:/TreeID/ModelInputs`
- Compact indexes and normalization metadata: `E:/TreeID/Indexes`
- Checkpoints, predictions, and reports: `E:/TreeID/ModelOutputs`

The preprocessing pass handles one source city at a time and writes bounded
25,000-tree train, validation, and test shards. Training workers load and
shuffle one shard at a time rather than loading the complete dataset into RAM.
For efficient preprocessing, Sentinel-2, ERA5, and Satellite Embedding files
should include their city token in the filename or reside in a city-named
subdirectory. Legacy global files can be enabled with
`--allow-global-modality-scan`, at the cost of repeated scans.

Sentinel-2 exports may use the reduced-cell `row_index` scheme from
`dataCollection/downloadSentinel.py`; the training script maps those rows back
to all trees in the same 10 m Sentinel cell through
`dataCollection/mccoy_sentinel_10m_cells_utm/<city>/tree_to_sentinel10m_cell.csv`.
Albuquerque also falls back to the legacy
`dataCollection/albuquerque_sentinel_10m_cells/tree_to_sentinel10m_cell.csv`
crosswalk.

BF16 mixed precision, pinned memory, prefetching, fused AdamW, and
channels-last NAIP tensors are enabled automatically on CUDA. Start the RTX
5090 with the default `--batch-size 256`; try `--batch-size 512` on the RTX PRO
6000. Add `--compile` only after the ordinary dry run succeeds.

Every tabular modality must identify a tree with `tree_uid`, `uniqueID`, or the
pair `mccoy_file` + `mccoy_row`. The last form is preferred for McCoy exports
because it corresponds directly to the source inventory row. NAIP caches are
referenced rather than copied into the split files, avoiding a duplicate copy
of all image pixels.

