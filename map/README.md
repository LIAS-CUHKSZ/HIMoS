# Map Data and Planning Map Generation

This folder contains the simplified coral reef map data used by HIMoS to build planning environments.

## Data Source

The original data comes from the supplementary dataset of Nieuwenhuis et al. (2022), "Integrating a UAV-Derived DEM in Object-Based Image Analysis Increases Habitat Classification Accuracy on Coral Reefs." The full original dataset can be downloaded from the link provided in the paper.

The original supplementary data includes UAV orthomosaics, DEMs, snorkel transect orthomosaics, habitat classification shapefiles, and manual classification data. We keep `README_original.txt` as a short reference to the original dataset description.

## What We Keep

To keep this repository lightweight, we do not include the full raw shapefile or orthomosaic dataset. Instead, we keep:

- `MAP_Preview/`: preview images of the original RGB orthomosaic and DEM maps.
- `complete_segmentation_map/`: complete rasterized habitat classification maps saved as `.npy` files, with preview images.
- `process_segmentation_50m.py`: the script that crops a selected region and converts the segmentation map into a planning map.

The complete segmentation maps are stored at `0.1 m/pixel`. The planning maps are downsampled to `0.25 m/pixel`.

## Planning Map Format

The original habitat classes are simplified into three planning classes:

- `0`: sand / traversable background
- `1`: rock or hard substrate
- `2`: coral target

The generated `map.npy` files are used by `main_game.py` as planning environments.

## Build Your Own Planning Map

Edit the configuration section in `process_segmentation_50m.py`.

The two most important parameters are:

- `area_index`: selects which complete segmentation map to use, for example `segmentation_array_Area2.npy`.
- `MAP_CONFIGS`: defines the crop regions you want to extract. Each region uses `range_x` and `range_y` in pixels on the original `0.1 m/pixel` segmentation map.

Then run:

```bash
python map/process_segmentation_50m.py
```

The script writes results to:

```text
map/planning_maps/Area_<area_index>_map_<rank>/
```

Each output folder contains:

- `map.npy`: planning grid used by the simulator
- `map_visualize.png`: visual preview of the planning grid
- `info.txt`: crop range, resolution, class counts, and coral statistics

## Citation

```bibtex
@article{nieuwenhuis2022integrating,
  title={Integrating a UAV-derived DEM in object-based image analysis increases habitat classification accuracy on coral reefs},
  author={Nieuwenhuis, Brian O and Marchese, Fabio and Casartelli, Marco and Sabino, Andrea and van der Meij, Sancia ET and Benzoni, Francesca},
  journal={Remote Sensing},
  volume={14},
  number={19},
  pages={5017},
  year={2022},
  publisher={MDPI}
}
```
