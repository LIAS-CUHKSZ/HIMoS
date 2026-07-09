import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import random
import os

# Use fixed seeds for reproducible map generation.
random.seed(42)
np.random.seed(42)

# Input segmentation arrays use 0.1 m/pixel. This script downsamples each crop
# to 0.25 m/pixel for planning.

# Configuration
area_index = 2

# Candidate 50 m x 50 m crop regions. Output map indices are ranked by coral count.

# map0    {"range_x": (1000, 1500), "range_y": (3200, 3700)}, # 1459

MAP_CONFIGS = [
    {"range_x": (500, 1000), "range_y": (3700, 4200)},  # 2096
    {"range_x": (600, 1100), "range_y": (4400, 4900)},  # 2095
    {"range_x": (500, 1000), "range_y": (1700, 2200)},  # 2086
    {"range_x": (1300, 1800), "range_y": (2700, 3200)}, # 2082


    {"range_x": (400, 900), "range_y": (2000, 2500)},   # 1621
    {"range_x": (1300, 1800), "range_y": (3000, 3500)}, # 1586
    {"range_x": (700, 1200), "range_y": (600, 1100)},   # 1585
    {"range_x": (700, 1200), "range_y": (3100, 3600)},  # 1585


    {"range_x": (800, 1300), "range_y": (1000, 1500)},  # 1105
    {"range_x": (600, 1100), "range_y": (2400, 2900)},  # 1103
    {"range_x": (1700, 2200), "range_y": (3800, 4300)}, # 1102
    {"range_x": (1400, 1900), "range_y": (3500, 4000)}, # 1094
]


# Region used by main_show() for preview-only visualization.
# SHOW_CONFIG = {"range_x": (2000, 2500), "range_y": (3600, 4100)}
SHOW_CONFIG = {"range_x": (250, 750), "range_y": (2500, 3000)}

# Run mode: "batch" or "show".
RUN_MODE = "batch"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_NPY_PATH = os.path.join(
    SCRIPT_DIR, "complete_segmentation_map", f"segmentation_array_Area{area_index}.npy"
)
PLANNING_ROOT = os.path.join(SCRIPT_DIR, "planning_maps")

# Downsample factor from 0.1 m/pixel to 0.25 m/pixel.
downsample_factor = 2.5

# Probability of converting source class 1 into planning class 2 (coral).
coral_spot_probability = 0.15

# Preview upscaling factor. This only affects PNG resolution, not map data.
preview_upscale_factor = 4


def downsample_array(arr, factor):
    """Downsample an array using the modal source value for each output cell."""
    new_height = int(arr.shape[0] / factor)
    new_width = int(arr.shape[1] / factor)
    downsampled = np.zeros((new_height, new_width), dtype=arr.dtype)
    
    for i in range(new_height):
        for j in range(new_width):
            # Source range for this output cell.
            y_start = int(i * factor)
            y_end = int((i + 1) * factor)
            x_start = int(j * factor)
            x_end = int((j + 1) * factor)
            
            # Clamp to array bounds.
            y_end = min(y_end, arr.shape[0])
            x_end = min(x_end, arr.shape[1])
            
            block = arr[y_start:y_end, x_start:x_end]
            if block.size > 0:
                values, counts = np.unique(block, return_counts=True)
                downsampled[i, j] = values[np.argmax(counts)]
    
    return downsampled


def build_candidate_result(original_array, candidate_id, range_x, range_y):
    """Process one candidate crop and return its planning-map data."""
    x_start, x_end = range_x
    y_start, y_end = range_y

    # Clamp crop range to array bounds.
    x_start = max(0, x_start)
    x_end = min(original_array.shape[1], x_end)
    y_start = max(0, y_start)
    y_end = min(original_array.shape[0], y_end)

    print(f"\n[Candidate {candidate_id}] Extracting region: x=[{x_start}:{x_end}], y=[{y_start}:{y_end}]")
    cropped_array = original_array[y_start:y_end, x_start:x_end].copy()
    print(f"[Candidate {candidate_id}] Cropped shape: {cropped_array.shape}")

    print(f"[Candidate {candidate_id}] Downsampling by factor {downsample_factor} (0.1m -> 0.25m)")
    cropped_array = downsample_array(cropped_array, downsample_factor)
    print(f"[Candidate {candidate_id}] Downsampled shape: {cropped_array.shape}")

    # Class mapping:
    # source classes 1, 2, 3 -> planning class 1 (rock / hard substrate)
    # source classes 4, 5, 6 -> planning class 0 (sand)
    # a subset of source class 1 -> planning class 2 (coral target)
    processed_array = np.zeros_like(cropped_array)
    for i in range(cropped_array.shape[0]):
        for j in range(cropped_array.shape[1]):
            original_class = cropped_array[i, j]
            if original_class in [1, 2, 3]:
                if original_class == 1 and random.random() < coral_spot_probability:
                    processed_array[i, j] = 2
                else:
                    processed_array[i, j] = 1
            else:
                processed_array[i, j] = 0

    sand_count = int(np.sum(processed_array == 0))
    rock_count = int(np.sum(processed_array == 1))
    coral_count = int(np.sum(processed_array == 2))
    total_pixels = int(processed_array.size)

    return {
        "candidate_id": candidate_id,
        "range_x": range_x,
        "range_y": range_y,
        "bounded_range_x": (x_start, x_end),
        "bounded_range_y": (y_start, y_end),
        "processed_array": processed_array,
        "sand_count": sand_count,
        "rock_count": rock_count,
        "coral_count": coral_count,
        "total_pixels": total_pixels,
    }


def save_ranked_map(area_idx, rank_map_idx, candidate):
    """Save one ranked planning map and its metadata."""
    processed_array = candidate["processed_array"]
    range_x = candidate["range_x"]
    range_y = candidate["range_y"]
    sand_count = candidate["sand_count"]
    rock_count = candidate["rock_count"]
    coral_count = candidate["coral_count"]
    total_pixels = candidate["total_pixels"]
    candidate_id = candidate["candidate_id"]

    area_folder = os.path.join(PLANNING_ROOT, f"Area_{area_idx}_map_{rank_map_idx}")
    os.makedirs(area_folder, exist_ok=True)
    planning_map_npy_path = os.path.join(area_folder, f"map.npy")
    planning_map_preview_path = os.path.join(area_folder, f"map_visualize.png")
    info_txt_path = os.path.join(area_folder, "info.txt")

    np.save(planning_map_npy_path, processed_array)
    print(f"[Rank {rank_map_idx}] Saved npy: {planning_map_npy_path}")

    # Save a clean image without axes, ticks, title, or legend.
    color_lut = np.array(
        [
            [240, 230, 140],  # 0 Sand  -> #F0E68C
            [47, 79, 79],     # 1 Rock  -> #2F4F4F
            [255, 192, 203],  # 2 Coral -> #FFC0CB
        ],
        dtype=np.uint8,
    )
    rgb_image = color_lut[processed_array]
    if preview_upscale_factor > 1:
        rgb_image = np.repeat(np.repeat(rgb_image, preview_upscale_factor, axis=0), preview_upscale_factor, axis=1)
    plt.imsave(planning_map_preview_path, rgb_image)
    print(f"[Rank {rank_map_idx}] Saved preview: {planning_map_preview_path}")

    with open(info_txt_path, "w", encoding="utf-8") as info_file:
        info_file.write("=== Processing Statistics ===\n")
        info_file.write("Class mapping: 0=Sand, 1=Rock, 2=Coral\n")
        info_file.write(f"Rank map index: {rank_map_idx}\n")
        info_file.write(f"Source candidate index: {candidate_id}\n")
        info_file.write(f"Total pixels: {total_pixels} (each pixel = 0.25m x 0.25m = 0.0625m^2)\n")
        info_file.write(f"Sand: {sand_count} ({sand_count / total_pixels * 100:.1f}%)\n")
        info_file.write(f"Rock: {rock_count} ({rock_count / total_pixels * 100:.1f}%)\n")
        info_file.write(
            f"Corals: {coral_count} ({coral_count / total_pixels * 100:.1f}%) - {coral_count} corals total\n"
        )
        info_file.write(
            f"Original resolution: 0.1m/pixel -> Downsampled to: 0.25m/pixel (factor {downsample_factor})\n"
        )
        info_file.write(f"Total area covered: {total_pixels * 0.0625:.1f} m^2\n")
        info_file.write(f"Range X(pixel): {range_x}\n")
        info_file.write(f"Range Y(pixel): {range_y}\n")
        info_file.write(f"Range X(m): {range_x[0] * 0.1} - {range_x[1] * 0.1}\n")
        info_file.write(f"Range Y(m): {range_y[0] * 0.1} - {range_y[1] * 0.1}\n")
    print(f"[Rank {rank_map_idx}] Saved stats: {info_txt_path}")

    return info_txt_path


def main():
    print(f"Loading segmentation array from: {INPUT_NPY_PATH}")
    original_array = np.load(INPUT_NPY_PATH)
    print(f"Original array shape: {original_array.shape}")
    print(f"Original unique classes: {np.unique(original_array)}")

    candidates = []
    for idx, cfg in enumerate(MAP_CONFIGS, start=1):
        candidate = build_candidate_result(
            original_array=original_array,
            candidate_id=idx,
            range_x=cfg["range_x"],
            range_y=cfg["range_y"],
        )
        candidates.append(candidate)

    ranked_candidates = sorted(
        candidates,
        key=lambda item: (-item["coral_count"], item["candidate_id"]),
    )

    results = []
    for rank_map_idx, candidate in enumerate(ranked_candidates, start=1):
        save_ranked_map(area_idx=area_index, rank_map_idx=rank_map_idx, candidate=candidate)
        results.append(
            {
                "rank_map_index": rank_map_idx,
                "source_candidate_index": candidate["candidate_id"],
                "coral_count": candidate["coral_count"],
            }
        )

    print("\n=== Coral Summary (Ranked Map1 -> Map10) ===")
    for item in results:
        print(
            f"Map {item['rank_map_index']}: {item['coral_count']} corals "
            f"(source candidate {item['source_candidate_index']})"
        )

    print("\n=== Source Candidate Coral Counts ===")
    for candidate in sorted(candidates, key=lambda x: x["candidate_id"]):
        print(f"Candidate {candidate['candidate_id']}: {candidate['coral_count']} corals")


def main_show():
    """Preview one crop and print coral counts without saving files."""
    print(f"Loading segmentation array from: {INPUT_NPY_PATH}")
    original_array = np.load(INPUT_NPY_PATH)

    result = build_candidate_result(
        original_array=original_array,
        candidate_id="show",
        range_x=SHOW_CONFIG["range_x"],
        range_y=SHOW_CONFIG["range_y"],
    )

    processed_array = result["processed_array"]
    coral_count = result["coral_count"]
    total_pixels = result["total_pixels"]
    print(
        f"[SHOW] range_x={SHOW_CONFIG['range_x']}, range_y={SHOW_CONFIG['range_y']} | "
        f"corals={coral_count} ({coral_count / total_pixels * 100:.1f}%)"
    )

    colors = ["#F0E68C", "#2F4F4F", "#FFC0CB"]  # 0 Sand, 1 Rock, 2 Coral
    custom_cmap = ListedColormap(colors)
    plt.figure(figsize=(10, 8))
    plt.imshow(processed_array, cmap=custom_cmap, vmin=0, vmax=2)
    # plt.title(
    #     f"Show Map (No Save)\n"
    #     f"range_x={SHOW_CONFIG['range_x']}, range_y={SHOW_CONFIG['range_y']}, corals={coral_count}"
    # )
    # plt.xlabel("X pixel")
    # plt.ylabel("Y pixel")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    if RUN_MODE == "show":
        main_show()
    else:
        main()
