import os
import sys
import warnings
from typing import List, Tuple

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

from predict import *
from preprocess import *
from utils import *

# Ignore the warning
warnings.filterwarnings("ignore")

from torchmetrics import CharErrorRate, MatchErrorRate, WordErrorRate

data_path = "./data/champions/natation.txt"
dic_names = get_champions_names(data_path)
scoreboard_path = "./data/natation/ROI/CLIP_visual_prompt/scoreboard.png"
default_visual_prompt = Image.open(scoreboard_path)
default_visual_prompt = np.array(default_visual_prompt)[:,:, 0:3]

def get_ground_truth(path):
    """Returns ground truth data

    Args:
        path (str): path to the ground truth txt file

    Returns:
        list: list of the ground truth data
    """

    with open(path, "r") as f:
        return [line.strip() for line in f.readlines()]
    
def compute_metrics(result, groundtruth):
    wer = WordErrorRate()
    cer = CharErrorRate()

    word_error_rate = 0
    character_error_rate = 0

    for yhat, y in zip(result, groundtruth):
        word_error_rate += wer(yhat, y)
        character_error_rate += cer(yhat, y)

    return word_error_rate/min(len(result), len(groundtruth)), character_error_rate/min(len(result), len(groundtruth))


data_path ="./data/champions/natation.txt"
dic_names = get_champions_names(data_path)


def process_img(img, orientation, clip):
    if clip:
        modified_img = extract_roi_clipseg_visual(img, prompt=default_visual_prompt, thresh=0.5)
        if orientation:
            modified_img = correct_orientation(modified_img)
        return modified_img
    else:
        if orientation:
            modified_img = correct_orientation(np.array(Image.open(img)))
            return modified_img
        else:
            return np.array(Image.open(img))

def evaluate(ocr_engine, orientation, clip, correct_names=False):
    total_wer, total_cer = 0, 0
    img_paths = create_images_path_list("./data/natation")
    for img in img_paths:
        processed_img = process_img(img, orientation, clip)

        predictions = infer(ocr_engine, processed_img)
        input_string = ' '.join(predictions)
        result = extract_names_scores(input_string, dic_names, correct_names=correct_names, min_edit_distance=3)

        groundtruth_path = img.split('.')
        groundtruth_path[-1] = ".txt"
        groundtruth_path = './' + ''.join(groundtruth_path)
        groundtruth = get_ground_truth(groundtruth_path)

        try:
            wer, cer = compute_metrics(result, groundtruth)
        except:
            wer, cer = 1, 1
        total_wer += wer
        total_cer += cer
    return total_wer/len(img_paths), total_cer/len(img_paths)



# # Experiments

def evaluate_all_combinations() -> Tuple[List[float], List[float]]:
    wer_list = []
    cer_list = []
    ocr_engines = ["doctr", "easyocr", "pytesseract"]
    orientations = [True, False]
    clips = [True, False]
    correct_names_options = [True, False]

    # Evaluate all combinations of the parameters
    for engine in ocr_engines:
        for orientation in orientations:
            for clip in clips:
                for correct_names in correct_names_options:
                    print(f"------ engine = {engine}, orientation={orientation}, clip={clip}, correct_names={correct_names} ----")
                    wer, cer = evaluate(engine, orientation, clip, correct_names)
                    wer_list.append(wer)
                    cer_list.append(cer)

    return wer_list, cer_list

def plot_results(wer_list: List[float], cer_list: List[float]) -> None:
    fig, ax = plt.subplots()

    ocr_engines = ["doctr", "easyocr", "pytesseract"]
    orientations = [True, False]
    clips = [True, False]
    correct_names_options = [True, False]
    
    # Create a custom colormap
    cmap = mcolors.LinearSegmentedColormap.from_list(
    'pastel_rainbow', 
    #['#fca3b1', '#f7c1ad', '#ffdea7', '#e4f4a4', '#b2dfb5', '#82d7e6', '#8cb5e5', '#c0a5e8'],
    #['#FF9AA2', '#FFB7B2', '#FFDAC1', '#E2F0CB', '#B5EAD7', '#C7CEEA', '#B5B8FF'],
    ["#3F5D7D", "#F2B447", "#E76F51", "#2A9D8F"],
    N=len(wer_list))
    
    # Evaluate all combinations of the parameters
    for i, engine in enumerate(ocr_engines):
        for j, orientation in enumerate(orientations):
            for k, clip in enumerate(clips):
                for l, correct_names in enumerate(correct_names_options):
                    idx = i * len(orientations) * len(clips) * len(correct_names_options) \
                          + j * len(clips) * len(correct_names_options) \
                          + k * len(correct_names_options) \
                          + l
                    if idx >= len(wer_list):
                        continue
                    
                    label = f"{engine}, orientation={orientation}, clip={clip}, correct_names={correct_names}"
                    color = cmap(idx)
                    ax.scatter(wer_list[idx], cer_list[idx], label=label, color=color)
                    
    ax.set_xlabel("WER")
    ax.set_ylabel("CER")
    ax.set_title("OCR engine evaluation")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xticks(np.arange(0, 1.1, 0.1))
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=7)
    plt.show()
    fig.savefig("ocr_evaluation.png", dpi=300, bbox_inches="tight")

if __name__ == "__main__": 
    wer_list, cer_list = evaluate_all_combinations()
    plot_results(wer_list, cer_list)



