import sys
import os
from utils import *
from predict import *
from preprocess import *
import warnings

# Ignore the warning
warnings.filterwarnings("ignore")

# TODO: test correction d'orientation
# TODO: combinaison EasyOCR et PyTesseract
# TODO: ajout zone d'intérêt

# this is the default visual prompt used to identify the ROIs.
cur_path = os.path.abspath(__file__)
dir_path = os.path.dirname(cur_path)
scoreboard_path = os.path.join(dir_path, "data/natation/ROI/CLIP_visual_prompt/scoreboard.png")
tennis_scoreboard_path = os.path.join(dir_path, "data/tennis-table/ROI/CLIP_visual_prompt/tennis_score_board.png")

if __name__ == "__main__":
    # Get args
    sport = sys.argv[1]
    ocr_engine = sys.argv[2]
    image_path = sys.argv[3]

    data_path = os.path.join(dir_path, f"data/champions/{sport}.txt")

    if sport == "tennis_table":
        default_visual_prompt = Image.open(tennis_scoreboard_path)
        default_visual_prompt = np.array(default_visual_prompt)[:,:, 0:3]
    elif sport == "natation":
        default_visual_prompt = Image.open(scoreboard_path)
        default_visual_prompt = np.array(default_visual_prompt)[:,:, 0:3]

    # Get paths
    imgs_paths = create_images_path_list(image_path)

    # OCR
    dic_names = get_champions_names(data_path)

    for img in imgs_paths:
        # Interest zone detection
        if sport == "natation":
            cropped_image = extract_roi_clipseg_visual(img, prompt=default_visual_prompt, thresh=0.5)
            oriented_image = correct_orientation(cropped_image)
        else:
            oriented_image = img
        predictions = infer(ocr_engine, oriented_image)
        input_string = ' '.join(predictions)
        result = extract_names_scores(input_string, dic_names, min_edit_distance=3)
        if not os.path.exists("result"):
            os.makedirs('result')
        with open(f"result/{img.split('/')[-1].split('.')[0]}.csv", "w") as outf:
            for r in result:
                row = r.split()
                score = row[-1]
                firstname = row[-2]
                lastname = ' '.join(row[:-2])
                outf.write(f"{lastname}\t{firstname}\t{score}\n")

