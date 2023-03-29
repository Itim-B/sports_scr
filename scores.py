import sys
from utils import *
from predict import *
from preprocess import *

# TODO: test correction d'orientation
# TODO: combinaison EasyOCR et PyTesseract
# TODO: ajout zone d'intérêt

# this is the default visual prompt used to identify the ROIs.
cur_path = os.path.abspath(__file__)
dir_path = os.path.dirname(cur_path)
scoreboard_path = os.path.join(dir_path, "data/natation/ROI/CLIP_visual_prompt/scoreboard.png")
default_visual_prompt = Image.open(scoreboard_path)
default_visual_prompt = np.array(default_visual_prompt)[:,:, 0:3]

if __name__ == "__main__":
    # Get args
    ocr_engine = sys.argv[1]
    image_path = sys.argv[2]
    output_name = sys.argv[3]

    # Get paths
    imgs_paths = create_images_path_list(image_path)

    # OCR
    dic_names = get_champions_names()
    with open(output_name, "w") as outf:
        for img in imgs_paths:
            # Interest zone detection
            cropped_image = extract_roi_clipseg_visual(img, prompt=default_visual_prompt, thresh=0.5)

            outf.write(f"----- Predictions for: {img}\n")
            predictions = infer(ocr_engine, cropped_image)
            # cleaned_predictions = get_predictions_of_interest(predictions, dic_names)
            # for p in cleaned_predictions:
            #     outf.write(f"{p}\n")
            for p in predictions:
                outf.write(f"{p}\n")
        
