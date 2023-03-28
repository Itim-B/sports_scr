import sys
from utils import *
from predict import *

# TODO: test correction d'orientation
# TODO: combinaison EasyOCR et PyTesseract
# TODO: ajout zone d'intérêt

if __name__ == "__main__":
    # Get args
    ocr_engine = sys.argv[1]
    image_path = sys.argv[2]
    output_name = sys.argv[3]

    # Get paths
    imgs_paths = create_images_path_list(image_path)

    # Interest zone detection

    # OCR
    dic_names = get_champions_names()
    with open(output_name, "w") as outf:
        for img in imgs_paths:
            outf.write(f"----- Predictions for: {img}\n")
            predictions = infer(ocr_engine, img)
            #cleaned_predictions = get_predictions_of_interest(predictions, dic_names)
            # for p in cleaned_predictions:
            #     outf.write(f"{p}\n")
            for p in predictions:
                outf.write(f"{p}\n")
        
