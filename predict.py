import re
import warnings
import os
import easyocr
import editdistance
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
import pytesseract
from PIL import Image

# Ignore the warning
warnings.filterwarnings("ignore")
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'
reader = easyocr.Reader(['en'])
predictor = ocr_predictor(pretrained=True, export_as_straight_boxes=True)

def get_champions_names(data_path):
    dic_names = {}
    with open(data_path, "r") as file:
        for line in file.readlines():
            split_name = line.strip().split()
            last_name, first_name = ' '.join(split_name[:-1]), split_name[-1]
            dic_names[last_name] = first_name
    return dic_names

def reformat_easyocr(raw_predictions):
    return [pred[1] for pred in raw_predictions]

def reformat_doctr(raw_predictions):
    predictions = ''
    for block in raw_predictions:
        for line in block['lines']:
            for word in line['words']:
                predictions += f"{word['value']} "
            predictions += '\n'
    return predictions.split('\n')

def infer(ocr_engine, img):
    if ocr_engine == "pytesseract":
        return pytesseract.image_to_string(img).split('\n')
    elif ocr_engine == "easyocr":
        return reformat_easyocr(reader.readtext(img))
    else:
        result = predictor([img])
        return reformat_doctr(result.export()['pages'][0]["blocks"])

# def get_cleaned_prediction(prediction, dic_names, min_edit_distance=2):
#     """Returns a cleaned version of the raw prediction

#     Args:
#         prediction (str): raw prediction
#         dic_names (dict): dictionary of the names of players
#         min_edit_distance (int, optional): minimum allowed edit distance to keep a predicted last name. Defaults to 2.

#     Returns:
#         list: list of the clean elements in the raw prediction
#     """

#     cprediction = []

#     real_first_name = '' # Will keep in memory the "real" first name
#     added_first_name = False # Keep a tab if the first name was added or not

#     # We use the last name as a reference for deciding to keep a prediction
#     for comp in prediction.strip().split(' '):
#         added = False # A component of the prediction has to be corrected and added only one time
#         # If a first name was found, check if this part of the 
#         # prediction is that first name
#         if len(real_first_name) > 0:
#             if editdistance.eval(real_first_name, comp) <= min_edit_distance:
#                 cprediction.append(comp)
#         for n in dic_names.keys():
#             if not added:
#                 if editdistance.eval(n, comp) <= min_edit_distance:
#                     # We add all the last names of the player
#                     cprediction.append(comp)
#                     # We keep in memory the first name corresponding to that last name
#                     real_first_name = dic_names[n]
#                     added = True
    
#     # Now we add the score if we find one
#     found_score = re.findall(r"\d{2}[.,]\d{2}", prediction.strip())
#     if len(found_score) > 0:
#         if "," in found_score[0]:
#             # We noramlize the scores
#             cprediction.append(".".join(found_score[0].split(",")))
#         else:
#             cprediction.append(found_score[0])
                
#     return cprediction

# def get_predictions_of_interest(predictions, dic_names):
#     """Returns cleaned predictions that are probably correct

#     Args:
#         predictions (list): raw predictions
#         dic_names (dict): dictionary of all the players names

#     Returns:
#         list: cleaned predictions
#     """
    
#     preds = []
#     for pred in predictions:
#         #cpred = get_prediction_last_name(pred, dic_names)
#         cpred = get_cleaned_prediction(pred, dic_names)
#         if len(cpred) == 0:
#             continue
#         elif len(cpred) == 1:
#             preds.extend(cpred)
#         else:
#             cpred = ' '.join(cpred)
#             preds.append(cpred)
#     return preds

def separate_cursed_names(string):

    # Define the regular expression pattern
    pattern = r"([A-Z]+)([A-Z]{1}[a-z]+)"

    # Remove non-alphanumeric charcaters
    string = re.sub(r'[^\w\s\-.,:]', '', string)

    # Find all words that match the pattern
    matches = re.findall(pattern, string)
    
    # Replace each match with the desired format
    for match in matches:
        old_word = ''.join(match)
        new_word = ' '.join(match)
        string = string.replace(old_word, new_word)
    return string

def extract_names_scores(input_string, dic_names, correct_names=False, min_edit_distance=3):
    # Separate cursed names
    input_string = separate_cursed_names(input_string)
    # Define the regular expression pattern
    name_pattern = r"[A-Z]+ [A-Z]{1}[a-z]{1}[\-aA-zZ]*"
    time_pattern = r"(?:\d+:)?\d+[.,]\d{2}"

    # Use the re.findall() function to find all matches in the input string
    name_matches = re.findall(name_pattern, input_string, re.MULTILINE)

    # Extract the names from the matches
    last_first_name_map = {}
    for match in name_matches:
        # Split the match into its constituent parts
        parts = match.split()
        # Extract the name
        name = " ".join(parts[:-1])

        last_first_name_map[name] = parts[-1]

    # Use the re.findall() function to find all matches in the input string
    time_matches = re.findall(time_pattern, input_string, re.MULTILINE)
    
    # Extract the names from the matches
    times = []
    for match in time_matches:
        # Split the match into its constituent parts
        parts = match.split()
        # Extract the name
        time = parts[0]
        times.append(time)
    
    # Filter names
    good_names = []
    good_first_names = []
    for name in last_first_name_map.keys():
        for dic_name in dic_names.keys():
            if editdistance.eval(name, dic_name) <= min_edit_distance:
                if correct_names:
                    good_names.append(dic_name)
                    good_first_names.append(dic_names[dic_name])

                else:
                    good_names.append(name)
                    good_first_names.append(last_first_name_map[name])
                break
            else:
                pass

    try :
        times = times[::-1]
        good_times = times[:len(good_names)]
        good_times = good_times[::-1]

    except Exception as e:
        print(e)
        good_times = times[0:len(good_names)]
    
    good_first_names = [v for k, v in last_first_name_map.items() if k in good_names]
    # final result : Last Name, First Name, Time
    result = [f"{last} {first} {time}" for last, first, time in zip(good_names, good_first_names, good_times)]
    return result