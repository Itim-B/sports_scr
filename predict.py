import editdistance
import re

def get_champions_names(sport="natation"):
    dic_names = {}
    with open("champions/natation.txt", "r") as file:
        for line in file.readlines():
            last_name, first_name = line.strip().split()
            dic_names[last_name] = first_name
    return dic_names

def reformat_predictions(raw_predictions):
    return [pred[1] for pred in raw_predictions]

def get_cleaned_prediction(prediction, dic_names, min_edit_distance=2):
    """Returns a cleaned version of the raw prediction

    Args:
        prediction (str): raw prediction
        dic_names (dict): dictionary of the names of players
        min_edit_distance (int, optional): minimum allowed edit distance to keep a predicted last name. Defaults to 2.

    Returns:
        list: list of the clean elements in the raw prediction
    """

    cprediction = []

    real_first_name = '' # Will keep in memory the "real" first name

    # We use the last name as a reference for deciding to keep a prediction
    for comp in prediction.strip().split(' '):
        # If a first name was found, check if this part of the 
        # prediction is that first name
        if len(real_first_name) > 0:
            if editdistance.eval(real_first_name, comp) <= min_edit_distance:
                cprediction.append(comp)
        for n in dic_names.keys():
            if editdistance.eval(n, comp) <= min_edit_distance:
                # We add all the last names of the player
                cprediction.append(comp)
                # We keep in memory the first name corresponding to that last name
                real_first_name = dic_names[n]
    
    # Now we add the score if we find one
    found_score = re.findall(r"\d{2}[.,]\d{2}", prediction.strip())
    if len(found_score) > 0:
        if "," in found_score[0]:
            # We noramlize the scores
            cprediction.append(".".join(found_score[0].split(",")))
        else:
            cprediction.append(found_score[0])
                
    return cprediction

def get_predictions_of_interest(predictions, dic_names):
    """Returns cleaned predictions that are probably correct

    Args:
        predictions (list): raw predictions
        dic_names (dict): dictionary of all the players names

    Returns:
        list: cleaned predictions
    """
    
    preds = []
    for pred in predictions:
        #cpred = get_prediction_last_name(pred, dic_names)
        cpred = get_cleaned_prediction(pred, dic_names)
        if len(cpred) == 0:
            continue
        elif len(cpred) == 1:
            preds.extend(cpred)
        else:
            cpred = ' '.join(cpred)
            preds.append(cpred)
    return preds