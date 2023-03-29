# Here we put all the functions used to preprocess, clean and format the input images in order
# to prepare them for use in the prediction satge.

import warnings

import os
import cv2
import mmcv
import numpy as np
import torch
# SegFormer imports
from mmseg.apis import inference_segmentor, init_segmentor
# CLIPSeg imports
from PIL import Image
from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor

# Ignore the warning
warnings.filterwarnings("ignore")


def extract_roi_segformer(model, path, crop_percent=0.05):

    """

    Get the Region of Interest (ROI) from a given image using a segmentation model.

    Args:
    - model: the segmentation model to use for obtaining the segmentation result.
    - path: the path to the input image file.
    - crop_percent: the percentage of the image to crop around the bounding box of the ROI.

    Returns:
    - cropped_image: the cropped ROI image.

    """

    # load the image and obtain the segmentation result
    image = cv2.imread(path)
    result = inference_segmentor(model, path)

    # obtain the segmentation mask and set pixels corresponding to regions of interest
    seg = result[0]
    binary_image = np.zeros((seg.shape[0], seg.shape[1]), dtype=np.uint8)

    classes_roi = {43}  # signboard
    for label in classes_roi:
        binary_image[seg == label] = 1

    # find connected components and obtain the biggest component (excluding background)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_image)

    max_label = 1
    max_size = stats[1, cv2.CC_STAT_AREA]

    for i in range(2, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > max_size:
            max_label = i
            max_size = stats[i, cv2.CC_STAT_AREA]

    # create mask for biggest component and obtain its bounding box
    mask = (labels == max_label).astype(np.uint8)
    x, y, w, h = cv2.boundingRect(mask)

    # crop the image to the bounding box and calculate the size reduction
    crop_width = int(image.shape[1] * crop_percent)
    crop_height = int(image.shape[0] * crop_percent)

    x += crop_width
    y += crop_height
    w -= 2 * crop_width
    h -= 2 * crop_height

    cropped_image = image[y:y+h, x:x+w]

    # return the ROI image
    return cropped_image

# instantiate the processor and the segmentation model for text embedding in the textual prompt task.
processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")

# this is the default prompt used to describe what the model needs to look for in the image using words.
default_prompt = ["""
            A swimming scoreboard is a rectangular panel that shows important information about a swimming competition.
            It uses bright LED lights to provide real-time updates on the progress of the race. The scoreboard is divided
            into several sections, each displaying different information about the race, including the name of the event,
            the swimmers' names, lane assignments, countries, times, and finishing order.
            """]


def extract_roi_clipseg_text(image, prompt=default_prompt, thresh=0.5):

    """

    Extracts a region of interest (ROI) from an input image based on the presence of text in the image.

    Args:
    image (numpy.ndarray): The input image as a numpy array.
    prompt (str): The prompt for text detection in the image. Default is default_prompt.
    thresh (float): The threshold value for text detection. Default is 0.5.

    Returns:
    numpy.ndarray: The ROI image as a numpy array.

    Raises:
    None.

    """

    inputs = processor(text=prompt, images=[image], padding="max_length", return_tensors="pt")

    # predict
    with torch.no_grad():
        outputs = model(**inputs)
    preds = outputs.logits.unsqueeze(0)

    # crop the image using the segmented image obtained from the model
    cropped_image = crop_image(image, preds, thresh=thresh)

    # return the ROI image
    return cropped_image

def extract_roi_clipseg_visual(image, prompt, thresh=0.5):
    
    """

    Extracts a region of interest (ROI) from an input image based on the presence of a visual prompt in the image.

    Args:
    image (numpy.ndarray): The input image as a numpy array.
    prompt (numpy.ndarray): The visual prompt for detection in the image. Default is default_visual_prompt.
    thresh (float): The threshold value for visual detection. Default is 0.5.

    Returns:
    numpy.ndarray: The ROI image as a numpy array.

    Raises:
    None.

    """

    image = Image.open(image)

    encoded_image = processor(images=[image], return_tensors="pt")
    encoded_prompt = processor(images=[prompt], return_tensors="pt")
    # predict
    with torch.no_grad():
        outputs = model(**encoded_image, conditional_pixel_values=encoded_prompt.pixel_values)
    preds = outputs.logits.unsqueeze(1)
    preds = torch.transpose(preds, 0, 1)

    # crop image
    cropped_image = crop_image(image, preds[0], thresh=0.5)

    # return the ROI image
    return cropped_image


def crop_image(original_image, preds, thresh=0.5):
    
    """

    Crops an image based on the presence of a mask.

    Args:
    original_image (numpy.ndarray): The input image as a numpy array.
    preds (torch.Tensor): The predicted output tensor obtained from the model.
    thresh (float): The threshold value for text detection. Default is 0.5.

    Returns:
    numpy.ndarray: The cropped image as a numpy array.

    Raises:
    None.
    
    """
  
    mask = np.array(torch.sigmoid(preds) > thresh, dtype=np.uint8)
    height, width, _ = np.array(original_image).shape

    # Resize the image
    mask = cv2.resize(mask, (width, height))

    # Find the bounding box coordinates
    x, y, w, h = cv2.boundingRect(mask)

    # Crop original image
    original_image = np.array(original_image)
    cropped_image = original_image[y:y+h, x:x+w, :]

    # return the cropped image
    return cropped_image