# Sports SCR

The Sports SCR (Sports Scene Character Recognition) project is an application that uses computer vision and machine learning techniques to detect and recognize players and their scores in different sports, such as table tennis and swimming. The aim of this project is to provide a tool that can automate the scorekeeping process and provide accurate and reliable results.

# Installation

Before using the application, we need to install some things first.

## Create a virtual environement:

It is recommended to use a virtual env (using venv or conda or any other package manager) to avoid conflicts and dependecy issues.

**The Python version used is : Python 3.8.16**

## Install Tesseract:

Refer to the documentation of Tesseract : https://tesseract-ocr.github.io/tessdoc/Installation.html.

Once installed, add the tesseract's binary path to your system like this :

In this case it's in */usr/bin/tesseract*

```
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'
```

To install the Sports SCR application, follow these steps:

- Clone the repository:

    ```console
    $ git clone https://github.com/Itim-B/sports_scr.git
    ```

- Install the required dependencies: 

    ```console
    $ pip install -r requirements.txt
    ```
- Download the necessary models and data files (done automatically once)

# Usage

To use the Sports SCR application, follow these steps:

- Open a terminal
- Select the sport you want to detect and recognize players and scores for : 'tennis' or 'natation' .
- Select an OCR engine to use from 3 choices : *easyocr*, *doctr* or *pytesseract*. *doctr* is the default
- Indicate the path of an image (video later) file of the sports match
- (optional) Indicate if you want some processing functions to be DISABLED. <u>All enabled by default</u>. Key letters : 
    - r for ROI detection
    - o for orientation correction (recommanded but a little time consuming)
    - n for names corrections using champions folder file
As follows:

    ```console
    $ python3 scores.py [sport] [ocr engine] [path/to/image] [processing_exceptions](optional)
    ```

for example : 

```console
$ python3 scores.py natation easyocr data/natation/000006.png ron
```

will run prediction for 'natation' on image 000006.png using doctr ocr engine, will not correct the extracted names from a dictionnary, the orientation, and will not detect a roi before infering. The result will be available in *result/000006.csv*

- Wait for the application to process the file and generate the results in CSV file in */result*.

The application will output a CSV file with the detected player names and scores.

# Supported Sports

The Sports SCR application currently supports the following sports:

- Swimming
- Table Tennis

# Contribution

If you would like to contribute to the Sports SCR project, you can do so by:

- Reporting issues or bugs
- Suggesting new features or improvements
- Submitting code changes through pull requests

# License
The Sports SCR project is licensed under the MIT License. 
See the LICENSE file for more information.