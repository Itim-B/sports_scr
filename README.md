# Sports SCR

The Sports SCR (Sports Scene Character Recognition) project is an application that uses computer vision and machine learning techniques to detect and recognize players and their scores in different sports, such as table tennis and swimming. The aim of this project is to provide a tool that can automate the scorekeeping process and provide accurate and reliable results.

# Installation

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
- Indicate the path of an image (video later) file of the sports match, as follows:

    ```console
    $ python3 scores.py [sport] [ocr engine] path/to/image.png
    ```

for example : 

```console
$ python3 scores.py natation doctr data/natation/000006.png
```

will run prediction for 'natation' on image 000006.png using doct ocr engine. The result will be available in *result/000006.csv*

- Wait for the application to process the file and generate the results in CSV file in */result*.

The application will generate a video or image file with the recognized players and their scores highlighted. It will also output a JSON file with the detected player names and scores.

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