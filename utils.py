import os

def is_image(path):
    """
    Check if path corresponds to an image
    """
    extensions = [".jpg", ".jpeg", ".png", ".bmp", ".gif"]
    _, file_extension = os.path.splitext(path)
    return file_extension.lower() in extensions

def create_images_path_list(path):

    if os.path.isfile(path):
        return [path]
    
    paths = []
    for f in os.listdir(path):
        if is_image(f):
            paths.append(f)
    
    return paths
