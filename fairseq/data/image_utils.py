


from typing import BinaryIO, List, Optional, Tuple, Union
from pathlib import Path
from PIL import Image
import cv2
IMAGE_FILE_EXTENSIONS = {".jpg", ".png"}

def parse_path(path: str) -> Tuple[str, List[int]]:
    """Parse data path which is a path to a .jpg/.png file

      Args:
          path (str): the data path to parse

      Returns:
          file_path (str): the file path
    """

    if Path(path).suffix in IMAGE_FILE_EXTENSIONS:
        return path
    else:
        raise Exception("Unknown image suffix")
 

def compute_ratio_and_resize(img,width,height,model_height):
    '''
    Calculate ratio and resize correctly for both horizontal text
    and vertical case
    '''
    ratio = width/height
    # if ratio<1.0:
    #     ratio = calculate_ratio(width,height)
    #     img = cv2.resize(img,(model_height,int(model_height*ratio)), interpolation=Image.ANTIALIAS)
    # else:
    img = cv2.resize(img,(int(model_height*ratio),model_height),interpolation=Image.ANTIALIAS)
    return img,ratio