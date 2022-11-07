


from typing import BinaryIO, List, Optional, Tuple, Union
from pathlib import Path

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
 