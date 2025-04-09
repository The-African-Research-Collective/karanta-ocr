import os
from pathlib import Path
import tempfile
import pandas as pd
from pdf2image import convert_from_path

def split_pdf2image_and_add_to_dataframe(data_path: Path, df:pd.DataFrame, for_segmentation: bool, train_or_test: str = "train"):
    with tempfile.TemporaryDirectory() as path:
        pages = convert_from_path(data_path, output_folder=path)
        label = 1 if for_segmentation else 0 # the labels are set to 1 and 0, this can easily be modified
        for count, page in enumerate(pages):
            basename = os.path.basename(data_path)
            name, _ = os.path.splitext(basename)
            filename = f'{name}_pg_{count}.jpg'
            save_path = f'dataset/{train_or_test}/{label}/{filename}'
            page.save(save_path, 'JPEG')
            df.loc[len(df)] = [data_path, filename, label]