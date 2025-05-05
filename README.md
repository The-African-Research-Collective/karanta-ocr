<div align="center">
<img src="assets/karanta.png" alt="Karanta OCR Logo" width="300"/>
<br/>
  <br>
  <h1>Karanta OCR</h1>
</div>

# Karanta OCR
Karanta means "read" in Hausa, a language spoken in Nigeria and other West African countries. This project is a OCR toolkit for processing scanned documents containing content in african languages and extracting the text in them at scale.

## OCR Pipeline

### Document Type Classification
The classifier identifies documents that require segmentation as segment and those that do not as no_segment. The model is available on the HuggingFace library and can be accessed as described below:

```python
from transformers import pipeline

# Load the pipeline
pipe = pipeline("image-classification", model="taresco/newspaper_classifier_segformer")

# Classify an image
image_path = "path_to_your_image.jpg"
result = pipe(image_path)
print(result)
```

#### Example Output:
The `result` will be a list of dictionaries, where each dictionary contains:

The predicted label (`segment` or `no_segment`).
The confidence score for the prediction.
For example:

```python
[{'label': 'no_segment', 'score': 0.9999988079071045}, {'label': 'segment', 'score': 1.2489092569012428e-06}]
```

### Document Article Segmentation
...

### OCR
...


### Installation
...

## Team

- Dugeri James - [@dugerij](https://github.com/dugerij)
- Ogundepo Odunayo - [@ToluClassics](https://github.com/ToluClassics)
- Akintunde Oladipo - [@theyorubayesian](https://github.com/theyorubayesian)

## License
...

## Citing
