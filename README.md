# River Plastic Detection

### 1.Import preprocessed Dataset from roboflow

```bash
import locale
locale.getpreferredencoding = lambda: "UTF-8"
!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="8PmKE7lnezZGqlFq7yxm")
project = rf.workspace("mylab").project("thai_laos")
dataset = project.version(1).download("yolov8")
```
### ROBOFLOW LINK (DATASET-- https://app.roboflow.com/mylab/thai_laos/1) --> where we can annotate, preprocess and augment datasets(IMAGES)
### 2. Train from scratch

```bash
from ultralytics import YOLO
import os
model = YOLO("yolov8n.pt")
model.train(data=os.path.join(root_dir,'data.yaml'), epochs=40, batch=16,cache=True)  

```

### 3. Download the contents

```bash
import locale
def getpreferredencoding(do_setlocale = True):
    return "UTF-8"
locale.getpreferredencoding = getpreferredencoding
!zip -r /content/file.zip /content/runs

```
### 4. Prediction

```bash
model.predict("/content/file20_95.png",save=True)

```
