# Object Detection using torchvision's pretrained fast-rcnn model.

* Download the pre-trained fast-rcnn object detection model's state_dict from the following URL :

https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth

```bash
wget https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth
```

* Create a model archive file and serve the fastrcnn model in TorchServe using below commands

    ```bash
    torch-model-archiver --model-name fastrcnn --version 1.0 --model-file model.py --serialized-file fasterrcnn_resnet50_fpn_coco-258fb6c6.pth --handler detector_handler.py --extra-files index_to_name.json
    mkdir model_store
    mv fastrcnn.mar model_store/
    torchserve --start --model-store model_store --models fastrcnn=fastrcnn.mar --ncs
    curl http://127.0.0.1:8080/predictions/fastrcnn -T persons.jpg
    ```
* Output

```json
[
  {
    "person": "[(167.42229, 57.038273), (301.30536, 436.6867)]",
    "quality": "{'blur_status': 'ok', 'exposure_status': 'ok'}"
  },
  {
    "person": "[(89.61491, 64.89805), (191.40207, 446.66058)]",
    "quality": "{'blur_status': 'ok', 'exposure_status': 'ok'}"
  },
  {
    "person": "[(362.3454, 161.98764), (515.5366, 385.23428)]",
    "quality": "{'blur_status': 'ok', 'exposure_status': 'ok'}"
  },
  {
    "handbag": "[(67.37423, 277.63788), (111.681015, 400.2647)]",
    "quality": "{'blur_status': 'ok', 'exposure_status': 'ok'}"
  },
  {
    "handbag": "[(228.71594, 145.87755), (303.50662, 231.10515)]",
    "quality": "{'blur_status': 'ok', 'exposure_status': 'ok'}"
  },
  {
    "handbag": "[(379.42468, 259.97763), (419.01486, 317.95105)]",
    "quality": "{'blur_status': 'ok', 'exposure_status': 'ok'}"
  },
  {
    "person": "[(517.89954, 149.54932), (636.594, 365.5247)]",
    "quality": "{'blur_status': 'ok', 'exposure_status': 'ok'}"
  },
  {
    "bench": "[(268.9992, 217.24332), (423.9517, 390.4785)]",
    "quality": "{'blur_status': 'ok', 'exposure_status': 'ok'}"
  },
  {
    "person": "[(539.683, 157.81769), (616.1672, 253.09447)]",
    "quality": "{'blur_status': 'ok', 'exposure_status': 'ok'}"
  },
  {
    "person": "[(477.13763, 147.92567), (611.02576, 297.9288)]",
    "quality": "{'blur_status': 'ok', 'exposure_status': 'ok'}"
  },
  {
    "bench": "[(286.66873, 216.35727), (550.454, 383.19553)]",
    "quality": "{'blur_status': 'ok', 'exposure_status': 'ok'}"
  },
  {
    "person": "[(627.44574, 177.21443), (640.0, 247.32768)]",
    "quality": "{'blur_status': 'ok', 'exposure_status': 'ok'}"
  },
  {
    "bench": "[(88.399635, 226.47939), (560.9191, 421.66183)]",
    "quality": "{'blur_status': 'ok', 'exposure_status': 'ok'}"
  },
  {
    "handbag": "[(406.96024, 261.82852), (453.762, 357.5365)]",
    "quality": "{'blur_status': 'ok', 'exposure_status': 'ok'}"
  },
  {
    "chair": "[(451.366, 207.4905), (504.65698, 287.6619)]",
    "quality": "{'blur_status': 'ok', 'exposure_status': 'ok'}"
  },
  {
    "chair": "[(454.38974, 207.96114), (487.7692, 270.3133)]",
    "quality": "{'blur_status': 'ok', 'exposure_status': 'ok'}"
  }
]
```