"""
Module for object detection default handler
"""
import io
import torch
import numpy as np
from cv2 import cv2
from PIL import Image
from torchvision import transforms
from torchvision import __version__ as torchvision_version
from torch.autograd import Variable
from ts.torch_handler.vision_handler import VisionHandler

class ObjectDetector(VisionHandler):
    """
    ObjectDetector handler class. This handler takes an image
    and returns list of detected classes and bounding boxes respectively
    """

    def __init__(self):
        super(ObjectDetector, self).__init__()

    def initialize(self, ctx):
        super(ObjectDetector, self).initialize(ctx)
        version = torchvision_version.split(".")

        if int(version[0]) == 0 and int(version[1]) < 6:
            self.initialized = False
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            self.model.eval()
            self.initialized = True

    def preprocess(self, data):
        """
         Scales, crops, and normalizes a image for a PyTorch model,
         returns an Numpy array
        """
        image = data[0].get("data")
        if image is None:
            image = data[0].get("body")

        my_preprocess = transforms.Compose([transforms.ToTensor()])
        image = Image.open(io.BytesIO(image))
        image_data = np.array(image)
        image = my_preprocess(image)
        return [image, image_data]

    def inference(self, data):
        
        # Store a copy of the original image data
    
        tensor = data[0]
        np_img = data[1]

        threshold = 0.5
        # Predict the classes and bounding boxes in an image using a trained deep learning model.
        data = Variable(tensor).to(self.device)
        pred = self.model([data])  # Pass the image to the model
        pred_class = list(pred[0]['labels'].cpu().numpy()) # Get the Prediction Score
        # Bounding boxes
        pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].cpu().detach().numpy())]
        pred_score = list(pred[0]['scores'].cpu().detach().numpy())
        # Get list of index with score greater than threshold.
        pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1]
        pred_boxes = pred_boxes[:pred_t + 1]
        pred_class = pred_class[:pred_t + 1]

        # Assess each bounding box for blur and contrast checks
        print(pred_boxes)

        pred_quality = []
        for i in range(len(pred_boxes)):
            print(i)
            pred_quality.append(check_quality(np_img, pred_boxes[i]))

        return [pred_class, pred_boxes, pred_quality]

    def postprocess(self, data):
        pred_class = data[0]
        pred_quality = data[2]
        try:
            if self.mapping:
                pred_class = [self.mapping['object_type_names'][i] for i in pred_class]  # Get the Prediction Score

            retval = []
            for idx, box in enumerate(data[1]):
                class_name = pred_class[idx]
                quality = pred_quality[idx]
                retval.append({class_name: str(box), "quality": str(quality)})
            return [retval]
        except Exception as e:
            raise Exception('Object name list file should be json format - {"object_type_names":["person","car"...]}"'
                            + e)
_service = ObjectDetector()

def handle(data, context):
    """
    Entry point for object detector default handler
    """
    try:
        if not _service.initialized:
            _service.initialize(context)

        if data is None:
            return None

        data = _service.preprocess(data)
        data = _service.inference(data)
        data = _service.postprocess(data)

        return data
    except Exception as e:
        raise Exception("Please provide a custom handler in the model archive." + e)


def estimate_blur(image: np.array, threshold: int = 100):
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur_map = cv2.Laplacian(image, cv2.CV_64F)
    score = np.var(blur_map)
    return blur_map, score, bool(score < threshold)

def pretty_blur_map(blur_map: np.array, sigma: int = 5, min_abs: float = 0.5):
    abs_image = np.abs(blur_map).astype(np.float32)
    abs_image[abs_image < min_abs] = min_abs

    abs_image = np.log(abs_image)
    cv2.blur(abs_image, (sigma, sigma))
    return cv2.medianBlur(abs_image, sigma)

def test_exposure(image: np.array, over_thresh=0.05, under_thresh=0.05):
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    total = np.sum(hist)
    bright = np.sum(hist[255])
    bright /= total
    dark = np.sum(hist[0])
    dark /= total
    print("overexposed proportion: " + str(bright))
    print("underexposed proportion: " + str(dark))
    return (bright > over_thresh) + (dark > under_thresh) * 2


def check_quality(image, box):
    print(box)
    x = round(box[0][0])
    y = round(box[0][1])
    w = round(box[1][0] - box[0][0])
    h = round(box[1][1] - box[0][1])
    cropped = image[y:y + h, x:x + w]
    _, _, is_blurred = estimate_blur(cropped)
    exposure_result = test_exposure(cropped)
    exposure_statuses = ['ok', 'overexposed', 'underexposed', 'both']
    exposure_status = exposure_statuses[exposure_result]
    blur_status = "blurred" if is_blurred else "ok"
    print("Exposure: " + exposure_status)
    print("Blur: " + blur_status)

    return {
        "blur_status": blur_status,
        "exposure_status": exposure_status
    }
    
