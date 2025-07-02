import os
import torch
import onnxruntime as ort
import numpy as np
from models.segmentation.STELLE_Pokemon_Segmentation.model.STELLE_model import STELLE_Seg
from models.segmentation.STELLE_Pokemon_Segmentation.utils.prepocessing import preprocess_image
from models.segmentation.STELLE_Pokemon_Segmentation.utils.visualization import apply_overlay

class STELLEInferencer:
    def __init__(self, torch_model_path="models/segmentation/STELLE_Pokemon_Segmentation/weights/STELLE_Seg.pth", onnx_model_path="models/segmentation/STELLE_Pokemon_Segmentation/weights/STELLE_Seg.onnx"):
        """
        Initialize the STELLEInferencer.

        Loads the model (PyTorch or ONNX) depending on the availability of a GPU.
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.use_onnx = self.device == 'cpu'
        print(f"[Vision Model] Using device: {self.device}")

        if self.use_onnx:
            if not os.path.isfile(onnx_model_path):
                raise FileNotFoundError("[Vision Model] ONNX model not found.")
            self.session = ort.InferenceSession(onnx_model_path)
        else:
            self.model = STELLE_Seg().to(self.device)
            self.model.load_state_dict(torch.load(torch_model_path, map_location=self.device))
            self.model.eval()

    def predict(self, np_image):
        """
        Perform segmentation on the input image and return the prediction mask.

        Args:
            np_image (np.ndarray): Input image as a NumPy array.

        Returns:
            np.ndarray: Segmentation prediction mask.
        """
        tensor = preprocess_image(np_image)
        tensor = tensor.unsqueeze(0)  # Add batch dimension

        if self.use_onnx:
            ort_inputs = {self.session.get_inputs()[0].name: tensor.numpy()}
            ort_outs = self.session.run(None, ort_inputs)
            output = torch.from_numpy(ort_outs[0])
        else:
            with torch.no_grad():
                output = self.model(tensor.to(self.device)).cpu()

        pred = torch.argmax(output, dim=1).squeeze(0).numpy().astype(np.uint8)
        return pred

    def predict_with_overlay(self, np_image):
        """
        Perform segmentation and return both the prediction mask and an image overlay.

        Args:
            np_image (np.ndarray): Input image as a NumPy array.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple containing the segmentation mask and the overlay image.
        """
        tensor = preprocess_image(np_image)
        tensor = tensor.unsqueeze(0)  # Add batch dimension

        if self.use_onnx:
            ort_inputs = {self.session.get_inputs()[0].name: tensor.numpy()}
            ort_outs = self.session.run(None, ort_inputs)
            output = torch.from_numpy(ort_outs[0])
        else:
            with torch.no_grad():
                output = self.model(tensor.to(self.device)).cpu()

        pred = torch.argmax(output, dim=1).squeeze(0).numpy().astype(np.uint8)
        overlay = apply_overlay(np_image, pred)
        return pred, overlay
