from typing import List, Dict, Any

import lightgbm as lgb
import numpy as np
from onnxmltools.convert.lightgbm.operator_converters.LightGbm import convert_lightgbm
import onnxruntime as rt
from skl2onnx import convert_sklearn, update_registered_converter
from skl2onnx.common.shape_calculator import calculate_linear_classifier_output_shapes
from skl2onnx.common.data_types import FloatTensorType

from s2and import logger


class ONNXConverter():
    """
    Class that implementing ONNX conversion utilities
    """

    def __init__(
        self,
        model: lgb.LGBMClassifier,
        features: List[Dict[str, Any]]
    ) -> None:
        """
        Initialize the class
        :param model: lightgbm classifier to be converted to ONNX
        :param features: list of features used by model
        """
        self.model = model
        self.features = features

    def convert(self, destination_path: str) -> None:
        """
        Performs the conversion
        :param destination_path: path to store onnx file
        """
        update_registered_converter(
            lgb.LGBMClassifier, 'LightGbmLGBMClassifier',
            calculate_linear_classifier_output_shapes, convert_lightgbm,
            options={'nocl': [True, False], 'zipmap': [True, False, 'columns']}
        )
        model_onnx = convert_sklearn(
            self.model, 'lightgbm',
            [('input', FloatTensorType([None, len(self.features)]))],
            target_opset=12)
        with open(destination_path, "wb") as f:
            f.write(model_onnx.SerializeToString())

    def test_conversion(self, destination_path: str, dataset: np.ndarray) -> None:
        """
        Test conversion under a test dataset
        :param destination_path: str
        :param dataset: np.ndarray of features
        """
        sess = rt.InferenceSession(destination_path)
        y_onnx = sess.run(None, {"input": dataset.astype(np.float32)})
        y_onnx = np.asarray([pred[0] for pred in y_onnx[1]])
        y_target = self.model.predict_proba(dataset)[:, 0]
        try:
            np.testing.assert_array_almost_equal(y_target, y_onnx, decimal=5)
        except AssertionError as e:
            logger.warning(e)
