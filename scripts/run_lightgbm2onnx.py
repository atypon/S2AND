import joblib
import onnxruntime as rt
import numpy as np
import lightgbm as lgb
from s2and.utils.configs import load_configurations
from s2and.extentions.featurization.utils import get_matrices, featurizing_function
from skl2onnx import convert_sklearn, update_registered_converter
from skl2onnx.common.shape_calculator import calculate_linear_classifier_output_shapes
from skl2onnx.common.data_types import FloatTensorType
from onnxmltools.convert.lightgbm.operator_converters.LightGbm import convert_lightgbm


if __name__ == "__main__":

    cfg = load_configurations('configs/onnx_conf.yml')
    # Get GBMClassifier out of LIghtGBMWrapper
    model = joblib.load(cfg.model_source).model
    X_train, y_train, X_val, y_val, X_test, y_test = get_matrices(
        datasets=['aminer', 'pubmed', 'zbmath', 'kisti', 'arnetminer'],
        featurizing_function=featurizing_function,
        remove_nan=False,
        default_embeddings=True,
        external_emb_dir=None
    )
    # Convertion
    update_registered_converter(
        lgb.LGBMClassifier, 'LightGbmLGBMClassifier',
        calculate_linear_classifier_output_shapes, convert_lightgbm,
        options={'nocl': [True, False], 'zipmap': [True, False, 'columns']})
    model_onnx = convert_sklearn(
        model, 'lightgbm',
        [('input', FloatTensorType([None, 9]))],
        target_opset=12)
    with open(cfg.onnx_dest, "wb") as f:
        f.write(model_onnx.SerializeToString())
    # Testing
    sess = rt.InferenceSession(cfg.onnx_dest)
    pred_onx = sess.run(None, {"input": X_train.astype(np.float32)})
    y_onnx = np.asarray([pred[0] for pred in pred_onx[1]])
    y_target = model.predict_proba(X_train)[:, 0]
    print(sess.run(None, {"input": X_train.astype(np.float32)[: 5, :]}))
    try:
        np.testing.assert_array_almost_equal(y_target, y_onnx, decimal=5)
        print('Test passed')
    except Exception:
        print('Test failed')
