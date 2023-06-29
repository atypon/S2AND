import joblib
from s2and.utils.configs import load_configurations
from s2and.extentions.featurization.utils import get_matrices
from s2and.utils.onnx_converter import ONNXConverter


if __name__ == "__main__":

    cfg = load_configurations('configs/onnx_conf.yml')

    # Get GBMClassifier out of LIghtGBMWrapper
    model = joblib.load(cfg.model.source).model
    X_train, _, _, _, _, _ = get_matrices(
        datasets=cfg.datasets,
        features=cfg.features,
        remove_nan=False,
        external_emb_dir=cfg.external_embeddings_dir
    )
    converter = ONNXConverter(
        model=model,
        features=cfg.features
    )
    converter.convert(destination_path=cfg.model.destination)
    converter.test_conversion(
        destination_path=cfg.model.destination,
        dataset=X_train
    )
