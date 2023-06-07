from s2and.utils.configs import load_configurations
from s2and.extentions.dataset_embedding import ONNXModel, embed_s2and

if __name__ == "__main__":

    cfg = load_configurations(path='configs/inference_conf.yaml')
    model = ONNXModel(
        path_to_onnx=cfg.model.path,
        tokenizer_pretrained_model=cfg.tokenizer.pretrained_model,
        tokenizer_max_length=cfg.tokenizer.max_length,
        inputs=cfg.model.inputs
    )
    embed_s2and(
        model=model,
        model_name=cfg.model.name,
        data_dir=cfg.data_dir,
        extended_data_dir=cfg.extended_data_dir,
        embeddings_dir=cfg.embeddings_dir
    )
