import argparse
import asyncio

from s2and.extentions.dataset_embedding import GeminiModel, embed_s2and
from s2and.utils.configs import load_configurations

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Run inference for embedding generation."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/inference_conf.yaml",
        help="Path to the inference configuration YAML file (default: configs/inference_conf.yaml)",
    )
    args = parser.parse_args()

    cfg = load_configurations(path=args.config)
    if cfg.model.type == "gemini":
        model = GeminiModel(
            gcp_project=cfg.model.gcp_project,
            gcp_location=cfg.model.gcp_location,
            model=cfg.model.name,
            task_type=cfg.model.task_type,
            truncate_dim=cfg.model.truncate_dim,
        )
    else:
        raise ValueError(f"Unknown model type: {cfg.model.type}.")

    asyncio.run(
        embed_s2and(
            model=model,
            model_name=cfg.model.name,
            data_dir=cfg.data_dir,
            extended_data_dir=cfg.extended_data_dir,
            embeddings_dir=cfg.embeddings_dir,
            dataset_names=cfg.datasets,
            batch_size=cfg.batch_size,
        )
    )
