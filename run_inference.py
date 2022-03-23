import yaml, argparse
from s2and_ext.dataset_embedding import ONNXModel, embed_s2and

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--conf',
        default='configs/inference_conf.yml',
        help='Path to configuration file'
    )
    args = parser.parse_args()
    with open(args.conf) as f:
        conf = yaml.safe_load(f)

    model = ONNXModel(conf['model']['path'], 
                      conf['tokenizer']['pretrained_model'],
                      conf['tokenizer']['max_length'])

    embed_s2and(model=model, 
                model_name=conf['model']['name'],
                data_dir=conf['data_dir'],
                extended_data_dir=conf['extended_data_dir'],
                embeddings_dir=conf['embeddings_dir'])