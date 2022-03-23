import os, onnxruntime, json, logging

from os.path import join
from transformers import AutoTokenizer
from tqdm import tqdm

logger = logging.getLogger("s2and")
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
ch = logging.StreamHandler()
ch.setFormatter(formatter)
ch.setLevel(logging.INFO)
logger.addHandler(ch)

class ONNXModel():
    """Class that implements model using ONNX runtime"""

    def __init__(self, path_to_onnx, tokenizer_pretrained_model, tokenizer_max_length):
        """Initiliaze model"""
        self.model = onnxruntime.InferenceSession(path_to_onnx, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.tokenizer_max_length = tokenizer_max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_pretrained_model)
    
    def forward(self, text):
        """Implement models forward method"""
        tokens = self.tokenizer(text,
                    add_special_tokens=True,
                    max_length=self.tokenizer_max_length,
                    return_token_type_ids=True,
                    padding='max_length',
                    return_attention_mask=True,
                    return_tensors='pt',
                    truncation=True)

        inputs_ids = tokens['input_ids']
        token_type_ids = tokens['token_type_ids']
        attention_mask = tokens['attention_mask']
        # Calculating embeddings
        ort_inputs = {
            'input_ids': inputs_ids.detach().cpu().numpy(),
            'attention_mask': attention_mask.detach().cpu().numpy(),
            'token_type_ids': token_type_ids.detach().cpu().numpy()
            }
        return self.model.run(None, ort_inputs)[0][:,0,:].squeeze(0).tolist()

def embed_s2and(model, model_name, data_dir, extended_data_dir, embeddings_dir):
    """
    Embed datasets regarding s2and framework
    
    :param model: Model that implements forward method and infers vectors
    :param model_name: Name of the model
    :param data_dir: Directory of s2and dataset
    :param extended_data_dir: Directory of s2and extended dataset
    :param embeddings_dir: Directory embeddings will be saved
    """
    embeddings_dir = join(embeddings_dir, model_name)

    def parse_jsonl(path):
        """Parses jsonl file"""
        with open(path) as f:
            content = f.read().split('\n')
        content.remove('')
        return list(map(json.loads, content))

    dataset_names = ['pubmed', 'aminer', 'zbmath', 'kisti', 'arnetminer']

    for dataset_name in dataset_names:
        logger.info(f'Embedding S2AND {dataset_name}...')
        with open(join(data_dir, dataset_name, f'{dataset_name}_papers.json')) as f:
            papers = json.load(f)
        signatures = parse_jsonl(join(extended_data_dir, dataset_name, f'{dataset_name}-signatures.json'))

        if not os.path.isdir(embeddings_dir):
            os.mkdir(embeddings_dir)

        if not os.path.isdir(join(embeddings_dir, dataset_name)):
            os.mkdir(join(embeddings_dir, dataset_name))
        embeddings = {}
        for entry in tqdm(signatures, total=len(signatures)):
            paper_id = str(entry['paper_id'])
            if paper_id in embeddings:
                # ALready embeded document
                continue
            title = papers[paper_id]['title']
            abstract = papers[paper_id]['abstract']
            if title is None:
                title = ''
            if abstract is None:
                abstract = ''
            model_input = title + '[SEP]' + abstract
            embeddings[paper_id] = model.forward(model_input)
        with open(join(embeddings_dir, dataset_name, f'{dataset_name}_embeddings.json'), 'w') as f:
            json.dump(embeddings, f)