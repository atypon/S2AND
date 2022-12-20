import requests
import time
import os
import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.sql.types import StringType, ArrayType, FloatType
from typing import List


#DATASETS = ['arnetminer', 'kisti', 'pubmed', 'zbmath', 'inspire', 'qian', 'aminer']
DATASETS = ['inspire']
URL = 'https://rnd-rsbert-triton-gpu.literatumonline.com/v2/models/ensemble_model/infer'
DELAY = 0.01

s = requests.Session()

def infer_vector(text: str) -> requests.Response:
    request = {
        "inputs": [
            {
                "name": "TEXT",
                "shape": [1],
                "datatype": "BYTES",
                "data": [text]
            }
        ],
        "outputs": [
            {
                "name": "sentence_emb",
                "parameters": {"binary_data": False}
            }
        ]
    }
    reply = s.post(
        url=URL,
        json=request,
        headers={'Content-Type': 'application/json'}
    )
    return reply

@F.udf(returnType=ArrayType(elementType=FloatType()))
def infer_vectors(text: str) -> List[float]:
    """
    Infer vectors for entries that do noy have vector repr
    """
    if text is None:
        return None
    while True:
        # Sleep otherwise the service cannot handle the load
        time.sleep(DELAY)
        try:
            reply = infer_vector(text=text)
        except:
            print('Connection exception, retrying...')
            continue            
        if reply.ok:
            break
        else:
            print(f'Received status code: {reply.status_code}, retrying...')
    vector = reply.json()['outputs'][0]['data']
    return vector

if __name__ == "__main__":
    
    spark = SparkSession.builder. \
        appName('data_preprocessing'). \
        config('spark.driver.memory', '58g'). \
        config('spark.executor.cores', '16'). \
        getOrCreate()

    while True:
        reply = infer_vector('test test test')
        if reply.ok: break
        time.sleep(0.3)
    print('Service is up')

    for dataset in DATASETS:
        t0 = time.time()
        print(f'Dataset: {dataset}')
        df = spark.read.json(f'extended_data_new/dataset={dataset}')
        print(f'Initial entries: {df.count()}')

        df_missing = df.filter(df.vector.isNull()). \
            withColumn('text', F.concat('title', F.lit('. '), 'abstract')). \
            withColumn('vector', infer_vectors('text')). \
            drop('text')
        initial_missing = df_missing.count()
        print(f'Initial missing vectors: {initial_missing}')

        df = df.filter(df.vector.isNotNull()). \
            union(df_missing). \
            drop('title', 'abstract'). \
            persist()
        
        print(f'Final entries: {df.count()}')
        final_missing = df.filter(df.vector.isNull()).count()
        print(f'Final missing vectors: {final_missing}')

        if not os.path.isdir(f'extended_data_new/{dataset}'):
            os.mkdir(f'extended_data_new/{dataset}')
        #df.coalesce(1).toPandas().to_json(f'extended_data/{dataset}/{dataset}-signatures.json', lines=True, orient='records')
        df.coalesce(1).write.json(f'extended_data/{dataset}')
        print(f'Overall processing duration {(time.time()-t0):.2f}')
        print('______________\n')