import json
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType


@udf(returnType=StringType())
def cleanse_signature(signature: str) -> str:
    """
    Converts signature of type xxx-yyy-dataset_name to xxx-yyy
    """
    signature_components = signature.split('-')
    return '-'.join(signature_components[:-1])


datasets = ['aminer', 'arnetminer', 'inspire', 'kisti', 'pubmed', 'zbmath']


if __name__ == "__main__":

    spark = SparkSession.builder. \
        appName('data_preprocessing'). \
        config('spark.driver.memory', '58g'). \
        config('spark.executor.cores', '16'). \
        getOrCreate()

    df = spark.read. \
        json('extended_data/trainingSignaturesJson2.json')

    df.printSchema()
    with open('extended_data/schema.json', 'w') as f:
        schema = json.loads(df.schema.json())
        json.dump(schema, f)

    for dataset in datasets:
        df_subdataset = df.filter(df.sampleDataset == dataset). \
            withColumn('signature_id', cleanse_signature('signatureS2AndId')). \
            dropDuplicates(['signature_id'])

        df_subdataset.coalesce(1).write. \
            mode('overwrite'). \
            json(f'extended_data/{dataset}')

        with open(f'data/{dataset}/{dataset}_signatures.json') as f:
            data = json.load(f)

        df_prev = spark.read.json(f'legacy/{dataset}-signatures.json'). \
            dropDuplicates(['signature_id'])

        print(f'{dataset}:')
        print(f'original: {len(data)}')
        print(f'previous: {df_prev.count()}')
        print(f'new: {df_subdataset.count()}\n\n')
