from os.path import join
from pyspark.sql import SparkSession


LEGACY_DIR = 'legacy'
EXTENDED_DATA_DIR = 'extended_data'
DATASETS = ['aminer', 'arnetminer', 'inspire', 'kisti', 'pubmed', 'zbmath']


if __name__ == "__main__":

    spark = SparkSession.builder. \
        appName('data_preprocessing'). \
        config('spark.driver.memory', '58g'). \
        config('spark.executor.cores', '16'). \
        getOrCreate()

    for dataset in DATASETS:

        df_old = spark.read.json(join(LEGACY_DIR, f'{dataset}-signatures.json')). \
            select(['signature_id', 'coAuthorShortNormNames'])

        df_gold = spark.read.json(join(EXTENDED_DATA_DIR, f'{dataset}-signatures.json')). \
            join(df_old, on='signature_id', how='left_outer')
        df_gold.coalesce(1). \
            write. \
            mode('overwrite'). \
            json(join(EXTENDED_DATA_DIR, dataset))
