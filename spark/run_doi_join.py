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
            withColumnRenamed('orcid', 'orcId'). \
            select(['signature_id', 'orcId'])

        df_gold = spark.read.json(join(EXTENDED_DATA_DIR, f'{dataset}-signatures.json'))
        orcids_before = df_gold.select('orcId').dropna().count()
        df_gold = df_gold.drop('orcId'). \
            join(df_old, on='signature_id', how='left_outer')
        orcids_after = df_gold.select('orcId').dropna().count()
        df_gold.coalesce(1). \
            write. \
            mode('overwrite'). \
            json(f'latest_extensions/{dataset}')
        print(dataset)
        print(f'orcids before enrichment: {orcids_before}')
        print(f'orcids after enrichment: {orcids_after}\n')
