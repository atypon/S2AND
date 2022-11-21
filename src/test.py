from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType

@udf(returnType=StringType())
def extract_signature_id(x: str) -> str:
    """split input string on - and return the first part"""
    x = x.split('-')
    return x[0]



spark = SparkSession.builder.appName('data_preprocessing').getOrCreate()

df = spark.read.json('extended_data_new/trainingSignaturesJson.json'). \
    withColumn(colName='signature_id', col=extract_signature_id('signatureS2AndId')). \
    withColumnRenamed(existing='orcIds', new='orcId')

for dataset in ['aminer', 'arnetminer', 'kisti', 'pubmed', 'zbmath']:
    count = df.filter(df.sampleDataset == dataset).count()
    print(f'{dataset} entries: {count}')

print(df.head())