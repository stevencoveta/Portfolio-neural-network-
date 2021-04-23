import json
import boto3


def S3():
    s3 = boto3.resource("s3", region_name='us-east-2',
        aws_access_key_id="AKIAVVZS666ZTUVXRS4A",
        aws_secret_access_key="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").Bucket("data-close")
    return s3




def S3_t():
    s3 = boto3.resource("s3", region_name='us-east-2',
        aws_access_key_id="AKIAVVZS666ZTUVXRS4A",
        aws_secret_access_key="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").Bucket("data-targetdict")
    return s3