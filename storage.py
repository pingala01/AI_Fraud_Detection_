"""Model storage in Amazon S3. NO credentials are stored in code.

boto3 finds credentials automatically: an IAM role (EC2/Lambda), `aws configure`,
or the AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY environment variables.
Required environment variable: MODEL_S3_BUCKET. Optional: MODEL_S3_PREFIX, AWS_REGION.
"""
import argparse
import os

from . import config as C


def _settings(bucket=None, prefix=None):
    bucket = bucket or os.getenv("MODEL_S3_BUCKET")
    if not bucket:
        raise RuntimeError("Set the MODEL_S3_BUCKET environment variable (name of your S3 bucket).")
    prefix = (prefix if prefix is not None else os.getenv("MODEL_S3_PREFIX", "fraud-model/"))
    if prefix and not prefix.endswith("/"):
        prefix += "/"
    return bucket, prefix


def _client():
    import boto3
    return boto3.client("s3", region_name=os.getenv("AWS_REGION") or None)


def upload_artifacts(bucket=None, prefix=None, model_dir=None):
    bucket, prefix = _settings(bucket, prefix)
    model_dir = model_dir or C.MODEL_DIR
    s3 = _client()
    for name in (C.MODEL_FILENAME, C.REFERENCE_FILENAME):
        path = model_dir / name
        if not path.exists():
            raise FileNotFoundError(f"{path} not found - train the model first.")
        s3.upload_file(str(path), bucket, prefix + name, ExtraArgs={"ServerSideEncryption": "AES256"})
        print(f"Uploaded {name} -> s3://{bucket}/{prefix}{name}")


def download_artifacts(bucket=None, prefix=None, model_dir=None):
    bucket, prefix = _settings(bucket, prefix)
    model_dir = model_dir or C.MODEL_DIR
    model_dir.mkdir(parents=True, exist_ok=True)
    s3 = _client()
    for name in (C.MODEL_FILENAME, C.REFERENCE_FILENAME):
        s3.download_file(bucket, prefix + name, str(model_dir / name))
        print(f"Downloaded s3://{bucket}/{prefix}{name} -> {model_dir / name}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Upload/download model files to/from S3")
    ap.add_argument("action", choices=["upload", "download"])
    a = ap.parse_args()
    (upload_artifacts if a.action == "upload" else download_artifacts)()
