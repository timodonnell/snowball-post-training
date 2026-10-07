"""Stage checksum-verified inputs in CoreWeave object storage."""

import argparse
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse

from huggingface_hub import HfApi, hf_hub_download

from .pins import MODEL, MODEL_REVISION


def upload_directory(local, uri):
    from rigging.filesystem.storage_path import StoragePath

    def upload(path):
        target = StoragePath(uri) / str(path.relative_to(local))
        payload = path.read_bytes()
        expected = hashlib.sha256(payload).hexdigest()
        if target.exists():
            with target.open("rb") as source:
                if hashlib.sha256(source.read()).hexdigest() == expected:
                    return
            raise ValueError(f"Immutable artifact collision: {target}")
        with target.open("wb") as output:
            output.write(payload)
        print(f"uploaded {path.name} ({len(payload)} bytes)", flush=True)

    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(upload, (p for p in local.rglob("*") if p.is_file() and "__pycache__" not in p.parts)))


def mirror_model(uri, cache, keep_cache=False):
    import boto3
    from boto3.s3.transfer import TransferConfig
    from botocore.config import Config

    root = urlparse(uri)
    if root.scheme != "s3" or not root.netloc:
        raise ValueError("Model mirror requires an s3:// URI")
    bucket, prefix = root.netloc, root.path.strip("/")
    cache.mkdir(parents=True, exist_ok=True)
    client = boto3.client(
        "s3",
        endpoint_url=os.environ["AWS_ENDPOINT_URL"],
        config=Config(
            s3={"addressing_style": "virtual"},
            connect_timeout=30,
            read_timeout=120,
            retries={"max_attempts": 5, "mode": "standard"},
        ),
    )
    info = HfApi().model_info(MODEL, revision=MODEL_REVISION, files_metadata=True)
    if info.sha != MODEL_REVISION:
        raise ValueError("HF returned a different revision")
    files = [s for s in info.siblings if s.rfilename.endswith((".json", ".jinja", ".safetensors"))]

    def mirror(entry):
        target = prefix + "/" + entry.rfilename
        receipt = prefix + "/checksums/" + entry.rfilename + ".json"
        expected = entry.lfs.sha256 if entry.lfs else None
        try:
            result = json.loads(client.get_object(Bucket=bucket, Key=receipt)["Body"].read())
            head = client.head_object(Bucket=bucket, Key=target)
        except client.exceptions.ClientError as error:
            if error.response["Error"]["Code"] not in {"NoSuchKey", "404"}:
                raise
        else:
            if result["revision"] != MODEL_REVISION or (expected and result["sha256"] != expected):
                raise ValueError("Existing model receipt does not match pinned source")
            if head["ContentLength"] != entry.size:
                raise ValueError("Existing model object has the wrong size")
            print(f"verified existing {entry.rfilename}", flush=True)
            return result
        digest = hashlib.sha256()
        size = 0
        downloaded = cache / entry.rfilename
        if not downloaded.exists():
            hf_hub_download(MODEL, entry.rfilename, revision=MODEL_REVISION, local_dir=cache)
        with downloaded.open("rb") as source:
            for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
                size += len(chunk)
        if size != entry.size or (expected and digest.hexdigest() != expected):
            raise ValueError(f"Model integrity failure: {entry.rfilename}")
        print(f"uploading {entry.rfilename}: {size} verified bytes", flush=True)
        client.upload_file(
            str(downloaded),
            bucket,
            target,
            ExtraArgs={"Metadata": {"sha256": digest.hexdigest()}},
            Config=TransferConfig(
                multipart_threshold=32 * 1024 * 1024, multipart_chunksize=64 * 1024 * 1024, max_concurrency=2
            ),
        )
        if client.head_object(Bucket=bucket, Key=target)["ContentLength"] != size:
            raise ValueError(f"Incomplete model object: {entry.rfilename}")
        result = {"file": entry.rfilename, "sha256": digest.hexdigest(), "size": size, "revision": MODEL_REVISION}
        client.put_object(Bucket=bucket, Key=receipt, Body=json.dumps(result).encode())
        if not keep_cache:
            downloaded.unlink()
        print(f"mirrored {entry.rfilename}: {size} bytes", flush=True)
        return result

    with ThreadPoolExecutor(max_workers=4) as pool:
        manifest = list(pool.map(mirror, files))
    client.put_object(
        Bucket=bucket,
        Key=prefix + "/source.json",
        Body=json.dumps({"model": MODEL, "revision": MODEL_REVISION, "files": manifest}, indent=2).encode(),
    )
    print("Model mirror complete", uri, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path)
    parser.add_argument("--data-uri")
    parser.add_argument("--model-uri")
    parser.add_argument("--model-cache", type=Path, default=Path("data/model-download"))
    parser.add_argument("--keep-cache", action="store_true", help="Preserve an existing shared model cache")
    args = parser.parse_args()
    if args.prepared:
        upload_directory(args.prepared, args.data_uri)
    if args.model_uri:
        mirror_model(args.model_uri, args.model_cache, args.keep_cache)


if __name__ == "__main__":
    main()
