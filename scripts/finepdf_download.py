# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "africanlanguages",
#     "datasets",
#     "datatrove",
#     "huggingface_hub",
#     "surt",
#     "tenacity",
#     "typer",
#     "warcio",
# ]
#
# [tool.uv.sources]
# africanlanguages = { git = "https://github.com/theyorubayesian/africanlanguages.git" }
# ///
# Usage: uv run scripts/finepdf_download.py download-fineweb-african-pdfs --download-dir data/finepdfs -e dag_Latn -e pcm_Latn
# Notes:
#   - We ignore dag_Latn and pcm_Latn because the eye test shows most of the PDFs are not the correct language
#   - Together, they consititure 1.9M out of the 2.0M PDFs listed as African languages in FinePDF
#   - We're crawling politely as CommonCrawl suggests here: https://commoncrawl.org/blog/oct-nov-2023-performance-issues
import asyncio
import hashlib
import os
import uuid
from functools import lru_cache
from io import BytesIO
from typing import Annotated

import aiohttp
import pyarrow.parquet as pq
from africanlanguages import AfricanLanguages
from datasets import get_dataset_config_names, load_dataset, Dataset
from huggingface_hub import snapshot_download
from surt import surt
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type
from typer import Option, Typer
from warcio.archiveiterator import ArchiveIterator

# NOTE: This limits the number of downloads that happens at once
download_semaphore = asyncio.Semaphore(5)

app = Typer(no_args_is_help=True)


def deterministic_id(string: str) -> str:
    return str(uuid.UUID(hashlib.md5(string).hexdigest()))


@retry(
    stop=stop_after_attempt(1000),
    wait=wait_fixed(1),
    retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
)
def query_commoncrawl_parquet(crawl_filepath: str, surt_key: str) -> dict | None:
    pf = pq.ParquetFile(crawl_filepath)

    for i in range(pf.num_row_groups):
        rg = pf.metadata.row_group(i)
        col_idx = pf.schema_arrow.get_field_index("url_surtkey")
        stats = rg.column(col_idx).statistics

        if stats and stats.min <= surt_key <= stats.max:
            df = pf.read_row_groups(
                [i],
                columns=[
                    "url_surtkey",
                    "url",
                    "warc_filename",
                    "warc_record_offset",
                    "warc_record_length",
                ],
            ).to_pandas()

            match = df[df["url_surtkey"] == surt_key]
            if len(match) > 0:
                # TODO: @theyorubayesian: Why would there be multiple matches?
                return match.iloc[0].to_dict()
            break

    return None


@retry(
    stop=stop_after_attempt(1000),
    wait=wait_fixed(1),
    retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
)
async def get_record_length_from_warc_headers(
    session: aiohttp.ClientSession, warc_filename: str, offset: str
) -> int:
    chunk_size = 1024 * 1024
    url = f"https://data.commoncrawl.org/{warc_filename}"
    headers = {"Range": f"bytes={offset}-{offset + chunk_size - 1}"}

    async with session.get(url, headers=headers) as resp:
        resp.raise_for_status()
        content = await resp.read()

    buffer = BytesIO(content)
    record = next(ArchiveIterator(buffer))
    return int(record.rec_headers.get_header("Content-Length"))


@retry(
    stop=stop_after_attempt(1000),
    wait=wait_fixed(1),
    retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
)
async def download_warc_record(
    download_path: str,
    warc_filename: str,
    offset: int,
    length: int = None,
) -> None:
    """
    Download a single WARC record from Common Crawl asynchronously.
    Returns the content of the record as bytes.
    """
    url = f"https://data.commoncrawl.org/{warc_filename}"

    async with aiohttp.ClientSession() as session:
        if length is None:
            length = await get_record_length_from_warc_headers(
                session, warc_filename, offset
            )

        headers = {"Range": f"bytes={offset}-{offset + length - 1}"}

        async with session.get(url, headers=headers) as resp:
            resp.raise_for_status()
            content = await resp.read()

    # Extract the record from the WARC content
    record = next(ArchiveIterator(BytesIO(content)))
    with open(download_path, "wb") as f:
        f.write(record.content_stream().read())


async def download_pdf(row: dict, download_dir: str) -> dict:
    """
    Full async function: get WARC content for a given URL.
    """
    os.makedirs(f"{download_dir}/{row['language']}", exist_ok=True)
    filename = f"{deterministic_id(row['id'].encode('utf-8'))}.pdf"
    download_path = f"{download_dir}/{row['language']}/{filename}"

    row["file_name"] = filename

    if os.path.exists(download_path):
        return row

    warc_filename = row["file_path"].replace("s3://commoncrawl/", "")
    offset = int(row["offset"])
    length = None

    async with download_semaphore:
        if warc_filename.startswith("cc-index"):
            cdx_results = query_commoncrawl_parquet(row["file_path"], surt(row["url"]))

            if cdx_results:
                warc_filename = cdx_results["warc_filename"]
                # Because we select the first warc_file that has a url match, there's a chance the offset is different
                # HF may have collected from a different warc_file
                offset = cdx_results["warc_record_offset"]
                length = int(cdx_results["warc_record_length"])
            else:
                row["file_name"] = None

        if warc_filename.startswith("crawl-data"):
            await download_warc_record(
                warc_filename=warc_filename,
                offset=offset,
                length=length,
                download_path=download_path,
            )
    return row


@lru_cache()
def list_african_languages_in_finepdf() -> list[str]:
    configs = get_dataset_config_names("HuggingFaceFW/finepdfs")
    finepdf_african_languages = [
        language
        for language in configs
        if language.split("_")[0].upper() in AfricanLanguages._member_names_
    ]
    return finepdf_african_languages


@app.command()
def download_hf_fineweb_african_data(
    download_dir: Annotated[str, Option(help="Directory to download data to")],
    exclude_languages: Annotated[
        list[str] | None,
        Option("-e", "--exclude-language", help="language codes to exclude"),
    ] = None,
):
    snapshot_download(
        repo_id="HuggingFaceFW/finepdfs",
        repo_type="dataset",
        allow_patterns=[
            f"data/{language}/**/*.parquet"
            for language in list_african_languages_in_finepdf()
            if language not in exclude_languages
        ],
        local_dir=download_dir,
    )


@app.command()
def download_fineweb_african_pdfs(
    download_dir: Annotated[str, Option(help="Directory to download data to")],
    exclude_languages: Annotated[
        list[str] | None,
        Option("-e", "--exclude-language", help="language codes to exclude"),
    ] = None,
):
    PDF_DOWNLOAD_FOLDER = os.path.join(download_dir, "pdfs")

    download_hf_fineweb_african_data(
        download_dir=download_dir, exclude_languages=exclude_languages
    )

    ds: dict[str, Dataset] = load_dataset(
        "parquet",
        data_files={
            language: f"{download_dir}/data/{language}/*/*.parquet"
            for language in list_african_languages_in_finepdf()
            if language not in exclude_languages
        },
    )

    for _, language_ds in ds.items():
        language_ds.map(
            download_pdf, batched=False, fn_kwargs={"download_dir": PDF_DOWNLOAD_FOLDER}
        )

    # TODO: Do we rewrite the dataset to file?


if __name__ == "__main__":
    app()
