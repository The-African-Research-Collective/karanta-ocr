# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "africanlanguages",
#     "datasets",
#     "datatrove",
#     "huggingface_hub",
#     "typer",
# ]
#
# [tool.uv.sources]
# africanlanguages = { git = "https://github.com/theyorubayesian/africanlanguages.git" }
# ///
# Usage: uv run scripts/finepdf_download.py download-fineweb-african-pdfs --download-dir data/finepdfs -e dag_Latn -e pcm_Latn
# Note that we ignore dag_Latn and pcm_Latn because the eye test shows most of the PDFs are not the correct language
# And together they consititure 1.9M out of the 2.0M PDFs listed as African languages in FinePDF
from __future__ import annotations

import asyncio
import hashlib
import os
from collections import namedtuple
from functools import lru_cache
from typing import Annotated

from africanlanguages import AfricanLanguages
from datasets import get_dataset_config_names, load_dataset, Dataset
from datatrove.pipeline.readers import WarcReader
from huggingface_hub import snapshot_download
from typer import Option, Typer

UrlInfo = namedtuple("UrlInfo", ["language", "url"])
urlstore: dict[str, UrlInfo] = {}

download_semaphore = asyncio.Semaphore(10)

PDF_DOWNLOAD_FOLDER = None

app = Typer(no_args_is_help=True)


class WarcPDFReader(WarcReader):
    """Need to subclass to read objects of mimetype application/pdf"""

    def read_file(self, filepath: str):
        from warcio.archiveiterator import ArchiveIterator

        with self.data_folder.open(filepath, "rb", compression=self.compression) as f:
            for ri, record in enumerate(ArchiveIterator(f)):
                with self.track_time():
                    extracted_data = self.process_record(record)
                    if extracted_data:
                        yield from []
                        return

    @staticmethod
    def process_record(record: "ArcWarcRecord") -> dict | None:
        import magic

        # record type
        if (
            record.rec_type != "response" and record.rec_type != "conversion"
        ):  # wet files have "conversion" type
            return

        # content type filtering
        mime_type = record.rec_headers.get("WARC-Identified-Payload-Type", None)
        if mime_type is not None and mime_type != "application/pdf":
            return

        url = record.rec_headers.get("WARC-Target-URI", None)

        # handle older formats
        if not url:
            url = dict(record.rec_headers.headers)["uri"]

        if url not in urlstore:
            return

        content_bytes = record.content_stream().read()
        if mime_type is None:
            # fallback for older crawls without payload types
            mime_type = magic.from_buffer(content_bytes, mime=True)
            if mime_type != "application/pdf":
                return

        filename = url.split("/")[-1]
        if filename.split(".")[-1].lower() != "pdf":
            filename = hashlib.sha1(url.encode("utf-8")).hexdigest() + ".pdf"

        output_filename = os.path.join(
            PDF_DOWNLOAD_FOLDER, urlstore[url].language, filename
        )

        os.makedirs(os.path.dirname(output_filename), exist_ok=True)
        if not os.path.exists(output_filename):
            with open(output_filename, "wb") as f:
                f.write(content_bytes)

        return True


async def download_pdf(row: dict):
    if "cc-index" in row["file_path"]:
        # For now, we are unable to download from cc-index. Need to figure it out
        return row
    
    filename = row["url"].split("/")[-1]
    if filename.split(".")[-1].lower() != "pdf":
        filename = hashlib.sha1(row["url"].encode("utf-8")).hexdigest() + ".pdf"

    output_filename = os.path.join(
        PDF_DOWNLOAD_FOLDER, row["language"], filename
    )
    if os.path.exists(output_filename):
        return row

    urlstore[row["url"]] = UrlInfo(language=row["language"], url=row["url"])

    data_folder, paths_file = row["file_path"].split("/segments/")
    reader = WarcPDFReader(
        f"{data_folder}/segments",
        glob_pattern=f"*/warc/{paths_file.split('/warc/')[-1]}",
        shuffle_files=False,
        limit=-1,
    )

    async with download_semaphore:
        for _ in reader.run():
            pass
    
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
    global PDF_DOWNLOAD_FOLDER
    PDF_DOWNLOAD_FOLDER = os.path.join(download_dir, "pdfs")

    download_hf_fineweb_african_data(download_dir=download_dir, exclude_languages=exclude_languages)

    ds: dict[str, Dataset] = load_dataset(
        "parquet",
        data_files={
            language: f"{download_dir}/data/{language}/*/*.parquet"
            for language in list_african_languages_in_finepdf()
            if language not in exclude_languages
        },
    )

    for _, language_ds in ds.items():
        language_ds.map(download_pdf, batched=False)


# app.command(download_all_fineweb_african_data)


if __name__ == "__main__":
    app()
