import io
import os
import json
import hashlib
import concurrent.futures
from threading import Lock
from datetime import datetime
from collections import defaultdict

import pandas as pd
from pypdf import PdfReader
from dotenv import load_dotenv
from azure.storage.blob import ContainerClient

load_dotenv()

AZURE_ACCOUNT_NAME = os.getenv("AZURE_STORAGE_ACCOUNT_NAME")
AZURE_SAS_TOKEN = os.getenv("AZURE_SAS_TOKEN")


class AzureBlobPDFAnalyzer:
    def __init__(
        self,
        sas_token,
        container_name,
        max_workers=5,
        state_file="pdf_analysis_state.json",
    ):
        account_url = f"https://{AZURE_ACCOUNT_NAME}.blob.core.windows.net"
        self.container_client = ContainerClient(
            account_url,
            container_name,
            credential=sas_token,
            connection_timeout=60,
            read_timeout=300,
        )
        self.max_workers = max_workers
        self.state_file = state_file
        self.results = []
        self.folder_stats = defaultdict(
            lambda: {"pdf_count": 0, "total_pages": 0, "processed_pdfs": []}
        )
        self.lock = Lock()
        self.state = self._load_state()

    def _load_state(self):
        """Load previous processing state from file"""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, "r") as f:
                    state = json.load(f)
                    processed_count = len(state.get("processed_files", {}))
                    directories_count = len(
                        set(
                            details.get("folder", "root")
                            for details in state.get("file_details", {}).values()
                        )
                    )
                    print(
                        f"📂 Loaded state: {processed_count} files from {directories_count} directories"
                    )
                    return state
            except Exception as e:
                print(f"⚠️ Could not load state file: {e}")
                return {"processed_files": {}, "file_details": {}, "last_updated": None}
        return {"processed_files": {}, "file_details": {}, "last_updated": None}

    def _save_state(self):
        """Save current processing state to file"""
        try:
            self.state["last_updated"] = datetime.now().isoformat()

            # Add summary statistics to state file
            self.state["summary"] = {
                "total_files_processed": len(self.state["processed_files"]),
                "total_directories": len(
                    set(
                        details.get("folder", "root")
                        for details in self.state.get("file_details", {}).values()
                    )
                ),
                "last_run_timestamp": datetime.now().isoformat(),
            }

            with open(self.state_file, "w") as f:
                json.dump(self.state, f, indent=2)

            files_count = len(self.state["processed_files"])
            print(f"💾 State saved: {files_count} processed files")
        except Exception as e:
            print(f"⚠️ Could not save state file: {e}")

    def _get_blob_hash(self, blob_name, last_modified):
        """Generate a hash for blob identification including last modified date"""
        hash_string = f"{blob_name}_{last_modified.isoformat()}"
        return hashlib.md5(hash_string.encode()).hexdigest()

    def _extract_folder_path(self, blob_name):
        """Extract folder path from blob name"""
        if "/" in blob_name:
            return "/".join(blob_name.split("/")[:-1])
        return "root"

    def _is_already_processed(self, blob_name, blob_hash):
        """Check if blob has already been processed"""
        return self.state["processed_files"].get(blob_name) == blob_hash

    def process_single_pdf(self, blob_info):
        """Process a single PDF blob"""
        blob_name, blob_hash = blob_info

        try:
            # Check if already processed
            if self._is_already_processed(blob_name, blob_hash):
                # Get cached page count from state if available
                cached_pages = (
                    self.state.get("file_details", {})
                    .get(blob_name, {})
                    .get("pages", "CACHED")
                )

                cached_result = {
                    "filename": blob_name,
                    "folder": self._extract_folder_path(blob_name),
                    "pages": cached_pages,
                    "processed_at": "FROM_CACHE",
                }

                print(f"📋 {blob_name}: CACHED")
                return cached_result

            # Download and process PDF
            blob_client = self.container_client.get_blob_client(blob_name)
            pdf_data = blob_client.download_blob(timeout=300).readall()

            pdf_reader = PdfReader(io.BytesIO(pdf_data))
            page_count = len(pdf_reader.pages)
            folder_path = self._extract_folder_path(blob_name)

            result = {
                "filename": blob_name,
                "folder": folder_path,
                "pages": page_count,
                "processed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }

            with self.lock:
                # Update state with file details
                self.state["processed_files"][blob_name] = blob_hash
                if "file_details" not in self.state:
                    self.state["file_details"] = {}
                self.state["file_details"][blob_name] = {
                    "pages": page_count,
                    "folder": folder_path,
                    "last_processed": result["processed_at"],
                }
                print(f"✅ {blob_name}: {page_count} pages")

            return result

        except Exception as e:
            error_result = {
                "filename": blob_name,
                "folder": self._extract_folder_path(blob_name),
                "pages": "ERROR",
                "error": str(e),
                "processed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }

            print(f"❌ {blob_name}: ERROR - {e}")
            return error_result

    def _organize_blobs_by_directory(self, pdf_blobs_info):
        """Organize PDF blobs by directory for sequential processing"""
        directories = defaultdict(list)

        for blob_name, blob_hash in pdf_blobs_info:
            directory = self._extract_folder_path(blob_name)
            directories[directory].append((blob_name, blob_hash))

        return directories

    def _process_directory(self, directory_name, directory_blobs):
        """Process all PDFs in a single directory and save state"""
        print(f"\n📁 Processing directory: {directory_name}")
        print(f"   Found {len(directory_blobs)} PDF files")

        # Check which files need processing
        needs_processing = [
            info
            for info in directory_blobs
            if not self._is_already_processed(info[0], info[1])
        ]
        cached_count = len(directory_blobs) - len(needs_processing)

        if cached_count > 0:
            print(f"   📋 {cached_count} files already cached")
        if len(needs_processing) > 0:
            print(f"   🔄 {len(needs_processing)} files need processing")

        directory_results = []

        # Process this directory's PDFs in parallel
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers
        ) as executor:
            futures = [
                executor.submit(self.process_single_pdf, blob_info)
                for blob_info in directory_blobs
            ]

            # Collect results as they complete
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                directory_results.append(result)

        # Update folder stats for this directory
        self._update_folder_stats_from_results(directory_name, directory_results)

        # Save state after completing this directory
        self._save_state()

        # Calculate directory statistics
        valid_pages = [
            r["pages"] for r in directory_results if isinstance(r.get("pages"), int)
        ]
        total_pages = sum(valid_pages)
        error_count = len([r for r in directory_results if r.get("pages") == "ERROR"])

        print(
            f"   ✅ Directory complete: {len(directory_results)} files, {total_pages} pages"
        )
        if error_count > 0:
            print(f"   ⚠️  {error_count} files had errors")

        return directory_results

    def _update_folder_stats_from_results(self, directory_name, directory_results):
        """Update folder statistics from directory results"""
        if directory_name not in self.folder_stats:
            self.folder_stats[directory_name] = {
                "pdf_count": 0,
                "total_pages": 0,
                "processed_pdfs": [],
            }

        for result in directory_results:
            if (
                result["filename"]
                not in self.folder_stats[directory_name]["processed_pdfs"]
            ):
                self.folder_stats[directory_name]["processed_pdfs"].append(
                    result["filename"]
                )
                self.folder_stats[directory_name]["pdf_count"] += 1

                if isinstance(result.get("pages"), int):
                    self.folder_stats[directory_name]["total_pages"] += result["pages"]

    def analyze_container(self):
        """Analyze all PDFs in the container with directory-by-directory processing"""
        print("🔍 Scanning container for PDF files...")

        # Get all blobs with metadata
        all_blobs = list(self.container_client.list_blobs())
        pdf_blobs_info = []

        for blob in all_blobs:
            if blob.name.lower().endswith(".pdf") and not blob.name.endswith("/"):
                blob_hash = self._get_blob_hash(blob.name, blob.last_modified)
                pdf_blobs_info.append((blob.name, blob_hash))

        print(f"📊 Found {len(pdf_blobs_info)} PDF files to analyze")

        if not pdf_blobs_info:
            print("No PDF files found.")
            return [], {}

        # Organize PDFs by directory
        directories = self._organize_blobs_by_directory(pdf_blobs_info)

        print(f"📂 Found {len(directories)} directories to process")

        # Sort directories for consistent processing order
        sorted_directories = sorted(directories.items())

        # Process each directory sequentially, saving state after each
        all_results = []

        # Load cached folder stats first
        self._update_folder_stats_from_cache()

        for i, (directory_name, directory_blobs) in enumerate(sorted_directories, 1):
            print(f"\n🏷️  [{i}/{len(directories)}] Processing: {directory_name}")

            directory_results = self._process_directory(directory_name, directory_blobs)
            all_results.extend(directory_results)

        # Final summary
        valid_results = [r for r in all_results if isinstance(r.get("pages"), int)]
        total_pages = sum(r["pages"] for r in valid_results)
        error_count = len([r for r in all_results if r.get("pages") == "ERROR"])
        cached_count = len([r for r in all_results if r.get("pages") == "CACHED"])

        print("\n🎉 ANALYSIS COMPLETE:")
        print(f"Total PDFs found: {len(pdf_blobs_info)}")
        print(f"Successfully processed: {len(valid_results)}")
        print(f"From cache: {cached_count}")
        print(f"Errors: {error_count}")
        print(f"Total pages across all PDFs: {total_pages}")
        print(f"Directories processed: {len(directories)}")

        self.results = all_results
        return self.results, dict(self.folder_stats)

    def _update_folder_stats_from_cache(self):
        """Update folder statistics with cached results from state file"""
        if "file_details" not in self.state:
            return

        for filename, details in self.state["file_details"].items():
            folder = details.get("folder", "root")
            pages = details.get("pages", 0)

            if folder not in self.folder_stats:
                self.folder_stats[folder] = {
                    "pdf_count": 0,
                    "total_pages": 0,
                    "processed_pdfs": [],
                }

            if filename not in self.folder_stats[folder]["processed_pdfs"]:
                self.folder_stats[folder]["processed_pdfs"].append(filename)
                self.folder_stats[folder]["pdf_count"] += 1
                if isinstance(pages, int):
                    self.folder_stats[folder]["total_pages"] += pages

    def generate_reports(self, results, folder_stats):
        """Generate Excel reports with multiple sheets"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create detailed results DataFrame
        df_details = pd.DataFrame(results)

        # Create folder summary DataFrame
        folder_summary = []
        for folder, stats in folder_stats.items():
            folder_summary.append(
                {
                    "folder_path": folder,
                    "pdf_count": stats["pdf_count"],
                    "total_pages": stats["total_pages"],
                    "avg_pages_per_pdf": stats["total_pages"] / stats["pdf_count"]
                    if stats["pdf_count"] > 0
                    else 0,
                }
            )

        df_folders = pd.DataFrame(folder_summary)
        df_folders = df_folders.sort_values("total_pages", ascending=False)

        # Save to Excel with multiple sheets
        output_filename = f"pdf_analysis_report_{timestamp}.xlsx"

        with pd.ExcelWriter(output_filename, engine="openpyxl") as writer:
            df_details.to_excel(writer, sheet_name="Detailed_Results", index=False)
            df_folders.to_excel(writer, sheet_name="Folder_Summary", index=False)

            # Create a summary statistics sheet
            summary_stats = {
                "Metric": [
                    "Total PDF Files",
                    "Total Pages",
                    "Total Folders",
                    "Average Pages per PDF",
                    "Largest Folder (by page count)",
                    "Largest Folder (by file count)",
                ],
                "Value": [
                    len(results),
                    sum(r["pages"] for r in results if isinstance(r.get("pages"), int)),
                    len(folder_stats),
                    round(
                        sum(
                            r["pages"]
                            for r in results
                            if isinstance(r.get("pages"), int)
                        )
                        / len([r for r in results if isinstance(r.get("pages"), int)]),
                        2,
                    )
                    if results
                    else 0,
                    df_folders.iloc[0]["folder_path"] if len(df_folders) > 0 else "N/A",
                    df_folders.loc[df_folders["pdf_count"].idxmax(), "folder_path"]
                    if len(df_folders) > 0
                    else "N/A",
                ],
            }

            df_summary = pd.DataFrame(summary_stats)
            df_summary.to_excel(writer, sheet_name="Summary_Statistics", index=False)

        print(f"📄 Analysis report saved as: {output_filename}")
        return output_filename


def main():
    """Main execution function"""
    print("🚀 Starting Azure Blob PDF Analysis...")

    # Initialize analyzer
    analyzer = AzureBlobPDFAnalyzer(
        sas_token=AZURE_SAS_TOKEN,
        container_name="bronze",  # Change this to your container name
        max_workers=5,
        state_file="pdf_analysis_state.json",
    )

    # Run analysis
    results, folder_stats = analyzer.analyze_container()

    # Generate reports
    if results:
        _report_file = analyzer.generate_reports(results, folder_stats)

        # Print top folders by page count
        print("\n📁 TOP 10 FOLDERS BY PAGE COUNT:")
        sorted_folders = sorted(
            folder_stats.items(), key=lambda x: x[1]["total_pages"], reverse=True
        )[:10]

        for i, (folder, stats) in enumerate(sorted_folders, 1):
            print(
                f"{i:2d}. {folder}: {stats['pdf_count']} PDFs, {stats['total_pages']} pages"
            )

    print("\n✨ Analysis complete!")


if __name__ == "__main__":
    main()
