# Example usage:
# python sample_pages.py data.csv 50000 --output train_samples.csv --strategy proportional
# python sample_pages.py data.csv 1000 --output eval_samples.csv --strategy balanced --coverage-ratio 0.2

import pandas as pd
import numpy as np
from typing import Dict, List
import argparse


def systematic_sample_pages(
    total_pages: int, num_samples: int, start_offset: int = 1
) -> List[int]:
    """
    Generate systematic sample of page numbers from a document.

    Args:
        total_pages: Total number of pages in document
        num_samples: Number of pages to sample
        start_offset: Starting page number (default 1)

    Returns:
        List of page numbers to sample
    """
    if num_samples >= total_pages:
        return list(range(start_offset, start_offset + total_pages))

    # Calculate interval
    interval = total_pages / num_samples

    # Generate systematic sample with random start within first interval
    random_start = np.random.uniform(0, interval)
    pages = []

    for i in range(num_samples):
        page_num = int(random_start + i * interval) + start_offset
        # Ensure we don't exceed total pages
        page_num = min(page_num, start_offset + total_pages - 1)
        pages.append(page_num)

    return sorted(list(set(pages)))  # Remove duplicates and sort


def calculate_folder_allocation(
    df: pd.DataFrame, total_sample_size: int, allocation_strategy: str = "proportional"
) -> Dict[str, int]:
    """
    Calculate how many pages to allocate to each folder.

    Args:
        df: DataFrame with filename, folder, pages columns
        total_sample_size: Total number of pages to sample
        allocation_strategy: 'proportional' or 'balanced'

    Returns:
        Dictionary mapping folder to allocated page count
    """
    folder_stats = df.groupby("folder")["pages"].agg(["sum", "count"]).reset_index()
    folder_stats.columns = ["folder", "total_pages", "num_docs"]

    if allocation_strategy == "proportional":
        # Allocate proportional to folder size
        total_pages = df["pages"].sum()
        folder_stats["allocation"] = (
            (folder_stats["total_pages"] / total_pages * total_sample_size)
            .round()
            .astype(int)
        )

    elif allocation_strategy == "balanced":
        # Give each folder a base allocation plus bonus based on size
        num_folders = len(folder_stats)
        base_per_folder = total_sample_size // (num_folders * 2)  # 50% for base
        remaining = total_sample_size - (base_per_folder * num_folders)

        # Distribute remaining proportionally
        total_pages = df["pages"].sum()
        bonus = (
            (folder_stats["total_pages"] / total_pages * remaining).round().astype(int)
        )
        folder_stats["allocation"] = base_per_folder + bonus

    # Ensure we don't exceed total sample size
    current_total = folder_stats["allocation"].sum()
    if current_total != total_sample_size:
        # Adjust the largest folder allocation
        diff = total_sample_size - current_total
        largest_idx = folder_stats["allocation"].idxmax()
        folder_stats.loc[largest_idx, "allocation"] += diff

    return dict(zip(folder_stats["folder"], folder_stats["allocation"]))


def sample_pages_from_folder(
    folder_df: pd.DataFrame, target_pages: int, coverage_ratio: float = 0.1
) -> List[Dict]:
    """
    Sample pages from documents in a single folder using hybrid approach.

    Args:
        folder_df: DataFrame with documents from one folder
        target_pages: Number of pages to sample from this folder
        coverage_ratio: Fraction of allocation for document coverage phase

    Returns:
        List of sampling records with filename, folder, page_number
    """
    if target_pages <= 0:
        return []

    samples = []
    total_pages_in_folder = folder_df["pages"].sum()
    num_docs = len(folder_df)

    # Phase 1: Document Coverage (ensure every doc gets at least 1 page)
    coverage_budget = max(1, int(target_pages * coverage_ratio))
    min_pages_per_doc = max(1, coverage_budget // num_docs)

    coverage_used = 0
    for _, doc in folder_df.iterrows():
        pages_to_sample = min(
            min_pages_per_doc, doc["pages"], coverage_budget - coverage_used
        )
        if pages_to_sample > 0:
            sampled_pages = systematic_sample_pages(doc["pages"], pages_to_sample)
            for page_num in sampled_pages:
                samples.append(
                    {
                        "filename": doc["filename"],
                        "folder": doc["folder"],
                        "page_number": page_num,
                        "phase": "coverage",
                    }
                )
            coverage_used += pages_to_sample

    # Phase 2: Proportional Sampling (distribute remaining pages proportionally)
    remaining_budget = target_pages - coverage_used

    if remaining_budget > 0:
        for _, doc in folder_df.iterrows():
            # Calculate proportional allocation
            doc_proportion = doc["pages"] / total_pages_in_folder
            proportional_pages = int(remaining_budget * doc_proportion)

            # Avoid over-sampling small documents
            max_additional = doc["pages"] - min_pages_per_doc
            proportional_pages = min(proportional_pages, max_additional)

            if proportional_pages > 0:
                # Sample additional pages, avoiding already sampled ones
                existing_pages = {
                    s["page_number"]
                    for s in samples
                    if s["filename"] == doc["filename"]
                }

                sampled_pages = systematic_sample_pages(
                    doc["pages"], min_pages_per_doc + proportional_pages
                )
                new_pages = [p for p in sampled_pages if p not in existing_pages]

                for page_num in new_pages:
                    samples.append(
                        {
                            "filename": doc["filename"],
                            "folder": doc["folder"],
                            "page_number": page_num,
                            "phase": "proportional",
                        }
                    )

    return samples


def create_page_samples(
    csv_file: str,
    sample_count: int,
    allocation_strategy: str = "proportional",
    coverage_ratio: float = 0.1,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Create stratified page samples from PDF collection.

    Args:
        csv_file: Path to CSV file with filename, folder, pages columns
        sample_count: Total number of pages to sample
        allocation_strategy: 'proportional' or 'balanced' folder allocation
        coverage_ratio: Fraction of allocation for document coverage
        random_seed: Random seed for reproducibility

    Returns:
        DataFrame with sampled pages
    """
    # Set random seed for reproducibility
    np.random.seed(random_seed)

    # Load the CSV
    print(f"Loading CSV file: {csv_file}")
    df = pd.read_csv(csv_file)

    # Validate required columns
    required_cols = ["filename", "folder", "pages"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    print(f"Loaded {len(df):,} documents with {df['pages'].sum():,} total pages")
    print(f"Target sample: {sample_count:,} pages")

    # Calculate folder allocations
    print(f"Calculating folder allocations using '{allocation_strategy}' strategy...")
    folder_allocations = calculate_folder_allocation(
        df, sample_count, allocation_strategy
    )

    # Display allocation summary
    print("\nFolder allocations:")
    for folder, allocation in sorted(folder_allocations.items()):
        folder_pages = df[df["folder"] == folder]["pages"].sum()
        folder_docs = len(df[df["folder"] == folder])
        pct = allocation / sample_count * 100
        print(
            f"  {folder}: {allocation:,} pages ({pct:.1f}%) from {folder_docs} docs ({folder_pages:,} total pages)"
        )

    # Sample pages from each folder
    all_samples = []
    total_sampled = 0

    print(f"\nSampling pages with coverage ratio: {coverage_ratio:.1%}")
    for folder, target_pages in folder_allocations.items():
        if target_pages > 0:
            folder_df = df[df["folder"] == folder].copy()
            folder_samples = sample_pages_from_folder(
                folder_df, target_pages, coverage_ratio
            )
            all_samples.extend(folder_samples)
            total_sampled += len(folder_samples)
            print(f"  {folder}: sampled {len(folder_samples):,} pages")

    # Create result DataFrame
    result_df = pd.DataFrame(all_samples)

    print("\nSampling complete!")
    print(f"Total pages sampled: {len(result_df):,}")
    print(f"Documents covered: {result_df['filename'].nunique():,} / {len(df):,}")
    print(
        f"Folders covered: {result_df['folder'].nunique():,} / {df['folder'].nunique():,}"
    )

    return result_df


def main():
    parser = argparse.ArgumentParser(description="Sample pages from PDF collection")
    parser.add_argument(
        "csv_file", help="Path to CSV file with filename, folder, pages columns"
    )
    parser.add_argument("sample_count", type=int, help="Number of pages to sample")
    parser.add_argument(
        "--output", "-o", help="Output CSV file path (default: sampled_pages.csv)"
    )
    parser.add_argument(
        "--strategy",
        choices=["proportional", "balanced"],
        default="proportional",
        help="Folder allocation strategy",
    )
    parser.add_argument(
        "--coverage-ratio",
        type=float,
        default=0.1,
        help="Fraction of allocation for document coverage (default: 0.1)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    # Create samples
    result_df = create_page_samples(
        csv_file=args.csv_file,
        sample_count=args.sample_count,
        allocation_strategy=args.strategy,
        coverage_ratio=args.coverage_ratio,
        random_seed=args.seed,
    )

    # Save results
    output_file = args.output or "sampled_pages.csv"
    result_df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")

    # Display sample of results
    print("\nSample of results:")
    print(result_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
