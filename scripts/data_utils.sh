#!/bin/bash

function download_files_with_prefix() {
    local prefix="$1"
    local output_dir="$2"
    local account_name="${3:-"tarescoblob"}"
    local container_name="${4:-"bronze"}"

    files=$(az storage blob list \
        --account-name "$account_name" \
        --container-name "$container_name" \
        --prefix "$prefix" \
        --query "[].name" --output json
    )

    echo "${files[@]}" | jq -r '.[]' | while read -r item; do
        file_dir=$(dirname "$item")
        [[ "$file_dir" != "." ]] && mkdir -p "$output_dir/$file_dir"

        az storage blob download \
            --account-name "$account_name" \
            --container-name "$container_name" \
            --name "$item" \
            --file "$output_dir/$item"
    done
}

# Example:
download_files_with_prefix train data