import os
import traceback
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed


MAX_ROWS_PER_FILE = 500_000
OLD_ROOT_DIR = "BlueLens/features_datasets"
NEW_ROOT_DIR = "BlueLensPos/features_datasets"


def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

def build_arrow_schema(conf, features_dim: int):
    return pa.schema(
        [
            ("image_id", pa.int32()),
            ("infer_id", pa.int32()),
            ("heads_id", pa.int32()),
            ("token_id", pa.int32()),
            ("filename", pa.string()),
            ("pred_box", pa.list_(pa.float32(), 4)),
            ("pred_cls", pa.int16()),
            ("pred_scr", pa.float32()),
            ("pred_ious", pa.float32()),
            ("conf_msk", pa.bool_()),
            ("token_ch", pa.string()),
            ("features", pa.list_(pa.float32(), features_dim)),
        ]
    )

def find_all_parquet_parts(root_dir):
    part_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.startswith("part.") and filename.split(".")[-1].isdigit():
                part_files.append(os.path.join(dirpath, filename))
    return part_files




def _process_single_file(args):
    part_file, conf = args
    try:
        df = pd.read_parquet(part_file)
        """
        Add your custom filter logic here.
        """
        df = df[df["conf_msk"] == True]

        if df.empty or df["features"].empty:
            return (part_file, True, "Empty after filtering")

        schema = build_arrow_schema(conf, len(df["features"].iloc[0]))
        table = pa.Table.from_pandas(df, schema=schema)

        new_part_file = part_file.replace("BlueLens", NEW_ROOT_DIR)
        os.makedirs(os.path.dirname(new_part_file), exist_ok=True)

        writer = pq.ParquetWriter(new_part_file, schema, compression="zstd", compression_level=7)
        try:
            writer.write(table)
        finally:
            writer.close()

        return (part_file, True, None)

    except Exception:
        return (part_file, False, traceback.format_exc())


def filter_and_rewrite_BlueLens(conf, num_workers=8, root_dir="BlueLens/features_datasets"):
    """
    Filters Parquet part files by conf_msk == True and rewrites them to a new location.
    
    Parameters:
        root_dir (str): Path to the root folder containing Parquet `part.*` files.
        conf (BLUEGLASSConf): Configuration object used to define feature dimensions.
        num_workers (int): Number of processes to use.
    """
    part_files = find_all_parquet_parts(OLD_ROOT_DIR)
    total = len(part_files)
    log(f"Found {total} files to process under: {OLD_ROOT_DIR}")

    success, failed = 0, 0

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(_process_single_file, (file, conf)): file
            for file in part_files
        }
        for future in tqdm(as_completed(futures), total=total, desc="Processing files", unit="file"):
            part_file, is_success, error = future.result()
            if is_success:
                success += 1
            else:
                failed += 1
                tqdm.write(f"❌ Failed: {part_file} | Error: {error}")

    log(f"✅ Finished. Total: {total}, Success: {success}, Failed: {failed}")

def find_folders_with_parts(root_dir):
    folders = []
    for dirpath, _, filenames in os.walk(root_dir):
        part_files = [f for f in filenames if f.startswith("part.") and f.split(".")[-1].isdigit()]
        if part_files:
            folders.append(dirpath)
    return folders

def build_arrow_schema(conf, features_dim: int):
    return pa.schema(
        [
            ("image_id", pa.int32()),
            ("infer_id", pa.int32()),
            ("heads_id", pa.int32()),
            ("token_id", pa.int32()),
            ("filename", pa.string()),
            ("pred_box", pa.list_(pa.float32(), 4)),
            ("pred_cls", pa.int16()),
            ("pred_scr", pa.float32()),
            ("pred_ious", pa.float32()),
            ("conf_msk", pa.bool_()),
            ("token_ch", pa.string()),
            ("features", pa.list_(pa.float32(), features_dim)),
        ]
    )

def process_single_folder(args):
    folder, conf = args
    part_files = sorted(
        [os.path.join(folder, f) for f in os.listdir(folder) if f.startswith("part.") and f.split(".")[-1].isdigit()]
    )

    if not part_files:
        return

    log(f"📁 Processing folder: {folder} with {len(part_files)} part files")
    buffer = None
    new_part_counter = 0

    new_folder = folder
    os.makedirs(new_folder, exist_ok=True)

    for part_file in part_files:
        try:
            df = pd.read_parquet(part_file)

            if df.empty:
                continue

            if buffer is None:
                buffer = df
            else:
                buffer = pd.concat([buffer, df], ignore_index=True)

            while len(buffer) >= MAX_ROWS_PER_FILE:
                current_batch = buffer.iloc[:MAX_ROWS_PER_FILE]
                buffer = buffer.iloc[MAX_ROWS_PER_FILE:]

                schema = build_arrow_schema(conf, len(current_batch["features"].iloc[0]))
                new_part_file = os.path.join(new_folder, f"part.{new_part_counter}")
                table = pa.Table.from_pandas(current_batch, schema=schema)

                pq.write_table(table, new_part_file, compression="zstd", compression_level=7)
                log(f"✅ Wrote {MAX_ROWS_PER_FILE} rows to {new_part_file}")
                new_part_counter += 1

        except Exception:
            log(f"❌ Failed at {part_file}")
            traceback.print_exc()

    if buffer is not None and len(buffer) > 0:
        schema = build_arrow_schema(conf, len(buffer["features"].iloc[0]))
        new_part_file = os.path.join(new_folder, f"part.{new_part_counter}")
        table = pa.Table.from_pandas(buffer, schema=schema)

        pq.write_table(table, new_part_file, compression="zstd", compression_level=7)
        log(f"✅ Final write {len(buffer)} rows to {new_part_file}")

def consolidate(conf, root_dir=NEW_ROOT_DIR, num_workers=8):
    folders = find_folders_with_parts(root_dir)
    log(f"Found {len(folders)} folders with part files.")

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(process_single_folder, (folder, conf)): folder
            for folder in folders
        }
        for _ in tqdm(as_completed(futures), total=len(futures), desc="Processing folders"):
            pass

if __name__ == "__main__":
    from blueglass.configs import BLUEGLASSConf
    conf = BLUEGLASSConf()
    num_workers = 75
    filter_and_rewrite_BlueLens(
        root_dir=OLD_ROOT_DIR,
        conf=conf,
        num_workers=num_workers
    )
    consolidate(conf, num_workers=num_workers)  # or 32 if you have CPUs
