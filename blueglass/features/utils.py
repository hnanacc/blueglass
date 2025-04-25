import traceback
import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime
from tqdm import tqdm
from blueglass.configs import BLUEGLASSConf
from concurrent.futures import ProcessPoolExecutor, as_completed

ROOT_DIR="BlueLens/features_datasets"
NEW_BLUELENS_DIR="BlueLensPos"

def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")


def find_parquet_parts(root_dir):
    part_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.startswith("part.") and filename.split(".")[-1].isdigit():
                part_files.append(os.path.join(dirpath, filename))
    return part_files


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

        new_part_file = part_file.replace("BlueLens", NEW_BLUELENS_DIR)
        os.makedirs(os.path.dirname(new_part_file), exist_ok=True)

        writer = pq.ParquetWriter(new_part_file, schema)
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
    part_files = find_parquet_parts(ROOT_DIR)
    total = len(part_files)
    log(f"Found {total} files to process under: {ROOT_DIR}")

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


if __name__ == "__main__":
    conf = BLUEGLASSConf()
    filter_and_rewrite_BlueLens(
        root_dir="BlueLens/features_datasets",
        conf=conf,
        num_workers=75
    )
