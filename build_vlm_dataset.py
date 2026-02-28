# -*- coding: utf-8 -*-
"""
Build a unified VLM training dataset from MIMIC-IV ECG data.

This script:
  1. Merges all 3 CSVs (record_list, machine_measurements, waveform_note_links)
     into a single comprehensive CSV, joined on study_id.
  2. Converts each .dat waveform file to per-lead ECG images + rhythm strip.
  3. Adds image path columns to every row so each record is self-contained.

Output:  vlm_dataset.csv  (one row per ECG study with ALL metadata + image paths)
         ecg_images_all/   (folder of per-study image directories)
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import wfdb
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import time

plt.switch_backend("Agg")  # headless backend

# ─── CONFIG ─────────────────────────────────────────────────────────────────────
DPI = 150
BASE_DIR = "mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0"
OUTPUT_DIR = "ecg_images_all"
OUTPUT_CSV = "vlm_dataset.csv"

# Set to None to process ALL records, or an int to limit (for testing)
LIMIT_RECORDS = None  # Set to an int (e.g. 10) for quick testing

# Paths to the three source CSVs
RECORD_LIST_CSV = os.path.join(BASE_DIR, "record_list.csv")
MACHINE_MEAS_CSV = os.path.join(BASE_DIR, "machine_measurements.csv")
WAVEFORM_LINKS_CSV = os.path.join(BASE_DIR, "waveform_note_links.csv")

# ─── STEP 1: MERGE ALL CSVs ────────────────────────────────────────────────────

def merge_all_csvs():
    """
    Merge record_list + machine_measurements + waveform_note_links
    on study_id into a single DataFrame.

    Join strategy:
      - record_list ← LEFT JOIN → machine_measurements   (1:1 on study_id)
      - result      ← LEFT JOIN → waveform_note_links    (many records have no note)

    Overlapping columns (subject_id, ecg_time) are kept from record_list;
    suffixed copies from the other CSVs are dropped.
    """
    print("📂 Loading record_list.csv ...")
    df_records = pd.read_csv(RECORD_LIST_CSV, low_memory=False)
    print(f"   {len(df_records):,} rows,  columns: {list(df_records.columns)}")

    print("📂 Loading machine_measurements.csv ...")
    df_meas = pd.read_csv(MACHINE_MEAS_CSV, low_memory=False)
    print(f"   {len(df_meas):,} rows,  columns: {list(df_meas.columns)}")

    print("📂 Loading waveform_note_links.csv ...")
    df_links = pd.read_csv(WAVEFORM_LINKS_CSV, low_memory=False)
    print(f"   {len(df_links):,} rows,  columns: {list(df_links.columns)}")

    # De-duplicate waveform_note_links on study_id (7 study_ids have >1 note)
    # Keep the first note for each study.
    df_links = df_links.drop_duplicates(subset="study_id", keep="first")

    # ── Merge 1: record_list + machine_measurements (1:1 on study_id) ──
    # Both share subject_id and ecg_time; keep record_list's version.
    df = df_records.merge(
        df_meas,
        on="study_id",
        how="left",
        suffixes=("", "_mm"),
    )
    # Drop the duplicated columns from machine_measurements
    drop_cols = [c for c in df.columns if c.endswith("_mm")]
    df.drop(columns=drop_cols, inplace=True)

    # ── Merge 2: + waveform_note_links (left join; not every ECG has a note) ──
    df = df.merge(
        df_links,
        on="study_id",
        how="left",
        suffixes=("", "_wn"),
    )
    drop_cols = [c for c in df.columns if c.endswith("_wn")]
    df.drop(columns=drop_cols, inplace=True)

    print(f"\n✅ Merged dataset: {len(df):,} rows × {len(df.columns)} columns")
    print(f"   Columns: {list(df.columns)}\n")
    return df


# ─── STEP 2: ECG IMAGE GENERATION (reused from original script) ────────────────

def draw_pixel_ecg_grid(ax, fig, block_px=5.6):
    """Draw ECG paper background grid."""
    try:
        fig.canvas.draw()
        ax.set_facecolor((1, 1, 1))
        width, height = fig.canvas.get_width_height()
        x_lines = np.arange(0, width, block_px)
        y_lines = np.arange(0, height, block_px)
        trans = fig.transFigure.inverted()
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        for x in x_lines:
            x_fig = trans.transform((x, 0))[0]
            ax.axvline(
                x=x0 + (x_fig * (x1 - x0)),
                color=(1, 0.7, 0.7, 0.5),
                linewidth=0.5,
                zorder=0,
            )
        for y in y_lines:
            y_fig = trans.transform((0, y))[1]
            ax.axhline(
                y=y0 + (y_fig * (y1 - y0)),
                color=(1, 0.7, 0.7, 0.5),
                linewidth=0.5,
                zorder=0,
            )
    except Exception:
        pass


# Expected lead names for a standard 12-lead ECG
STANDARD_LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def check_existing_images(record_out_dir):
    """If images already exist for this record, return their paths. Otherwise None."""
    if not os.path.isdir(record_out_dir):
        return None
    existing = set(os.listdir(record_out_dir))
    # Need at least 1 lead image to consider it done
    lead_images = {}
    for lead in STANDARD_LEADS:
        fname = f"{lead}.png"
        if fname in existing:
            lead_images[f"lead_{lead}_path"] = os.path.join(record_out_dir, fname)
    if not lead_images:
        return None  # no leads found → must regenerate
    # Check rhythm strip
    rhythm_file = ""
    rhythm_name = "rhythm_II.png"
    if rhythm_name in existing:
        rhythm_file = os.path.join(record_out_dir, rhythm_name)
    return {"lead_images": lead_images, "rhythm_file": rhythm_file}


def process_record(args):
    """Convert a single .dat waveform to per-lead PNG images.

    Returns a dict with all original columns + image path columns,
    or None on failure.  Only called for records that DON'T already have images.
    """
    idx, row_dict = args
    path_val = row_dict["path"]

    full_path = os.path.normpath(os.path.join(BASE_DIR, str(path_val)))
    study_id = str(row_dict["study_id"])
    record_out_dir = os.path.join(OUTPUT_DIR, study_id)

    # ── Generate images from scratch ──
    os.makedirs(record_out_dir, exist_ok=True)

    try:
        record = wfdb.rdrecord(full_path)
        signals = record.p_signal
        fs = record.fs
        lead_names = record.sig_name
        n_leads = record.n_sig

        if signals is None or len(signals) == 0:
            return None

        short_samples = min(int(4 * fs), len(signals))
        long_samples = min(int(10 * fs), len(signals))
        rhythm_idx = lead_names.index("II") if "II" in lead_names else 0

        lead_images = {}

        # ── 12-lead images ──
        for i in range(min(12, n_leads)):
            try:
                fig, ax = plt.subplots(figsize=(5.5, 0.9), dpi=DPI)
                y = signals[:short_samples, i]
                x = np.arange(len(y)) / fs

                if np.all(np.isnan(y)):
                    plt.close(fig)
                    continue

                ax.plot(x, y, color="black", linewidth=1)
                ax.axis("off")

                y_min, y_max = np.nanmin(y), np.nanmax(y)
                if np.isnan(y_min) or np.isnan(y_max) or y_min == y_max:
                    plt.close(fig)
                    continue

                y_center = (y_max + y_min) / 2
                y_range = y_max - y_min
                scale_factor = 2.2
                ylim = (
                    y_center - scale_factor * y_range / 2,
                    y_center + scale_factor * y_range / 2,
                )
                ax.set_xlim(x[0], x[-1])
                ax.set_ylim(ylim)
                draw_pixel_ecg_grid(ax, fig, block_px=5.6)

                lead_filename = f"{lead_names[i]}.png"
                lead_file = os.path.join(record_out_dir, lead_filename)
                plt.savefig(lead_file, dpi=DPI, bbox_inches="tight", pad_inches=0)
                plt.close(fig)
                lead_images[f"lead_{lead_names[i]}_path"] = lead_file
            except Exception:
                try:
                    plt.close("all")
                except Exception:
                    pass

        # ── Rhythm strip ──
        rhythm_file = ""
        try:
            fig, ax = plt.subplots(figsize=(9.5, 1), dpi=DPI)
            y = signals[:long_samples, rhythm_idx]
            x = np.arange(len(y)) / fs
            if not (np.all(np.isnan(y)) or len(y) == 0):
                ax.plot(x, y, color="black", linewidth=1)
                ax.axis("off")
                y_min, y_max = np.nanmin(y), np.nanmax(y)
                if not (np.isnan(y_min) or np.isnan(y_max) or y_min == y_max):
                    y_center = (y_max + y_min) / 2
                    y_range = y_max - y_min
                    scale_factor = 2.0
                    ylim = (
                        y_center - scale_factor * y_range / 2,
                        y_center + scale_factor * y_range / 2,
                    )
                    ax.set_xlim(x[0], x[-1])
                    ax.set_ylim(ylim)
                    draw_pixel_ecg_grid(ax, fig, block_px=5.6)
                    rhythm_filename = f"rhythm_{lead_names[rhythm_idx]}.png"
                    rhythm_file = os.path.join(record_out_dir, rhythm_filename)
                    plt.savefig(
                        rhythm_file, dpi=DPI, bbox_inches="tight", pad_inches=0
                    )
            plt.close(fig)
        except Exception:
            try:
                plt.close("all")
            except Exception:
                pass

        # ── Assemble result row ──
        result = dict(row_dict)
        result["image_dir"] = record_out_dir
        result["rhythm_image_path"] = rhythm_file
        result["num_leads_generated"] = len(lead_images)
        result.update(lead_images)
        return result

    except Exception:
        try:
            plt.close("all")
        except Exception:
            pass
        return None


# ─── STEP 3: COMBINE REPORT COLUMNS INTO ONE (for VLM readability) ─────────────

def combine_reports(df):
    """
    Concatenate report_0 … report_17 into a single 'report' text column.
    Drops the individual report_N columns afterward.
    """
    report_cols = [c for c in df.columns if c.startswith("report_") and c[7:].isdigit()]
    report_cols.sort(key=lambda c: int(c.split("_")[1]))  # sort by number

    def _join(row):
        parts = [str(row[c]).strip() for c in report_cols if pd.notna(row[c]) and str(row[c]).strip()]
        return " | ".join(parts) if parts else ""

    df["report"] = df.apply(_join, axis=1)
    df.drop(columns=report_cols, inplace=True)
    return df


# ─── CHECKPOINT HELPER ───────────────────────────────────────────────────────────

CHECKPOINT_CSV = "vlm_dataset_checkpoint.csv"

def _save_checkpoint(results_list, done, total):
    """Save a partial CSV so progress survives crashes."""
    try:
        ckpt_df = pd.DataFrame(results_list)
        ckpt_df = combine_reports(ckpt_df)
        ckpt_df.to_csv(CHECKPOINT_CSV, index=False)
        print(f"  💾 Checkpoint saved ({done:,}/{total:,}): {CHECKPOINT_CSV}")
        sys.stdout.flush()
    except Exception as e:
        print(f"  ⚠️ Checkpoint save failed: {e}")


# ─── RESUME HELPERS ──────────────────────────────────────────────────────────────

def build_row_for_existing(row_dict, record_out_dir):
    """Build a result dict for a record whose images already exist on disk.
    Pure string ops — no filesystem access per record.
    """
    result = dict(row_dict)
    result["image_dir"] = record_out_dir
    result["rhythm_image_path"] = os.path.join(record_out_dir, "rhythm_II.png")
    count = 0
    for lead in STANDARD_LEADS:
        result[f"lead_{lead}_path"] = os.path.join(record_out_dir, f"{lead}.png")
        count += 1
    result["num_leads_generated"] = count
    return result


def reorder_and_save(df_out, output_path):
    """Combine reports, reorder columns, write CSV."""
    df_out = combine_reports(df_out)

    id_cols = ["subject_id", "study_id", "file_name", "ecg_time", "path"]
    meas_cols = [
        "cart_id", "bandwidth", "filtering",
        "rr_interval", "p_onset", "p_end",
        "qrs_onset", "qrs_end", "t_end",
        "p_axis", "qrs_axis", "t_axis",
    ]
    note_cols = ["waveform_path", "note_id", "note_seq", "charttime"]
    report_col = ["report"]
    image_cols = ["image_dir", "rhythm_image_path", "num_leads_generated"]
    lead_path_cols = sorted([c for c in df_out.columns if c.startswith("lead_") and c.endswith("_path")])

    ordered = []
    for c in id_cols + meas_cols + note_cols + report_col + image_cols + lead_path_cols:
        if c in df_out.columns:
            ordered.append(c)
    remaining = [c for c in df_out.columns if c not in ordered]
    ordered.extend(remaining)
    df_out = df_out[ordered]

    df_out.to_csv(output_path, index=False)
    return df_out


# ─── MAIN ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        from multiprocessing import freeze_support
        freeze_support()
    except Exception:
        pass

    t0 = time.time()

    # ── 1. Merge CSVs ──
    df = merge_all_csvs()

    # Filter to records with a valid path
    df["path"] = df["path"].fillna("")
    df = df[df["path"].str.len() > 0].reset_index(drop=True)

    if LIMIT_RECORDS is not None:
        df = df.head(LIMIT_RECORDS)
        print(f"⚠️  Testing mode: limiting to {LIMIT_RECORDS} records\n")

    total = len(df)
    print(f"🔢 {total:,} total records.\n")

    if total == 0:
        print("⚠️  No records found!")
        sys.exit(0)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── 2. FAST RESUME: scan output dir once to find already-done study_ids ──
    print("📂 Scanning existing images ...")
    existing_folders = set(os.listdir(OUTPUT_DIR))
    print(f"   Found {len(existing_folders):,} existing image folders.\n")

    # Split into done vs. todo
    done_mask = df["study_id"].astype(str).isin(existing_folders)
    df_done = df[done_mask]
    df_todo = df[~done_mask]

    print(f"   ✅ Already done: {len(df_done):,}")
    print(f"   🔧 Still need:  {len(df_todo):,}\n")

    # ── 3a. Build result rows for already-done records (instant, no I/O) ──
    print("⚡ Building rows for existing records ...")
    results_list = []
    for _, row in df_done.iterrows():
        study_id = str(row["study_id"])
        record_out_dir = os.path.join(OUTPUT_DIR, study_id)
        results_list.append(build_row_for_existing(row.to_dict(), record_out_dir))
    print(f"   {len(results_list):,} rows built in {time.time()-t0:.1f}s\n")

    # ── 3b. Process remaining records in parallel ──
    n_todo = len(df_todo)
    if n_todo > 0:
        tasks = [(idx, row.to_dict()) for idx, row in df_todo.iterrows()]
        max_workers = max(1, (os.cpu_count() or 4) - 1)
        print(f"🚀 Converting {n_todo:,} remaining records with {max_workers} workers ...")
        sys.stdout.flush()

        completed = 0
        failed = 0
        t_convert = time.time()

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_record, t): t for t in tasks}

            for fut in as_completed(futures):
                try:
                    result = fut.result()
                    if result is not None:
                        results_list.append(result)
                        completed += 1
                    else:
                        failed += 1

                    done = completed + failed
                    if done % 500 == 0 or done == n_todo:
                        elapsed = time.time() - t_convert
                        rate = done / elapsed if elapsed > 0 else 0
                        eta = (n_todo - done) / rate if rate > 0 else 0
                        print(
                            f"  [{done:,}/{n_todo:,}] "
                            f"✅ {completed:,}  ⚠️ {failed:,}  "
                            f"({rate:.0f} rec/s, ETA {eta/60:.1f} min)"
                        )
                        sys.stdout.flush()

                    # Checkpoint every 50k new records
                    if done % 50_000 == 0 and results_list:
                        _save_checkpoint(results_list, len(df_done) + done, total)

                except Exception as e:
                    failed += 1
                    print(f"Worker error: {str(e)[:120]}")
                    sys.stdout.flush()

        print(f"\n   New records: ✅ {completed:,} converted, ⚠️ {failed:,} failed")
    else:
        completed = 0
        failed = 0
        print("✅ All images already exist! Just building the CSV.\n")

    # ── 4. Build & save final CSV ──
    if not results_list:
        print("❌ No records processed successfully!")
        sys.exit(1)

    print(f"💾 Building final CSV ({len(results_list):,} rows) ...")
    df_out = pd.DataFrame(results_list)
    df_out = reorder_and_save(df_out, OUTPUT_CSV)

    if os.path.exists(CHECKPOINT_CSV):
        os.remove(CHECKPOINT_CSV)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"🎉 Done in {elapsed/60:.1f} minutes")
    print(f"   Already had images: {len(df_done):,}")
    print(f"   Newly converted:    {completed:,}")
    print(f"   Failed/skipped:     {failed:,}")
    print(f"   Output CSV: {OUTPUT_CSV}  ({len(df_out):,} rows × {len(df_out.columns)} cols)")
    print(f"   Columns: {list(df_out.columns)}")
    print(f"{'='*60}")
