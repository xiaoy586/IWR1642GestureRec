#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_INPUT = os.path.join(
    "Result-RPi-Realtime",
    "raspberrypi4",
    "lstm_gru_realtime_latency_fps.csv",
)


def read_realtime_csv(csv_path):
    records = []
    with open(csv_path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        required = {
            "preprocess_ms",
            "fusion_inference_ms",
            "pipeline_latency_ms",
            "pipeline_fps",
        }
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing columns in {csv_path}: {sorted(missing)}")

        for row in reader:
            try:
                records.append({
                    "timestamp": row.get("timestamp", ""),
                    "gesture": row.get("gesture", ""),
                    "preprocess_ms": float(row["preprocess_ms"]),
                    "fusion_inference_ms": float(row["fusion_inference_ms"]),
                    "pipeline_latency_ms": float(row["pipeline_latency_ms"]),
                    "pipeline_fps": float(row["pipeline_fps"]),
                    "device": row.get("device", ""),
                })
            except (TypeError, ValueError):
                continue

    if not records:
        raise ValueError(f"No valid records found in {csv_path}")
    return records


def summarize(records):
    def arr(key):
        return np.array([r[key] for r in records], dtype=np.float64)

    pipeline = arr("pipeline_latency_ms")
    fusion = arr("fusion_inference_ms")
    preprocess = arr("preprocess_ms")
    fps = arr("pipeline_fps")

    return {
        "samples": len(records),
        "mean_preprocess_ms": float(np.mean(preprocess)),
        "std_preprocess_ms": float(np.std(preprocess)),
        "p95_preprocess_ms": float(np.percentile(preprocess, 95)),
        "mean_fusion_inference_ms": float(np.mean(fusion)),
        "std_fusion_inference_ms": float(np.std(fusion)),
        "p95_fusion_inference_ms": float(np.percentile(fusion, 95)),
        "mean_pipeline_latency_ms": float(np.mean(pipeline)),
        "std_pipeline_latency_ms": float(np.std(pipeline)),
        "p95_pipeline_latency_ms": float(np.percentile(pipeline, 95)),
        "mean_pipeline_fps": float(np.mean(fps)),
        "std_pipeline_fps": float(np.std(fps)),
    }


def save_summary(summary, output_dir):
    path = os.path.join(output_dir, "rpi_realtime_performance_summary.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)
    print(f"Saved: {path}")


def configure_plot_style():
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42


def save_hist(values, xlabel, title, output_path, bins=20):
    values = np.array(values, dtype=np.float64)
    mean = np.mean(values)
    p95 = np.percentile(values, 95)

    plt.figure(figsize=(7.2, 4.8))
    plt.hist(values, bins=bins, color="#4C78A8", edgecolor="black", alpha=0.82)
    plt.axvline(mean, color="#D62728", linestyle="--", linewidth=2.0,
                label=f"Mean = {mean:.2f}")
    plt.axvline(p95, color="#2CA02C", linestyle="--", linewidth=2.0,
                label=f"P95 = {p95:.2f}")
    plt.xlabel(xlabel)
    plt.ylabel("Frequency")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def save_trace(values, ylabel, title, output_path):
    values = np.array(values, dtype=np.float64)
    x = np.arange(1, len(values) + 1)
    mean = np.mean(values)

    plt.figure(figsize=(7.2, 4.8))
    plt.plot(x, values, marker="o", markersize=3.5, linewidth=1.4,
             color="#4C78A8")
    plt.axhline(mean, color="#D62728", linestyle="--", linewidth=2.0,
                label=f"Mean = {mean:.2f}")
    plt.xlabel("Test index")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def save_latency_breakdown(records, output_path):
    preprocess = np.array([r["preprocess_ms"] for r in records], dtype=np.float64)
    fusion = np.array([r["fusion_inference_ms"] for r in records], dtype=np.float64)
    other = np.maximum(
        np.array([r["pipeline_latency_ms"] for r in records], dtype=np.float64)
        - preprocess - fusion,
        0.0,
    )

    labels = ["Preprocessing", "DSTG-Net inference", "Other overhead"]
    means = [np.mean(preprocess), np.mean(fusion), np.mean(other)]

    plt.figure(figsize=(6.8, 4.6))
    bars = plt.bar(labels, means, color=["#4C78A8", "#F58518", "#54A24B"],
                   edgecolor="black", alpha=0.88)
    for bar, value in zip(bars, means):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f"{value:.2f}", ha="center", va="bottom", fontsize=10)
    plt.ylabel("Mean latency (ms)")
    plt.title("End-to-End Latency Breakdown on Raspberry Pi 4B")
    plt.grid(axis="y", linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_all(records, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    pipeline = [r["pipeline_latency_ms"] for r in records]
    fusion = [r["fusion_inference_ms"] for r in records]
    preprocess = [r["preprocess_ms"] for r in records]
    fps = [r["pipeline_fps"] for r in records]

    save_hist(
        pipeline,
        "End-to-end pipeline latency (ms)",
        "Raspberry Pi 4B End-to-End Latency Distribution",
        os.path.join(output_dir, "RPi_Pipeline_Latency_Distribution.pdf"),
    )
    save_hist(
        fusion,
        "DSTG-Net inference latency (ms)",
        "Raspberry Pi 4B DSTG-Net Inference Latency Distribution",
        os.path.join(output_dir, "RPi_Fusion_Inference_Latency_Distribution.pdf"),
    )
    save_hist(
        preprocess,
        "Feature preprocessing latency (ms)",
        "Raspberry Pi 4B Preprocessing Latency Distribution",
        os.path.join(output_dir, "RPi_Preprocessing_Latency_Distribution.pdf"),
    )
    save_trace(
        fps,
        "Pipeline throughput (FPS)",
        "Raspberry Pi 4B End-to-End Throughput",
        os.path.join(output_dir, "RPi_Pipeline_FPS_Trace.pdf"),
    )
    save_latency_breakdown(
        records,
        os.path.join(output_dir, "RPi_Latency_Breakdown.pdf"),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Plot Raspberry Pi GUI end-to-end real-time performance."
    )
    parser.add_argument("--input", default=DEFAULT_INPUT,
                        help="Path to lstm_gru_realtime_latency_fps.csv")
    parser.add_argument("--output-dir", default=None,
                        help="Directory for output PDF figures and summary CSV")
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.dirname(args.input)
    configure_plot_style()
    records = read_realtime_csv(args.input)
    summary = summarize(records)
    save_summary(summary, output_dir)
    plot_all(records, output_dir)

    print("\nSummary:")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
