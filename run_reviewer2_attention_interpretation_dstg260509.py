#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Attention-weight interpretation for Reviewer 2 using the submitted 260509 DSTG-Net.

The script does not retrain the model. It loads the DSTG-Net checkpoint and
shared public-dataset split produced by run_reviewer2_sota_dstg260509.py, then
exports attention-weight visualizations and CSV summaries.

Default:
  python3 run_reviewer2_attention_interpretation_dstg260509.py
"""

import argparse
import os
import shutil

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

import run_reviewer2_sota_dstg260509 as sota

try:
    import scienceplots
    plt.style.use(["science", "no-latex"])
except ImportError:
    plt.style.use("seaborn-v0_8-paper")


DEFAULT_CHECKPOINT = os.path.join(
    "Result-Reviewer2-SoTA-DSTG260509",
    "best_model_DSTG_Net.pth",
)
DEFAULT_SPLIT = os.path.join(
    "Result-Reviewer2-SoTA-DSTG260509",
    "shared_split_indices.npz",
)
DEFAULT_RESULT_DIR = "Result-Reviewer2-AttentionInterpretation-DSTG260509"
DEFAULT_PAPER_FIG = os.path.join(
    "electronics-4459693",
    "figures",
    "fig10_attention_interpretation.pdf",
)
DEFAULT_PAPER_FIG_DIR = os.path.join("electronics-4459693", "figures")
ATTENTION_HEATMAP_SIZE = (7.2, 3.2)
ATTENTION_PANEL_SIZE = (5.2, 3.6)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def configure_plot_style():
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.serif": ["Times New Roman"],
        "font.sans-serif": ["Times New Roman"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "Times New Roman",
        "mathtext.it": "Times New Roman:italic",
        "mathtext.bf": "Times New Roman:bold",
        "axes.unicode_minus": False,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def load_checkpoint_state(path):
    checkpoint = torch.load(path, map_location=sota.DEVICE)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"], checkpoint
    return checkpoint, {}


@torch.no_grad()
def forward_with_attention(model, x_lstm, x_gru):
    h0 = torch.zeros(model.lstm_layers, x_lstm.size(0), model.lstm_hidden, device=x_lstm.device)
    c0 = torch.zeros(model.lstm_layers, x_lstm.size(0), model.lstm_hidden, device=x_lstm.device)
    out_lstm, _ = model.lstm(x_lstm, (h0, c0))
    feat_lstm = model.lstm_fc(out_lstm.reshape(out_lstm.shape[0], -1))

    batch_size, seq_len, _ = x_gru.shape
    x_g_proj = model.gru_proj(x_gru.reshape(-1, x_gru.size(-1))).reshape(batch_size, seq_len, -1)
    gru_out, _ = model.gru(x_g_proj)
    scores = model.attention(gru_out).squeeze(-1)
    attn = torch.softmax(scores, dim=1)
    feat_gru = model.gru_fc(torch.sum(attn.unsqueeze(-1) * gru_out, dim=1))
    logits = model.fusion_classifier(torch.cat([feat_lstm, feat_gru], dim=1))
    return logits, attn


def load_dataset(data_path, split_path, max_samples_per_class=None):
    samples, labels = sota.collect_samples(data_path, max_samples_per_class=max_samples_per_class)
    dataset = sota.SotaRadarDataset(samples)
    if os.path.exists(split_path):
        split = np.load(split_path)
        train_idx = split["train_idx"]
        val_idx = split["val_idx"]
        if max(train_idx.max(initial=0), val_idx.max(initial=0)) >= len(dataset):
            train_idx = None
            val_idx = None
    else:
        train_idx = None
        val_idx = None

    if train_idx is None or val_idx is None:
        from sklearn.model_selection import train_test_split

        indices = np.arange(len(dataset))
        train_idx, val_idx = train_test_split(
            indices,
            test_size=0.2,
            stratify=dataset.y,
            random_state=sota.SEED,
        )
    norm_dataset = sota.TrainSetNormalizedSotaDataset(dataset, train_idx)
    return dataset, norm_dataset, train_idx, val_idx, labels


def collect_attention(model, norm_dataset, val_idx, batch_size):
    loader = DataLoader(Subset(norm_dataset, val_idx), batch_size=batch_size, shuffle=False)
    all_attention = []
    all_labels = []
    all_preds = []
    all_frame_masks = []

    model.eval()
    for batch in loader:
        _, frame_mask, _, x_lstm, x_gru, y = batch
        x_lstm = x_lstm.to(sota.DEVICE)
        x_gru = x_gru.to(sota.DEVICE)
        logits, attn = forward_with_attention(model, x_lstm, x_gru)
        all_attention.append(attn.detach().cpu().numpy())
        all_labels.append(y.numpy())
        all_preds.append(logits.argmax(dim=1).detach().cpu().numpy())
        all_frame_masks.append(frame_mask.numpy())

    return (
        np.concatenate(all_attention, axis=0),
        np.concatenate(all_labels, axis=0),
        np.concatenate(all_preds, axis=0),
        np.concatenate(all_frame_masks, axis=0),
    )


def summarize_attention(attention, labels, preds, frame_masks):
    frames = np.arange(sota.MAX_FRAMES)
    rows = []
    class_mean = np.zeros((sota.NUM_CLASSES, sota.MAX_FRAMES), dtype=np.float32)
    class_std = np.zeros_like(class_mean)

    for class_idx, class_name in enumerate(sota.DISPLAY_GESTURES):
        mask = labels == class_idx
        cls_attn = attention[mask]
        cls_frame_masks = frame_masks[mask]
        if len(cls_attn) == 0:
            continue
        mean_attn = cls_attn.mean(axis=0)
        class_mean[class_idx] = mean_attn
        class_std[class_idx] = cls_attn.std(axis=0)

        top_frames = np.argsort(mean_attn)[-5:][::-1]
        early_mass = mean_attn[:27].sum()
        middle_mass = mean_attn[27:54].sum()
        late_mass = mean_attn[54:].sum()
        entropy = -np.sum(mean_attn * np.log(mean_attn + 1e-12))
        norm_entropy = entropy / np.log(sota.MAX_FRAMES)
        top20_count = max(1, int(round(0.2 * sota.MAX_FRAMES)))
        top20_mass = np.sort(mean_attn)[-top20_count:].sum()
        active_mass = float((cls_attn * cls_frame_masks.astype(np.float32)).sum(axis=1).mean())

        rows.append({
            "Class": class_name,
            "Samples": int(mask.sum()),
            "Accuracy (%)": float((preds[mask] == labels[mask]).mean() * 100.0),
            "Peak Frame": int(np.argmax(mean_attn)),
            "Peak Attention": float(mean_attn.max()),
            "Attention Center Frame": float((frames * mean_attn).sum()),
            "Early Mass (0-26)": float(early_mass),
            "Middle Mass (27-53)": float(middle_mass),
            "Late Mass (54-79)": float(late_mass),
            "Top-20% Frame Mass": float(top20_mass),
            "Normalized Entropy": float(norm_entropy),
            "Mean Active-Frame Mass": active_mass,
            "Top-5 Frames": ",".join(str(int(i)) for i in top_frames),
        })

    return pd.DataFrame(rows), class_mean, class_std


def save_csv_outputs(result_dir, attention, labels, preds, frame_masks, summary, class_mean, class_std):
    pd.DataFrame(class_mean, index=sota.DISPLAY_GESTURES, columns=[f"Frame_{i}" for i in range(sota.MAX_FRAMES)]).to_csv(
        os.path.join(result_dir, "Attention_ClassMean_ByFrame.csv"),
        encoding="utf-8-sig",
    )
    pd.DataFrame(class_std, index=sota.DISPLAY_GESTURES, columns=[f"Frame_{i}" for i in range(sota.MAX_FRAMES)]).to_csv(
        os.path.join(result_dir, "Attention_ClassStd_ByFrame.csv"),
        encoding="utf-8-sig",
    )
    summary.to_csv(os.path.join(result_dir, "Attention_Summary_ByClass.csv"), index=False, encoding="utf-8-sig")

    per_sample = pd.DataFrame({
        "True Label": [sota.DISPLAY_GESTURES[int(i)] for i in labels],
        "Predicted Label": [sota.DISPLAY_GESTURES[int(i)] for i in preds],
        "Correct": preds == labels,
        "Peak Frame": attention.argmax(axis=1),
        "Peak Attention": attention.max(axis=1),
        "Attention Center Frame": (attention * np.arange(sota.MAX_FRAMES)[None, :]).sum(axis=1),
        "Active-Frame Mass": (attention * frame_masks.astype(np.float32)).sum(axis=1),
    })
    per_sample.to_csv(os.path.join(result_dir, "Attention_PerSample_Summary.csv"), index=False, encoding="utf-8-sig")


def plot_attention_interpretation(result_dir, class_mean, summary):
    frames = np.arange(sota.MAX_FRAMES)
    fig = plt.figure(figsize=(12.0, 8.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.4, 1.0], width_ratios=[1.2, 1.0], hspace=0.35, wspace=0.25)

    ax0 = fig.add_subplot(gs[0, :])
    im = ax0.imshow(class_mean, aspect="auto", cmap="viridis")
    ax0.set_xlabel("Frame index")
    ax0.set_ylabel("Gesture class")
    ax0.set_yticks(np.arange(len(sota.DISPLAY_GESTURES)))
    ax0.set_yticklabels(sota.DISPLAY_GESTURES)
    ax0.set_xticks(np.arange(0, sota.MAX_FRAMES, 10))
    cbar = fig.colorbar(im, ax=ax0, pad=0.01)

    ax1 = fig.add_subplot(gs[1, 0])
    for idx, cls_name in enumerate(sota.DISPLAY_GESTURES):
        ax1.plot(frames, class_mean[idx], linewidth=1.0, alpha=0.75, label=cls_name)
    ax1.plot(frames, class_mean.mean(axis=0), color="black", linewidth=2.2, label="Mean")
    ax1.set_xlabel("Frame index")
    ax1.set_ylabel("Attention weight")
    ax1.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)
    ax1.legend(ncol=4, fontsize=7, frameon=False)

    ax2 = fig.add_subplot(gs[1, 1])
    segment = summary[["Early Mass (0-26)", "Middle Mass (27-53)", "Late Mass (54-79)"]].mean()
    ax2.bar(["Early", "Middle", "Late"], segment.values, color=["#4575b4", "#91bfdb", "#d73027"])
    ax2.set_ylabel("Attention mass")
    ax2.set_ylim(0, max(0.4, float(segment.max()) * 1.2))
    for i, value in enumerate(segment.values):
        ax2.text(i, value + 0.01, f"{value:.2f}", ha="center", va="bottom", fontsize=9)

    out_pdf = os.path.join(result_dir, "Attention_Interpretation.pdf")
    out_png = os.path.join(result_dir, "Attention_Interpretation.png")
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_pdf


def plot_attention_interpretation_separate(result_dir, class_mean, summary):
    frames = np.arange(sota.MAX_FRAMES)
    outputs = {}

    fig, ax = plt.subplots(figsize=ATTENTION_HEATMAP_SIZE)
    im = ax.imshow(class_mean, aspect="auto", cmap="viridis", interpolation="nearest")
    ax.set_xlabel("Frame index", fontsize=12)
    ax.set_ylabel("Gesture class", fontsize=12)
    ax.set_yticks(np.arange(len(sota.DISPLAY_GESTURES)))
    ax.set_yticklabels(sota.DISPLAY_GESTURES)
    ax.set_xticks(np.arange(0, sota.MAX_FRAMES, 10))
    ax.tick_params(axis="both", labelsize=10)
    cbar = fig.colorbar(im, ax=ax, pad=0.015)
    cbar.ax.tick_params(labelsize=8)
    fig.tight_layout()
    outputs["heatmap"] = os.path.join(result_dir, "Attention_ClassMean_Heatmap.pdf")
    fig.savefig(outputs["heatmap"], format="pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=ATTENTION_PANEL_SIZE)
    palette = plt.cm.tab20(np.linspace(0, 1, len(sota.DISPLAY_GESTURES)))
    for idx, cls_name in enumerate(sota.DISPLAY_GESTURES):
        ax.plot(frames, class_mean[idx], linewidth=1.1, alpha=0.82, color=palette[idx], label=cls_name)
    ax.plot(frames, class_mean.mean(axis=0), color="black", linewidth=2.2, label="Mean")
    ax.set_xlabel("Frame index", fontsize=12)
    ax.set_ylabel("Attention weight", fontsize=12)
    ax.set_xlim(0, sota.MAX_FRAMES - 1)
    ax.tick_params(axis="both", labelsize=10)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)
    ax.legend(ncol=4, fontsize=9, frameon=True, loc="upper right")
    fig.tight_layout()
    outputs["profiles"] = os.path.join(result_dir, "Attention_Temporal_Profiles.pdf")
    fig.savefig(outputs["profiles"], format="pdf", bbox_inches="tight")
    plt.close(fig)

    segment = summary[["Early Mass (0-26)", "Middle Mass (27-53)", "Late Mass (54-79)"]].mean()
    segment_percent = segment * 100.0
    fig, ax = plt.subplots(figsize=ATTENTION_PANEL_SIZE)
    bars = ax.bar(
        ["Early", "Middle", "Late"],
        segment_percent.values,
        color=["#4C78A8", "#F58518", "#54A24B"],
        edgecolor="black",
        alpha=0.88,
    )
    for bar, value in zip(bars, segment_percent.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.2f}%",
            ha="center",
            va="bottom",
            fontsize=11,
        )
    ax.set_ylabel("Attention mass (%)", fontsize=12)
    ax.set_ylim(0, max(100.0, float(segment_percent.max()) * 1.12))
    ax.tick_params(axis="both", labelsize=10)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.35)
    fig.tight_layout()
    outputs["segments"] = os.path.join(result_dir, "Attention_Segment_Mass.pdf")
    fig.savefig(outputs["segments"], format="pdf", bbox_inches="tight")
    plt.close(fig)

    return outputs


def plot_example_attention(result_dir, attention, labels, preds):
    selected = []
    for class_idx in range(sota.NUM_CLASSES):
        correct_idx = np.where((labels == class_idx) & (preds == labels))[0]
        if len(correct_idx) > 0:
            selected.append(int(correct_idx[0]))
    selected = selected[:12]
    if not selected:
        return None

    fig, axes = plt.subplots(4, 3, figsize=(12, 8), sharex=True, sharey=True)
    axes = axes.ravel()
    frames = np.arange(sota.MAX_FRAMES)
    for ax, idx in zip(axes, selected):
        ax.plot(frames, attention[idx], color="#2c7fb8", linewidth=1.8)
        ax.axvline(int(attention[idx].argmax()), color="#d7301f", linestyle="--", linewidth=1.0)
        ax.set_title(f"{sota.DISPLAY_GESTURES[int(labels[idx])]} peak={int(attention[idx].argmax())}", fontsize=9)
        ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)
    for ax in axes[len(selected):]:
        ax.axis("off")
    fig.supxlabel("Frame index")
    fig.supylabel("Attention weight")
    fig.tight_layout()

    out_pdf = os.path.join(result_dir, "Attention_Example_CorrectSamples.pdf")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    return out_pdf


def parse_args():
    parser = argparse.ArgumentParser(description="Interpret DSTG-Net temporal attention weights for Reviewer 2.")
    parser.add_argument("--data-path", default=sota.DATA_PATH)
    parser.add_argument("--result-dir", default=DEFAULT_RESULT_DIR)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-samples-per-class", type=int, default=None)
    parser.add_argument("--copy-to-paper-figures", action="store_true", default=True)
    parser.add_argument("--no-copy-to-paper-figures", dest="copy_to_paper_figures", action="store_false")
    parser.add_argument("--paper-figure-path", default=DEFAULT_PAPER_FIG)
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_dir(args.result_dir)
    configure_plot_style()

    sota.DATA_PATH = args.data_path
    sota.set_seed(sota.SEED)

    print("=" * 80)
    print("DSTG-Net attention interpretation")
    print("=" * 80)
    print(f"Data path : {args.data_path}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Split     : {args.split}")
    print(f"Result dir: {args.result_dir}")
    print(f"Device    : {sota.DEVICE}")

    raw_dataset, norm_dataset, train_idx, val_idx, _ = load_dataset(
        args.data_path,
        args.split,
        max_samples_per_class=args.max_samples_per_class,
    )
    print(f"Train samples: {len(train_idx)}")
    print(f"Eval samples : {len(val_idx)}")

    model = sota.FusionGestureModel(num_classes=sota.NUM_CLASSES).to(sota.DEVICE)
    state, metadata = load_checkpoint_state(args.checkpoint)
    model.load_state_dict(state)
    model.eval()
    if metadata:
        print(f"Checkpoint epoch   : {metadata.get('best_epoch', 'N/A')}")
        print(f"Checkpoint accuracy: {metadata.get('accuracy', 'N/A')}")

    attention, labels, preds, frame_masks = collect_attention(model, norm_dataset, val_idx, args.batch_size)
    summary, class_mean, class_std = summarize_attention(attention, labels, preds, frame_masks)
    save_csv_outputs(args.result_dir, attention, labels, preds, frame_masks, summary, class_mean, class_std)
    main_pdf = plot_attention_interpretation(args.result_dir, class_mean, summary)
    separate_pdfs = plot_attention_interpretation_separate(args.result_dir, class_mean, summary)
    example_pdf = plot_example_attention(args.result_dir, attention, labels, preds)

    if args.copy_to_paper_figures:
        paper_dir = os.path.dirname(args.paper_figure_path)
        ensure_dir(paper_dir)
        shutil.copyfile(main_pdf, args.paper_figure_path)
        print(f"Copied paper figure: {args.paper_figure_path}")
        paper_split_paths = {
            "heatmap": os.path.join(paper_dir, "fig10a_attention_class_mean_heatmap.pdf"),
            "profiles": os.path.join(paper_dir, "fig10b_attention_temporal_profiles.pdf"),
            "segments": os.path.join(paper_dir, "fig10c_attention_segment_mass.pdf"),
        }
        for key, paper_path in paper_split_paths.items():
            shutil.copyfile(separate_pdfs[key], paper_path)
            print(f"Copied paper figure: {paper_path}")

    print("\nOverall validation accuracy: {:.2f}%".format((preds == labels).mean() * 100.0))
    print("Mean active-frame attention mass: {:.4f}".format(summary["Mean Active-Frame Mass"].mean()))
    print("Mean top-20% frame attention mass: {:.4f}".format(summary["Top-20% Frame Mass"].mean()))
    print("Mean normalized attention entropy: {:.4f}".format(summary["Normalized Entropy"].mean()))
    print("\nSaved:")
    print(f"  {main_pdf}")
    for pdf_path in separate_pdfs.values():
        print(f"  {pdf_path}")
    if example_pdf:
        print(f"  {example_pdf}")
    print(f"  {os.path.join(args.result_dir, 'Attention_Summary_ByClass.csv')}")


if __name__ == "__main__":
    main()
