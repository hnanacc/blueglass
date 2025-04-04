# Copyright 2025 Intel Corporation
# SPDX: Apache-2.0

from argparse import ArgumentParser


def add_environment_arguments(p: ArgumentParser):
    p.add_argument("--num-cpus", type=int, default=10, help="No. of CPUs to use.")
    p.add_argument("--num-gpus", type=int, default=1, help="No. of GPUs to use.")
    p.add_argument("--rank", type=int, default=0, help="Rank of the current machine.")
    p.add_argument(
        "--num-machines",
        type=int,
        default=1,
        help="Number of machines for distributed run.",
    )
    p.add_argument(
        "--dist-url",
        type=str,
        default="auto",
        help="URL for distributed run communication.",
    )
    p.add_argument(
        "--random-seed",
        type=int,
        default=1337,
        help="Random seed value to initialize all random generators.",
    )


datasets = [
    "kitti_train",
    "kitti_val",
    "kitti_mini",
    "bdd100k_train",
    "bdd100k_mini",
    "bdd100k_val",
    "lvis_train",
    "lvis_minival",
    "lvis_mini",
    "lvis_val",
    "ecpersons_train",
    "ecpersons_val",
    "ecpersons_mini",
    "valerie22_train",
    "valerie22_val",
    "valerie22_mini",
    "coco_train",
    "coco_mini",
    "coco_val",
    "funnybirds_no_intervention",
    "funnybirds_no_beak",
    "funnybirds_no_eyes",
    "funnybirds_no_foot",
    "funnybirds_no_tail",
    "funnybirds_no_wings",
    "features",
    "openimages_train",
    "openimages_val",
    "openimages_mini",
    "openimages_train_s0",
    "openimages_train_s1",
    "openimages_train_s2",
    "openimages_train_s3",
    "openimages_train_s4",
    "openimages_train_s5",
    "openimages_train_s6",
    "openimages_train_s7",
    "openimages_train_s8",
    "openimages_train_s9",
    "openimages_train_sa",
    "openimages_train_sb",
    "openimages_train_sc",
    "openimages_train_sd",
    "openimages_train_se",
    "openimages_train_sf",
]


def add_dataset_arguments(p: ArgumentParser):
    p.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="batch size to use while inference or training.",
    )
    p.add_argument(
        "--trainset",
        type=str,
        default="coco_train",
        choices=datasets,
    )
    p.add_argument(
        "--testset",
        type=str,
        default="coco_mini",
        choices=datasets,
    )
    p.add_argument(
        "--labelset",
        type=str,
        default="coco_train",
        choices=datasets,
        help="Dataset to use for extracting classnames.",
    )
    p.add_argument(
        "--dataset",
        type=str,
        default="lvis_train",
        choices=datasets,
    )

    return p


def add_evaluation_arguments(p: ArgumentParser):
    p.add_argument(
        "--use-label-matching",
        action="store_true",
        help="Enable label matching evaluation.",
    )
    p.add_argument(
        "--use-multi-layer", action="store_true", help="Enable multi-layer evaluation."
    )
    p.add_argument(
        "--prompter-name",
        type=str,
        default="basic",
        choices=["basic", "ensemble"],
        help="Prompt template to use for adapting the classnames.",
    )
    p.add_argument(
        "--matcher-name",
        type=str,
        default="similarity",
        choices=["similarity"],
        help="Metric to use for comparions of text descriptions.",
    )
    p.add_argument(
        "--encoder-name",
        type=str,
        default="clip",
        choices=["clip", "nvembed", "b1ade", "bert", "siglip"],
        help="Text encoder model to use for converting prompts to embeds.",
    )
    p.add_argument(
        "--num-topk-matches",
        type=int,
        default=1,
        help="No. of classes to match each description to.",
    )
    p.add_argument(
        "--max-prediction",
        type=int,
        default=900,
        help="Limit max no. of prediction to use during evaluation (based on scores).",
    )
    p.add_argument(
        "--min-confidence",
        type=float,
        default=0.0,
        help="Filter predictions based on confidence scores.",
    )
    p.add_argument(
        "--use-descriptions",
        action="store_true",
        help="Uses descriptions to compute similarity score and class assignment.",
    )
    p.add_argument(
        "--use-box-ious",
        action="store_true",
        help="Uses box ious along with text match (ex. sim) scores.",
    )
    p.add_argument(
        "--use-box-objectness",
        action="store_true",
        help="Uses box objectness along with text match (ex. sim) scores.",
    )

    p.add_argument(
        "--use-negatives",
        action="store_true",
        help="Use negative prompts to catch non-positive objects.",
    )
    p.add_argument(
        "--use-parts",
        action="store_true",
        help="Use parts prompts to catch part vs wholes.",
    )
    p.add_argument(
        "--enable-analysis",
        action="store_true",
        help="Enables analysis of predictions in the label matching evaluation.",
    )
    p.add_argument(
        "--enable-common-metrics",
        action="store_true",
        help="Enables common metric (VOC) evaluation besides dataset-specific evalution.",
    )
    p.add_argument(
        "--num-vis-samples",
        type=int,
        default=10,
        help="No. of samples to visualize during evaluation.",
    )
    p.add_argument(
        "--biou_threshold",
        type=float,
        default=0,
        help="Threshold for filtering based on box threshold.",
    )
    p.add_argument("--evaluator", type=str, choices=["coco", "lvis", "bdd100k"])
    p.add_argument(
        "--use-analysis",
        action="store_true",
        help="Switch to enable prediction analysis during evaluation.",
    )
    p.add_argument(
        "--analyze-confusion",
        action="store_true",
        help="Switch to enable confusion mask during evaluation.",
    )
    p.add_argument(
        "--compute-tp-fp",
        action="store_true",
        help="Computes the TP and FP from predictions while parsing sequences.",
    )


def add_runner_arguments(p):
    p.add_argument(
        "--runner",
        type=str,
        choices=[
            "benchmark",
            "layers_patch",
            "feature_extract",
            "vlm_linear_probe",
            "sae_linear_probe",
            "cluster_probe",
            "attention_probe",
            "sae",
        ],
        help="Choose a runner to launch the run.",
    )
    p.add_argument("--use-wandb", action="store_true", help="Switch to enable wandb")
    p.add_argument(
        "--run-name",
        type=str,
        default="unnamed_run",
        help="Name of the current run. Will be used in Wandb",
    )
    p.add_argument(
        "--run-mode",
        type=str,
        choices=["train", "infer", "test"],
        default="test",
        help="Which mode to run the model in?",
    )
    p.add_argument(
        "--max-steps",
        type=int,
        default=100_000,
        help="Maximun no. of steps to train for.",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        help="Path to the common output folder of current model.",
    )
    p.add_argument(
        "--logs-period", type=int, default=20, help="Period to publish logs."
    )
    p.add_argument(
        "--eval-period", type=int, default=500, help="Period to validate current model."
    )


def add_probe_arguments(p):
    p.add_argument(
        "--extraction-mode",
        type=str,
        choices=["hooks", "manual"],
        help="method to use to extract features.",
    )
    p.add_argument(
        "--feature-pattern",
        type=str,
        default="decoder_features",
        choices=[
            "det_decoder_mlp",
            "det_decoder_mha",
            "det_decoder_resid",
            "det_decoder_resid_denormed",
            "llm_decoder_mlp",
            "llm_decoder_mha",
            "llm_decoder_resid",
            "llm_decoder_resid_denormed",
        ],
        help="Feature pattern name or pattern in unix style.",
    )
    p.add_argument(
        "--feature-subpattern",
        type=str,
        default="pos_img",
        choices=[
            "pre_img",
            "pos_img",
            "pre_txt",
            "pos_txt",
            "weights",
            "outputs",
            "true_cls",
            "true_box",
            "pred_cls",
            "pred_box",
            "pred_scr",
            "filename",
        ],
    )
    p.add_argument(
        "--feature-path",
        type=str,
        help="Path to store features for this run and model.",
    )
    p.add_argument(
        "--num-spatial-units",
        type=int,
        default=1024,
        help="Num units in the output dim of localization prober.",
    )
    p.add_argument(
        "--probe-fwd-period",
        type=int,
        default=1,
        help="Period for running probe train pass.",
    )
    p.add_argument(
        "--use-classified-boxes",
        action="store_true",
        help="Switch to enable localization hypothesis.",
    )
    p.add_argument(
        "--use-vlm-pred-as-true",
        action="store_true",
        help="use vlm predictions as ground truth for probe losses.",
    )
    p.add_argument(
        "--probe-mode",
        type=str,
        choices=[
            "box_linear",
            "cls_linear",
            "det_linear",
            "det_dropout",
            "det_feature_split",
        ],
        default="standard",
        help="Which probe head to use?",
    )
    p.add_argument(
        "--subfeature-dim",
        type=int,
        default=1024,
        help="Dimension of each subfeature in feature split probe.",
    )
    p.add_argument(
        "--mc-dropout-pct",
        type=float,
        default=0.5,
        help="Dropout percentage for dropout probes.",
    )
    p.add_argument(
        "--mc-infer-steps",
        type=int,
        default=10,
        help="No. of steps for monte carlo inference for dropout probes.",
    )


def add_shared_arguments(p: ArgumentParser):
    add_environment_arguments(p)
    add_probe_arguments(p)
    add_dataset_arguments(p)
    add_evaluation_arguments(p)
    add_runner_arguments(p)
