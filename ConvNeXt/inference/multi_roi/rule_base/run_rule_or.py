import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[4]))

from ConvNeXt.inference.multi_roi.rule_base import select_or_rule_prediction
from ConvNeXt.inference.multi_roi.rule_base.rule_fusion_common import build_parser, run_rule_fusion


def main():
    ap = build_parser(default_out_dir="/root/project/convnext/convnext_relabeled_tiny_rule_or")
    args = ap.parse_args()
    run_rule_fusion(args, rule_name="rule_or", selector=select_or_rule_prediction)


if __name__ == "__main__":
    main()
