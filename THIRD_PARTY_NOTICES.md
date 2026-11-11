# Third-party attribution and publication decisions

The official EoMT implementation is by the Mobile Perception Systems Lab at TU/e. The inspected revision is `aa017c34afbe023de9462b17ae6067b52fef9867` from https://github.com/tue-mps/eomt. Its MIT notice is preserved verbatim in [licenses/EoMT-MIT.txt](licenses/EoMT-MIT.txt). The inference wrapper's visualization boundary follows the upstream notebook; that notice applies to the incorporated material. The upstream checkout is excluded from Git and obtained separately.

The vendored COCO panoptic scorer is from https://github.com/cocodataset/panopticapi at `7bb4655548f98f3fedc07bf37e9040a992b054b0`, copyright 2018 Alexander Kirillov. Its BSD 2-clause notice remains in [scripts/vendor/panopticapi/LICENSE](scripts/vendor/panopticapi/LICENSE); provenance is in the adjacent vendor documentation.

Research attribution: Tommie Kerssies, Niccolò Cavagnero, Alexander Hermans, Narges Norouzi, Giuseppe Averta, Bastian Leibe, Gijs Dubbelman, and Daan de Geus, *Your ViT is Secretly an Image Segmentation Model*, CVPR 2025. This repository is an independent fixed-checkpoint reproduction and sensitivity study.

No project-wide license has been selected. The maintainers must confirm ownership and any course, institution or contributor requirements before choosing a license for original scripts, documentation and numerical summaries. The upstream MIT license does not establish ownership of those contributions or grant rights to datasets, checkpoints or report imagery.

COCO images, annotations and derived qualitative figures require a separate redistribution review. No dataset images, generated masks or qualitative exports were copied into the publication summaries. Existing report source/PDF and figures remain unchanged locally; decide which report artifacts may be published after reviewing embedded third-party material. Numerical summaries contain aggregate metrics, not image content.

Checkpoint and pretrained-backbone terms must be reviewed at their pinned acquisition sources before any redistribution. They are excluded here. Dependencies are installed separately by the container build; their licenses and any binary-distribution obligations need review if container images are distributed. Source preparation does not imply clearance to distribute those assets or images.
