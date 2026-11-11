# COCO panoptic reference scorer

`panopticapi/{evaluation.py,utils.py,__init__.py}` are unmodified copies from
https://github.com/cocodataset/panopticapi at revision
`7bb4655548f98f3fedc07bf37e9040a992b054b0`.
`panopticapi/LICENSE` is the repository's original `license.txt` (BSD 2-clause,
copyright 2018 Alexander Kirillov). Retain it when redistributing.

Only source files were fetched; no sample data or datasets were fetched.
The wrapper calls the official serial worker and `PQStat.pq_average` directly.
It handles empty metric groups as null rather than invoking division by zero.
