# Dissolution Statistics

Statistical analysis of immediate-release dissolution data, based on the FDA (1997) dissolution guidance,
the EMA (2017) dissolution-specification reflection paper and ICH M9 (BCS-based biowaivers).

- **Profile comparison:** f1/f2 with validity checks, multivariate (MSD) confidence region, bootstrap f2 (percentile, BCa, bootstrap-t)
- **Specification (EMA):** proposed Q and time point from biobatch data
- **BCS biowaiver (ICH M9):** classification and comparative dissolution assessment

Data format: a CSV/Excel table with one row per unit and one column per time point (minutes), values as % dissolved.
Examples are in `sample_data/`.

Decision support only. Verify results against your SOPs and validated systems.
