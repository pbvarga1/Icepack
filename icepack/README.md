# icepack

Python wrapper for running icepack as well as creating input data

## Install

Must be on python3.7 or greater.

Make sure you are in the ``icepack`` directory them

```
pip install .
```

or

```
pip install -e .
```

## Commands

* ``run_icepack``: Runs icepack for the entire arctic
* ``run_case_inputs`` runs a list of cases asynchronously (3 at a time).

## Process

* ``settings.json``:
    * case name
    * machine name
    * env name
    * pump amount
    * pump repeats
    * forcing data dir
    * forcing paths : own object
    