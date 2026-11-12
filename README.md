# Commuting gravity model for the North of England

A doubly constrained gravity model of journeys to work between the 72 local authority districts of the North East, North West and Yorkshire and The Humber, built on ONS Census 2021 origin–destination data.

**What it does**
1. Builds the observed commuting matrix (4.03 million people living and working in the North) from Census 2021 table ODWP01EW, and straight-line distances between district centroids from ONS Open Geography.
2. Calibrates exponential and power deterrence functions so the modelled mean trip length matches the observed one, with Furness balancing to match each district's workers and jobs.
3. Validates against flows not used in calibration (R², SRMSE) and the trip length distribution.
4. Tests an option: a 20% cut in effective distance between eight core cities, and checks whether the response is plausible.

**Result**: the chosen specification (exponential, flows between districts only, beta = 0.106 per km) explains 89% of the variance in 5,112 district-to-district flows. See `docs/debug_log.md` for how the specification was reached and what the model cannot yet tell you.

```
python src/data.py   # downloads geography, builds data/processed
python src/run.py    # calibration, validation, scenario -> results/, figures/
pytest tests         # 5 unit tests
```

Data: ONS Census 2021 ODWP01EW (Nomis bulk download, place in data/raw/); ONS Open Geography LAD and Region boundaries (Dec 2021). Contains public sector information licensed under the Open Government Licence v3.0.
