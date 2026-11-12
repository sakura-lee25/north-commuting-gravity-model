# Debug and plausibility log

Issues found while building and testing the model, in the order they happened.

## 1. Calibration overshoot broke the balancing step
- Symptom: `RuntimeError: Furness did not converge` with divide-by-zero and overflow warnings.
- Trace: the exponential form calibrated cleanly (beta = 0.140 in 7 steps). The power form failed. Hyman's first update assumes mean trip length scales roughly with 1/beta; for the power form it moved beta from 1.0 to 4.2 in one step. At beta = 4.2 the weight on a 2 km intrazonal trip is about 10^10 times the weight on a 240 km trip, so the balancing factors ran off to extreme values.
- Fix: replaced the unbracketed secant update with a bracketed root finder (Brent's method), since mean trip length falls monotonically as beta rises.

## 2. Bracket end too extreme
- Symptom: still no convergence at the first bracket evaluation.
- Trace: printed origin-total errors at iterations 10, 100, 1,000 and 20,000 for each bracket end. Exponential beta = 1.0 made the deterrence matrix almost diagonal (smallest entry about 1e-106); after 20,000 iterations origin totals were still off by 1e-5. The other three ends converged within 1,000 iterations.
- Fix: realistic brackets (exponential 0.01–0.4, power 0.1–6) and a convergence test on the constraint itself (origin totals within 1e-8) rather than on changes in the balancing factors.

## 3. Implausible results from the first specification
- Symptom: the model ran, but (a) neighbouring districts were badly under-predicted (East Riding to Hull: 21,836 observed vs 2,792 modelled) and (b) a 20% cut in effective distance between core cities raised their flows by about 77–102%.
- Diagnosis: 63% of commuters live and work in the same district. With those trips included, beta was fitted mainly to very short intrazonal trips, using an approximate intrazonal distance, which distorted the fit for flows between districts.
- Fix: held intrazonal trips at observed values and calibrated on flows between districts only. R² on those flows rose from 0.64 to 0.89, the East Riding to Hull flow is now 17,202 against 21,836 observed, and the scenario response fell to +40%.

## 4. Remaining limits (not fixed, reported)
- The model under-predicts long trips (80–250 km: 1.2% observed vs 0.2% modelled), so responses on long pairs such as Leeds to Manchester (+195%) are not reliable. A credible test needs generalised cost by mode (time, fare), not straight-line distance.
- Doubly constrained: total commuting and each district's workers and jobs are fixed, so the scenario redistributes trips (27,789 diverted from other pairs) rather than generating new ones.
- Census 2021 was taken in March 2021 under COVID-19 restrictions, which affected travel to work.
- Centroids are ONS label points from the generalised boundary layer, not population-weighted centroids.
