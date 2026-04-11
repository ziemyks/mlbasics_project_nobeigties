# TODO — Missing Items (return here)

## 1. `model_test.py` — currently empty
- [ ] Write basic tests for the Flask API endpoints (`/health`, `/generate`, `/predict`)
- [ ] Test input validation (missing fields, bad types)
- [ ] Test feature engineering logic

## 2. Notebook — re-run before submission
- [Done ] Run all cells top-to-bottom in a clean kernel before April 14
- [ Done] Confirm all outputs/charts render correctly with current data

## 3. Threshold optimisation (optional but adds points)
- [ Done ] Lower classification threshold from 0.5 → try 0.3–0.4 to boost Recall
- [ ] Show Precision/Recall tradeoff at different thresholds

## 4. API security (minor, low priority for academic context)
- [ ] Restrict CORS origins (currently wide open)
- [ ] Sanitise 500 error messages (currently leaks exception details)
- [ ] Add basic input range validation (`amt` > 0, lat/long in range)

## 5. Cosmetic / optional
- [ ] `data_collection.ipynb` mentioned in guidelines but missing — not critical if data loading is in main notebook
