# UNC Attendance Streamlit App

This bundle is wired to the finalized six-file prediction export for the split preseason and midseason models.

## Run locally

```bash
cd unc_attendance_streamlit_app_finalized
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Included behavior

- Uses the finalized preseason and midseason prediction CSVs from the `data/` folder
- Keeps the existing calendar, game detail, and data explorer layout
- Removes the `Notes about this version` section

The six finalized CSVs are already included in this package.
