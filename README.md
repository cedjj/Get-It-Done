# Get-It-Done

> Everything needed to run the notebooks is in this file. If you know nothing about the project, start at **Quick start**.

---

## Contents

1. [Quick start](#quick-start)
2. [Repository structure](#repository-structure)
3. [Setup](#setup)
4. [Data: where to get it and where to put it](#data-where-to-get-it-and-where-to-put-it)
5. [Notebooks](#notebooks)
6. [Setting file paths](#setting-file-paths)
7. [Example run](#example-run)
8. [Notes and limitations](#notes-and-limitations)
9. [Team and contributions](#team-and-contributions)

---

## Quick start

```bash
git clone https://github.com/cedjj/Get-It-Done.git
cd Get-It-Done
python -m venv .venv
.venv\Scripts\activate          # Windows. On macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook
```

Then:

1. Download or confirm the data files listed under [Data](#data-where-to-get-it-and-where-to-put-it).
2. Open a notebook from the [Notebooks](#notebooks) table.
3. Choose **Kernel > Restart & Run All**.

Requires Python [3.10 or newer] and an internet connection. Some notebooks download City data directly by URL.

---

## Repository structure

```
Get-It-Done/
├── README.md                 # this file
├── requirements.txt          # Python packages
├── data/                     # data files (large or restricted files are NOT stored in git; see Data)
├── [notebooks/]              # EDA notebooks  [update to match the actual folder layout]
│   ├── get_it_done_eda.ipynb
│   ├── get_it_done_img_link_eda.ipynb
│   └── [other notebooks]
└── .gitignore
```

> [Update this tree to match the repository. If any folder gets its own README, add a link to it here.]

---

## Setup

1. Install Python [3.10+] and Git.
2. Clone the repository and create a virtual environment (see [Quick start](#quick-start)).
3. Install packages with `pip install -r requirements.txt`.

Main packages: pandas, numpy, scipy, scikit-learn, plotly (5.24 or newer, needed for density maps), matplotlib, pillow, requests, and Jupyter.

Memory and time: the photo-attachments file has about 2.6 million rows (about 197 MB as a CSV), so notebooks that merge it with the case data need several GB of RAM and may take a few minutes.

---

## Data: where to get it and where to put it

All paths are relative to the repository root. Put downloaded files in `data/`.

| # | File | What it is | Where to get it | Stored in this repo? |
|---|------|------------|-----------------|----------------------|
| 1 | `get_it_done_requests_closed_<YEAR>_datasd.csv` (2016 to 2026, one file per year) | Closed GID cases | City of San Diego Open Data Portal (data.sandiego.gov, search "Get It Done"). Direct URL pattern: `https://seshat.datasd.org/get_it_done_reports/get_it_done_requests_closed_<YEAR>_datasd.csv` https://data.sandiego.gov/datasets/get-it-done-reports/ | No. Notebooks download these by URL, so no manual step is needed. |
| 2 | `get_it_done_requests_open_datasd.csv` | Currently open GID cases | Same portal as above | No. Notebooks download these by URL, so no manual step is needed. |
| 3 | `public-attachments-cases.csv` (about 2.6M rows) | One row per photo attached to a case. Columns include `Id`, `ParentId`, `Parent.Name` (the case's `service_request_id`), and `Name` (file name). | Provided to our team by the City [confirm whether it can be shared] | Yes, as `data/public-attachments-cases.csv.gz` |
| 4 | `referral_data_reduced.csv.gz` | Referral fields from a City-provided export, with personal contact columns removed | Provided to our team by the City [confirm whether it can be shared] | Yes, as `data/public-attachments-cases.csv.gz` |
| 5 | `gid_cases-001`, about 4.31M records | City case records with `Case Reason`, original/current type, and internal referral flags | Provided to our team by the City. **Not public.** | No. [Notebooks that need it: `referrals-data-eda.ipynb`.] |

**Photo links.** `img_link` is built as
`https://getitdone.sandiego.gov/servlet/servlet.FileDownload?file=<attachment Id>`.
The links are publicly reachable, and in a test of 200 sampled links all returned images. Photos are never stored in the repo.

**If a file is missing,** the notebook that needs it says so in its first cell. Files 4 and 5 are restricted. Without them, run only the notebooks marked "public data only" in the table below.

---

## Notebooks

| Notebook | Purpose | Inputs (from the Data table) | Public data only? | Author |
|----------|---------|------------------------------|-------------------|--------|
| `public-data-eda.ipynb` | Cleaning and EDA of GID cases | 1, 2 | Yes | Kai Arizabal |
| `gid_intake_rework_analysis.ipynb` | Exploring `Case Reason` field and "send back to intake" values | 1, 2, 5 | No | Christian Kumagai |
| `img-link-eda.ipynb` | Merges photo attachments with cases and explores the new `img_link` column | 1, 2, 3 | Yes | Cedric Jeng |
| `referrals-data-eda.ipynb` | Merges public and private GID cases with referral data and explores the referral data | 1, 2, 4, 5 | No | Kai Arizabal |

Run each notebook top to bottom. Each starts with a setup cell that imports packages and sets paths.

---

## Example run

Example: reproduce the photo-coverage analysis.

1. Complete [Setup](#setup).
2. Save `public-attachments-cases.csv.gz` in `data/`.
3. Open `img_link_eda.ipynb` and choose **Restart & Run All**.

Expected output:
- A printout of how many cases and photos were matched (the merge step), and what share of cases have at least one photo.
- Bar charts of cases with a photo by service category.
- A table of file extensions in the attachments file. Almost all are image files such as `.jpg` and `.png`.

The final cell sends web requests to the City's photo server. It is optional, so skip it if you want to avoid network calls.

---

## Notes and limitations

- **Misroutes are not directly observable.** Referral, reclassification, and "send back to intake" show that a routing change happened, not that it was an error. "Send back to intake" applies only to cases that passed through Parks and Recreation. The `Case Reason` field is overwritten when a case closes.
- **Photos are not always resident-taken.** Some categories, such as parking, may have photos taken by staff.
- **Counts depend on cleaning choices,** such as dropping child records and keeping the earliest closure for duplicated case IDs. The notebooks state each choice where it is made.
- **No model results are reported here.** The photo-based designs we propose have not been trained or tested.
- **Restricted data.** Do not commit files 4 or 5 without the City's permission.

---

## Team and contributions

| Team member | Contribution |
|-------------|--------------|
| Kai Arizabal | Public data EDA and Referral Data EDA |
| Cedric Jeng | `img_link` EDA |
| Christian Kumagai | `Case Reason` EDA |

---

## Contact

