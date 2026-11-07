# AI USE


## Contents

1. [img-link-eda.ipynb](#img-link-eda.ipynb)
2. [gid_intake_rework_analysis.ipynb](#gid-intake-rework-analysis.ipynb)

---



## img-link-eda.ipynb
### Prompt:
[pasted the extension-extraction regex] "for this can i see the rest of the information in the dataframe for weird files (docx, csv, xlsx, doc)"
### Resulting Code:
```bash
# Extension per attachment, keyed by attachment Id
img['ext'] = img['Name'].str.extract(r'\.([A-Za-z0-9]+)$')[0].str.lower()
ext_by_id = img.drop_duplicates('Id').set_index('Id')['ext']

# Pull the Id back out of the URL and attach the extension to merged_df
merged_df['att_id'] = merged_df['img_link'].str.extract(r'file=(\w+)$')[0]
merged_df['ext'] = merged_df['att_id'].map(ext_by_id)

weird_ext = ['docx', 'doc', 'csv', 'xlsx', 'xls']
weird_rows = merged_df[merged_df['ext'].isin(weird_ext)]

print(f"{len(weird_rows):,} attachment rows on {weird_rows['service_request_id'].nunique():,} cases")

pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', 80)
weird_rows.sort_values('ext').groupby('ext').head(5)
```
### How it was tested:
```bash
weird_rows['img_link'].head()
```
This resulted in image links which I then clicked to check if they are files that I wanted.

---

### Prompt:
"how do i get outputs that should be a good link to show what the whole link looks like: merged_df[merged_df['service_request_id'] == 6153869]['img_link']". This gave the full-link and clickable-link code.
### Resulting Code:
```bash
pd.set_option('display.max_colwidth', None)
```
### How it was tested:
I could now see the full link to the images and not just parts of it.

---

### Prompt:
[attached an excerpt of image link] “help me visualize these through graphs”
### Resulting Code:
```bash
first_img = (merged_df.dropna(subset=['img_link']).drop_duplicates('service_request_id')
             [['service_request_id', 'img_link', 'service_name']])

def fetch_image(url, size=(300, 300)):
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        im = Image.open(io.BytesIO(r.content)).convert('RGB')
        im.thumbnail(size)
        return im
    except Exception:
        return None

def category_sheet(categories, k=8, seed=21): # change seed for different random samples
    picks = []
    for c in categories:
        pool = first_img[first_img['service_name'] == c]
        picks.append(pool.sample(min(k, len(pool)), random_state=seed)['img_link'].tolist())
    with ThreadPoolExecutor(max_workers=4) as ex:
        loaded = [list(ex.map(fetch_image, urls)) for urls in picks]
    fig, axes = plt.subplots(len(categories), k, figsize=(2.5 * k, 2.6 * len(categories)), squeeze=False)
    for i, c in enumerate(categories):
        for j in range(k):
            ax = axes[i, j]
            ax.axis('off')
            if j < len(loaded[i]) and loaded[i][j] is not None:
                ax.imshow(loaded[i][j])
        axes[i, 0].text(-0.05, 0.5, c[:28], transform=axes[i, 0].transAxes, rotation=90, va='center', ha='right', fontsize=9, weight='bold')
    plt.tight_layout()
    plt.show()

# Top categories by backlog that also have enough photos to sample
cands = svc[svc['imaged'] >= 6].sort_values('open_over_1yr', ascending=False).head(5).index.tolist()
category_sheet(cands)
```
### How it was tested:
The graph resulted in randomly selected images from the dataset. Using the seed, I checked if the same image showed up when looking for its report ID.

---

## gid_intake_rework_analysis.ipynb

### AI Usage Statement
We used ChatGPT to help with writing, debugging, and improving our data analysis code. We were responsible for determining what to analyze, reviewing the results, and questioning assumptions made by AI.

### Main Prompt (Abbreviated)
"Create a Jupyter notebook analyzing whether cases sent back to intake take longer to resolve than similar cases. Clean the dataset, compare cases across departments and categories, account for open cases, create visualizations, and explain the findings and limitations."

### Follow-Up Prompts
Reconstructed summaries of questions about the analysis, not exact transcripts.

1. "Make sure we're using all 4.3 million cases. Are we accidentally dropping cases during cleaning?"
2. "Check for duplicate Case IDs, missing dates, and invalid closure times."
3. "Does Send back to intake actually mean a case was misrouted? We shouldn't assume that."
4. "Can we look at Action Taken and referral information to better understand why cases were returned?"
5. "Wouldn't comparing cases across different departments be misleading since some services naturally take longer?"
6. "Can we match cases based on service group, category, priority, submission channel, and year?"
7. "Why are we including Other Department cases? Can we exclude them and see if the results change?"
8. "A lot of cases are still open. Wouldn't excluding them bias the analysis?"
9. "Does Closed Date actually mean the issue was resolved or just administratively closed?"
10. "Can we test different matching conditions to see if the findings are consistent?"
11. "Can we really conclude that intake returns caused delays, or are we only finding a correlation?"
12. "Add validation checks to make sure our calculations and comparisons are correct."

### How the Code Was Tested
We tested the code by running the notebook and checking that the data was cleaned correctly. We verified case counts, checked for duplicates, missing dates, and invalid closure times. We also used assertions to check the matching groups and calculations. Lastly, we compared results using different matching conditions and assumptions to see if our findings were consistent. All validation assertions passed.1
