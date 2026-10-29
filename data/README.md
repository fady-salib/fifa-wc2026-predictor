# Source data and integrity

`international_football_2018_present.csv` is committed (3,866 rows, 60 columns).
The previous statement that it was absent was incorrect. It contains two team
perspectives per apparent match, malformed country-name affixes, missing fields
and contradictory/ambiguous match groups. Historical Elo and xG columns are
entirely missing. No authoritative upstream URL, scrape script, retrieval date
or dataset license is provided; resolve those before asserting data provenance.

The reviewed loader validates exact reciprocal perspectives, score/result
agreement and reversed scores. It quarantines whole inconsistent/ambiguous groups
rather than guessing which side is correct. The review retained 1,859 matches and
quarantined 60 groups; see `../docs/data-audit.json` for original line references.
The raw file is unchanged. Reciprocal consistency is not external truth checking.

Features use only earlier dated validated results. Present-day rankings/xG,
same-match statistics, ambiguous lineup/squad values and global imputation are
excluded. Ratings initialize at 1500 and update with K=20 from the retained
partial history; these are reconstructed ratings, not historical official Elo.
Use a sourced, independently validated match database to strengthen this study.
