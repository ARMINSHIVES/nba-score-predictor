# Data

The game table joins two public datasets. Seasons 1946-47 to 1953-54 come from FiveThirtyEight. Seasons 1954-55 to the current season come from a Kaggle dataset built from NBA.com. `nbapred-train` downloads both to `data/raw/` (not committed) and writes the joined table to `data/processed/games.csv` (not committed).

## Sources

### Kaggle: NBA box scores, 1954-55 onward

| | |
| --- | --- |
| Dataset | eoinamoore, "Historical NBA Data and Player Box Scores" (page title "NBA Dataset: Box Scores and Stats (1947 - Today)") |
| Page | <https://www.kaggle.com/datasets/eoinamoore/historical-nba-data-and-player-box-scores> |
| Licence | CC0-1.0 (public domain dedication), as listed on the dataset page |
| Upstream source | NBA.com, per the dataset page |
| Update rhythm | Daily, per the dataset page; this snapshot ends with games of 2026-11-25 |
| Downloaded with | The official Kaggle API (`kaggle` 2.2.5, `dataset_download_file`), retrieved 2026-11-26 |
| Files used | `Games.csv`, `TeamStatistics.csv`, `TeamHistories.csv` |

SHA-256 of the snapshot used for the published results:

| File | SHA-256 |
| --- | --- |
| `Games.csv` | `c2abe895be6c05fa8b6350acc1f62a0535dfc97d823ec658bfc1c6d107df7b7f` |
| `TeamStatistics.csv` | `bece860f7ee5ff6b98b653fcb5f464803a486147aebd5857152e3bf44462ced0` |
| `TeamHistories.csv` | `8e16158a963e6d66aef3cf053fee3c46d4c4e646c7c1c7446b865296ba188388` |

The dataset is updated daily, so a fresh download will have different checksums and more games. The loader does not pin these hashes.

Columns read:

- `Games.csv`: `gameId`, `gameDateTimeEst`, `hometeamId`, `awayteamId`, `gameSubtype`, `gameSubLabel`, `arenaCity`
- `TeamStatistics.csv`: `gameId`, `teamId`, `home`, `teamScore`, `q1Points` to `q4Points`, `otAllPoints`, `fieldGoalsAttempted`, `freeThrowsAttempted`, `reboundsOffensive`, `turnovers`, `numMinutes`
- `TeamHistories.csv`: team id, abbreviation and the season each name took effect, used to print the code a team had at the time

Player box scores are in the dataset but are not used.

### FiveThirtyEight: final scores, 1946-47 to 1953-54

| | |
| --- | --- |
| Dataset | FiveThirtyEight, "Historical NBA Elo" (`nba-elo/nbaallelo.csv`) |
| Download URL | <https://raw.githubusercontent.com/fivethirtyeight/data/master/nba-elo/nbaallelo.csv> |
| Repository | <https://github.com/fivethirtyeight/data/tree/master/nba-elo> |
| Licence | CC BY 4.0, as stated in the repository README: <https://github.com/fivethirtyeight/data/blob/master/README.md> |
| Upstream source | FiveThirtyEight credits Basketball-Reference.com |
| Coverage | 1946-47 to the 2015 Finals; 59,008 NBA games after dropping ABA games |
| SHA-256 | `d46ed3540ee8d9eca31b3e94cc8c777e0be5156173d814ebf65b8195e8d616bc` |
| Retrieved | 2026-11-26 |

The file no longer changes, so the loader checks this hash and refuses a file that differs. Columns read: `game_id`, `lg_id`, `_iscopy`, `year_id`, `date_game`, `is_playoffs`, `team_id`, `fran_id`, `pts`, `opp_id`, `opp_fran`, `opp_pts`, `game_location`. The `elo_i`, `elo_n`, `win_equiv` and `forecast` columns are not read; `elo_n` is the rating after the game, so it would leak the result.

`nbapred-train --source 538` still trains on this file alone (1946-47 to 2014-15).

## How the two are joined

**Which source per season.** The Kaggle dataset lists games from 1946-47, but before 1954-55 many are missing: it has 20 of FiveThirtyEight's 350 games for 1946-47 and 275 of 347 for 1953-54. So seasons up to 1953-54 come from FiveThirtyEight and every later season comes from Kaggle (`src/nbapred/combine.py`, `LAST_538_SEASON = 1954`).

**Overlap check.** Both sources cover 1954-55 to 2014-15. Each training run compares them game by game. A game matches when the same two teams meet on the same date or one day apart. The numbers below are stored under `overlap` in `reports/results.json`:

| Check, seasons 1954-55 to 2014-15 | Games |
| --- | --- |
| Games in FiveThirtyEight / Kaggle | 56,014 / 56,014 |
| Matched | 56,009 |
| Same final score | 55,942 |
| Different final score | 67 |
| Home and away listed the other way round | 610 |
| Dated one day apart | 3 |
| No match, each side | 5 |

Examples: 98-115 against 98-116 (FiveThirtyEight game `195712220CIN`), 100-83 against 100-93 (`195803150DET`) and 127-118 against 120-118 (`196103150LAL`). The first five are listed in `results.json`. Neither source is used to correct the other: the stitched table takes Kaggle for these seasons.

**Team ids across moves and renames.** Kaggle keys teams by NBA team id, which stays the same when a franchise moves or is renamed. The Minneapolis and Los Angeles Lakers share one id, as do Seattle and Oklahoma City, and New Jersey and Brooklyn. The eight FiveThirtyEight franchises that are still playing after 1953-54 are mapped to their NBA ids (`FRAN_TO_ID` in `combine.py`): Celtics, Hawks, Kings, Knicks, Lakers, Pistons, Sixers, Warriors. So a franchise's Elo and scoring history carries across the 1954 switch. Franchises that folded before 1954-55 keep a `538:` key, for example `538:Capitols`. For the overlap check, FiveThirtyEight franchises are matched to ids season by season, because the two sources split some histories differently. FiveThirtyEight files the 1988-2002 Charlotte Hornets under the Pelicans, for example.

**Game types.** The third digit of the Kaggle `gameId` gives the game type:

| Digit | Type | Kept | Flags |
| --- | --- | --- | --- |
| 1 | Preseason | No | |
| 2 | Regular season (includes NBA Cup group and knockout games) | Yes | |
| 3 | All-Star | No | |
| 4 | Playoffs | Yes | `playoffs = 1` |
| 5 | Play-in | Yes | `playoffs = 1` |
| 6 | NBA Cup final | Yes | `neutral = 1` |

The season comes from `gameId` digits 4-5, not from the date, because the 2019-20 season ended in October 2020. Games are marked neutral-site when they are NBA Cup finals, have the subtype "Global Games", were played in Las Vegas or labelled a semifinal, or were played in the 2020 bubble (2019-20 games from July 2020 on). FiveThirtyEight marks neutral games with `game_location = "N"`.

**Quarter scores and possessions.** Halftime and end-of-Q3 scores are kept only when the four quarter scores plus overtime add up to the final score. Possessions use the standard estimate FGA + 0.44 × FTA − OREB + TOV and are kept from 1985-86 on. Before that, only a few percent of games have box scores, too few to rate teams on.

## Coverage per season

From `reports/coverage.csv`, written by each training run. Seasons are labelled by the years they span. "Regular" and "Playoffs" (which includes play-in games) don't sum to "Games" in seasons with an NBA Cup final, which counts as neither.

| Season | Source | Games | Regular | Playoffs | With halftime | With possessions |
| --- | --- | --- | --- | --- | --- | --- |
| 1946-47 | FiveThirtyEight | 350 | 331 | 19 | 0 | 0 |
| 1947-48 | FiveThirtyEight | 215 | 192 | 23 | 0 | 0 |
| 1948-49 | FiveThirtyEight | 380 | 360 | 20 | 0 | 0 |
| 1949-50 | FiveThirtyEight | 593 | 561 | 32 | 0 | 0 |
| 1950-51 | FiveThirtyEight | 380 | 354 | 26 | 0 | 0 |
| 1951-52 | FiveThirtyEight | 355 | 330 | 25 | 0 | 0 |
| 1952-53 | FiveThirtyEight | 374 | 351 | 23 | 0 | 0 |
| 1953-54 | FiveThirtyEight | 347 | 324 | 23 | 0 | 0 |
| 1954-55 | Kaggle | 309 | 288 | 21 | 303 | 0 |
| 1955-56 | Kaggle | 311 | 290 | 21 | 302 | 0 |
| 1956-57 | Kaggle | 307 | 290 | 17 | 304 | 0 |
| 1957-58 | Kaggle | 309 | 288 | 21 | 304 | 0 |
| 1958-59 | Kaggle | 310 | 288 | 22 | 307 | 0 |
| 1959-60 | Kaggle | 325 | 300 | 25 | 320 | 0 |
| 1960-61 | Kaggle | 341 | 316 | 25 | 326 | 0 |
| 1961-62 | Kaggle | 389 | 360 | 29 | 375 | 0 |
| 1962-63 | Kaggle | 389 | 360 | 29 | 374 | 0 |
| 1963-64 | Kaggle | 387 | 360 | 27 | 367 | 0 |
| 1964-65 | Kaggle | 386 | 360 | 26 | 382 | 0 |
| 1965-66 | Kaggle | 387 | 360 | 27 | 386 | 0 |
| 1966-67 | Kaggle | 436 | 405 | 31 | 436 | 0 |
| 1967-68 | Kaggle | 532 | 492 | 40 | 530 | 0 |
| 1968-69 | Kaggle | 613 | 574 | 39 | 612 | 0 |
| 1969-70 | Kaggle | 614 | 574 | 40 | 614 | 0 |
| 1970-71 | Kaggle | 737 | 697 | 40 | 736 | 0 |
| 1971-72 | Kaggle | 734 | 697 | 37 | 734 | 0 |
| 1972-73 | Kaggle | 738 | 697 | 41 | 737 | 0 |
| 1973-74 | Kaggle | 738 | 697 | 41 | 738 | 0 |
| 1974-75 | Kaggle | 785 | 738 | 47 | 784 | 0 |
| 1975-76 | Kaggle | 788 | 738 | 50 | 788 | 0 |
| 1976-77 | Kaggle | 955 | 902 | 53 | 955 | 0 |
| 1977-78 | Kaggle | 953 | 902 | 51 | 953 | 0 |
| 1978-79 | Kaggle | 955 | 902 | 53 | 953 | 0 |
| 1979-80 | Kaggle | 950 | 902 | 48 | 949 | 0 |
| 1980-81 | Kaggle | 996 | 943 | 53 | 993 | 0 |
| 1981-82 | Kaggle | 990 | 943 | 47 | 990 | 0 |
| 1982-83 | Kaggle | 986 | 943 | 43 | 985 | 0 |
| 1983-84 | Kaggle | 1,022 | 943 | 79 | 1,020 | 0 |
| 1984-85 | Kaggle | 1,011 | 943 | 68 | 1,011 | 0 |
| 1985-86 | Kaggle | 1,011 | 943 | 68 | 1,008 | 1,011 |
| 1986-87 | Kaggle | 1,014 | 943 | 71 | 1,014 | 1,014 |
| 1987-88 | Kaggle | 1,023 | 943 | 80 | 1,023 | 1,023 |
| 1988-89 | Kaggle | 1,087 | 1,025 | 62 | 1,087 | 1,087 |
| 1989-90 | Kaggle | 1,179 | 1,107 | 72 | 1,166 | 1,179 |
| 1990-91 | Kaggle | 1,175 | 1,107 | 68 | 1,175 | 1,175 |
| 1991-92 | Kaggle | 1,180 | 1,107 | 73 | 1,180 | 1,180 |
| 1992-93 | Kaggle | 1,183 | 1,107 | 76 | 1,183 | 1,183 |
| 1993-94 | Kaggle | 1,184 | 1,107 | 77 | 1,183 | 1,184 |
| 1994-95 | Kaggle | 1,180 | 1,107 | 73 | 1,180 | 1,180 |
| 1995-96 | Kaggle | 1,257 | 1,189 | 68 | 1,256 | 1,257 |
| 1996-97 | Kaggle | 1,261 | 1,189 | 72 | 1,257 | 1,261 |
| 1997-98 | Kaggle | 1,260 | 1,189 | 71 | 1,260 | 1,260 |
| 1998-99 | Kaggle | 791 | 725 | 66 | 791 | 791 |
| 1999-00 | Kaggle | 1,264 | 1,189 | 75 | 1,264 | 1,264 |
| 2000-01 | Kaggle | 1,260 | 1,189 | 71 | 663 | 1,260 |
| 2001-02 | Kaggle | 1,260 | 1,189 | 71 | 1,260 | 1,260 |
| 2002-03 | Kaggle | 1,277 | 1,189 | 88 | 1,276 | 1,277 |
| 2003-04 | Kaggle | 1,271 | 1,189 | 82 | 1,268 | 1,271 |
| 2004-05 | Kaggle | 1,314 | 1,230 | 84 | 1,302 | 1,314 |
| 2005-06 | Kaggle | 1,319 | 1,230 | 89 | 1,319 | 1,319 |
| 2006-07 | Kaggle | 1,309 | 1,230 | 79 | 1,309 | 1,309 |
| 2007-08 | Kaggle | 1,316 | 1,230 | 86 | 1,316 | 1,316 |
| 2008-09 | Kaggle | 1,315 | 1,230 | 85 | 1,315 | 1,315 |
| 2009-10 | Kaggle | 1,312 | 1,230 | 82 | 1,311 | 1,312 |
| 2010-11 | Kaggle | 1,311 | 1,230 | 81 | 1,310 | 1,311 |
| 2011-12 | Kaggle | 1,074 | 990 | 84 | 1,073 | 1,074 |
| 2012-13 | Kaggle | 1,314 | 1,229 | 85 | 1,314 | 1,314 |
| 2013-14 | Kaggle | 1,319 | 1,230 | 89 | 1,319 | 1,319 |
| 2014-15 | Kaggle | 1,311 | 1,230 | 81 | 1,311 | 1,311 |
| 2015-16 | Kaggle | 1,316 | 1,230 | 86 | 1,316 | 1,316 |
| 2016-17 | Kaggle | 1,309 | 1,230 | 79 | 1,309 | 1,309 |
| 2017-18 | Kaggle | 1,312 | 1,230 | 82 | 1,311 | 1,312 |
| 2018-19 | Kaggle | 1,312 | 1,230 | 82 | 1,312 | 1,312 |
| 2019-20 | Kaggle | 1,143 | 1,059 | 84 | 1,143 | 1,143 |
| 2020-21 | Kaggle | 1,171 | 1,080 | 91 | 1,171 | 1,171 |
| 2021-22 | Kaggle | 1,323 | 1,230 | 93 | 1 | 1,323 |
| 2022-23 | Kaggle | 1,320 | 1,230 | 90 | 1,320 | 1,319 |
| 2023-24 | Kaggle | 1,319 | 1,231 | 88 | 1,318 | 1,318 |
| 2024-25 | Kaggle | 1,315 | 1,225 | 90 | 1,315 | 1,315 |
| 2025-26 | Kaggle | 1,322 | 1,231 | 91 | 1,322 | 1,322 |
| 2026-27 | Kaggle | 270 | 270 | 0 | 270 | 270 |

Gaps and short seasons:

- **1946-47 to 1953-54:** final scores only, no quarter scores or box scores.
- **1954-55 to 1984-85:** quarter scores for nearly every game, no possessions.
- **1998-99:** 791 games, lockout season.
- **2000-01:** halftime scores for 663 of 1,260 games. The source has no quarter scores for the rest.
- **2011-12:** 1,074 games, lockout season.
- **2019-20 and 2020-21:** 1,143 and 1,171 games, seasons shortened by COVID-19.
- **2021-22:** halftime scores for 1 of 1,323 games. The source has no quarter scores for the rest.
- **2026-27:** in progress, 270 games through 2026-11-25.

No season between 1946-47 and 2025-26 is missing.

## Test fixtures

All three are small subsets, so the tests run offline. The folder is about 0.5 MB.

| File | Contents | Licence |
| --- | --- | --- |
| `tests/fixtures/games_2024_2026.csv` | The stitched game table for 2023-24 to 2025-26 (3,956 games), as written by `nbapred-train` | CC0-1.0 (derived from the Kaggle data) |
| `tests/fixtures/kaggle/` | 18 games from the three Kaggle files (their `TeamStatistics.csv` rows, plus all of `TeamHistories.csv`), picked to test game types, neutral sites, the 2015 Finals and missing quarter scores | CC0-1.0 |
| `tests/fixtures/games_538_2015.csv` | FiveThirtyEight rows for the 2014-15 season, the columns listed above only | CC BY 4.0 |

## Attribution

- Kaggle data: eoinamoore, "Historical NBA Data and Player Box Scores", CC0-1.0, from NBA.com. No attribution is required; it is given here anyway.
- FiveThirtyEight data: "Historical NBA Elo", CC BY 4.0. This project changes it by filtering rows, dropping columns and reshaping each game into one home/away row.

## Sources considered and not used

- stats.nba.com through `nba_api`: its robots.txt returned "Access Denied" to a plain client, so no further requests were made and no data came from it.
- Scraping Basketball-Reference: not attempted. Both datasets above have open licences, and scraping does not.
