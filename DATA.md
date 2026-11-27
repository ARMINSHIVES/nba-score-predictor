# Data

## Source

| | |
| --- | --- |
| Dataset | FiveThirtyEight, "Historical NBA Elo" (`nba-elo/nbaallelo.csv`) |
| Publisher | FiveThirtyEight (ABC News) |
| Download URL | <https://raw.githubusercontent.com/fivethirtyeight/data/master/nba-elo/nbaallelo.csv> |
| Repository | <https://github.com/fivethirtyeight/data/tree/master/nba-elo> |
| Licence | Creative Commons Attribution 4.0 International (CC BY 4.0) |
| Licence terms | <https://creativecommons.org/licenses/by/4.0/>, as stated in the repository README: <https://github.com/fivethirtyeight/data/blob/master/README.md> |
| Upstream source | FiveThirtyEight credits Basketball-Reference.com for the game information |
| Coverage | 1946-47 to the 2015 Finals; 59,008 NBA games after filtering (ABA games dropped) |
| SHA-256 | `d46ed3540ee8d9eca31b3e94cc8c777e0be5156173d814ebf65b8195e8d616bc` |
| Retrieved | 2026-11-26 |

`nbapred-train` downloads the file once to `data/raw/` (not committed) and refuses it if the checksum differs.

## Columns used

`game_id`, `lg_id`, `_iscopy`, `year_id`, `date_game`, `is_playoffs`, `team_id`, `fran_id`, `pts`, `opp_id`, `opp_fran`, `opp_pts`, `game_location`.

The file's `elo_i`, `elo_n`, `win_equiv` and `forecast` columns are not read. `elo_n` is the rating after the game, so using it would leak the result.

## Test fixture

`tests/fixtures/games_2012_2015.csv` is a subset of the same file: NBA games from the 2011-12 to 2014-15 seasons, one row per game, the columns above only. It is redistributed under CC BY 4.0 with this attribution. Changes: rows filtered and columns dropped.

## Attribution

Data: FiveThirtyEight, "Historical NBA Elo", CC BY 4.0. This project changes the data by filtering rows, dropping columns, and reshaping each game into one home/away row.

## Sources considered and not used

- Kaggle NBA datasets: need Kaggle API credentials, which were not set up on the build machine.
- stats.nba.com through `nba_api`: its robots.txt returned "Access Denied" to a plain client, and further requests were not made, so no data came from it.
