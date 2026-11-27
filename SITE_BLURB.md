# NBA score predictor

Predicts both teams' final scores for an NBA game from what was known before tip-off: Elo ratings, season and last-10-game scoring, rest days and league scoring level. A test that rewrites every same-day and later result and checks the features don't change guards against leakage, and a held-out season compares ridge regression and gradient boosting with season-average and Elo baselines.

On 1,311 held-out games from 2014-15, the model's predicted totals miss by 13.85 points on average, against 14.18 for the best baseline. It picks the winner 67.2% of the time, which a plain Elo rating matches within noise.
