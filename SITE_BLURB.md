# NBA score predictor

Predicts both teams' final scores for an NBA game, either before tip-off or from the score at halftime or after three quarters. It is trained on every season from the 1946-47 BAA to the current one: 73,440 games stitched from NBA.com box scores (Kaggle, CC0) and FiveThirtyEight's early-season records. A game-by-game check of the 61 seasons both sources cover runs on every build. Features include Elo, season and last-10-game scoring, and offensive and defensive rating and pace from box scores. A test rewrites every same-day and later result and checks that no feature changes.

On 1,315 held-out games from 2024-25, the predicted totals miss by 14.63 points on average, against 15.05 for the best baseline. The model picks the winner 65.7% of the time, about one point better than plain Elo, a gap inside the noise. From the halftime score, the margin error falls from 11.1 to 8.7 points.
