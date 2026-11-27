## Validation 2024 (fit on 1947-2023)

| Model | Home pts MAE | Away pts MAE | Margin MAE | Total MAE | Winner acc. |
| --- | --- | --- | --- | --- | --- |
| Season average | 9.62 | 9.84 | 11.54 | 15.44 | 64.5% |
| Elo margin | 10.08 | 10.06 | 11.23 | 16.49 | 64.1% |
| Ridge regression | 9.22 | 9.29 | 11.08 | 14.55 | 65.0% |
| Gradient boosting | 9.33 | 9.40 | 11.12 | 14.78 | 65.1% |

## 2025 season, 1315 games, pre-game

| Model | Home pts MAE | Away pts MAE | Margin MAE | Total MAE | Winner acc. |
| --- | --- | --- | --- | --- | --- |
| Season average | 9.50 | 9.59 | 11.61 | 15.05 | 65.4% |
| Elo margin | 9.73 | 9.87 | 11.29 | 15.91 | 64.5% |
| Ridge regression | 9.21 | 9.24 | 11.11 | 14.63 | 65.7% |
| Gradient boosting | 9.25 | 9.32 | 11.12 | 14.75 | 65.9% |

### 2025 in-game from halftime, 1315 games

| Model | Home pts MAE | Away pts MAE | Margin MAE | Total MAE | Winner acc. |
| --- | --- | --- | --- | --- | --- |
| In-game ridge | 7.03 | 7.23 | 8.70 | 11.00 | 75.4% |
| Pace extrapolation | 9.87 | 10.12 | 13.78 | 14.45 | 73.0% |
| Score + pre-game share | 7.08 | 7.23 | 8.90 | 10.96 | 75.3% |

### 2025 in-game from end of Q3, 1315 games

| Model | Home pts MAE | Away pts MAE | Margin MAE | Total MAE | Winner acc. |
| --- | --- | --- | --- | --- | --- |
| In-game ridge | 4.97 | 5.18 | 6.20 | 7.80 | 81.8% |
| Pace extrapolation | 5.85 | 5.96 | 7.88 | 8.62 | 82.8% |
| Score + pre-game share | 5.06 | 5.22 | 6.32 | 7.87 | 82.3% |

## 2026 season, 1322 games, pre-game

| Model | Home pts MAE | Away pts MAE | Margin MAE | Total MAE | Winner acc. |
| --- | --- | --- | --- | --- | --- |
| Season average | 9.86 | 9.75 | 12.03 | 15.43 | 66.3% |
| Elo margin | 10.17 | 9.73 | 11.60 | 16.10 | 67.5% |
| Ridge regression | 9.72 | 9.49 | 11.49 | 15.20 | 69.3% |
| Gradient boosting | 9.74 | 9.48 | 11.47 | 15.21 | 67.0% |

### 2026 in-game from halftime, 1322 games

| Model | Home pts MAE | Away pts MAE | Margin MAE | Total MAE | Winner acc. |
| --- | --- | --- | --- | --- | --- |
| In-game ridge | 6.86 | 7.09 | 8.69 | 11.05 | 77.0% |
| Pace extrapolation | 9.88 | 10.07 | 13.44 | 14.60 | 73.5% |
| Score + pre-game share | 6.89 | 7.05 | 8.68 | 11.06 | 76.5% |

### 2026 in-game from end of Q3, 1322 games

| Model | Home pts MAE | Away pts MAE | Margin MAE | Total MAE | Winner acc. |
| --- | --- | --- | --- | --- | --- |
| In-game ridge | 5.02 | 5.21 | 6.43 | 8.06 | 82.8% |
| Pace extrapolation | 6.00 | 6.06 | 8.05 | 8.98 | 81.5% |
| Score + pre-game share | 5.14 | 5.20 | 6.50 | 8.13 | 82.6% |

## 2027 season to 2026-11-25 (in progress), 270 games, pre-game

| Model | Home pts MAE | Away pts MAE | Margin MAE | Total MAE | Winner acc. |
| --- | --- | --- | --- | --- | --- |
| Season average | 9.59 | 9.99 | 11.41 | 15.21 | 64.8% |
| Elo margin | 9.83 | 10.13 | 11.54 | 15.68 | 64.8% |
| Ridge regression | 9.61 | 10.06 | 11.23 | 15.33 | 65.6% |
| Gradient boosting | 9.72 | 10.08 | 11.40 | 15.40 | 65.6% |

### 2027 in-game from halftime, 270 games

| Model | Home pts MAE | Away pts MAE | Margin MAE | Total MAE | Winner acc. |
| --- | --- | --- | --- | --- | --- |
| In-game ridge | 7.03 | 7.32 | 8.43 | 11.60 | 77.0% |
| Pace extrapolation | 9.72 | 10.00 | 12.96 | 14.90 | 77.4% |
| Score + pre-game share | 7.05 | 7.22 | 8.55 | 11.58 | 77.0% |

### 2027 in-game from end of Q3, 270 games

| Model | Home pts MAE | Away pts MAE | Margin MAE | Total MAE | Winner acc. |
| --- | --- | --- | --- | --- | --- |
| In-game ridge | 4.73 | 5.16 | 6.34 | 7.86 | 80.7% |
| Pace extrapolation | 6.02 | 5.76 | 8.20 | 8.50 | 80.4% |
| Score + pre-game share | 4.96 | 5.18 | 6.54 | 7.98 | 80.4% |

## Error by decade (walk-forward)

| Seasons | Games | Model | Margin MAE | Total MAE |
| --- | --- | --- | --- | --- |
| 1950-1959 | 3595 | Season average | 9.66 | 15.30 |
| 1950-1959 | 3595 | Elo margin | 8.65 | 16.02 |
| 1950-1959 | 3595 | Ridge regression | 8.74 | 15.94 |
| 1960-1969 | 4185 | Season average | 9.96 | 15.26 |
| 1960-1969 | 4185 | Elo margin | 9.31 | 16.06 |
| 1960-1969 | 4185 | Ridge regression | 9.47 | 15.05 |
| 1970-1979 | 7997 | Season average | 10.01 | 14.74 |
| 1970-1979 | 7997 | Elo margin | 9.27 | 15.74 |
| 1970-1979 | 7997 | Ridge regression | 9.22 | 14.37 |
| 1980-1989 | 10090 | Season average | 9.75 | 14.94 |
| 1980-1989 | 10090 | Elo margin | 8.84 | 16.68 |
| 1980-1989 | 10090 | Ridge regression | 8.79 | 14.34 |
| 1990-1999 | 11650 | Season average | 9.95 | 14.75 |
| 1990-1999 | 11650 | Elo margin | 9.14 | 16.18 |
| 1990-1999 | 11650 | Ridge regression | 9.53 | 14.26 |
| 2000-2009 | 12905 | Season average | 9.89 | 14.68 |
| 2000-2009 | 12905 | Elo margin | 9.23 | 16.08 |
| 2000-2009 | 12905 | Ridge regression | 9.20 | 14.17 |
| 2010-2019 | 12890 | Season average | 10.18 | 14.44 |
| 2010-2019 | 12890 | Elo margin | 9.63 | 15.50 |
| 2010-2019 | 12890 | Ridge regression | 9.63 | 14.09 |
| 2020-2029 | 9183 | Season average | 11.37 | 15.24 |
| 2020-2029 | 9183 | Elo margin | 11.08 | 15.94 |
| 2020-2029 | 9183 | Ridge regression | 11.02 | 14.86 |
