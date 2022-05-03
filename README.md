# CPSC2021
Paroxysmal Atrial Fibrillation Events Detection from Dynamic ECG Recordings: The 4th China Physiological Signal Challenge 2021

![format-check](https://github.com/DeepPSP/cpsc2021/actions/workflows/check-formatting.yml/badge.svg)

## Offline generated training data

[CPSC2021-sliced](https://www.kaggle.com/wenh06/cpsc2021-sliced) on Kaggle.

## Graphical Abstract of the Solution
![res_pht](/images/graphical-abstract.svg)


## Results (Rankings)

### Results on the partial hidden test set (only the top of the table is included)

to update...

### Results on the partial hidden test set
![res_pht](/images/cpsc2021-final_validation.png)

### Results on the validation set

Raw results are gathered into one zip file, the `val_res.zip` in the [`results`](/results/) folder

|     Network(s)    | Merge Rule    | Score on Partial Hidden Test Set | Score on Validation Set|
|-------------------|---------------|----------------------------------|------------------------|
|   LSTM            |  NA           |   1.9392                         | 2.0621                 |
|   SeqTag          |  NA           |   1.9781                         | 2.1577                 |
|   U-Net           |  NA           |   1.3699                         | NA                     |
|  LSTM + U-Net     |  Union        |   1.7829                         | NA                     |
|  LSTM + SeqTag    | Intersection  |   1.9287                         | NA                     |
|  LSTM + SeqTag    |  Union        |   1.9766                         | 2.1682                 |
| **LSTM + SeqTag** | **New Union** |   ?                              | **2.2179**             |

### Confusion matrices of the LSTM model and the SeqTag model
<p align="middle">
  <img src="/images/rr-lstm-confusion-matrix.svg" width="33%" />
  &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp;
  <img src="/images/seq-tag-confusion-matrix.svg" width="33%" />
</p>

### More detailed analysis using `pandas`
<p align="middle">
  <img src="/images/res_ana_1.png" width="60%" />
  <img src="/images/res_ana_2.png" width="110%" />
</p>

## References
to add ...
