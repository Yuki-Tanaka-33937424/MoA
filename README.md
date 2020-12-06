# MoA
MoAコンペのコード置き場です。

### training<br>
- 各モデルの学習に使用したコード

### inference <br>
- 各モデルの推論に使用したコード

### submission
- inferenceのコードを切りはりして一つにしたコード
  - submission_final1: Public 0.01832 Private 0.01622
  - submission_final2: Public 0.01832 Private 0.01620

ファイル名は全て~1, ~2になっており、<br>
~1はCVをMultilabelStratifiedKFold, <br>
~2はCVをChris Method で行っています。<br>
それ以外は全て同じです。
