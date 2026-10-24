# 比赛输入

本分支包含三份运行所需的比赛文件，均原样复制，未删行、改标签或调整顺序。

| 文件 | 行数 | 用途 |
| --- | ---: | --- |
| `train.csv` | 150,000 | 使用已知 `MONTHLY_RENT` 核验公开数据重建误差 |
| `test.csv` | 50,000 | 生成租金结果；文件不含比赛真实租金标签 |
| `example-submission.csv` | 50,000 | 确定 `Id,Predicted` 格式及 ID 顺序；示例预测值不参与计算 |

来源：[CS5228-2610 比赛](https://www.kaggle.com/competitions/cs-5228-2610-project)。
原始字节哈希见 `results/summary.json` 的 `inputs`；程序每次运行都会重新计算，并检查运行前后保持一致。
如替换任何输入数据，匹配覆盖率和误差可能变化，历史结果不再自动适用。
