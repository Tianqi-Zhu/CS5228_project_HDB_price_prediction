# Appendix：替代假设、解释性实验与取舍依据

main 负责最小充分主线；这里保存补充证据、反例和未采用的方案。每张 X 卡按“问题—对照—观察—取舍”组织，具体状态见[执行表](../../docs/planning/appendix-execution-status.md)，机器可读版本为[决策目录](decision-catalog.json)。报告正文保持不变。

## 按要解释的问题选择实验

| 主题 | 入口 | 对 main 的价值 |
|---|---|---|
| 数据处理的依据与代价 | [X01 特征/空间控制](X01/README.md)、[X02 清洗](X02/README.md)、[X08 设施时间](X08/README.md) | 解释保留/删除/规范化的条件；区分关联和因果 |
| 替代表示为什么未进入终点 | [X05 RNN/LSTM、动态面与Concat](X05/README.md)、[X06 神经/基础模型](X06/README.md) | 展示不同假设、匹配对照和负结果 |
| 融合与最终选择依据 | [X03 另两版本](X03/README.md)、[X04 种子/分组/结构](X04/README.md)、[X11 实际组件互补](X11/README.md) | 解释选择规则与窗口依赖，保留有利/不利结果 |
| 差值可信度与适用边界 | [X07 匹配重复待做](X07/README.md)、[X09 长跨度边界](X09/README.md)、[X12 配对区间](X12/README.md) | 区分点估计、固定预测敏感性和训练随机性 |
| 信息来源与复现层次 | [X10 目标重建审计](X10/README.md)、[X13 复现](X13/README.md) | 排除不可比成绩；区分数据身份、重放、推理和完整重训 |

## 13 张卡的真实执行状态

| ID | 状态 | 本轮补充/保留内容 |
|---|---|---|
| [X01](X01/README.md) | archived | 删列/组冗余＋新增空间调整、MRT×商场控制 |
| [X02](X02/README.md) | archived | ReLU/IQR＋MLP匹配清洗，不混拟合/校准期 |
| [X03](X03/README.md) | archived | 历史/宏观/结构三份原CSV与固定组件重组 |
| [X04](X04/README.md) | archived | 8/9/10轮、覆盖条件、PLR年度回退等负结果 |
| [X05](X05/README.md) | archived | 补入RNN/LSTM、动态租金面；保留Concat时空探索 |
| [X06](X06/README.md) | archived | MLP解释图/预测、Concat架构、TabPFN替代模型 |
| [X07](X07/README.md) | planned | 匹配外部42/43/44全管线重复未跑；前置条件已列 |
| [X08](X08/README.md) | partial | 独立来源/特征审计完成；历史日期重训未跑 |
| [X09](X09/README.md) | partial | 共同月/查询跨度已审计；两终点匹配L未跑 |
| [X10](X10/README.md) | archived | 公开目标重合只作来源审计，不进性能排名 |
| [X11](X11/README.md) | partial | J组件消融、交叉矩、月/支持组已完成；Y缺预测 |
| [X12](X12/README.md) | partial | J月份区块/楼栋簇各3,000次已完成；Y待补 |
| [X13](X13/README.md) | partial | 三提交与F2组件重放；新Conda/完整GPU重训未做 |

`archived` 表示所列历史材料已按来源保存，可用于既有结果论证；不表示本轮重训过全部实验。`partial` 始终写明已完成的子任务。本轮新增科学模型拟合数为0。

## 保留的作者工作与来源主题档案

- [MLP 解释实验、三组矢量图和逐行验证](X06/mlp_explain/README.md)
- [Concat 时空特征探索](X05/concat_explorations/README.md)、[Concat 架构对照](X06/concat_explorations/README.md)
- [RNN/LSTM 源报告](X05/external_studies/rnn_lstm/docs/rnn_lstm_report.md)、[动态租金面源报告](X05/external_studies/dynamic_surface/reports/factored_rental_surface/report.md)

T 档案按原来源专题细分，服务上面的 X 问题，不作为额外最终方案：

| 来源专题 | 对应解释问题 |
|---|---|
| [T01 数据/清洗](T01/README.md) | X02/X08，含原MRT状态审计 |
| [T02 时间/经济](T02/README.md) | X05/X09，固定原点与更新条件 |
| [T03 空间/外部特征](T03/README.md) | X01/X05/X08，来源/日期限制 |
| [T04 上下文/输入表示](T04/README.md) | X06，实际上下文与schema |
| [T05 适配/残差](T05/README.md) | X06，训练和损失对照 |
| [T06 历史融合](T06/README.md) | X03/X11，原成员和选择记录 |
| [T07 TabM/Transformer](T07/README.md) | X06，方法/预算差异 |
| [T08 环境/合并来源](T08/README.md) | X13，来源和运行范围 |

## 使用边界

最终PDF只挑最能支持main结论的少量补充图表，不把本目录整体塞进报告。不同原点、目标、样本范围、宏观可用性和拟合预算分开比较。完整论文核心解释仍在main；源快照或待做训练不会自动变成已完成证据。
