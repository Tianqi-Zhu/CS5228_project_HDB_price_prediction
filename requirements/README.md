# 依赖分层

从项目根目录运行安装命令，`-r` 的递归包含路径以依赖文件所在目录为基准。

| 文件 | 用途 | 安装命令 |
| --- | --- | --- |
| `eda.txt` | 数据读取、统计、图表与预处理 | `python -m pip install -r requirements/eda.txt` |
| `model.txt` | EDA基础上增加LightGBM、CatBoost等 | `python -m pip install -r requirements/model.txt` |
| `search.txt` | 官方FT、TabNet和XGBoost适配包 | `python -m pip install -r requirements/search.txt` |
| `deep.txt` | 模型依赖、PyTorch CUDA 12.8及搜索包 | `python -m pip install -r requirements/deep.txt` |
| 根目录 `requirements.txt` | 常规完整工具，含PDF和Excel输入 | `python -m pip install -r requirements.txt` |

完整Conda环境仍使用根目录 `environment.yml`。`deep.txt` 的CUDA wheel适合当前NVIDIA环境；
其他平台应选择兼容的PyTorch安装方式，不直接照搬GPU版本。

本机额外模型包保存在Git忽略的 `.cache/model-deps/`，由运行器自动发现。
此文件夹是本地依赖安装位置，和可以随时删除的特征缓存用途不同；删除后需重新安装这些包。
