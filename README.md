
# [JDATA算法大赛(JDATA Algorithm Competition) ](https://www.datafountain.cn/projects/jdata/)


## 依赖库(Dependent Package)

- pandas
- sklearn
- xgboost

## 项目结构(Project Structure)

- data: 储存数据目录(raw data save path)
- cache: 缓存目录(interval data save path)
- sub: 结果目录(submission file save path)

---------------------user model----------------------------
- user_model.py: user model execute py, single model.
- user_feat.py: user feature generate py.

--------------------ui model---------------------------
- gen_ui_feat.py: ui feature generate py.
- ui_feat_ensemble.py: ui model execute py, model based on feature improtance weight ensemble.

-------------------------results----------------------------
- function.py: self-define function used in results ensemble and offline test.
- sub.py: sub file generate py.

## 使用说明(execute order):
Notice: Step by Step, and the process time would cost few hours.
- python user_model.py
- python ui_feat_ensemble.py
- python sub.py

