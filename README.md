# JDataCompetitionFinalCode
# JDATA算法大赛

## 依赖库

- pandas
- sklearn
- xgboost

## 项目结构

- data: 储存数据目录(raw data save path)
- cache: 缓存目录(interval data save path)
- sub: 结果目录(submission file save path)

-----------------ui--------------------------------
- gen_feat: 生成ui特征
- train.py: 单模型ui模型执行文件
- ui_mix_xgb_ave.py: 参数不同的xgboost average 模型融合
- ui_feat_ensemble.py: 特征重要性不同的模型融合
- ui_mix_weight.py: 权重投票模型 权重xgb.get
- M1_cate.py : 训练样本（5天内加入购物车ui)（未执行）
- M2_not_cate.py : 训练样本（5天内加未入购物车ui)（未执行）
- M3_5_7.py : 训练样本（5-7天ui)(未执行)
-----------------user---------------------------
- user_feat: 生产user特征
- user_train.py: 单模型user模型执行文件
- user_cart_train.py: 只选取将cate8加入过购物车的用户加入训练样本（解决类别不平衡问题）
- user_feat_ensemble.py: 特征重要性不同的模型融合
- user_mix_xgb_ave.py: 参数不同的xgboost average 模型融合

## 使用说明

python train.py
