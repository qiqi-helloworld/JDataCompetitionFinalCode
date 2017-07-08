#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "QIQI"

import xgboost as xgb
from sklearn.model_selection import train_test_split
from gen_feat import M2_make_train_set
from gen_feat import M2_make_test_set
from gen_feat import get_cate8_labels
from user_feat import get_user_labels
from function import get_important_feat_names
from function import offline_ui
from function import get_new_ui_trainning_dat
from function import new_idea_predict_dat
import pandas as pd
import random
import numpy as np



def new_idea_ui_xgboost_make_submission(make_train_set, make_test_set, savename, online_user_path, days = 0):
    ''' 将重要特征单独训练一个模型，权重为0.3
     在其余的非重要特征中， 随机选取10次采样组成10个不同训练集，训练10个不同模型，每个权重为0.07
    并对这11和模型的结果取平均'''


    print("xgbsub ui=========================================================")
    #训练日期
    train_start_date = '2016-02-06'
    train_end_date = '2016-04-11'
    test_start_date = '2016-04-11'
    test_end_date = '2016-04-16'
    #预测日期
    sub_start_date = '2016-02-06'
    sub_end_date = '2016-04-16'

    #读取全部训练特征集
    user_index, training_data, label = make_train_set(train_start_date, train_end_date, test_start_date, test_end_date, days = days)

    #构造训练集，我们使用训练特征集中真实购买用户的数据作为训练集
    user_label = get_user_labels(test_start_date, test_end_date)       #训练数据中真实购买用户
    user_index, training_data, label = get_new_ui_trainning_dat(user_index, training_data, label, user_label)#真实购买用户对应的训练集


    #训练集 训练，eval数据划分
    X_train, X_test, y_train, y_test = train_test_split(training_data.values, label.values, test_size=0.2,
                                                        random_state=0)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label= y_test)
    param = {'eta': 0.05, 'n_estimators': 1000, 'max_depth': 5, 'eval_metric': 'auc',
             'min_child_weight': 1, 'gamma': 0, 'subsample': 1, 'colsample_bytree': 0.8,
             'scale_pos_weight': 1,  'silent': 1, 'objective': 'binary:logistic'}

    num_round = 1000
    param['nthread'] = 20
    evallist = [(dtrain, 'train'),(dtest, 'eval')]
    bst = xgb.train(param, dtrain, num_round, evallist, early_stopping_rounds = 50)
    inportance_feature = pd.Series(bst.get_fscore()).sort_values(ascending=False)  #获取重要特征

    important_feat_names = get_important_feat_names(inportance_feature, training_data.columns.values)
    important_trainning_dat = training_data[important_feat_names]    #重要特征对应的训练集

    trivial_feat_names = [x for x in training_data.columns.values if x not in important_feat_names]
    trivial_trainning_dat = training_data[trivial_feat_names] #非重要特征（重要特征之外的所有特征）对应的训练集
    print trivial_trainning_dat.shape



    #构造预测集， 预测的真实购买用户中所对应的所有用户商品对作为预测集
    sub_user_index, sub_trainning_data = make_test_set(sub_start_date, sub_end_date, days = days)   #获取所有预测特征集
    # 在预测特征集中找到所有真实购买用户所对应的用户商品对作为预测集
    sub_user_index, sub_trainning_data = new_idea_predict_dat(sub_user_index, sub_trainning_data, online_user_path)
    sub_user_index = pd.DataFrame(sub_user_index)
    trivial_sub_trainning_dat = sub_trainning_data[trivial_feat_names]
    print trivial_sub_trainning_dat.shape

    # #线下测试集 在此不做注释
    # cv_train_start_date = '2016-02-01'
    # cv_train_end_date = '2016-04-06'
    # cv_test_start_date = '2016-04-06'
    # cv_test_end_date = '2016-04-11'
    # cv_user_index, cv_training_data, cv_label = make_train_set(cv_train_start_date, cv_train_end_date, cv_test_start_date,
    #                                                            cv_test_end_date, days = days)
    # cv_user_index, cv_training_data= new_idea_predict_dat(cv_user_index, cv_training_data, offline_path)
 #   cv_unique_dat = cv_user_index.drop_duplicates('user_id',keep='first')

    weights = [0.3, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07]  #不同模型权重
    pred_mat = np.zeros((sub_user_index.shape[0], len(weights)))
    #cv_pred_mat = np.zeros((cv_user_index.shape[0], len(weights)))
    for i in range(len(weights)):  # 11个模型分别进行训练
        #print("mix i", i)
        if i == 0:
            X_train, X_test, y_train, y_test = train_test_split(important_trainning_dat.values, label.values,
                                                                test_size=0.2,
                                                                random_state=0)  #对重要特征训练数据进行 训练，eval 数据划分
            tmp_sub_trainning_data = sub_trainning_data[important_feat_names]    #预测数据
   #         cv_tmp_sub_trainning_data = cv_training_data[important_feat_names]   #线下评估数据

        else:

            num_sam = int(len(trivial_feat_names)* 0.4)  #非重要特征随机选取特征数
            print("num_sam:", num_sam)
            sample_feat_names = random.sample(trivial_feat_names, num_sam)    #随机选取非重要特征
            sample_train_dat = trivial_trainning_dat[sample_feat_names]       #构成非重要特征训练集
            X_train, X_test, y_train, y_test = train_test_split(sample_train_dat.values, label.values,
                                                                test_size=0.2,
                                                                random_state=0) #训练，eval 数据划分
            tmp_sub_trainning_data = trivial_sub_trainning_dat[sample_feat_names]      #预测数据
    #        cv_tmp_sub_trainning_data = cv_training_data[sample_feat_names]     #线下评估数据

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)
        evallist = [(dtrain, 'train'), (dtest, 'eval')]
        tmp_sub_trainning_data = xgb.DMatrix(tmp_sub_trainning_data.values)
        gbm = xgb.train(param, dtrain, num_boost_round=1000, evals=evallist, early_stopping_rounds=50)
        pred_mat[:, i] = gbm.predict(tmp_sub_trainning_data)

   #     cv_tmp_sub_trainning_data = xgb.DMatrix(cv_tmp_sub_trainning_data.values)
   #     cv_pred_mat[:,i] = gbm.predict(cv_tmp_sub_trainning_data)

    pred_mat = pred_mat*weights  #结果乘以权重
    y = pred_mat.sum(1)
    sub_user_index['label'] = y
    sub_user_index = sub_user_index.sort(['label'], ascending=False).reset_index(drop=True)
    sub_user_index.drop_duplicates('user_id', keep='first', inplace=True)
    sub_user_index.reset_index(drop=True)
    prob_path = "./cache/" + str(savename) + ".csv"
    print("sub_user_index", sub_user_index.shape)
    sub_user_index.to_csv(prob_path, index=False, index_label=False)


    # #线下测试， 不做注释
    # cv_pred_mat = cv_pred_mat*weights
    # cv_user_index['label'] = cv_pred_mat.sum(1)
    # print("cv_user_index:", cv_user_index.head())
    # cv_user_index = cv_user_index.sort(['label'], ascending=False).reset_index(drop=True)
    # cv_user_index.drop_duplicates('user_id', keep='first', inplace=True)
    # cv_user_index.reset_index(drop=True)
    # print(cv_user_index.shape)
    # prob_path = "./cache/offline_" + str(savename) + ".csv"
    # cv_user_index.to_csv(prob_path, index=False, index_label=False)

    return (sub_user_index)


if __name__ == '__main__':


    cv_new_idea_ui_index_30 = new_idea_ui_xgboost_make_submission(M2_make_train_set, M2_make_test_set,
                                                                     "final_sub_prob",
                                                                     "./cache/pred_user.csv",
                                                                     days=30)
