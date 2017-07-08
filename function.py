#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "QIQI"

from user_feat import get_user_labels
from gen_feat import get_cate8_labels
import numpy as np
import pandas as pd
def report_F11(pred, y_true):
    all_user_set = y_true['user_id'].unique()
    all_user_test_set = pred['user_id'].unique()
    pos = 0
    neg = 0
    for user_id in all_user_test_set:
        if user_id in all_user_set:
            pos += 1
        else:
            neg += 1
    all_user_acc = 1.0 * pos / (pos + neg)
    all_user_recall = 1.0 * pos / len(all_user_set)
   # print('user(accuracy): ' + str(all_user_acc), len(all_user_set))
  #  print('user(recall):' + str(all_user_recall), len(all_user_set))
    F11 = 6.0 * all_user_recall * all_user_acc / (5.0 * all_user_recall + all_user_acc)
    print('accuracy: ' + str(all_user_acc),'recall:' + str(all_user_recall),"F11 score:", F11, pos, len(all_user_set))

def offline_user(cv_ui_index):
    cv_test_start_date = '2016-04-06'
    cv_test_end_date = '2016-04-11'
    true_labels = get_user_labels(cv_test_start_date, cv_test_end_date)
    print(true_labels)
    print("900----------------------------------------------")
    report_F11(cv_ui_index.head(900), true_labels)
    print("950----------------------------------------------")
    report_F11(cv_ui_index.head(950), true_labels)
    print("1000--------------------------------------------")
    report_F11(cv_ui_index.head(1000), true_labels)


def report(pred, label):

    actions = label

    result = pred

    # 所有用户商品对
    all_user_item_pair = actions['user_id'].map(str) + '-' + actions['sku_id'].map(str)
    all_user_item_pair = np.array(all_user_item_pair)
    # 所有购买用户
    all_user_set = actions['user_id'].unique()

    # 所有品类中预测购买的用户
    all_user_test_set = result['user_id'].unique()
    all_user_test_item_pair = result['user_id'].map(str) + '-' + result['sku_id'].map(str)
    all_user_test_item_pair = np.array(all_user_test_item_pair)

    # 计算所有用户购买评价指标
    pos, neg = 0,0
    for user_id in all_user_test_set:
        if user_id in all_user_set:
            pos += 1
        else:
            neg += 1


    all_user_acc = 1.0 * pos / ( pos + neg)
    all_user_recall = 1.0 * pos / len(all_user_set)
    print('user predict correct num, all buy user num', pos, len(all_user_set))
#    print ('所有用户中预测购买用户的准确率为 ' + str(all_user_acc))
#    print ('所有用户中预测购买用户的召回率' + str(all_user_recall))

    pos, neg = 0, 0
    for user_item_pair in all_user_test_item_pair:
        if user_item_pair in all_user_item_pair:
            pos += 1
        else:
            neg += 1
    all_item_acc = 1.0 * pos / ( pos + neg)
    all_item_recall = 1.0 * pos / len(all_user_item_pair)
    print('user-sku predict correct num, all buy user num', pos, len(all_user_item_pair))
#    print ('所有用户中预测购买商品的准确率为 ' + str(all_item_acc))
#    print ('所有用户中预测购买商品的召回率' + str(all_item_recall))
    F11 = 6.0 * all_user_recall * all_user_acc / (5.0 * all_user_recall + all_user_acc)
    F12 = 5.0 * all_item_acc * all_item_recall / (2.0 * all_item_recall + 3 * all_item_acc)
    score = 0.4 * F11 + 0.6 * F12
    print ('F11=' + str(F11), ' F12= ' + str(F12), ' score= ' + str(score))
    return (str(F11), str(F12), str(score))

def offline_ui(cv_ui_index, cv_test_start_date, cv_test_end_date):
   # print(str(cv_ui_index))
    true_labels = get_cate8_labels(cv_test_start_date, cv_test_end_date)
    print("600=================================================================")
    report(cv_ui_index.head(600), true_labels)
    print("800=================================================================")
    report(cv_ui_index.head(800), true_labels)
    print("900==================================================================")
    report(cv_ui_index.head(900), true_labels)
    print("1000================================================================")
    report(cv_ui_index.head(1000), true_labels)


def get_important_feat_names(dat, feature_names):

    dat = dat.reset_index(drop=False)                                               #dat: pd.Series
    #print(dat.columns.values)
    dat.rename(columns={'index': 'feat_index', 0: 'important'}, inplace=True)
    dat['feat_index'] = dat['feat_index'].map(lambda x: x.replace('f', ''))
    dat['feat_index'] = dat['feat_index'].astype(int)
    feat_important_index = list(dat['feat_index'])
    #form index return feat names
    feature_names = np.array(feature_names)
    important_feature_name = feature_names[feat_important_index]
    return important_feature_name #np.array


def get_new_ui_trainning_dat(user_index, training_data, label, user_label):
    new_training_dat = pd.concat([user_index, training_data, label], axis=1)
    user_label = user_label[['user_id']]
    new_training_dat = pd.merge(new_training_dat, user_label, how='inner', on='user_id')
    user_index = new_training_dat[['user_id', 'sku_id']]
    label = new_training_dat['label']
    del new_training_dat['user_id']
    del new_training_dat['sku_id']
    del new_training_dat['label']
    training_data = new_training_dat
    return user_index, training_data, label


def new_idea_predict_dat(sub_user_index, sub_trainning_data, path, topi = 1000):
    test_user_label = pd.read_csv(path)
    test_user_label = test_user_label.head(topi)
    sub_trainning_data = pd.concat([sub_user_index, sub_trainning_data], axis=1)
    sub_trainning_data = pd.merge(sub_trainning_data, test_user_label[['user_id']], how='inner', on='user_id')
    sub_user_index = sub_trainning_data[['user_id', 'sku_id']]

    del sub_trainning_data['user_id']
    del sub_trainning_data['sku_id']
    return sub_user_index, sub_trainning_data


def show_common_res(user_res, ui_res, top_user_accont, top_ui_account):
    merge_dat = pd.merge(user_res.head(top_user_accont), ui_res.head(top_ui_account), how='inner', on=['user_id'])
    print("Top"+ str(top_user_accont)+"user result+ top" + str(top_ui_account) +"ui result common number:",  merge_dat.shape)
    return merge_dat