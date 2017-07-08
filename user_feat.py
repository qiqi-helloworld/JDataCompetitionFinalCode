#!/usr/local/bin/env python
# -*- coding: UTF-8 -*-

__author__ = "QIQI"



import time
from datetime import datetime
from datetime import timedelta
import pandas as pd
import pickle
import os
import math
import numpy as np
import copy
from sklearn.utils import shuffle
from collections import Counter

action_1_path = "./data/JData_Action_201602.csv"
action_2_path = "./data/JData_Action_201603.csv"
action_3_path = "./data/JData_Action_201604.csv"
comment_path = "./data/JData_Comment.csv"
product_path = "./data/JData_Product.csv"
user_path = "./data/JData_User.csv"

comment_date = ["2016-02-01", "2016-02-08", "2016-02-15", "2016-02-22", "2016-02-29", "2016-03-07", "2016-03-14",
                "2016-03-21", "2016-03-28",
                "2016-04-04", "2016-04-11", "2016-04-15"]

from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
from sklearn.preprocessing import Imputer


def get_actions_1():
    action = pd.read_csv(action_1_path)
    year_start = "2016-02-07"
    years_end = "2016-02-13"
    action = action[-(action.time >= year_start) & (action.time <= years_end)]
    return action


def get_actions_2():
    action2 = pd.read_csv(action_2_path)
    return action2


def get_actions_3():
    action3 = pd.read_csv(action_3_path)
    return action3


def get_actions(start_date, end_date):
    """
    combine all action data from different action data file
    """
    dump_path = './cache/all_action_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path))
    else:
        action_1 = get_actions_1()
        action_2 = get_actions_2()
        action_3 = get_actions_3()
        actions = pd.concat([action_1, action_2, action_3])  # type: pd.DataFrame
        actions = actions[(actions.time >= start_date) & (actions.time < end_date)]
        pickle.dump(actions, open(dump_path, 'w'))
    return actions


def convert_age(age_str):
    """transfer age to number"""
    if age_str == u'-1':
        return 0
    elif age_str == u'15岁以下':
        return 1
    elif age_str == u'16-25岁':
        return 2
    elif age_str == u'26-35岁':
        return 3
    elif age_str == u'36-45岁':
        return 4
    elif age_str == u'46-55岁':
        return 5
    elif age_str == u'56岁以上':
        return 6
    else:
        return -1


# 排序函数
def relative_order(a):
    l = sorted(a, reverse=True)
    # hash table of element -> index in ordered list
    d = dict(zip(l, range(len(l))))
    return [d[e] for e in a]


###########################################用户特征###################################################
# 基本用户信息 年龄 性别 用户等级
def get_basic_user_feat():
    """Function :get basic user informations"""
    dump_path = './cache/basic_user.pkl'
    if os.path.exists(dump_path):
        user = pickle.load(open(dump_path))
    else:
        user = pd.read_csv(user_path, encoding='gbk')
        # user['age'] = user['age'].map(lambda x:convert_age(x))
        age_df = pd.get_dummies(user["age"], prefix="age")
        sex_df = pd.get_dummies(user["sex"], prefix="sex")
        user_lv_df = pd.get_dummies(user["user_lv_cd"], prefix="user_lv_cd")
        user = pd.concat([user['user_id'], age_df, sex_df, user_lv_df], axis=1)
        pickle.dump(user, open(dump_path, 'w'))
    return user

# 注册天数
def get_reg_day(start_date, end_date):
    '''4.22'''
    dump_path = './cache/basic_user_reg_day_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        user_dat = pickle.load(open(dump_path))
    else:
        user_dat = pd.read_csv(user_path, encoding='gbk')
        max_interval_day = (datetime.strptime(end_date, '%Y-%m-%d') - datetime.strptime(start_date, '%Y-%m-%d')).days
        print(type(user_dat['user_reg_tm']))
        user_dat = user_dat.dropna(axis=0, how='any')
        user_dat['user_reg_tm'] = user_dat['user_reg_tm'].astype(str).map(
            lambda x: (datetime.strptime(end_date, '%Y-%m-%d') - datetime.strptime(x, '%Y/%m/%d')).days)
        user_dat = user_dat[['user_id', 'user_reg_tm']]
        user_dat.loc[user_dat['user_reg_tm'] < 0, 'user_reg_tm'] = -1
        pickle.dump(user_dat, open(dump_path, 'w'))
    return user_dat

# 首次注册与首次购买时间间隔
def get_register_first_buy_interval_day(start_date, end_date):
    """4.22 first distance day...
    N = 'USR_ID'
    """
    dump_path = './cache/register_first_buy_interval_day_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        dat = pickle.load(open(dump_path))
    else:
        max_interval_day = (datetime.strptime(end_date, '%Y-%m-%d') - datetime.strptime(start_date, '%Y-%m-%d')).days
        print(max_interval_day)
        actions = get_actions(start_date, end_date)
        del actions['model_id']
        df = pd.get_dummies(actions['type'], prefix='action')
        actions = pd.concat([actions[['user_id', 'time']], df], axis=1)
        actions['inter_day'] = actions['time'].map(
            lambda x: (datetime.strptime(end_date, '%Y-%m-%d') - datetime.strptime(x, '%Y-%m-%d %H:%M:%S')).days)
        print(1)
        actions['first_buy'] = actions['action_4'] * actions['inter_day']
        actions = actions.loc[actions['action_4'] != 0]
        first_buy_dat = actions.groupby(['user_id'], as_index=False)['first_buy'].max()
        user_dat = pd.read_csv(user_path, encoding='gbk')
        user_dat = user_dat.dropna(axis=0, how='any')
        print(1)
        user_dat['user_reg_tm'] = user_dat['user_reg_tm'].map(
            lambda x: (datetime.strptime(end_date, '%Y-%m-%d') - datetime.strptime(x, '%Y/%m/%d')).days)
        user_dat = user_dat[['user_id', 'user_reg_tm']]
        dat = pd.merge(user_dat, first_buy_dat, how='left', on='user_id')
        print(dat.columns.values)
        dat = dat.fillna(-1)
        dat['inter_reg_first_buy'] = dat['user_reg_tm'] - dat['first_buy']
        dat['inter_reg_first_buy'] = dat['inter_reg_first_buy'].astype(int)
        dat['inter_reg_first_buy'] = dat['inter_reg_first_buy'].map(
            lambda x: int((x + 1) / 5))
        dat = dat[['user_id', 'inter_reg_first_buy']]
        pickle.dump(dat, open(dump_path, 'w'))

    return dat

# 用户购买周期：时间间隔/购买件数
def get_user_buy_period(start_date, end_date, i):
    dump_path = './cache/%d_user_buy_period_%s_%s.pkl' % (i, start_date, end_date)
    if os.path.exists(dump_path):
        user_buy_period_dat = pickle.load(open(dump_path))
    else:
        user_dat = get_basic_user_feat()
        user_dat = pd.DataFrame(user_dat['user_id'])
        max_interval_day = (datetime.strptime(end_date, '%Y-%m-%d') - datetime.strptime(start_date, '%Y-%m-%d')).days
        actions = get_actions(start_date, end_date)
        actions = actions[['user_id', 'type']]
        df = pd.get_dummies(actions['type'], prefix='%d_action_period' % i)
        # print(df.head())
        actions = pd.concat([actions['user_id'], df], axis=1)  # type: pd.DataFrame
        actions = actions.groupby(['user_id'], as_index=False).count()
        user_buy_period_dat = pd.merge(user_dat, actions, how='left', on='user_id')
        user_buy_period_dat = user_buy_period_dat.fillna(-1)
        user_buy_period_dat[user_buy_period_dat == 0] = -1
        user_buy_period_dat['%d_action_period_1' % i] = max_interval_day * 1.0 / user_buy_period_dat[
            '%d_action_period_1' % i]
        user_buy_period_dat['%d_action_period_2' % i] = max_interval_day * 1.0 / user_buy_period_dat[
            '%d_action_period_2' % i]
        user_buy_period_dat['%d_action_period_3' % i] = max_interval_day * 1.0 / user_buy_period_dat[
            '%d_action_period_3' % i]
        user_buy_period_dat['%d_action_period_4' % i] = max_interval_day * 1.0 / user_buy_period_dat[
            '%d_action_period_4' % i]
        user_buy_period_dat['%d_action_period_5' % i] = max_interval_day * 1.0 / user_buy_period_dat[
            '%d_action_period_5' % i]
        user_buy_period_dat['%d_action_period_6' % i] = max_interval_day * 1.0 / user_buy_period_dat[
            '%d_action_period_6' % i]
        user_buy_period_dat[user_buy_period_dat < 0] = -1
        pickle.dump(user_buy_period_dat, open(dump_path, 'w'))
    user_buy_period_dat = user_buy_period_dat[['user_id', '%d_action_period_4' % i, '%d_action_period_6' % i]]
    return user_buy_period_dat

# 每个用户的商品行为（流量，加入购物车， 关注）转化比率
def get_user_conv_ratio(start_date, end_date, i):
    dump_path = './cache/%d_user_conv_ratio%s_%s.pkl' % (i, start_date, end_date)
    if os.path.exists(dump_path):
        user_conv_ratio = pickle.load(open(dump_path))
    else:
        actions = get_actions(start_date, end_date)
        df = pd.get_dummies(actions['type'], prefix='%d_user_conv_ratio' % i)
        actions = pd.concat([actions[['user_id']], df], axis=1)
        actions = actions.groupby(['user_id'], as_index=False).sum()
        actions['%d_user_conv_ratio_1' % i] = actions['%d_user_conv_ratio_4' % i] / actions['%d_user_conv_ratio_1' % i]
        actions['%d_user_conv_ratio_2' % i] = actions['%d_user_conv_ratio_4' % i] / actions['%d_user_conv_ratio_2' % i]
        actions['%d_user_conv_ratio_5' % i] = actions['%d_user_conv_ratio_4' % i] / actions['%d_user_conv_ratio_5' % i]
        actions['%d_user_conv_ratio_3' % i] = actions['%d_user_conv_ratio_4' % i] / actions['%d_user_conv_ratio_3' % i]
        actions['%d_user_conv_ratio_6' % i] = actions['%d_user_conv_ratio_4' % i] / actions['%d_user_conv_ratio_6' % i]
        user_conv_ratio = actions[
            ['user_id', '%d_user_conv_ratio_1' % i, '%d_user_conv_ratio_2' % i, '%d_user_conv_ratio_3' % i,
             '%d_user_conv_ratio_5' % i, '%d_user_conv_ratio_6' % i]]
        user_conv_ratio = user_conv_ratio.replace(np.inf, np.nan)
        user_conv_ratio = user_conv_ratio.fillna(0)
        pickle.dump(user_conv_ratio, open(dump_path, 'w'))
    return user_conv_ratio

#点击量
def get_user_action(start_date, end_date, i):
    dump_path = './cache/%d_user_action_%s_%s.pkl' % (i, start_date, end_date)
    if os.path.exists(dump_path):
        user_action = pickle.load(open(dump_path))
    else:
        actions = get_actions(start_date, end_date)
        df = pd.get_dummies(actions['type'], prefix='%d_user_action' % i)
        actions = pd.concat([actions[['user_id']], df], axis=1)
        actions = actions.groupby(['user_id'], as_index=False).sum()
        user_action = actions[
            ['user_id', '%d_user_action_1' % i, '%d_user_action_2' % i, '%d_user_action_3' % i,
             '%d_user_action_5' % i, '%d_user_action_6' % i]]
        user_action = user_action.replace(np.inf, np.nan)
        user_action = user_action.fillna(0)
        pickle.dump(user_action, open(dump_path, 'w'))
    return user_action

#比率 for test
def get_user_ratio(start_date, end_date, i):
    dump_path = './cache/%d_user_cate8_ratio_%s_%s.pkl' % (i, start_date, end_date)
    if os.path.exists(dump_path):
        user_action = pickle.load(open(dump_path))
    else:
        actions = get_actions(start_date, end_date)
        actions = actions[actions['cate'] == 8]
        sum_actions= actions[['user_id', 'cate']]
        sum_actions = sum_actions.groupby(['user_id'], as_index= False).count()
        sum_actions.rename(columns={'cate': 'cate_action_num'}, inplace=True)
        df = pd.get_dummies(actions['type'], prefix='%d_user_ratio' % i)
        actions = pd.concat([actions[['user_id']], df], axis=1)
        actions = actions.groupby(['user_id'], as_index=False).sum()

        actions = pd.merge(actions, sum_actions, how = 'left', on='user_id')
        actions['%d_user_ratio_1' % i] = actions['%d_user_ratio_1' % i] / actions['cate_action_num']
        actions['%d_user_ratio_2' % i] = actions['%d_user_ratio_2' % i] / actions['cate_action_num']
        actions['%d_user_ratio_3' % i] = actions['%d_user_ratio_3' % i] / actions['cate_action_num']
        actions['%d_user_ratio_4' % i] = actions['%d_user_ratio_4' % i] / actions['cate_action_num']
        actions['%d_user_ratio_5' % i] = actions['%d_user_ratio_5' % i] / actions['cate_action_num']
        actions['%d_user_ratio_6' % i] = actions['%d_user_ratio_6' % i] / actions['cate_action_num']
        del actions['cate_action_num']
        user_action = actions.replace(np.inf, np.nan)
        user_action = user_action.fillna(0)
        pickle.dump(user_action, open(dump_path, 'w'))
    return user_action

####新加特征
# 每个user 购买的cate品种，即关联规则
def get_uc_correlation(start_date, end_date):
    dump_path = './cache/uc_cor%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        uc_cor = pickle.load(open(dump_path))
    else:
        user_dat = get_basic_user_feat()
        user_dat = pd.DataFrame(user_dat['user_id'])
        actions = get_actions(start_date, end_date)
        actions = actions[['user_id', 'cate', 'type']]
        df = pd.get_dummies(actions['type'], prefix='action')
        dc = pd.get_dummies(actions['cate'], prefix='cate')
        actions = pd.concat([actions['user_id'], df['action_4'], dc], axis=1)
        # dc,
        actions = actions.loc[actions['action_4'] == 4]
        actions = actions.groupby('user_id', as_index=False).sum()
        # print(actions.columns.values)
        uc_cor = pd.merge(user_dat, actions, how='left', on='user_id')
        pickle.dump(uc_cor, open(dump_path, 'w'))
    # (uc_cor.columns.values)
    return uc_cor

# 每个用户购买的商品cate排名
def get_cate_rank_in_user(start_date, end_date):
    '''on : user_id, cate'''
    dump_path = './cache/cate_rank_in_user%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        sku_rank_in_user = pickle.load(open(dump_path))
    else:
        actions = get_actions(start_date, end_date)
        df1 = pd.get_dummies(actions['type'], prefix='ratio_action')
        df2 = pd.get_dummies(actions['type'], prefix='user_action')
        df1 = pd.concat([actions[['user_id', 'cate']], df1], axis=1)
        df2 = pd.concat([actions['user_id'], df2], axis=1)
        actions = actions[['user_id', 'cate']]
        actions = pd.concat([actions, df1.groupby(['user_id', 'cate']).transform(sum)], axis=1)
        actions = pd.concat([actions, df2.groupby(['user_id']).transform(sum)], axis=1)
        actions = actions.groupby(['user_id', 'cate'], as_index=False).first()
        actions['ratio_action_1'] = actions['ratio_action_1'] / actions['user_action_1']
        actions['ratio_action_2'] = actions['ratio_action_2'] / actions['user_action_2']
        actions['ratio_action_3'] = actions['ratio_action_3'] / actions['user_action_3']
        actions['ratio_action_4'] = actions['ratio_action_4'] / actions['user_action_4']
        actions['ratio_action_5'] = actions['ratio_action_5'] / actions['user_action_5']
        actions['ratio_action_6'] = actions['ratio_action_6'] / actions['user_action_6']
        actions['sku_user_rank'] = actions['ratio_action_1'] + actions['ratio_action_2'] + actions['ratio_action_3'] + \
                                   actions['ratio_action_4'] + actions['ratio_action_5'] + actions['ratio_action_6']
        sku_rank_in_user = actions[['user_id', 'cate', 'sku_user_rank', 'ratio_action_4']]
        pickle.dump(sku_rank_in_user, open(dump_path, 'w'))


    df1 = pd.get_dummies(sku_rank_in_user['cate'], prefix='cate_rank')
    rank_cate = pd.concat([sku_rank_in_user[['user_id', 'sku_user_rank']], df1], axis=1)
    df2 = pd.get_dummies(sku_rank_in_user['cate'], prefix='cate_buy_ratio')
    buy_ratio_cate = pd.concat([sku_rank_in_user[['user_id', 'ratio_action_4']], df2], axis=1)
    rank_cate['cate_rank_4'] = rank_cate['cate_rank_4'] * rank_cate['sku_user_rank']
    rank_cate['cate_rank_5'] = rank_cate['cate_rank_5'] * rank_cate['sku_user_rank']
    rank_cate['cate_rank_6'] = rank_cate['cate_rank_6'] * rank_cate['sku_user_rank']
    rank_cate['cate_rank_7'] = rank_cate['cate_rank_7'] * rank_cate['sku_user_rank']
    rank_cate['cate_rank_8'] = rank_cate['cate_rank_8'] * rank_cate['sku_user_rank']
    rank_cate['cate_rank_9'] = rank_cate['cate_rank_9'] * rank_cate['sku_user_rank']
    rank_cate['cate_rank_10'] = rank_cate['cate_rank_10'] * rank_cate['sku_user_rank']
    rank_cate['cate_rank_11'] = rank_cate['cate_rank_11'] * rank_cate['sku_user_rank']

    buy_ratio_cate['cate_buy_ratio_4'] = buy_ratio_cate['cate_buy_ratio_4'] * buy_ratio_cate['ratio_action_4']
    buy_ratio_cate['cate_buy_ratio_5'] = buy_ratio_cate['cate_buy_ratio_5'] * buy_ratio_cate['ratio_action_4']
    buy_ratio_cate['cate_buy_ratio_6'] = buy_ratio_cate['cate_buy_ratio_6'] * buy_ratio_cate['ratio_action_4']
    buy_ratio_cate['cate_buy_ratio_7'] = buy_ratio_cate['cate_buy_ratio_7'] * buy_ratio_cate['ratio_action_4']
    buy_ratio_cate['cate_buy_ratio_8'] = buy_ratio_cate['cate_buy_ratio_8'] * buy_ratio_cate['ratio_action_4']
    buy_ratio_cate['cate_buy_ratio_9'] = buy_ratio_cate['cate_buy_ratio_9'] * buy_ratio_cate['ratio_action_4']
    buy_ratio_cate['cate_buy_ratio_10'] = buy_ratio_cate['cate_buy_ratio_10'] * buy_ratio_cate['ratio_action_4']
    buy_ratio_cate['cate_buy_ratio_11'] = buy_ratio_cate['cate_buy_ratio_11'] * buy_ratio_cate['ratio_action_4']

    rank_cate = rank_cate.groupby(['user_id'], as_index=False).sum()
    del rank_cate['sku_user_rank']
    buy_ratio_cate = buy_ratio_cate.groupby(['user_id'], as_index=False).sum()
    del buy_ratio_cate['ratio_action_4']
    sku_rank_in_user = pd.merge(rank_cate, buy_ratio_cate, how ='left', on='user_id')
    return sku_rank_in_user

def get_cart_focus_user_cate8(start_date, end_date):
    dump_path = './cache/cookly_cart_focus_cate8_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path, 'rb+'))
    else:
        actions = get_actions(start_date, end_date)
        actions = actions[((actions['cate'] == 8) & ((actions['type'] == 4) | (actions['type'] == 2) | (actions['type'] == 5)))]
        pickle.dump(actions, open(dump_path, 'wb+'))
    user_id = pd.DataFrame(actions['user_id'])
    user_id = user_id.drop_duplicates()
    return user_id

#用户最近一次加入购物车
def get_user_last_cart(start_date, end_date):
    dump_path = './cache/user_last_cart_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        cart_dat = pickle.load(open(dump_path, 'rb+'))
    else:
        actions = get_actions(start_date, end_date)
        actions = actions[actions['type'] == 3]
        actions.drop_duplicates(['user_id',  'type'],  keep = 'last', inplace=True)
        print(actions.head())
        print("action time head:", actions['time'].head(5))
        actions['user_last_cart'] = actions['time'].map(lambda x: (datetime.strptime(end_date, '%Y-%m-%d') - datetime.strptime(x, '%Y-%m-%d %H:%M:%S')).days)
        cart_dat = actions[['user_id',  'user_last_cart']]
        pickle.dump(cart_dat, open(dump_path, 'w'))
    return cart_dat

#用户最近一次购买
def get_user_last_buy(start_date, end_date):
    dump_path = './cache/user_last_buy_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        buy_dat = pickle.load(open(dump_path, 'rb+'))
    else:
        actions = get_actions(start_date, end_date)
        actions = actions[actions['type'] == 4]
        actions.drop_duplicates(['user_id',  'type'],keep='last', inplace=True)
        print("action time head:", actions['time'].head(5))
        actions['user_last_buy'] = actions['time'].map(lambda x: (datetime.strptime(end_date, '%Y-%m-%d') - datetime.strptime(x, '%Y-%m-%d %H:%M:%S')).days)
        buy_dat = actions[['user_id', 'user_last_buy']]
        pickle.dump(buy_dat, open(dump_path, 'w'))

    return buy_dat

#target
def get_user_labels(start_date, end_date):
    """Each prdiction label"""
    dump_path = './cache/user_labels_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path))
    else:
        actions = get_actions(start_date, end_date)
        actions = actions[(actions['type'] == 4) & (actions['cate'] == 8)]
        actions = actions.groupby(['user_id'], as_index=False).sum()
        actions['label'] = 1
        actions = actions[['user_id', 'label']]
        pickle.dump(actions, open(dump_path, 'w'))
    return actions

#model_id
def get_user_model_id(start_date, end_date):
    dump_path = './cache/user_model_id_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        model_id_dat = pickle.load(open(dump_path))
    else:
        actions = get_actions(start_date, end_date)
        actions = actions[['user_id', 'model_id']]
        actions = actions[actions['model_id'].isin([14, 21, 28, 110, 210])]
        df = pd.get_dummies(actions['model_id'], prefix="user_model_id")
        actions = pd.concat([actions[['user_id']], df], axis= 1)
        model_id_dat = actions.groupby(['user_id'], as_index = False).count()
    return model_id_dat

#用户交互天数
def get_user_action_day(start_date, end_date, i):
    dump_path = './cache/user_action_tm_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path))
    else:
        actions = get_actions(start_date, end_date)
        actions['time'] = actions['time'].map(
            lambda x: (datetime.strptime(end_date, '%Y-%m-%d') - datetime.strptime(x, '%Y-%m-%d %H:%M:%S')).days)
        actions = actions[['user_id', 'time']]
        actions = actions.drop_duplicates()
        actions = actions.groupby(['user_id'], as_index=False).count()
        actions.rename(columns = {'time':'%d_user_action_tm'%i}, inplace=True)
        pickle.dump(actions, open(dump_path, 'w'))

    return actions

#用户购买天数
def get_user_buy_action_day(start_date, end_date, i):
    dump_path = './cache/user_buy_action_tm_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path))
    else:
        actions = get_actions(start_date, end_date)
        actions = actions[actions['type'] == 4]
        actions['time'] = actions['time'].map(
            lambda x: (datetime.strptime(end_date, '%Y-%m-%d') - datetime.strptime(x, '%Y-%m-%d %H:%M:%S')).days)
        actions = actions[['user_id', 'time']]
        actions = actions.drop_duplicates()
        actions = actions.groupby(['user_id'], as_index=False).count()
        actions.rename(columns={'time': '%d_user_action_buy_tm' % i}, inplace=True)
        pickle.dump(actions, open(dump_path, 'w'))

    return actions

#用户最近一次购买
def get_user_cate_last_buy(start_date, end_date):
    dump_path = './cache/user_cate_last_buy_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        last_buy_cate = pickle.load(open(dump_path, 'rb+'))
    else:
        actions = get_actions(start_date, end_date)
        actions = actions[actions['type'] == 4]
        cate_fill_action = pd.DataFrame({'cate': [4, 5, 6, 7, 8, 9, 10, 11]})
        actions = pd.concat([cate_fill_action, actions], axis=0)
        #重置下标以删除
        actions = actions.reset_index(drop=True)
        df = pd.get_dummies(actions['cate'], prefix="last_buy_cate" )

        #删除填充的几列
        actions = actions.drop(np.arange(cate_fill_action.shape[0]))
        df = df.drop(np.arange(cate_fill_action.shape[0]))

        actions = pd.concat([actions, df], axis=1)
        actions.drop_duplicates(['user_id','cate'],keep='last', inplace=True)
        actions['user_last_buy'] = actions['time'].map(lambda x: (datetime.strptime(end_date, '%Y-%m-%d') - datetime.strptime(x, '%Y-%m-%d %H:%M:%S')).days)
        actions['last_buy_cate_4'] = actions['last_buy_cate_4'] * actions['user_last_buy']
        actions['last_buy_cate_5'] = actions['last_buy_cate_5'] * actions['user_last_buy']
        actions['last_buy_cate_6'] = actions['last_buy_cate_6'] * actions['user_last_buy']
        actions['last_buy_cate_7'] = actions['last_buy_cate_7'] * actions['user_last_buy']
        actions['last_buy_cate_8'] = actions['last_buy_cate_8'] * actions['user_last_buy']
        actions['last_buy_cate_9'] = actions['last_buy_cate_9'] * actions['user_last_buy']
        actions['last_buy_cate_10'] = actions['last_buy_cate_10'] * actions['user_last_buy']
        actions['last_buy_cate_11'] = actions['last_buy_cate_11'] * actions['user_last_buy']
        actions = actions[['user_id', 'last_buy_cate_4', 'last_buy_cate_5', 'last_buy_cate_6', 'last_buy_cate_7',
                           'last_buy_cate_8', 'last_buy_cate_9', 'last_buy_cate_10', 'last_buy_cate_11']]
        last_buy_cate = actions.groupby(['user_id'], as_index=False).sum()
        pickle.dump(last_buy_cate, open(dump_path, 'w'))

    return last_buy_cate

#用户最近一次加入购物车
def get_user_cate_last_cart(start_date, end_date):
    dump_path = './cache/user_cate_last_cart_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions_last = pickle.load(open(dump_path, 'rb+'))
    else:
        actions = get_actions(start_date, end_date)
        actions = actions[actions['cate'] == 8]

        actions_cart = actions[actions['type'] == 2]
        actions_cart = actions_cart[['user_id', 'time']]
        actions_cart['user_last_cart'] = actions_cart['time'].map(lambda x: (datetime.strptime(end_date, '%Y-%m-%d') - datetime.strptime(x, '%Y-%m-%d %H:%M:%S')).days)
        del actions_cart['time']

        actions_del_cart = actions[actions['type'] == 3]
        actions_del_cart = actions_del_cart[['user_id', 'time']]
        actions_del_cart['user_last_del_cart'] = actions_del_cart['time'].map(lambda x: (datetime.strptime(end_date, '%Y-%m-%d') - datetime.strptime(x, '%Y-%m-%d %H:%M:%S')).days)
        del actions_del_cart['time']

        actions_last = pd.merge(actions_cart, actions_del_cart, how='left', on='user_id')
        actions_last = actions_last.fillna(0)
        actions_last['buy_possible'] = actions_last['user_last_del_cart'] - actions_last['user_last_cart']
        actions_last['buy_possible'] = actions_last['buy_possible'].map(lambda x: x>=0)
        actions_last['buy_possible'] = actions_last['buy_possible'].astype(int)
        pickle.dump(actions_last, open(dump_path, 'w'))
    print(actions_last.columns.values)
    return actions_last


####################################cate8特征#################################################
#model_id
def get_user_model_id_cate8(start_date, end_date):
    dump_path = './cache/user_model_id_cate8_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        model_id_dat = pickle.load(open(dump_path))
    else:
        actions = get_actions(start_date, end_date)
        actions = actions[actions['cate'] == 8]
        actions = actions[['user_id', 'model_id']]
        actions = actions[actions['model_id'].isin([14, 21, 28, 110, 210])]
        df = pd.get_dummies(actions['model_id'], prefix="user_cate8_model_id")
        actions = pd.concat([actions[['user_id']], df], axis= 1)
        model_id_dat = actions.groupby(['user_id'], as_index = False).count()
    return model_id_dat


#用户cate交互天数
def get_user_cate_action_day(start_date, end_date, i):
    dump_path = './cache/user_cate_action_tm_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path))
    else:
        actions = get_actions(start_date, end_date)
        cate_fill_action = pd.DataFrame({'cate': [4, 5, 6, 7, 8, 9, 10, 11]})
        actions = pd.concat([cate_fill_action, actions], axis= 0)
        actions =actions.reset_index(drop = True)
        df = pd.get_dummies(actions['cate'], prefix="%i_tm_cate"%i)
        actions = actions.drop(np.arange(cate_fill_action.shape[0]))
        df = df.drop(np.arange(cate_fill_action.shape[0]))

        actions['time'] = actions['time'].map(
            lambda x: (datetime.strptime(end_date, '%Y-%m-%d') - datetime.strptime(x, '%Y-%m-%d %H:%M:%S')).days)
        actions = actions[['user_id', 'time', 'cate']]
        actions = pd.concat([actions,df], axis= 1)
        actions = actions.drop_duplicates(['user_id', 'time', 'cate'])
        actions = actions.groupby(['user_id'], as_index=False).sum()
        del actions['time']
        del actions['cate']
        pickle.dump(actions, open(dump_path, 'w'))
    if i>=0:
        actions = actions[['user_id', "%i_tm_cate_8" % i]]
    return actions

#用户cate购买交互天数
def get_user_cate_buy_action_day(start_date, end_date, i):
    dump_path = './cache/user_cate_buy_action_tm_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path))
    else:
        actions = get_actions(start_date, end_date)
        actions = actions[actions['type'] == 4]

        cate_fill_action = pd.DataFrame({'cate': [4, 5, 6, 7, 8, 9, 10, 11]})
        actions = pd.concat([cate_fill_action, actions], axis=0)
        actions = actions.reset_index(drop=True)
        df = pd.get_dummies(actions['cate'], prefix="%i_tm_buy_cate" % i)
        actions = actions.drop(np.arange(cate_fill_action.shape[0]))
        df = df.drop(np.arange(cate_fill_action.shape[0]))

        actions['time'] = actions['time'].map(
            lambda x: (datetime.strptime(end_date, '%Y-%m-%d') - datetime.strptime(x, '%Y-%m-%d %H:%M:%S')).days)
        actions = actions[['user_id', 'time', 'cate']]
        actions = pd.concat([actions,df], axis= 1)
        actions = actions.drop_duplicates(['user_id', 'time', 'cate'])
        actions = actions.groupby(['user_id'], as_index=False).sum()
        del actions['time']
        del actions['cate']
        pickle.dump(actions, open(dump_path, 'w'))

    if i>=0:
        actions = actions[['user_id', "%i_tm_buy_cate_8" % i]]
    return actions

#用户交互过cate 8
def get_watch_user_cate8(start_date, end_date):
    dump_path = './cache/cookly_cate8_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path, 'rb+'))
    else:
        actions = get_actions(start_date, end_date)
        actions = actions[actions['cate'] == 8]
        pickle.dump(actions, open(dump_path, 'wb+'))

    user_id = pd.DataFrame(actions['user_id'])
    user_id = user_id.drop_duplicates()
    return user_id

#买了cate8的user
def get_buy_cate8(start_date, end_date):
    dump_path = './cache/buy_cate8_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path, 'rb+'))
    else:
        actions = get_actions(start_date, end_date)
        actions = actions[(actions['cate'] == 8) & (actions['type'] == 4)]
        pickle.dump(actions, open(dump_path, 'wb+'))

    user_id = pd.DataFrame(actions['user_id'])
    user_id = user_id.drop_duplicates()
    return user_id

# 每个（用户,cate）的购买行为的数量
def get_uc_num_feat(start_date, end_date, i):
    """
    就是用户在哪个里面买的多 哪个里面买的少
    return(user_id, cate) action type 1:6 number in different date period.
    """
    dump_path = './cache/uc_action_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        ui_feat_num = pickle.load(open(dump_path))
    else:
        actions = get_actions(start_date, end_date)
        unique_buy_sku_actions = actions.loc[actions['type'] == 4, ['user_id', 'cate', 'sku_id']]
        del actions['model_id']
        del actions['time']
        df = pd.get_dummies(actions['type'], prefix='uc_action_%s_%s' % (start_date, end_date))
        actions = pd.concat([actions[['user_id', 'cate']], df], axis=1)
        actions = actions.groupby(['user_id', 'cate'], as_index=False).sum()
        sku_names = 'sku_%s_%s' % (start_date, end_date)
        unique_buy_sku_actions = unique_buy_sku_actions.rename(index=str, columns={'sku_id': sku_names})
        unique_buy_sku_actions = unique_buy_sku_actions.groupby(['user_id', 'cate'], as_index=False).count()
        ui_feat_num = pd.merge(actions, unique_buy_sku_actions, how="left", on=["user_id", 'cate'])
        pickle.dump(ui_feat_num, open(dump_path, 'w'))

    return ui_feat_num

def get_user_uc_num_feat(start_date, end_date, i):
    dump_path = './cache/user_uc_action_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        new_user_cate_action = pickle.load(open(dump_path))
    else:
        ui_feat_num = get_uc_num_feat(start_date, end_date, i)
        ui_name = ui_feat_num.columns.values
        new_user_cate_action = ui_feat_num[['user_id']]
        for j in range(2, 8):
            df1 = pd.get_dummies(ui_feat_num['cate'], prefix="%d_action_cate" % (j - 1))
            new_df1 = my_dummies(df1, ui_feat_num[ui_name[j]], j - 1)
            new_user_cate_action = pd.concat([new_user_cate_action, new_df1], axis=1)
        new_user_cate_action = new_user_cate_action.groupby(['user_id'], as_index=False).sum()
        pickle.dump(new_user_cate_action,  open(dump_path, 'w'))

    if i>= 1:
        new_user_cate_action = new_user_cate_action[['user_id', '1_action_cate_8', '2_action_cate_8', '3_action_cate_8', '4_action_cate_8', '5_action_cate_8', '6_action_cate_8']]
        new_user_cate_action.rename(columns={'1_action_cate_8': '1_action_cate8_%d' % i, '2_action_cate_8': '2_action_cate8_%d' % i, '3_action_cate_8': '3_action_cate8_%d' % i,
                                           '4_action_cate_8': '4_action_cate8_%d' % i, '5_action_cate_8': '5_action_cate8_%d' % i, '6_action_cate_8': '6_action_cate8_%d' % i
                                           }, inplace=True)

    return new_user_cate_action

#cate one hot
def my_dummies(df1, b, i):
    df1['%d_action_cate_4'%i] = df1['%d_action_cate_4'%i] * b
    df1['%d_action_cate_5'%i] = df1['%d_action_cate_5'%i] * b
    df1['%d_action_cate_6'%i] = df1['%d_action_cate_6'%i] * b
    df1['%d_action_cate_7'%i] = df1['%d_action_cate_7'%i] * b
    df1['%d_action_cate_8'%i] = df1['%d_action_cate_8'%i] * b
    df1['%d_action_cate_9'%i] = df1['%d_action_cate_9'%i] * b
    df1['%d_action_cate_10'%i] = df1['%d_action_cate_10'%i] * b
    df1['%d_action_cate_11'%i] = df1['%d_action_cate_11'%i] * b

    return df1

# 每个（用户,cate）的转化比率
def get_uc_conv_ratio(start_date, end_date, i):
    dump_path = './cache/%d_uc_conv_ratio%s_%s.pkl' % (i, start_date, end_date)
    if os.path.exists(dump_path):
        uc_conv_ratio = pickle.load(open(dump_path))
    else:
        actions = get_actions(start_date, end_date)
        df = pd.get_dummies(actions['type'], prefix='%d_uc_conv_ratio'% i)
        actions = pd.concat([actions[['user_id', 'cate']], df], axis=1)
        actions = actions.groupby(['user_id', 'cate'], as_index=False).sum()
        actions['%d_uc_conv_ratio_1'% i] = actions['%d_uc_conv_ratio_4'% i] / actions['%d_uc_conv_ratio_1'% i]
        actions['%d_uc_conv_ratio_2'% i] = actions['%d_uc_conv_ratio_4'% i] / actions['%d_uc_conv_ratio_2'% i]
        actions['%d_uc_conv_ratio_3' % i] = actions['%d_uc_conv_ratio_4' % i] / actions['%d_uc_conv_ratio_3' % i]
        actions['%d_uc_conv_ratio_5'% i] = actions['%d_uc_conv_ratio_4'% i] / actions['%d_uc_conv_ratio_5'% i]
        actions['%d_uc_conv_ratio_6'% i] = actions['%d_uc_conv_ratio_4'% i] / actions['%d_uc_conv_ratio_6'% i]
        uc_conv_ratio = actions[['user_id', 'cate', '%d_uc_conv_ratio_1'% i, '%d_uc_conv_ratio_2'% i, '%d_uc_conv_ratio_3'% i,
                                 '%d_uc_conv_ratio_5'% i, '%d_uc_conv_ratio_6'% i]]
        uc_conv_ratio = uc_conv_ratio.replace(np.inf, np.nan)
        uc_conv_ratio = uc_conv_ratio.fillna(0)
        pickle.dump(uc_conv_ratio, open(dump_path, 'w'))
    print(uc_conv_ratio.columns.values)


    return uc_conv_ratio

# user cate8的转化率
def get_user_uc_conv_ratio(start_date, end_date, i):
    dump_path = './cache/%d_user_uc_conv_ratio%s_%s.pkl' % (i, start_date, end_date)
    if os.path.exists(dump_path):
        new_uc_conv_action = pickle.load(open(dump_path))
    else:
        uc_conv_ratio = get_uc_conv_ratio(start_date, end_date, i)
        ui_name = uc_conv_ratio.columns.values
        print("uc_conv_name:", ui_name)
        new_uc_conv_action = uc_conv_ratio[['user_id']]
        for j in range(2, 7):
            df1 = pd.get_dummies(uc_conv_ratio['cate'], prefix="%d_action_cate" % (j - 1))
            new_df1 = my_dummies(df1, uc_conv_ratio[ui_name[j]], j - 1)
            new_uc_conv_action = pd.concat([new_uc_conv_action, new_df1], axis=1)

        new_uc_conv_action = new_uc_conv_action.groupby(['user_id'], as_index=False).sum()

        pickle.dump(new_uc_conv_action, open(dump_path, 'wb+'))

    if i>= 1:
        new_uc_conv_action = new_uc_conv_action[['user_id', '1_action_cate_8', '2_action_cate_8', '3_action_cate_8', '4_action_cate_8', '5_action_cate_8' ]]
        new_uc_conv_action.rename(columns={'1_action_cate_8': '1_action_conv_cate8_%d' % i, '2_action_cate_8': '2_action_conv_cate8_%d' % i, '3_action_cate_8': '3_action_conv_cate8_%d' % i,
                                           '4_action_cate_8': '4_action_conv_cate8_%d' % i, '5_action_cate_8': '5_action_conv_cate8_%d' % i}, inplace=True)
    return new_uc_conv_action


#cate8点击量
def get_user_cate8_action(start_date, end_date, i):
    dump_path = './cache/%d_user_cate8_action_%s_%s.pkl' % (i, start_date, end_date)
    if os.path.exists(dump_path):
        user_action = pickle.load(open(dump_path))
    else:
        actions = get_actions(start_date, end_date)
        actions = actions[actions['cate'] == 8]
        df = pd.get_dummies(actions['type'], prefix='%d_user_action' % i)
        actions = pd.concat([actions[['user_id']], df], axis=1)
        actions = actions.groupby(['user_id'], as_index=False).sum()
        user_action = actions[
            ['user_id', '%d_user_cate_action_1' % i, '%d_user_cate_action_2' % i, '%d_user_cate_action_3' % i,
             '%d_user_cate_action_5' % i, '%d_user_cate_action_6' % i]]
        user_action = user_action.replace(np.inf, np.nan)
        user_action = user_action.fillna(0)
        pickle.dump(user_action, open(dump_path, 'w'))
    return user_action

#cate8比率
def get_user_cate8_ratio(start_date, end_date, i):
    dump_path = './cache/%d_user_cate8_ratio_%s_%s.pkl' % (i, start_date, end_date)
    if os.path.exists(dump_path):
        user_action = pickle.load(open(dump_path))
    else:
        actions = get_actions(start_date, end_date)
        actions = actions[actions['cate'] == 8]
        sum_actions= actions[['user_id', 'cate']]
        sum_actions = sum_actions.groupby(['user_id'], as_index= False).count()
        sum_actions.rename(columns={'cate': 'cate_action_num'}, inplace=True)
        df = pd.get_dummies(actions['type'], prefix='%d_user_cate8_ratio' % i)
        actions = pd.concat([actions[['user_id']], df], axis=1)
        actions = actions.groupby(['user_id'], as_index=False).sum()

        actions = pd.merge(actions, sum_actions, how = 'left', on='user_id')
        actions['%d_user_cate8_ratio_1' % i] = actions['%d_user_cate8_ratio_1' % i] / actions['cate_action_num']
        actions['%d_user_cate8_ratio_2' % i] = actions['%d_user_cate8_ratio_2' % i] / actions['cate_action_num']
        actions['%d_user_cate8_ratio_3' % i] = actions['%d_user_cate8_ratio_3' % i] / actions['cate_action_num']
        actions['%d_user_cate8_ratio_4' % i] = actions['%d_user_cate8_ratio_4' % i] / actions['cate_action_num']
        actions['%d_user_cate8_ratio_5' % i] = actions['%d_user_cate8_ratio_5' % i] / actions['cate_action_num']
        actions['%d_user_cate8_ratio_6' % i] = actions['%d_user_cate8_ratio_6' % i] / actions['cate_action_num']
        del actions['cate_action_num']
        user_action = actions.replace(np.inf, np.nan)
        user_action = user_action.fillna(0)
        pickle.dump(user_action, open(dump_path, 'w'))
    return user_action












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
    pos, neg = 0, 0
    for user_id in all_user_test_set:
        if user_id in all_user_set:
            pos += 1
        else:
            neg += 1
    all_user_acc = 1.0 * pos / (pos + neg)
    all_user_recall = 1.0 * pos / len(all_user_set)
    print('所有用户中预测购买用户的准确率为 ' + str(all_user_acc))
    print('所有用户中预测购买用户的召回率' + str(all_user_recall))

    pos, neg = 0, 0
    for user_item_pair in all_user_test_item_pair:
        if user_item_pair in all_user_item_pair:
            pos += 1
        else:
            neg += 1
    all_item_acc = 1.0 * pos / (pos + neg)
    all_item_recall = 1.0 * pos / len(all_user_item_pair)
    print('所有用户中预测购买商品的准确率为 ' + str(all_item_acc))
    print('所有用户中预测购买商品的召回率' + str(all_item_recall))
    F11 = 6.0 * all_user_recall * all_user_acc / (5.0 * all_user_recall + all_user_acc)
    F12 = 5.0 * all_item_acc * all_item_recall / (2.0 * all_item_recall + 3 * all_item_acc)
    score = 0.4 * F11 + 0.6 * F12
    print('F11=' + str(F11))
    print('F12=' + str(F12))
    print('score=' + str(score))
    return (str(F11), str(F12), str(score))


# 下采样
def subsampe(ratio, user_index, training_data, label, random_seed, name):
    dump_path = './cache/'+ str(name) + '_subsample_' + str(ratio) + "_" + str(random_seed) + ".pkl"
    if os.path.exists(dump_path):
        with open(dump_path) as f:
            user_index = pickle.load(f)
            X_train = pickle.load(f)
            y_train = pickle.load(f)
    else:
        buy_ind = label[label == 1].index
        nobuy_ind = label[label == 0].index

        X_train_buy = training_data.loc[buy_ind].reset_index(drop=True)
        y_train_buy = label.loc[buy_ind].reset_index(drop=True)
        X_train_nobuy = training_data.loc[nobuy_ind].reset_index(drop=True)
        y_train_nobuy = label.loc[nobuy_ind].reset_index(drop=True)
        user_index_buy = user_index.loc[buy_ind].reset_index(drop=True)
        user_index_nobuy = user_index.loc[nobuy_ind].reset_index(drop=True)

        sam_nobuy_index = pd.Series(nobuy_ind).sample(len(buy_ind) * ratio, random_state=random_seed).index
        X_train_nobuy, y_train_nobuy, user_index_nobuy = (
            X_train_nobuy.loc[sam_nobuy_index], y_train_nobuy.loc[sam_nobuy_index],
            user_index_nobuy.loc[sam_nobuy_index])
        X_train = pd.concat([X_train_buy, X_train_nobuy], axis=0, ignore_index=True)
        y_train = pd.concat([y_train_buy, y_train_nobuy], axis=0, ignore_index=True)
        user_index = pd.concat([user_index_buy, user_index_nobuy], axis=0, ignore_index=True)
        X_train, y_train, user_index = shuffle(X_train, y_train, user_index, random_state=random_seed)
        # reset index
        user_index, X_train, y_train = (
        user_index.reset_index(drop=True), X_train.reset_index(drop=True), y_train.reset_index(drop=True))
        with open(dump_path, 'w') as f:
            pickle.dump(user_index, f)
            pickle.dump(X_train, f)
            pickle.dump(y_train, f)
    return user_index, X_train, y_train


# 降维
def reduce_dimension(training_data, label, dim, cv=None):
    if cv is None:
        fi_path = "./fi.pkl"
    else:
        fi_path = "./fi_" + str(cv) + ".pkl"
    colname = training_data.columns.values
    if os.path.exists(fi_path):
        fi = pickle.load(open(fi_path))
    else:
        training_data = Imputer().fit_transform(training_data)
        clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
                                         max_depth=6, random_state=0).fit(training_data, label)
        fi = clf.feature_importances_
        pickle.dump(fi, open(fi_path, 'w'))

    fi = np.array(fi).astype(float)
    colname = colname[np.where(fi > 0)]
    fi = fi[np.where(fi > 0)]
    ind_fi = np.argsort(-fi)
    name_fi = colname[ind_fi]
    return name_fi[0:dim], ind_fi[0:dim]




def get_user_cate8_statistical(start_date, end_date, i):
    dump_path = './cache/%d_user_cate8_statistical_%s_%s.pkl' % (i, start_date, end_date)
    if os.path.exists(dump_path):
        statistical_action = pickle.load(open(dump_path))
    else:
        actions = get_actions(start_date, end_date)
        actions = actions[actions['cate'] == 8]
        actions = actions[['user_id', 'sku_id', 'type']]


        df_mean = pd.get_dummies(actions['type'], prefix= "%d_action_mean" %i)
        mean_actions = pd.concat([actions[['user_id', 'sku_id']], df_mean], axis= 1)
        mean_actions = mean_actions.groupby(['user_id', 'sku_id'], as_index = False).sum()
        mean_actions.drop('sku_id', inplace=True, axis = 1)
        mean_actions = mean_actions.groupby(['user_id'], as_index = False).mean()

        df_max = pd.get_dummies(actions['type'], prefix="%d_action_max" % i)
        max_actions = pd.concat([actions[['user_id', 'sku_id']], df_max], axis=1)
        max_actions = max_actions.groupby(['user_id', 'sku_id'], as_index=False).sum()
        max_actions.drop('sku_id', inplace=True, axis=1)
        max_actions = max_actions.groupby(['user_id'], as_index=False).max()
        del max_actions['user_id']
        statistical_action = pd.concat([mean_actions, max_actions], axis=1)


        df_min = pd.get_dummies(actions['type'], prefix="%d_action_min" % i)
        min_actions = pd.concat([actions[['user_id', 'sku_id']], df_min], axis=1)
        min_actions = min_actions.groupby(['user_id', 'sku_id'], as_index=False).sum()
        min_actions.drop('sku_id', inplace=True, axis=1)
        min_actions = min_actions.groupby(['user_id'], as_index=False).min()
        del min_actions['user_id']
        statistical_action = pd.concat([statistical_action, min_actions], axis= 1)


        df_var = pd.get_dummies(actions['type'], prefix="%d_action_var" % i)
        var_actions = pd.concat([actions[['user_id', 'sku_id']], df_var], axis=1)
        var_actions = var_actions.groupby(['user_id', 'sku_id'], as_index=False).sum()
        var_actions.drop('sku_id', inplace=True, axis=1)
        var_actions = var_actions.groupby(['user_id'], as_index= False).var()
        del var_actions['user_id']
        statistical_action = pd.concat([statistical_action, var_actions], axis=1)


        df_std = pd.get_dummies(actions['type'], prefix="%d_action_std" % i)
        std_actions = pd.concat([actions[['user_id', 'sku_id']], df_std], axis=1)
        std_actions = std_actions.groupby(['user_id', 'sku_id'], as_index=False).sum()
        std_actions.drop('sku_id', inplace=True, axis=1)
        std_actions = std_actions.groupby(['user_id'], as_index= False).std()
        del std_actions['user_id']
        statistical_action = pd.concat([statistical_action, std_actions], axis=1)


        pickle.dump(statistical_action, open(dump_path, 'w'))

    #print statistical_action.shape, statistical_action.columns.values
    return statistical_action





# 构建训练集 action_windows_day "2016-02-01" or "2016-04-06" or "2016-04-01"
def make_train_set(train_start_date, train_end_date, test_start_date, test_end_date,days = 5):
    dump_path = './cache/M2_35_user_train_set_%s_%s_%s_%s_%d.pkl' % (train_start_date, train_end_date, test_start_date, test_end_date, days)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path))
    else:
        start_days = "2016-02-01"
        user = get_basic_user_feat()
        reg_acc_day = get_reg_day(train_start_date, train_end_date)
        reg_first_buy_day = get_register_first_buy_interval_day(train_start_date, train_end_date)
        user_last_cart = get_user_last_cart(start_days, train_end_date)
        user_last_buy = get_user_last_buy(start_days, train_end_date)
        user_cate_rank = get_cate_rank_in_user(start_days, train_end_date)
        uc_cor = get_uc_correlation(start_days, train_end_date)
        user_cate_last_buy = get_user_cate_last_buy(start_days, train_end_date)
        user_cate_last_cart = get_user_cate_last_cart(start_days, train_end_date)

        #labels
        labels = get_user_labels(test_start_date, test_end_date)

        # 商品比率
        user_ratio = get_user_ratio(start_days, train_end_date, 0)
        statistical_cate8 = get_user_cate8_statistical(start_days, train_end_date, 0)
        # 取最近5天的
        if days == 0:
            action_windows_day = start_days
        else:
            action_windows_day = datetime.strptime(train_end_date, '%Y-%m-%d') - timedelta(days)
            action_windows_day = action_windows_day.strftime('%Y-%m-%d')


        uc_conv_ratio = get_user_uc_conv_ratio(train_start_date, train_end_date, 0)
        user_buy_acc = get_user_buy_period(start_days, train_end_date, 0)
        user_conv_ratio_acc = get_user_conv_ratio(train_start_date, train_end_date, 0)
        user_action_acc = get_user_action(start_days, train_end_date, 0)
        uc_buy_action_tm = get_user_cate_buy_action_day(start_days, train_end_date, 0)
        uc_action_tm = get_user_cate_action_day(start_days, train_end_date, 0)
        user_buy_action_tm = get_user_buy_action_day(start_days, train_end_date, 0)
        user_action_tm = get_user_action_day(start_days, train_end_date, 0)

        # model_id
   #     user_model_id = get_user_model_id(start_days, train_end_date)
    #      user_model_cate8_id = get_user_model_id_cate8(start_days, train_end_date)
        #       user_cate8_ratio = get_user_cate8_ratio(start_days, train_end_date, 0)
        #   uc_num_feat = get_user_uc_num_feat(start_days, train_end_date, 0)


        user_id_fileter = get_watch_user_cate8(action_windows_day, train_end_date)

        actions = user_id_fileter
        actions = pd.merge(actions, user, how='left', on='user_id')
        actions = pd.merge(actions, reg_acc_day, how='left', on='user_id')
        actions = pd.merge(actions, reg_first_buy_day, how='left', on='user_id')
        actions = pd.merge(actions, user_buy_acc, how='left', on='user_id')
        actions = pd.merge(actions, user_conv_ratio_acc, how='left', on='user_id')
        actions = pd.merge(actions, user_action_acc, how='left', on='user_id')
        actions = pd.merge(actions, labels, how='left', on='user_id')
        actions = pd.merge(actions, user_last_cart, how='left', on='user_id')
        actions = pd.merge(actions, user_last_buy, how='left', on='user_id')
        actions = pd.merge(actions, user_cate_rank, how='left', on='user_id')
        actions = pd.merge(actions, uc_cor, how='left', on='user_id')
        actions = pd.merge(actions, uc_conv_ratio, how='left', on='user_id')

        actions = pd.merge(actions, user_cate_last_buy, how='left', on='user_id')
        actions = pd.merge(actions, user_cate_last_cart, how='left', on='user_id')
        actions = pd.merge(actions, uc_buy_action_tm, how='left', on='user_id')
        actions = pd.merge(actions, user_buy_action_tm, how='left', on='user_id')
        actions = pd.merge(actions, uc_action_tm, how='left', on='user_id')
        actions = pd.merge(actions, user_action_tm, how='left', on='user_id')
        actions = pd.merge(actions, statistical_cate8, how='left', on=['user_id'])
        # 商品比率
        actions = pd.merge(actions, user_ratio, how='left', on="user_id")
 #       actions = pd.merge(actions, user_cate8_ratio, how='left', on='user_id')
  #      actions = pd.merge(actions, user_model_id, how='left', on=['user_id'])
  #      actions = pd.merge(actions, user_model_cate8_id, how='left', on=['user_id'])
    #     actions = pd.merge(actions, uc_num_feat, how='left', on='user_id')


        for i in (1, 2, 3, 5, 10, 15, 21, 28, 35):
            print("feature phrase: ", i)
            start_days = datetime.strptime(train_end_date, '%Y-%m-%d') - timedelta(days=i)
            start_days = start_days.strftime('%Y-%m-%d')
            actions = pd.merge(actions, get_user_conv_ratio(start_days, train_end_date, i), how='left', on='user_id')
            actions = pd.merge(actions, get_user_action(start_days, train_end_date, i), how='left', on = 'user_id')
            actions = pd.merge(actions, get_user_buy_period(start_days, train_end_date, i), how='left', on = 'user_id')
            actions = pd.merge(actions, get_user_uc_conv_ratio(start_days, train_end_date, i), how='left', on = 'user_id')
            actions = pd.merge(actions, get_user_cate_buy_action_day(start_days, train_end_date, i), how='left', on='user_id')
            actions = pd.merge(actions, get_user_cate_action_day(start_days, train_end_date, i), how='left', on='user_id')
            actions = pd.merge(actions, get_user_buy_action_day(start_days, train_end_date, i), how='left', on='user_id')
            actions = pd.merge(actions, get_user_action_day(start_days, train_end_date, i), how='left', on='user_id')
            # ratio
            actions = pd.merge(actions, get_user_ratio(start_days, train_end_date, i), how='left', on='user_id')
   #         actions = pd.merge(actions, get_user_cate8_ratio(start_days, train_end_date, i), how='left', on='user_id')
        #         actions = pd.merge(actions, get_user_uc_num_feat(start_days, train_end_date, i), how='left', on='user_id')

        for i in (5, 30):
            start_days = datetime.strptime(train_end_date, '%Y-%m-%d') - timedelta(days=i)
            start_days = start_days.strftime('%Y-%m-%d')
            actions = pd.merge(actions, get_user_cate8_statistical(start_days, train_end_date, i), how='left',
                               on='user_id')

        actions = actions.fillna(0)
        pickle.dump(actions, open(dump_path, 'w'))

    # 训练集也只用cate == 8
    users = actions[['user_id']].copy()
    labels = actions['label'].copy()
    del actions['user_id']
    del actions['label']
    del actions['sex_0.0']
    del actions['sex_1.0']
    del actions['sex_2.0']
    print("user_train:", actions.shape)
    return users, actions, labels

# 构建测试集 action_windows_day "2016-02-01" or "2016-04-06" or "2016-04-01"
def make_test_set(train_start_date, train_end_date, days = 5):
    dump_path = './cache/M2_35_user_test_set_%s_%s_%d.pkl' % (train_start_date, train_end_date, days)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path))
    else:
        start_days = "2016-02-01"
        user = get_basic_user_feat()
        reg_acc_day = get_reg_day(train_start_date, train_end_date)
        reg_first_buy_day = get_register_first_buy_interval_day(train_start_date, train_end_date)
        user_last_cart = get_user_last_cart(start_days, train_end_date)
        user_last_buy = get_user_last_buy(start_days, train_end_date)
        user_cate_rank = get_cate_rank_in_user(start_days, train_end_date)
        uc_cor = get_uc_correlation(start_days, train_end_date)
        user_cate_last_buy = get_user_cate_last_buy(start_days, train_end_date)
        user_cate_last_cart = get_user_cate_last_cart(start_days, train_end_date)

        #商品比率
        user_ratio = get_user_ratio(start_days, train_end_date, 0)
        statistical_cate8 = get_user_cate8_statistical(start_days, train_end_date, 0)
        # 取最近5天的
        if days == 0:
            action_windows_day = start_days
        else:
            action_windows_day = datetime.strptime(train_end_date, '%Y-%m-%d') - timedelta(days)
            action_windows_day = action_windows_day.strftime('%Y-%m-%d')

        uc_conv_ratio = get_user_uc_conv_ratio(train_start_date, train_end_date, 0)
        user_buy_acc = get_user_buy_period(start_days, train_end_date, 0)
        user_conv_ratio_acc = get_user_conv_ratio(train_start_date, train_end_date, 0)
        user_action_acc = get_user_action(start_days, train_end_date, 0)
        uc_buy_action_tm = get_user_cate_buy_action_day(start_days, train_end_date, 0)
        uc_action_tm = get_user_cate_action_day(start_days, train_end_date, 0)
        user_buy_action_tm = get_user_buy_action_day(start_days, train_end_date, 0)
        user_action_tm = get_user_action_day(start_days, train_end_date, 0)

        user_id_fileter = get_watch_user_cate8(action_windows_day, train_end_date)

        # model_id
#        user_model_id = get_user_model_id(start_days, train_end_date)
#        user_model_cate8_id = get_user_model_id_cate8(start_days, train_end_date)
        #       user_cate8_ratio = get_user_cate8_ratio(start_days, train_end_date, 0)
        #        uc_num_feat = #(start_days, train_end_date, 0)

        #全局信息
        actions = user_id_fileter
        actions = pd.merge(actions, user, how='left', on='user_id')
        actions = pd.merge(actions, reg_acc_day, how='left', on='user_id')
        actions = pd.merge(actions, reg_first_buy_day, how='left', on='user_id')
        actions = pd.merge(actions, user_buy_acc, how='left', on='user_id')
        actions = pd.merge(actions, user_conv_ratio_acc, how='left', on='user_id')
        actions = pd.merge(actions, user_action_acc, how='left', on='user_id')
        actions = pd.merge(actions, user_last_cart, how='left', on='user_id')
        actions = pd.merge(actions, user_last_buy, how='left', on='user_id')
        actions = pd.merge(actions, user_cate_rank, how='left', on='user_id')
        actions = pd.merge(actions, uc_cor, how='left', on='user_id')
        actions = pd.merge(actions, uc_conv_ratio, how='left', on='user_id')

        actions = pd.merge(actions, user_cate_last_buy, how='left', on='user_id')
        actions = pd.merge(actions, user_cate_last_cart, how='left', on='user_id')
        actions = pd.merge(actions, uc_buy_action_tm, how='left', on='user_id')
        actions = pd.merge(actions, user_buy_action_tm, how='left', on='user_id')
        actions = pd.merge(actions, uc_action_tm, how='left', on='user_id')
        actions = pd.merge(actions, user_action_tm, how='left', on='user_id')
        actions = pd.merge(actions, statistical_cate8, how='left', on=['user_id'])
        # 商品比率
        actions = pd.merge(actions, user_ratio, how='left', on="user_id")
 #       actions = pd.merge(actions, user_cate8_ratio, how = 'left', on = 'user_id')
        #model_id
 #       actions = pd.merge(actions, user_model_id, how='left', on=['user_id'])
#        actions = pd.merge(actions, user_model_cate8_id, how='left', on=['user_id'])
        #        actions = pd.merge(actions, uc_num_feat, how='left', on='user_id')


        for i in (1, 2, 3, 5, 10, 15, 21, 28, 35):
            print("feature phrase: ", i)
            start_days = datetime.strptime(train_end_date, '%Y-%m-%d') - timedelta(days=i)
            start_days = start_days.strftime('%Y-%m-%d')
            actions = pd.merge(actions, get_user_conv_ratio(start_days, train_end_date, i), how='left', on='user_id')
            actions = pd.merge(actions, get_user_action(start_days, train_end_date, i), how='left', on='user_id')
            actions = pd.merge(actions, get_user_buy_period(start_days, train_end_date, i), how='left', on='user_id')
            actions = pd.merge(actions, get_user_uc_conv_ratio(start_days, train_end_date, i), how='left', on='user_id')
            actions = pd.merge(actions, get_user_cate_buy_action_day(start_days, train_end_date, i), how='left',
                               on='user_id')
            actions = pd.merge(actions, get_user_cate_action_day(start_days, train_end_date, i), how='left',
                               on='user_id')
            actions = pd.merge(actions, get_user_buy_action_day(start_days, train_end_date, i), how='left',
                               on='user_id')
            actions = pd.merge(actions, get_user_action_day(start_days, train_end_date, i), how='left', on='user_id')
            # ratio
            actions = pd.merge(actions, get_user_ratio(start_days, train_end_date, i), how='left', on='user_id')
            #         actions = pd.merge(actions, get_user_cate8_ratio(start_days, train_end_date, i), how='left', on='user_id')

        for i in (5, 30):
            start_days = datetime.strptime(train_end_date, '%Y-%m-%d') - timedelta(days=i)
            start_days = start_days.strftime('%Y-%m-%d')
            actions = pd.merge(actions, get_user_cate8_statistical(start_days, train_end_date, i), how='left',
                               on='user_id')


        actions = actions.fillna(0)
        pickle.dump(actions, open(dump_path, 'w'))


    users = actions[['user_id']].copy()
    del actions['user_id']
    print("user_test:", actions.shape)
    del actions['sex_0.0']
    del actions['sex_1.0']
    del actions['sex_2.0']
    return (users, actions)



if __name__ == '__main__':


    # train_start_date = '2016-02-01'
    # train_end_date = '2016-04-06'
    # test_start_date = '2016-04-06'
    # test_end_date = '2016-04-11'
    #
    # make_train_set(train_start_date, train_end_date, test_start_date, test_end_date, days= 30)
    #
    # train_start_date = '2016-02-06'
    # train_end_date = '2016-04-11'
    # test_start_date = '2016-04-11'
    # test_end_date = '2016-04-16'
    #
    # make_train_set(train_start_date, train_end_date, test_start_date, test_end_date, days=30)

    train_start_date = '2016-02-06'
    train_end_date = '2016-04-11'
    make_test_set(train_start_date, train_end_date, days=30)