#!/usr/local/bin/env python
# -*- coding: UTF-8 -*-


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

# 去掉了过年7天的信息
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
        return 1
    elif age_str == u'15岁以下':
        return 2
    elif age_str == u'16-25岁':
        return 3
    elif age_str == u'26-35岁':
        return 4
    elif age_str == u'36-45岁':
        return 5
    elif age_str == u'46-55岁':
        return 6
    elif age_str == u'56岁以上':
        return 7
    else:
        return 0


# 排序函数
def relative_order(a):
    l = sorted(a, reverse=True)
    # hash table of element -> index in ordered list
    d = dict(zip(l, range(len(l))))
    return [d[e] for e in a]


#model id
def get_ui_model_id(start_date, end_date):
    dump_path = './cache/ui_model_id_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        model_id_dat = pickle.load(open(dump_path))
    else:
        actions = get_actions(start_date, end_date)
        actions = actions[['user_id', 'sku_id', 'model_id']]
        actions = actions[actions['model_id'].isin([14, 21, 28, 110, 210])]
        df = pd.get_dummies(actions['model_id'], prefix="ui_model_id")
        actions = pd.concat([actions[['user_id', 'sku_id']], df], axis= 1)
        model_id_dat = actions.groupby(['user_id', 'sku_id'], as_index = False).count()

    return model_id_dat

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


###########################################用户特征###################################################
# 基本用户信息 年龄 性别 用户等级
def get_basic_user_feat():
    """Function :get basic user informations"""
    dump_path = './cache/basic_user.pkl'
    if os.path.exists(dump_path):
        user = pickle.load(open(dump_path))
    else:
        user = pd.read_csv(user_path, encoding='gbk')
        user['age'] = user['age'].map(lambda x:convert_age(x))
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
            lambda x: (datetime.strptime(end_date, '%Y-%m-%d') - datetime.strptime(x, '%Y-%m-%d')).days)
        user_dat = user_dat[['user_id', 'user_reg_tm']]
        user_dat.loc[user_dat['user_reg_tm'] < 0, 'user_reg_tm'] = -1
        pickle.dump(user_dat, open(dump_path, 'w'))
    return user_dat

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
        user_dat['user_reg_tm'] = user_dat['user_reg_tm'].map(
            lambda x: (datetime.strptime(end_date, '%Y-%m-%d') - datetime.strptime(x, '%Y-%m-%d')).days)
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
    # stat_first_buy = Counter(dat['inter_reg_first_buy'])
    # stat_list = sorted(stat_first_buy.items(), key=lambda d: d[0])
    # x_day = [stat_list[i][0] for i in range(len(stat_list))]
    # y_day = [stat_list[i][1] for i in range(len(stat_list))]
    # plt.plot(x_day, y_day)
    # plt.show()
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
        user_buy_period_dat['%d_action_period_1'%i] = max_interval_day * 1.0 / user_buy_period_dat['%d_action_period_1'%i]
        user_buy_period_dat['%d_action_period_2'%i] = max_interval_day * 1.0 / user_buy_period_dat['%d_action_period_2'%i]
        user_buy_period_dat['%d_action_period_3'%i] = max_interval_day * 1.0 / user_buy_period_dat['%d_action_period_3'%i]
        user_buy_period_dat['%d_action_period_4'%i] = max_interval_day * 1.0 / user_buy_period_dat['%d_action_period_4'%i]
        user_buy_period_dat['%d_action_period_5'%i] = max_interval_day * 1.0 / user_buy_period_dat['%d_action_period_5'%i]
        user_buy_period_dat['%d_action_period_6'%i] = max_interval_day * 1.0 / user_buy_period_dat['%d_action_period_6'%i]
        user_buy_period_dat[user_buy_period_dat < 0] = -1
        pickle.dump(user_buy_period_dat, open(dump_path, 'w'))
    # print(user_buy_period_dat.columns.values)
    return user_buy_period_dat

#用户购买周期
def get_user_first_buy_last_buy_interval(start_date, end_date, i):
    dump_path = './cache/%d_user_first_buy_last_buy_day_%s_%s.pkl' % (i, start_date, end_date)
    if os.path.exists(dump_path):
        buy_dat = pickle.load(open(dump_path, 'rb+'))
    else:
        actions = get_actions(start_date, end_date)
        first_buy = actions.drop_duplicates(['user_id', 'type'], keep='first')
        laste_buy =actions.drop_duplicates(['user_id', 'type'], keep='last')
        print("first_buy: ", first_buy.shape[0],"laste_buy: ", laste_buy.shape[0])
        first_buy['first_buy'] = first_buy['time'].map(
            lambda x: (datetime.strptime(end_date, '%Y-%m-%d') - datetime.strptime(x, '%Y-%m-%d %H:%M:%S')).days)
        laste_buy['last_buy'] = laste_buy['time'].map(
            lambda x: (datetime.strptime(end_date, '%Y-%m-%d') - datetime.strptime(x, '%Y-%m-%d %H:%M:%S')).days)
        first_buy = first_buy[['user_id', 'type', 'first_buy']]
        laste_buy = laste_buy[['user_id', 'type', 'last_buy']]
        buy_dat = pd.merge(first_buy, laste_buy, how='left', on = ['user_id', 'type'])
        buy_dat = buy_dat[buy_dat['type'] == 4]
        buy_dat['interval'] = buy_dat['first_buy'] - buy_dat['last_buy']
        del buy_dat['last_buy']
        buy_dat.rename(columns={"interval": "%d_user_interval"%i, 'first_buy': '%d_user_first_buy'%i}, inplace = True)
        pickle.dump(buy_dat, open(dump_path, 'w'))

    #(buy_dat.columns.values)
    return buy_dat


# 每个用户的商品行为（流量，加入购物车， 关注）转化比率
def get_user_conv_ratio(start_date, end_date, i):
    dump_path = './cache/%d_user_conv_ratio%s_%s.pkl' % (i, start_date, end_date)
    if os.path.exists(dump_path):
        user_conv_ratio = pickle.load(open(dump_path))
    else:
        actions = get_actions(start_date, end_date)
        df = pd.get_dummies(actions['type'], prefix='%d_user_conv_ratio'%i)
        actions = pd.concat([actions[['user_id']], df], axis=1)
        actions = actions.groupby(['user_id'], as_index=False).sum()
        actions['%d_user_conv_ratio_1'%i] = actions['%d_user_conv_ratio_4'%i] / actions['%d_user_conv_ratio_1'%i]
        actions['%d_user_conv_ratio_2'%i] = actions['%d_user_conv_ratio_4'%i] / actions['%d_user_conv_ratio_2'%i]
        actions['%d_user_conv_ratio_5'%i] = actions['%d_user_conv_ratio_4'%i] / actions['%d_user_conv_ratio_5'%i]
        actions['%d_user_conv_ratio_3'%i] = actions['%d_user_conv_ratio_4'%i] / actions['%d_user_conv_ratio_3'%i]
        actions['%d_user_conv_ratio_6'%i] = actions['%d_user_conv_ratio_4'%i] / actions['%d_user_conv_ratio_6'%i]
        user_conv_ratio = actions[['user_id', '%d_user_conv_ratio_1'%i, '%d_user_conv_ratio_2'%i, '%d_user_conv_ratio_3'%i,
                                   '%d_user_conv_ratio_5'%i, '%d_user_conv_ratio_6'%i]]
        user_conv_ratio = user_conv_ratio.replace(np.inf, np.nan)
        user_conv_ratio = user_conv_ratio.fillna(0)
        pickle.dump(user_conv_ratio, open(dump_path, 'w'))
    return user_conv_ratio

#user_num_feart
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

#用户交互天数
def get_user_action_tm(start_date, end_date, i):
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
def get_user_buy_action_tm(start_date, end_date, i):
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




#'''''''''''''''''''''''商品特征''''''''''''''''''''''''''''
# 基本商品信息
def get_basic_product_feat():
    dump_path = './cache/basic_product.pkl'
    if os.path.exists(dump_path):
        product = pickle.load(open(dump_path))
    else:
        product = pd.read_csv(product_path)
        attr1_df = pd.get_dummies(product["a1"], prefix="a1")
        attr2_df = pd.get_dummies(product["a2"], prefix="a2")
        attr3_df = pd.get_dummies(product["a3"], prefix="a3")
        product = pd.concat([product[['sku_id', 'cate', 'brand']], attr1_df, attr2_df, attr3_df], axis=1)
        pickle.dump(product, open(dump_path, 'w'))
    return product



# 商品转化率
def get_sku_conv_ratio(start_date, end_date, i):
    dump_path = './cache/%d_sku_conv_ratio%s_%s.pkl' % (i, start_date, end_date)
    if os.path.exists(dump_path):
        pro_conv_ratio = pickle.load(open(dump_path))
    else:
        actions = get_actions(start_date, end_date)
        df = pd.get_dummies(actions['type'], prefix='%d_sku_conv_ratio' % i)
        actions = pd.concat([actions[['sku_id']], df], axis=1)
        actions = actions.groupby(['sku_id'], as_index=False).sum()
        actions['%d_sku_conv_ratio_1'%i] = actions['%d_sku_conv_ratio_4'%i] / actions['%d_sku_conv_ratio_1'%i]
        actions['%d_sku_conv_ratio_2'%i] = actions['%d_sku_conv_ratio_4'%i] / actions['%d_sku_conv_ratio_2'%i]
        actions['%d_sku_conv_ratio_5'%i] = actions['%d_sku_conv_ratio_4'%i] / actions['%d_sku_conv_ratio_5'%i]
        actions['%d_sku_conv_ratio_6'%i] = actions['%d_sku_conv_ratio_4'%i] / actions['%d_sku_conv_ratio_6'%i]
        actions['%d_sku_conv_ratio_3'%i] = actions['%d_sku_conv_ratio_4'%i] / actions['%d_sku_conv_ratio_3'%i]
        pro_conv_ratio = actions[['sku_id', '%d_sku_conv_ratio_1'%i, '%d_sku_conv_ratio_2'%i, '%d_sku_conv_ratio_3'%i, '%d_sku_conv_ratio_5'%i, '%d_sku_conv_ratio_6'%i]]
        pro_conv_ratio = pro_conv_ratio.replace(np.inf, np.nan)
        pro_conv_ratio = pro_conv_ratio.fillna(0)
        pickle.dump(pro_conv_ratio, open(dump_path, 'w'))
    return pro_conv_ratio


# 基本商品信息购买行为数量
def get_sku_num_feat(start_date, end_date, i):
    """
     different action count for different number.
    """
    dump_path = './cache/product_action_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        product = pickle.load(open(dump_path))
    else:
        actions = get_actions(start_date, end_date)

        df = pd.get_dummies(actions['type'], prefix='sku_action_%s_%s' % (start_date, end_date))
        actions = pd.concat([actions['sku_id'], df], axis=1)
        actions = actions.groupby(['sku_id'], as_index=False).sum()
        product = actions
        pickle.dump(product, open(dump_path, 'w'))

    product.rename(columns={ 'sku_action_%s_%s_1' % (start_date, end_date):'%d_sku_action_1'%i,
                            'sku_action_%s_%s_2' % (start_date, end_date):'%d_sku_action_2'%i,
                                'sku_action_%s_%s_3' % (start_date, end_date):'%d_sku_action_3'%i,
                            'sku_action_%s_%s_4' % (start_date, end_date):'%d_sku_action_4'%i,
                                'sku_action_%s_%s_5' % (start_date, end_date):'%d_sku_action_5'%i,
                            'sku_action_%s_%s_6' % (start_date, end_date): '%d_sku_action_6'%i}, inplace=True)

    return product


#'''''''''''''''''''''''Cate特征'''''''''''''''''''''''''''
# 每个cate中的商品排名
def get_sku_rank_in_cate(start_date, end_date):
    '''on = sku_id, cate_id'''
    dump_path = './cache/sku_rank_in_cate%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        sku_rank_in_cate = pickle.load(open(dump_path))
    else:
        actions = get_actions(start_date, end_date)
        df1 = pd.get_dummies(actions['type'], prefix='ratio_action')
        df2 = pd.get_dummies(actions['type'], prefix='cate_action')
        df1 = pd.concat([actions[['sku_id', 'cate']], df1], axis=1)
        df2 = pd.concat([actions['cate'], df2], axis=1)
        actions = actions[['cate', 'sku_id']]
        actions = pd.concat([actions, df1.groupby(['sku_id', 'cate']).transform(sum)], axis=1)
        actions = pd.concat([actions, df2.groupby(['cate']).transform(sum)], axis=1)
        actions = actions.groupby(['cate', 'sku_id'], as_index=False).first()
        actions['ratio_action_1'] = actions['ratio_action_1'] / actions['cate_action_1']
        actions['ratio_action_2'] = actions['ratio_action_2'] / actions['cate_action_2']
        actions['ratio_action_3'] = actions['ratio_action_3'] / actions['cate_action_3']
        actions['ratio_action_4'] = actions['ratio_action_4'] / actions['cate_action_4']
        actions['ratio_action_5'] = actions['ratio_action_5'] / actions['cate_action_5']
        actions['ratio_action_6'] = actions['ratio_action_6'] / actions['cate_action_6']
        actions['sku_cate_rank'] = actions['ratio_action_1'] + actions['ratio_action_2'] + actions['ratio_action_3'] + \
                                   actions['ratio_action_4'] + actions['ratio_action_5'] + actions['ratio_action_6']
        sku_rank_in_cate = actions[['sku_id', 'cate', 'sku_cate_rank', 'ratio_action_4']]
        pickle.dump(sku_rank_in_cate, open(dump_path, 'w'))

    return sku_rank_in_cate


# 每个cate中的brand排名
def get_brand_rank_in_cate(start_date, end_date):
    """
    同cate 中 sku 排名
    """
    dump_path = './cache/brand_in_cate_rank_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        brand_in_cate = pickle.load(open(dump_path))
    else:
        actions = get_actions(start_date, end_date)
        df1 = pd.get_dummies(actions['type'], prefix='ratio_action')
        df2 = pd.get_dummies(actions['type'], prefix='cate_action')
        df1 = pd.concat([actions[['brand', 'cate']], df1], axis=1)
        df2 = pd.concat([actions['cate'], df2], axis=1)
        actions = actions[['cate', 'brand']]
        actions = pd.concat([actions, df1.groupby(['brand', 'cate']).transform(sum)], axis=1)
        actions = pd.concat([actions, df2.groupby(['cate']).transform(sum)], axis=1)
        actions = actions.groupby(['cate', 'brand'], as_index=False).first()
        actions['ratio_action_1'] = actions['ratio_action_1'] / actions['cate_action_1']
        actions['ratio_action_2'] = actions['ratio_action_2'] / actions['cate_action_2']
        actions['ratio_action_3'] = actions['ratio_action_3'] / actions['cate_action_3']
        actions['ratio_action_4'] = actions['ratio_action_4'] / actions['cate_action_4']
        actions['ratio_action_5'] = actions['ratio_action_5'] / actions['cate_action_5']
        actions['ratio_action_6'] = actions['ratio_action_6'] / actions['cate_action_6']
        actions['sku_cate_rank'] = actions['ratio_action_1'] + actions['ratio_action_2'] + actions['ratio_action_3'] + \
                                   actions['ratio_action_4'] + actions['ratio_action_5'] + actions['ratio_action_6']
        brand_in_cate = actions[['brand', 'cate', 'sku_cate_rank', 'ratio_action_4']]
        pickle.dump(brand_in_cate, open(dump_path, 'w'))

    #print("brand_in_cate:", brand_in_cate.columns.values)
    return brand_in_cate


###########################################用户/商品交互特征（ui）##################################################
# 用户商品点击量
def get_ui_num_feat(start_date, end_date, i):
    dump_path = './cache/ui_action_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path))
    else:
        actions = get_actions(start_date, end_date)
        actions = actions[['user_id', 'sku_id', 'type']]
        df = pd.get_dummies(actions['type'], prefix='%s-%s-action' % (start_date, end_date))
        actions = pd.concat([actions, df], axis=1)  # type: pd.DataFrame
        actions = actions.groupby(['user_id', 'sku_id'], as_index=False).sum()
        del actions['type']
        pickle.dump(actions, open(dump_path, 'w'))

    actions.rename(columns={'%s-%s-action_1' % (start_date, end_date): '%d_action_1' % i,
                            '%s-%s-action_2' % (start_date, end_date): '%d_action_2' % i,
                            '%s-%s-action_3' % (start_date, end_date): '%d_action_3' % i,
                            '%s-%s-action_4' % (start_date, end_date): '%d_action_4' % i,
                            '%s-%s-action_5' % (start_date, end_date): '%d_action_5' % i,
                            '%s-%s-action_6' % (start_date, end_date): '%d_action_6' % i}, inplace=True)

    return actions


# 用户商品点击量 近期行为按时间衰减
def get_accumulate_action_feat(start_date, end_date):
    """
    different action with decay function 近期行为按时间衰减
    """
    dump_path = './cache/action_accumulate_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path))
    else:
        actions = get_actions(start_date, end_date)
        df = pd.get_dummies(actions['type'], prefix='action')
        actions = pd.concat([actions, df], axis=1)  # type: pd.DataFrame
        # 近期行为按时间衰减
        actions['weights'] = actions['time'].map(
            lambda x: datetime.strptime(end_date, '%Y-%m-%d') - datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
        # actions['weights'] = time.strptime(end_date, '%Y-%m-%d') - actions['datetime']
        actions['weights'] = actions['weights'].map(lambda x: math.exp(-x.days))
        # print (actions.head(10))
        actions['action_1'] = actions['action_1'] * actions['weights']
        actions['action_2'] = actions['action_2'] * actions['weights']
        actions['action_3'] = actions['action_3'] * actions['weights']
        actions['action_4'] = actions['action_4'] * actions['weights']
        actions['action_5'] = actions['action_5'] * actions['weights']
        actions['action_6'] = actions['action_6'] * actions['weights']
        del actions['model_id']
        del actions['type']
        del actions['time']
        # del actions['datetime']
        del actions['weights']
        actions = actions.groupby(['user_id', 'sku_id', 'cate', 'brand'], as_index=False).sum()
        pickle.dump(actions, open(dump_path, 'w'))
    return actions


# 每个用户购买的商品排名
def get_sku_rank_in_user(start_date, end_date, i):
    """
    each ui action rank order
    """
    dump_path = './cache/sku_rank_in_user%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        sku_rank_in_user = pickle.load(open(dump_path))
    else:
        actions = get_actions(start_date, end_date)
        df1 = pd.get_dummies(actions['type'], prefix='ratio_action')
        df2 = pd.get_dummies(actions['type'], prefix='user_action')
        df1 = pd.concat([actions[['user_id', 'sku_id']], df1], axis=1)
        df2 = pd.concat([actions['user_id'], df2], axis=1)
        actions = actions[['user_id', 'sku_id']]
        actions = pd.concat([actions, df1.groupby(['user_id', 'sku_id']).transform(sum)], axis=1)
        actions = pd.concat([actions, df2.groupby(['user_id']).transform(sum)], axis=1)
        actions = actions.groupby(['user_id', 'sku_id'], as_index=False).first()
        actions['ratio_action_1'] = actions['ratio_action_1'] / actions['user_action_1']
        actions['ratio_action_2'] = actions['ratio_action_2'] / actions['user_action_2']
        actions['ratio_action_3'] = actions['ratio_action_3'] / actions['user_action_3']
        actions['ratio_action_4'] = actions['ratio_action_4'] / actions['user_action_4']
        actions['ratio_action_5'] = actions['ratio_action_5'] / actions['user_action_5']
        actions['ratio_action_6'] = actions['ratio_action_6'] / actions['user_action_6']
        actions['sku_user_rank_%s_%s'%(start_date, end_date)] = actions['ratio_action_1'] + actions['ratio_action_2'] + actions['ratio_action_3'] + \
                                   actions['ratio_action_4'] + actions['ratio_action_5'] + actions['ratio_action_6']
        sku_rank_in_user = actions[['user_id', 'sku_id', 'sku_user_rank_%s_%s'%(start_date, end_date), 'ratio_action_4']]
        pickle.dump(sku_rank_in_user, open(dump_path, 'w'))


    sku_rank_in_user.rename(columns={'sku_user_rank_%s_%s' % (start_date, end_date): '%d_sku_user_rank' % i,
                                     "ratio_action_4", ''}, inplace=True)

    return sku_rank_in_user


# 用户商品点击量（刚刚发现和 gen_feat.get_action_feat 功能一样）
# def get_ui_num(start_date, end_date):
#     dump_path = './cache/ui_num%s_%s.pkl' % (start_date, end_date)
#     if os.path.exists(dump_path):
#         ui_feat_num = pickle.load(open(dump_path))
#     else:
#         actions = get_actions(start_date, end_date)
#         df = pd.get_dummies(actions['type'], prefix='ui_action_%s_%s'%(start_date, end_date))
#         actions = pd.concat([actions[['user_id', 'sku_id']], df], axis=1)
#         ui_feat_num = actions.groupby(['user_id', 'sku_id'], as_index=False).sum()
#     return ui_feat_num


# 每个用户商品对 动作转化率
def get_ui_conv_ratio(start_date, end_date, i):
    dump_path = './cache/%d_ui_conv_ratio%s_%s.pkl' % (i, start_date, end_date)
    if os.path.exists(dump_path):
        ui_conv_ratio = pickle.load(open(dump_path))
    else:
        actions = get_actions(start_date, end_date)
        df = pd.get_dummies(actions['type'], prefix='%d_ui_conv_ratio'%i)
        actions = pd.concat([actions[['user_id', 'sku_id']], df], axis=1)
        actions = actions.groupby(['user_id', 'sku_id'], as_index=False).sum()
        actions['%d_ui_conv_ratio_1'%i] = actions['%d_ui_conv_ratio_4'%i] / actions['%d_ui_conv_ratio_1'%i]
        actions['%d_ui_conv_ratio_2'%i] = actions['%d_ui_conv_ratio_4'%i] / actions['%d_ui_conv_ratio_2'%i]
        actions['%d_ui_conv_ratio_5'%i] = actions['%d_ui_conv_ratio_4'%i] / actions['%d_ui_conv_ratio_5'%i]
        actions['%d_ui_conv_ratio_3'%i] = actions['%d_ui_conv_ratio_4'%i] / actions['%d_ui_conv_ratio_3'%i]
        actions['%d_ui_conv_ratio_6'%i] = actions['%d_ui_conv_ratio_4'%i] / actions['%d_ui_conv_ratio_6'%i]
        ui_conv_ratio = actions[['user_id', 'sku_id', '%d_ui_conv_ratio_1'%i, '%d_ui_conv_ratio_2'%i,'%d_ui_conv_ratio_3'%i,
                                 '%d_ui_conv_ratio_5'%i,'%d_ui_conv_ratio_6'%i]]
        ui_conv_ratio = ui_conv_ratio.replace(np.inf, np.nan)
        ui_conv_ratio = ui_conv_ratio.fillna(0)
        pickle.dump(ui_conv_ratio, open(dump_path, 'w'))
    return ui_conv_ratio

#用户最近一次加入购物车
def get_ui_last_cart(start_date, end_date):
    dump_path = './cache/ui_last_cart_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        cart_dat = pickle.load(open(dump_path, 'rb+'))
    else:
        actions = get_actions(start_date, end_date)
        actions = actions[actions['cate'] == 8]
        actions = actions[actions['type'] == 3]
        actions.drop_duplicates(['user_id', 'sku_id', 'type'],keep = 'last', inplace=True)

        actions['ui_last_cart'] = actions['time'].map(
                lambda x: (datetime.strptime(end_date, '%Y-%m-%d') - datetime.strptime(x, '%Y-%m-%d %H:%M:%S')).days)

        cart_dat = actions[['user_id', 'sku_id', 'ui_last_cart']]
        pickle.dump(cart_dat, open(dump_path, 'w'))
    return cart_dat


#用户最近一次加入购物车
def get_ui_last_del_cart(start_date, end_date):
    dump_path = './cache/ui_last_del_cart_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        cart_dat = pickle.load(open(dump_path, 'rb+'))
    else:
        actions = get_actions(start_date, end_date)
        actions = actions[actions['cate'] == 8]
        actions = actions[actions['type'] == 2]
        actions.drop_duplicates(['user_id', 'sku_id', 'type'],keep = 'last', inplace=True)

        actions['ui_last_del_cart'] = actions['time'].map(
                lambda x: (datetime.strptime(end_date, '%Y-%m-%d') - datetime.strptime(x, '%Y-%m-%d %H:%M:%S')).days)

        cart_dat = actions[['user_id', 'sku_id', 'ui_last_del_cart']]
        pickle.dump(cart_dat, open(dump_path, 'w'))
    return cart_dat


#用户最近一次购买
def get_ui_last_buy(start_date, end_date):
    dump_path = './cache/ui_last_buy_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        buy_dat = pickle.load(open(dump_path, 'rb+'))
    else:
        actions = get_actions(start_date, end_date)
        actions = actions[actions['cate'] == 8]
        actions = actions[actions['type'] == 4]
        actions.drop_duplicates(['user_id', 'sku_id', 'type'], keep = 'last', inplace=True)

        actions['ui_last_buy'] = actions['time'].map(
            lambda x: (datetime.strptime(end_date, '%Y-%m-%d') - datetime.strptime(x, '%Y-%m-%d %H:%M:%S')).days)
        buy_dat = actions[['user_id', 'sku_id', 'ui_last_buy']]
        pickle.dump(buy_dat, open(dump_path, 'w'))

    return buy_dat





#用户商品对交互天数
def get_ui_action_tm(start_date, end_date, i):
    dump_path = './cache/ui_ui_action_tm_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path))
    else:
        actions = get_actions(start_date, end_date)
        actions['time'] = actions['time'].map(
            lambda x: (datetime.strptime(end_date, '%Y-%m-%d') - datetime.strptime(x, '%Y-%m-%d %H:%M:%S')).days)
        actions = actions[['user_id','sku_id', 'time']]
        actions = actions.drop_duplicates()
        actions = actions.groupby(['user_id', 'sku_id'], as_index=False).count()
        actions.rename(columns = {'time':'%d_ui_tm'%i}, inplace=True)
        pickle.dump(actions, open(dump_path, 'w'))

    return actions

#用户商品对购买天数
def get_ui_buy_action_tm(start_date, end_date, i):
    dump_path = './cache/ui_ui_buy_action_tm_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path))
    else:
        actions = get_actions(start_date, end_date)
        actions = actions[actions['type'] == 4]
        actions['time'] = actions['time'].map(
            lambda x: (datetime.strptime(end_date, '%Y-%m-%d') - datetime.strptime(x, '%Y-%m-%d %H:%M:%S')).days)
        actions = actions[['user_id', 'sku_id', 'time']]
        actions = actions.drop_duplicates()
        actions = actions.groupby(['user_id', 'sku_id'], as_index=False).count()
        actions.rename(columns={'time': '%d_ui_buy_tm' % i}, inplace=True)
        pickle.dump(actions, open(dump_path, 'w'))

    return actions


###########################################用户/Cate交互特征（uc）##################################################

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

    sku_rank_in_user.rename(columns={'ratio_action_4': 'sku_user_rank_ratio_4'}, inplace=True)

    return sku_rank_in_user


# 每个用户购买的同一cate中的商品排名
def get_sku_rank_in_user_cate(start_date, end_date):
    '''
    同 uc中 ui 排名
    '''
    dump_path = './cache/sku_rank_in_user_cate%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        sku_rank_in_user_cate = pickle.load(open(dump_path))
    else:

        actions = get_actions(start_date, end_date)
        df1 = pd.get_dummies(actions['type'], prefix='ratio_ui_uc_action')
        df2 = pd.get_dummies(actions['type'], prefix='uc_action')
        df1 = pd.concat([actions[['user_id', 'cate', 'sku_id']], df1], axis=1)
        df2 = pd.concat([actions[['user_id', 'cate']], df2], axis=1)
        actions = actions[['user_id', 'cate', 'sku_id']]
        actions = pd.concat([actions, df1.groupby(['user_id', 'cate', 'sku_id']).transform(sum)], axis=1)
        actions = pd.concat([actions, df2.groupby(['user_id', 'cate']).transform(sum)], axis=1)
        actions = actions.groupby(['user_id', 'cate', 'sku_id'], as_index=False).first()
        actions['ratio_ui_uc_action_1'] = actions['ratio_ui_uc_action_1'] / actions['uc_action_1']
        actions['ratio_ui_uc_action_2'] = actions['ratio_ui_uc_action_2'] / actions['uc_action_2']
        actions['ratio_ui_uc_action_3'] = actions['ratio_ui_uc_action_3'] / actions['uc_action_3']
        actions['ratio_ui_uc_action_4'] = actions['ratio_ui_uc_action_4'] / actions['uc_action_4']
        actions['ratio_ui_uc_action_5'] = actions['ratio_ui_uc_action_5'] / actions['uc_action_5']
        actions['ratio_ui_uc_action_6'] = actions['ratio_ui_uc_action_6'] / actions['uc_action_6']
        actions['sku_in_uc_rank'] = actions['ratio_ui_uc_action_1'] + actions['ratio_ui_uc_action_2'] + actions[
            'ratio_ui_uc_action_3'] + \
                                    actions['ratio_ui_uc_action_4'] + actions['ratio_ui_uc_action_5'] + actions[
                                        'ratio_ui_uc_action_6']
        sku_rank_in_user_cate = actions[['user_id', 'cate', 'sku_id', 'sku_in_uc_rank', 'ratio_ui_uc_action_4']]
        pickle.dump(sku_rank_in_user_cate, open(dump_path, 'w'))

    # print ("sku_rank_in_user_cate")
    return sku_rank_in_user_cate


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

    ui_feat_num.rename(columns={'uc_action_%s_%s_1' % (start_date, end_date): '%d_uc_action_1' % i,
                            'uc_action_%s_%s_2' % (start_date, end_date): '%d_uc_action_2' % i,
                            'uc_action_%s_%s_3' % (start_date, end_date): '%d_uc_action_3' % i,
                            'uc_action_%s_%s_4' % (start_date, end_date): '%d_uc_action_4' % i,
                            'uc_action_%s_%s_5' % (start_date, end_date): '%d_uc_action_5' % i,
                            'uc_action_%s_%s_6' % (start_date, end_date): '%d_uc_action_6' % i,
                            'sku_%s_%s' % (start_date, end_date): '%d_sku_buy_num_in_cate' %  i}, inplace=True)

    return ui_feat_num


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
    #print(uc_conv_ratio.columns.values)
    return uc_conv_ratio


#用户cate交互天数
def get_uc_action_tm(start_date, end_date, i):
    dump_path = './cache/ui_uc_action_tm_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path))
    else:
        actions = get_actions(start_date, end_date)
        actions['time'] = actions['time'].map(
            lambda x: (datetime.strptime(end_date, '%Y-%m-%d') - datetime.strptime(x, '%Y-%m-%d %H:%M:%S')).days)
        actions = actions[['user_id', 'time', 'cate']]
        actions = actions.drop_duplicates(['user_id', 'time', 'cate'])
        actions = actions.groupby(['user_id', 'cate'], as_index=False).count()
        actions.rename(columns={'time': '%d_uc_tm' % i}, inplace=True)
        pickle.dump(actions, open(dump_path, 'w'))

    return actions

#用户cate购买交互天数
def get_uc_buy_action_tm(start_date, end_date, i):
    dump_path = './cache/ui_uc_buy_action_tm_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path))
    else:
        actions = get_actions(start_date, end_date)
        actions = actions[actions['type'] == 4]
        actions['time'] = actions['time'].map(
            lambda x: (datetime.strptime(end_date, '%Y-%m-%d') - datetime.strptime(x, '%Y-%m-%d %H:%M:%S')).days)
        actions = actions[['user_id', 'time', 'cate']]
        actions = actions.drop_duplicates(['user_id', 'time', 'cate'])
        actions = actions.groupby(['user_id',  'cate'], as_index=False).count()
        actions.rename(columns={'time': '%d_uc_buy_tm' % i}, inplace=True)
        pickle.dump(actions, open(dump_path, 'w'))
    return actions

#cate8统计量
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

#cate8统计量
def get_product_cate8_statistical(start_date, end_date, i):
    dump_path = './cache/%d_product_cate8_statistical_%s_%s.pkl' % (i, start_date, end_date)
    if os.path.exists(dump_path):
        statistical_action = pickle.load(open(dump_path))
    else:
        actions = get_actions(start_date, end_date)
        actions = actions[actions['cate'] == 8]
        actions['time'] = actions['time'].map(
            lambda x: (datetime.strptime(end_date, '%Y-%m-%d') - datetime.strptime(x, '%Y-%m-%d %H:%M:%S')).days)
        actions = actions[['sku_id', 'time', 'type']]

        df_mean = pd.get_dummies(actions['type'], prefix= "%d_pro_action_mean" %i)
        mean_actions = pd.concat([actions[['sku_id', 'time']], df_mean], axis= 1)
        mean_actions = mean_actions.groupby(['sku_id', 'time'], as_index = False).sum()
        mean_actions.drop('time', inplace=True, axis = 1)
        mean_actions = mean_actions.groupby(['sku_id'], as_index = False).mean()

        df_max = pd.get_dummies(actions['type'], prefix="%d_pro_action_max" % i)
        max_actions = pd.concat([actions[['sku_id', 'time']], df_max], axis=1)
        max_actions = max_actions.groupby(['sku_id', 'time'], as_index=False).sum()
        max_actions.drop('time', inplace=True, axis=1)
        max_actions = max_actions.groupby(['sku_id'], as_index=False).max()
        del max_actions['sku_id']
        statistical_action = pd.concat([mean_actions, max_actions], axis=1)


        df_min = pd.get_dummies(actions['type'], prefix="%d_pro_action_min" % i)
        min_actions = pd.concat([actions[['sku_id', 'time']], df_min], axis=1)
        min_actions = min_actions.groupby(['sku_id', 'time'], as_index=False).sum()
        min_actions.drop('time', inplace=True, axis=1)
        min_actions = min_actions.groupby(['sku_id'], as_index=False).min()
        del min_actions['sku_id']
        statistical_action = pd.concat([statistical_action, min_actions], axis= 1)


        df_var = pd.get_dummies(actions['type'], prefix="%d_pro_action_var" % i)
        var_actions = pd.concat([actions[['sku_id', 'time']], df_var], axis=1)
        var_actions = var_actions.groupby(['sku_id', 'time'], as_index=False).sum()
        var_actions.drop('time', inplace=True, axis=1)
        var_actions = var_actions.groupby(['sku_id'], as_index= False).var()
        del var_actions['sku_id']
        statistical_action = pd.concat([statistical_action, var_actions], axis=1)


        df_std = pd.get_dummies(actions['type'], prefix="%d_pro_action_std" % i)
        std_actions = pd.concat([actions[['sku_id', 'time']], df_std], axis=1)
        std_actions = std_actions.groupby(['sku_id', 'time'], as_index=False).sum()
        std_actions.drop('time', inplace=True, axis=1)
        std_actions = std_actions.groupby(['sku_id'], as_index= False).std()
        del std_actions['sku_id']
        statistical_action = pd.concat([statistical_action, std_actions], axis=1)


        pickle.dump(statistical_action, open(dump_path, 'w'))

    #print statistical_action.shape, statistical_action.columns.values
    return statistical_action






#''''''''''用户/商品/Cate交互特征（syn）'''''''''''

# 用户/商品/Cate 首次购买和最后一次购买天数
def get_accumulate_first_buy_last_by_syn_feat(start_date, end_date, i):
    """
	Function: Get each (sku_id, user_id, cate) first buy and last buy time interval to the predict day.
	Therefore, due to the difference days between train and test data could be different,
	the length of dummy vector would be different, therefore cause different data dimension in train and test dataset.
	TODO:some problems in this function, different vector in train and test data.
	"""
    dump_path = './cache/syn_first_last_buy_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        buy_dat_first_interval = pickle.load(open(dump_path))
    else:
        actions = get_actions(start_date, end_date)
        actions = actions[actions['type'] == 4]
        actions['inter_day'] = actions['time'].map(
            lambda x: (datetime.strptime(end_date, '%Y-%m-%d') - datetime.strptime(x, '%Y-%m-%d %H:%M:%S')).days)
        actions = actions[['user_id', 'sku_id', 'cate', 'inter_day']]
        first_buy = actions.drop_duplicates(['user_id', 'sku_id', 'cate'], keep = 'first')
        last_buy = actions.drop_duplicates(['user_id', 'sku_id', 'cate'], keep = 'last')
        first_buy.rename(columns={'inter_day': '%d_first_buy_day' % i}, inplace=True)
        last_buy.rename(columns={'inter_day': '%d_last_buy_day' % i}, inplace=True)
        buy_dat_first_interval = pd.merge(first_buy, last_buy, how="left", on = ['user_id', 'sku_id', 'cate'])
        buy_dat_first_interval['%d_ui_interval'%i] = buy_dat_first_interval['%d_first_buy_day' % i] - buy_dat_first_interval['%d_last_buy_day' % i]
        pickle.dump(buy_dat_first_interval, open(dump_path, 'w'))

    #print("get_accumulate_first_buy_last_by_syn_feat:", i, buy_dat_first_interval.columns.values)
    return buy_dat_first_interval


# 用户/商品/Cate 行为个数
def get_syn_num_feat(start_date, end_date, i):
    """return(user_id, sku_id, cate) action type 1:6 in different date period."""
    dump_path = './cache/syn_action_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        syn_feat_num = pickle.load(open(dump_path))
    else:
        actions = get_actions(start_date, end_date)
        actions = actions[['user_id', 'sku_id', 'cate', 'type']]
        df = pd.get_dummies(actions['type'], prefix='syn_action_%s_%s' % (start_date, end_date))
        actions = pd.concat([actions, df], axis=1)
        syn_feat_num = actions.groupby(['user_id', 'sku_id', 'cate'], as_index=False).sum()
        del syn_feat_num['type']
        pickle.dump(syn_feat_num, open(dump_path, 'w'))

    syn_feat_num.rename(columns={'syn_action_%s_%s_1' % (start_date, end_date): '%d_syn_action_1' % i,
                                'syn_action_%s_%s_2' % (start_date, end_date): '%d_syn_action_2' % i,
                                'syn_action_%s_%s_3' % (start_date, end_date): '%d_syn_action_3' % i,
                                'syn_action_%s_%s_4' % (start_date, end_date): '%d_syn_action_4' % i,
                                'syn_action_%s_%s_5' % (start_date, end_date): '%d_syn_action_5' % i,
                                'syn_action_%s_%s_6' % (start_date, end_date): '%d_syn_action_6' % i}, inplace=True)
    # print "get_syn_num_feat", actions.columns.values
    return syn_feat_num


# 用户/商品/Cate 转化率
def get_syn_conv_ratio(start_date, end_date, i):
    """return(user_id, sku_id, cate) action type 1:6 in different date period."""
    dump_path = './cache/%d_syn_conv_ratio%s_%s.pkl' % (i, start_date, end_date)
    if os.path.exists(dump_path):
        syn_conv_ratio = pickle.load(open(dump_path))
    else:
        actions = get_actions(start_date, end_date)
        actions = actions[['user_id', 'sku_id', 'cate', 'type']]
        df = pd.get_dummies(actions['type'], prefix='%d_syn_conv_ratio'%i)
        actions = pd.concat([actions, df], axis=1)
        actions = actions.groupby(['user_id', 'sku_id', 'cate'], as_index=False).sum()
        actions['%d_syn_conv_ratio_1'%i] = actions['%d_syn_conv_ratio_4'%i] / actions['%d_syn_conv_ratio_1'%i]
        actions['%d_syn_conv_ratio_2'%i] = actions['%d_syn_conv_ratio_4'%i] / actions['%d_syn_conv_ratio_2'%i]
        actions['%d_syn_conv_ratio_3'%i] = actions['%d_syn_conv_ratio_4'%i] / actions['%d_syn_conv_ratio_3'%i]
        actions['%d_syn_conv_ratio_5'%i] = actions['%d_syn_conv_ratio_4'%i] / actions['%d_syn_conv_ratio_5'%i]
        actions['%d_syn_conv_ratio_6'%i] = actions['%d_syn_conv_ratio_4'%i] / actions['%d_syn_conv_ratio_6'%i]
        syn_conv_ratio = actions[['user_id', 'sku_id', 'cate', '%d_syn_conv_ratio_1'%i, '%d_syn_conv_ratio_2'%i, '%d_syn_conv_ratio_3'%i, '%d_syn_conv_ratio_5'%i, '%d_syn_conv_ratio_6'%i]]
        syn_conv_ratio = syn_conv_ratio.replace(np.inf, np.nan)
        syn_conv_ratio = syn_conv_ratio.fillna(0)
        pickle.dump(syn_conv_ratio, open(dump_path, 'w'))

    #print "get_syn_conv_ratio", syn_conv_ratio.columns.values
    return syn_conv_ratio


#'''''''''''''''''''''''''''''''''评论信息'''''''''''''''''''''''''''''''''''''''
# 不同评价的个数
def get_comments_product_feat(start_date, end_date):
    """comments information"""
    dump_path = './cache/comments_accumulate_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        comments = pickle.load(open(dump_path))
    else:
        comments = pd.read_csv(comment_path)
        comment_date_end = end_date
        comment_date_begin = comment_date[0]
        for date in reversed(comment_date):
            if date < comment_date_end:
                comment_date_begin = date
                break
        comments = comments[(comments.dt >= comment_date_begin) & (comments.dt < comment_date_end)]
        df = pd.get_dummies(comments['comment_num'], prefix='comment_num')
        comments = pd.concat([comments, df], axis=1)  # type: pd.DataFrame
        comments = comments[
            ['sku_id', 'has_bad_comment', 'bad_comment_rate', 'comment_num_1', 'comment_num_2', 'comment_num_3',
             'comment_num_4']]
        pickle.dump(comments, open(dump_path, 'w'))
    return comments



def get_cate8_labels(start_date, end_date):
    """Each prdiction label"""
    dump_path = './cache/cate8_labels_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path))
    else:
        actions = get_actions(start_date, end_date)
        actions = actions[(actions['type'] == 4) & (actions['cate'] == 8)]
        actions = actions.groupby(['user_id', 'sku_id'], as_index=False).sum()
        actions['label'] = 1
        actions = actions[['user_id', 'sku_id', 'label']]
        pickle.dump(actions, open(dump_path, 'w'))
    return actions





#构建训练集 去掉最近五天的
def M2_make_train_set(train_start_date, train_end_date, test_start_date, test_end_date, days=0):
    dump_path = './cache/M2_35_0524_period_train_cate8_set_%s_%s_%s_%s_%d.pkl' % (
        train_start_date, train_end_date, test_start_date, test_end_date, days)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path))
    else:
        start_days = "2016-02-01"
        #basic information
        user = get_basic_user_feat()                                                                                 #基本用户信息
        product = get_basic_product_feat()                                                                           #基本商品信息
        comment_acc = get_comments_product_feat(start_days, train_end_date)                                          #累积评论
        first_last_acc = get_accumulate_first_buy_last_by_syn_feat(train_start_date, train_end_date, 0)              #第一次购买， 以及和最后一次购买时长

        labels = get_cate8_labels(test_start_date, test_end_date)   #target

        # model_id
        ui_moel_id = get_ui_model_id(start_days, train_end_date)
        user_model_id = get_user_model_id(start_days, train_end_date)
        user_model_cate8_id = get_user_model_id_cate8(start_days, train_end_date)

        #rank
        sku_rank_in_user_acc = get_sku_rank_in_user(train_start_date, train_end_date,0)                                #商品在user浏览排名
        sku_rank_in_user_cate_acc = get_sku_rank_in_user_cate(train_start_date, train_end_date)                      #商品在每个用户购买类目下的排名
        sku_rank_in_cate_acc = get_sku_rank_in_cate(train_start_date, train_end_date)                                #商品在每个cate下排名
        cate_rank_in_user_acc = get_cate_rank_in_user(train_start_date, train_end_date)                              #用户下 每个cate的排名
        brand_rank_in_cate_acc = get_brand_rank_in_cate(train_start_date, train_end_date)                            # brand cate排名


        #time interval information
        reg_acc_day = get_reg_day(train_start_date, train_end_date)                                                  #注册天数
        reg_first_buy_day = get_register_first_buy_interval_day(train_start_date, train_end_date)                    #第一次注册跟第一次购买时间间隔
        user_first_buy_acc = get_user_first_buy_last_buy_interval(train_start_date, train_end_date, 0)               #用户第一次购买时长
        ui_last_cart = get_ui_last_cart(start_days, train_end_date)                                                  #最后一次加购
        ui_last_buy = get_ui_last_buy(start_days, train_end_date)                                                    #最后一次购买
        user_last_cart = get_user_last_cart(start_days, train_end_date)                                              #用户最后一次加够
        user_last_buy = get_user_last_buy(start_days, train_end_date)                                                #用户最后一次购买
        uc_cor_acc = get_uc_correlation(train_start_date, train_end_date)                                            #一个user 买了哪些cate




        #活动数量
        sku_num_acc = get_sku_num_feat(start_days, train_end_date, 0)
        syn_num_acc = get_syn_num_feat(start_days, train_end_date, 0)
        uc_num_acc = get_uc_num_feat(start_days, train_end_date, 0)
        user_num_acc = get_user_action(start_days, train_end_date, 0)
        ui_num_acc = get_ui_num_feat(start_days, train_end_date, 0)

        #活动天数
        ui_buy_action_tm = get_ui_buy_action_tm(start_days, train_end_date, 0)                                  #截至至交互日前5天 用户商品购买活跃度
        ui_action_tm = get_ui_action_tm(start_days, train_end_date, 0)                                          #用户商品活跃天
        uc_action_tm = get_uc_action_tm(start_days, train_end_date, 0)                                          #用户cate活跃天数
        uc_buy_action_tm = get_uc_buy_action_tm(start_days, train_end_date, 0)                                  #用户cate购买天数
        user_action_tm = get_user_action_tm(start_days, train_end_date, 0)                                      #用户活跃天数
        user_buy_action_tm = get_user_buy_action_tm(start_days, train_end_date, 0)                              #用户购买活跃天数

        # transfer ratio
        user_buy_acc = get_user_buy_period(start_days, train_end_date, 0)                                       #用户购买总件数/时长
        ui_conv_ratio_acc = get_ui_conv_ratio(train_start_date, train_end_date, 0)                              #用户商品转化率
        pro_conv_ratio_acc = get_sku_conv_ratio(train_start_date, train_end_date, 0)                            #商品转化率
        user_conv_ratio_acc = get_user_conv_ratio(train_start_date, train_end_date, 0)                          #用户转化率
        uc_conv_ratio_acc = get_uc_conv_ratio(train_start_date, train_end_date, 0)                              #用户cate 转化率
        syn_conv_ratio_acc = get_syn_conv_ratio(train_start_date, train_end_date, 0)                            #用户商品cate转化率
        statistical_cate8 = get_user_cate8_statistical(start_days, train_end_date, 0)


        if days == 0:
            action_windows_day = start_days
        else:
            action_windows_day = datetime.strptime(train_end_date, '%Y-%m-%d') - timedelta(days = days)
            action_windows_day = action_windows_day.strftime('%Y-%m-%d')

        print "action_windows_day", action_windows_day
        actions = get_accumulate_action_feat(action_windows_day, train_end_date)                                    #累积行为
        actions = actions[actions['cate'] == 8]
        for i in (1, 2, 3, 5, 10, 15, 21, 28, 35):
            start_days = datetime.strptime(train_end_date, '%Y-%m-%d') - timedelta(days=i)
            start_days = start_days.strftime('%Y-%m-%d')
            if actions is None:
                actions = get_ui_num_feat(start_days, train_end_date, i)                                               #各个商品行为类型
            else:
                actions = pd.merge(actions, get_ui_num_feat(start_days, train_end_date, i), how='left',
                                   on=['user_id', 'sku_id'])

        for i in  (5, 30):
            start_days = datetime.strptime(train_end_date, '%Y-%m-%d') - timedelta(days=i)
            start_days = start_days.strftime('%Y-%m-%d')
            actions = pd.merge(actions, get_user_cate8_statistical(start_days, train_end_date, i), how='left',
                               on='user_id')

        actions = pd.merge(actions, user, how='left', on='user_id')
        actions = pd.merge(actions, product, how='left', on=['sku_id', 'cate', 'brand'])
        actions = pd.merge(actions, comment_acc, how='left', on='sku_id')
        actions = pd.merge(actions, first_last_acc, how='left', on =['user_id', 'sku_id', 'cate'])
        actions = pd.merge(actions, sku_rank_in_user_acc, how='left', on=['user_id', 'sku_id'])
        actions = pd.merge(actions, sku_rank_in_user_cate_acc, how='left', on=['user_id', 'cate', 'sku_id'])
        actions = pd.merge(actions, brand_rank_in_cate_acc, how='left', on=['cate', 'brand'])
        actions = pd.merge(actions, cate_rank_in_user_acc, how='left', on=['cate', 'user_id'])
        actions = pd.merge(actions, sku_rank_in_cate_acc, how='left', on=['cate', 'sku_id'])
        actions = pd.merge(actions, reg_acc_day, how='left', on='user_id')
        actions = pd.merge(actions, reg_first_buy_day, how='left', on= 'user_id')
        actions = pd.merge(actions, user_buy_acc, how='left', on='user_id')
        actions = pd.merge(actions, uc_cor_acc, how='left', on='user_id')
        actions = pd.merge(actions, ui_conv_ratio_acc, how='left', on=['user_id', 'sku_id'])
        actions = pd.merge(actions, pro_conv_ratio_acc, how='left', on='sku_id')
        actions = pd.merge(actions, user_conv_ratio_acc, how='left', on='user_id')
        actions = pd.merge(actions, uc_conv_ratio_acc, how='left', on=['user_id', 'cate'])
        #print("actions:", actions.columns.values)
        actions = pd.merge(actions, syn_conv_ratio_acc, how='left', on=['user_id', 'sku_id', 'cate'])
        actions = pd.merge(actions, ui_last_cart, how='left', on=['user_id', 'sku_id'])
        actions = pd.merge(actions, ui_last_buy, how='left', on=['user_id', 'sku_id'])
        #活跃天数
        actions = pd.merge(actions, ui_buy_action_tm, how='left', on=['user_id', 'sku_id'])
        actions = pd.merge(actions, ui_action_tm, how='left', on=['user_id', 'sku_id'])
        actions = pd.merge(actions, uc_action_tm, how='left', on=['user_id', 'cate'])
        actions = pd.merge(actions, uc_buy_action_tm, how='left', on=['user_id', 'cate'])
        actions = pd.merge(actions, user_action_tm, how='left', on='user_id')
        actions = pd.merge(actions, user_buy_action_tm, how='left', on='user_id')
        # 购买量
        actions = pd.merge(actions, sku_num_acc, how='left', on='sku_id')
        actions = pd.merge(actions, syn_num_acc, how='left',
                           on=['user_id', 'sku_id', 'cate'])
        actions = pd.merge(actions, uc_num_acc, how='left', on=['user_id', 'cate'])
        actions = pd.merge(actions, user_num_acc, how='left', on=['user_id'])
        actions = pd.merge(actions, ui_num_acc, how='left', on=['user_id', 'sku_id'])

        #user feat
        actions = pd.merge(actions, user_first_buy_acc, how='left', on=['user_id'])
        actions = pd.merge(actions, user_last_buy, how='left', on=['user_id'])
        actions = pd.merge(actions, user_last_cart, how='left', on=['user_id'])

        # model_id
        actions = pd.merge(actions, ui_moel_id, how='left', on=['user_id', 'sku_id'])
        actions = pd.merge(actions, user_model_id, how='left', on=['user_id'])
        actions = pd.merge(actions, user_model_cate8_id, how='left', on=['user_id'])

        # statistical
        actions = pd.merge(actions, statistical_cate8, how='left', on=['user_id'])
        for i in (1, 2, 3, 5, 10, 15, 21, 28, 35):
            print("feature phrase: ", i)
            start_days = datetime.strptime(train_end_date, '%Y-%m-%d') - timedelta(days=i)
            start_days = start_days.strftime('%Y-%m-%d')
            actions = pd.merge(actions, get_sku_num_feat(start_days, train_end_date, i), how='left', on='sku_id')
            actions = pd.merge(actions, get_syn_num_feat(start_days, train_end_date, i), how='left', on=['user_id', 'sku_id', 'cate'])
            actions = pd.merge(actions, get_uc_num_feat(start_days, train_end_date, i), how='left', on=['user_id', 'cate'])
           # actions = pd.merge(actions, get_user_buy_period(start_days, train_end_date, i), how='left',  on=['user_id'])
           # actions = pd.merge(actions, get_ui_conv_ratio(start_days, train_end_date, i), how='left',  on=['user_id', 'sku_id'])
            actions = pd.merge(actions, get_user_conv_ratio(start_days, train_end_date, i), how='left', on='user_id')
           # actions = pd.merge(actions, get_uc_conv_ratio(start_days, train_end_date, i), how='left',  on=['user_id', 'cate'])
           # actions = pd.merge(actions, get_syn_conv_ratio(start_days, train_end_date, i), how='left', on=['user_id', 'sku_id', 'cate'])
           # actions = pd.merge(actions, get_accumulate_first_buy_last_by_syn_feat(start_days, train_end_date, i),how='left', on=['user_id', 'sku_id', 'cate'])
            actions = pd.merge(actions, get_ui_buy_action_tm(start_days, train_end_date, i), how='left',on=['user_id', 'sku_id'])
            actions = pd.merge(actions, get_ui_action_tm(start_days, train_end_date, i), how='left',on=['user_id', 'sku_id'])
            actions = pd.merge(actions, get_uc_action_tm(start_days, train_end_date, i), how='left', on=['user_id', 'cate'])
            actions = pd.merge(actions, get_uc_buy_action_tm(start_days, train_end_date, i), how='left', on=['user_id', 'cate'])
            actions = pd.merge(actions, get_user_action_tm(start_days, train_end_date, i), how='left', on='user_id')
            actions = pd.merge(actions, get_user_buy_action_tm(start_days, train_end_date, i), how='left', on='user_id')

        actions = pd.merge(actions, labels, how='left', on=['user_id', 'sku_id'])
        actions = actions.fillna(0)
        actions = pd.merge(actions, get_ui_last_del_cart("2016-02-01", train_end_date), how='left',
                           on=['user_id', 'sku_id'])
        actions = pd.merge(actions, get_product_cate8_statistical(start_days, train_end_date, 0), how='left',
                           on='sku_id')
        for i in (5, 30):
            start_days = datetime.strptime(train_end_date, '%Y-%m-%d') - timedelta(days=i)
            start_days = start_days.strftime('%Y-%m-%d')
            actions = pd.merge(actions, get_product_cate8_statistical(start_days, train_end_date, i), how='left',
                               on='sku_id')
        actions = actions.fillna(0)
        actions['cart_or_not'] = actions['ui_last_del_cart'] - actions['ui_last_cart']  # 加够天数 - 删够天数 >0 说明不大会买， < 0 会买
        actions['cart_or_not'] = actions['cart_or_not'].map(lambda x: x <= 0)
        actions['cart_or_not'] = actions['cart_or_not'].astype(int)

        pickle.dump(actions, open(dump_path, 'w'))

    #训练集也只用cate == 8
    actions = actions[actions['cate'] == 8]
    users = actions[['user_id', 'sku_id']].copy()
    labels = actions['label'].copy()
    del actions['user_id']
    del actions['sku_id']
    del actions['label']
    del actions['sex_0.0']
    del actions['sex_1.0']
    del actions['sex_2.0']
    print ("train data shape: ", actions.shape)
    return users, actions, labels


# 构建测试集 去掉最近5天
def M2_make_test_set(train_start_date, train_end_date, days = 0):
    dump_path = './cache/M2_35_0524_period_test_set_%s_%s_%d.pkl' % (train_start_date, train_end_date, days)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path))
    else:
        start_days = "2016-02-01"

        #basic infromation
        user = get_basic_user_feat()
        product = get_basic_product_feat()
        comment_acc = get_comments_product_feat(start_days, train_end_date)
        first_last_acc = get_accumulate_first_buy_last_by_syn_feat(train_start_date, train_end_date, 0)

        # model_id
        ui_moel_id = get_ui_model_id(start_days, train_end_date)
        user_model_id = get_user_model_id(start_days, train_end_date)
        user_model_cate8_id = get_user_model_id_cate8(start_days, train_end_date)


        #rank information
        sku_rank_in_user_acc = get_sku_rank_in_user(train_start_date, train_end_date, 0)
        sku_rank_in_user_cate_acc = get_sku_rank_in_user_cate(train_start_date, train_end_date)
        sku_rank_in_cate_acc = get_sku_rank_in_cate(train_start_date, train_end_date)
        cate_rank_in_user_acc = get_cate_rank_in_user(train_start_date, train_end_date)
        brand_rank_in_cate_acc = get_brand_rank_in_cate(train_start_date, train_end_date)

        #time interval information
        reg_acc_day = get_reg_day(train_start_date, train_end_date)
        reg_first_buy_day = get_register_first_buy_interval_day(train_start_date, train_end_date)
        uc_cor_acc = get_uc_correlation(train_start_date, train_end_date)
        ui_last_cart = get_ui_last_cart(start_days, train_end_date)
        ui_last_buy = get_ui_last_buy(start_days, train_end_date)
        user_last_cart = get_user_last_cart(start_days, train_end_date)
        user_last_buy = get_user_last_buy(start_days, train_end_date)



        #活动数量
        sku_num_acc = get_sku_num_feat(start_days, train_end_date, 0)
        syn_num_acc = get_syn_num_feat(start_days, train_end_date, 0)
        uc_num_acc = get_uc_num_feat(start_days, train_end_date, 0)
        user_num_acc = get_user_action(start_days, train_end_date, 0)
        ui_num_acc = get_ui_num_feat(start_days, train_end_date, 0)
        user_first_buy_acc = get_user_first_buy_last_buy_interval(train_start_date, train_end_date, 0)
        #活动天数
        ui_buy_action_tm = get_ui_buy_action_tm(start_days, train_end_date, 0)
        ui_action_tm = get_ui_action_tm(start_days, train_end_date, 0)
        uc_action_tm = get_uc_action_tm(start_days, train_end_date, 0)
        uc_buy_action_tm = get_uc_buy_action_tm(start_days, train_end_date, 0)
        user_action_tm = get_user_action_tm(start_days, train_end_date, 0)
        user_buy_action_tm = get_user_buy_action_tm(start_days, train_end_date, 0)

        #tranfer ration
        user_buy_acc = get_user_buy_period(start_days, train_end_date, 0)
        ui_conv_ratio_acc = get_ui_conv_ratio(train_start_date, train_end_date, 0)
        pro_conv_ratio_acc = get_sku_conv_ratio(train_start_date, train_end_date, 0)
        user_conv_ratio_acc = get_user_conv_ratio(train_start_date, train_end_date, 0)
        uc_conv_ratio_acc = get_uc_conv_ratio(train_start_date, train_end_date, 0)
        syn_conv_ratio_acc = get_syn_conv_ratio(train_start_date, train_end_date, 0)
        statistical_cate8 = get_user_cate8_statistical(start_days, train_end_date, 0)

        if days == 0:
            action_windows_day = start_days
        else:
            action_windows_day = datetime.strptime(train_end_date, '%Y-%m-%d') - timedelta(days = days)
            action_windows_day = action_windows_day.strftime('%Y-%m-%d')

        print "action_windows_day", action_windows_day
        actions = get_accumulate_action_feat(action_windows_day, train_end_date)
        actions = actions[actions['cate'] == 8]
        for i in (1, 2, 3, 5, 10, 15, 21, 28, 35):
            start_days = datetime.strptime(train_end_date, '%Y-%m-%d') - timedelta(days=i)
            start_days = start_days.strftime('%Y-%m-%d')
            if actions is None:
                actions = get_ui_num_feat(start_days, train_end_date, i)
            else:
                actions = pd.merge(actions, get_ui_num_feat(start_days, train_end_date, i), how='left',
                                   on=['user_id', 'sku_id'])

        for i in  (5, 30):
            start_days = datetime.strptime(train_end_date, '%Y-%m-%d') - timedelta(days=i)
            start_days = start_days.strftime('%Y-%m-%d')
            actions = pd.merge(actions,  get_user_cate8_statistical(start_days, train_end_date, i),
                               how='left', on = 'user_id')

        actions = pd.merge(actions, user, how='left', on='user_id')
        actions = pd.merge(actions, product, how='left', on=['sku_id', 'cate', 'brand'])
        actions = pd.merge(actions, comment_acc, how='left', on='sku_id')
        actions = pd.merge(actions, first_last_acc, how='left', on=['user_id', 'sku_id', 'cate'])
        actions = pd.merge(actions, sku_rank_in_user_acc, how='left', on=['user_id', 'sku_id'])
        actions = pd.merge(actions, sku_rank_in_user_cate_acc, how='left', on=['user_id', 'cate', 'sku_id'])
        actions = pd.merge(actions, brand_rank_in_cate_acc, how='left', on=['cate', 'brand'])
        actions = pd.merge(actions, cate_rank_in_user_acc, how='left', on=['cate', 'user_id'])
        actions = pd.merge(actions, sku_rank_in_cate_acc, how='left', on=['cate', 'sku_id'])
        actions = pd.merge(actions, reg_acc_day, how='left', on='user_id')
        actions = pd.merge(actions, reg_first_buy_day, how='left', on='user_id')
        actions = pd.merge(actions, user_buy_acc, how='left', on='user_id')
        actions = pd.merge(actions, uc_cor_acc, how='left', on='user_id')
        actions = pd.merge(actions, ui_conv_ratio_acc, how='left', on=['user_id', 'sku_id'])
        actions = pd.merge(actions, pro_conv_ratio_acc, how='left', on='sku_id')
        actions = pd.merge(actions, user_conv_ratio_acc, how='left', on='user_id')
        actions = pd.merge(actions, uc_conv_ratio_acc, how='left', on=['user_id', 'cate'])
        actions = pd.merge(actions, syn_conv_ratio_acc, how='left', on=['user_id', 'sku_id', 'cate'])
        actions = pd.merge(actions, ui_last_cart, how='left', on=['user_id', 'sku_id'])
        actions = pd.merge(actions, ui_last_buy, how='left', on=['user_id', 'sku_id'])


        #购买量
        actions = pd.merge(actions, sku_num_acc, how='left', on='sku_id')
        actions = pd.merge(actions, syn_num_acc, how='left',
                           on=['user_id', 'sku_id', 'cate'])
        actions = pd.merge(actions, uc_num_acc, how='left', on=['user_id', 'cate'])
        actions = pd.merge(actions, user_num_acc, how='left', on=['user_id'])
        actions = pd.merge(actions, ui_num_acc, how='left', on=['user_id', 'sku_id'])

        # 活跃天数
        actions = pd.merge(actions, ui_buy_action_tm, how='left', on=['user_id', 'sku_id'])
        actions = pd.merge(actions, ui_action_tm, how='left', on=['user_id', 'sku_id'])
        actions = pd.merge(actions, uc_action_tm, how='left', on=['user_id', 'cate'])
        actions = pd.merge(actions, uc_buy_action_tm, how='left', on=['user_id', 'cate'])
        actions = pd.merge(actions, user_action_tm, how='left', on='user_id')
        actions = pd.merge(actions, user_buy_action_tm, how='left', on='user_id')

        # user feat
        actions = pd.merge(actions, user_first_buy_acc, how='left', on=['user_id'])
        actions = pd.merge(actions, user_last_buy, how='left', on=['user_id'])
        actions = pd.merge(actions, user_last_cart, how='left', on=['user_id'])

        # model_id
        actions = pd.merge(actions, ui_moel_id, how='left', on=['user_id', 'sku_id'])
        actions = pd.merge(actions, user_model_id, how='left', on=['user_id'])
        actions = pd.merge(actions, user_model_cate8_id, how='left', on=['user_id'])
        # statistical
        actions = pd.merge(actions, statistical_cate8, how='left', on=['user_id'])
        for i in (1, 2, 3, 5, 10, 15, 21, 28, 35):
            print("feature phrase: ", i)
            start_days = datetime.strptime(train_end_date, '%Y-%m-%d') - timedelta(days=i)
            start_days = start_days.strftime('%Y-%m-%d')
            actions = pd.merge(actions, get_sku_num_feat(start_days, train_end_date, i), how='left', on='sku_id')
            actions = pd.merge(actions, get_syn_num_feat(start_days, train_end_date, i), how='left', on=['user_id', 'sku_id', 'cate'])
            actions = pd.merge(actions, get_uc_num_feat(start_days, train_end_date, i), how='left', on=['user_id', 'cate'])
            #actions = pd.merge(actions, get_user_buy_period(start_days, train_end_date, i), how='left', on=['user_id'])
            #actions = pd.merge(actions, get_ui_conv_ratio(start_days, train_end_date, i), how='left', on=['user_id', 'sku_id'])
            actions = pd.merge(actions, get_user_conv_ratio(start_days, train_end_date, i), how='left', on='user_id')
            #actions = pd.merge(actions, get_uc_conv_ratio(start_days, train_end_date, i), how='left', on=['user_id', 'cate'])
            #actions = pd.merge(actions, get_syn_conv_ratio(start_days, train_end_date, i), how='left', on=['user_id', 'sku_id', 'cate'])
            #actions = pd.merge(actions, get_accumulate_first_buy_last_by_syn_feat(start_days, train_end_date, i), how='left', on=['user_id', 'sku_id', 'cate'])
            actions = pd.merge(actions, get_ui_buy_action_tm(start_days, train_end_date, i), how='left', on=['user_id', 'sku_id'])
            actions = pd.merge(actions, get_ui_action_tm(start_days, train_end_date, i), how='left', on=['user_id', 'sku_id'])
            actions = pd.merge(actions, get_uc_action_tm(start_days, train_end_date, i), how='left',on=['user_id', 'cate'])
            actions = pd.merge(actions, get_uc_buy_action_tm(start_days, train_end_date, i), how='left',on=['user_id', 'cate'])
            actions = pd.merge(actions, get_user_action_tm(start_days, train_end_date, i), how='left', on='user_id')
            actions = pd.merge(actions, get_user_buy_action_tm(start_days, train_end_date, i), how='left', on='user_id')


        actions = actions.fillna(0)
        actions = actions[actions['cate'] == 8]

        actions = pd.merge(actions, get_ui_last_del_cart("2016-02-01", train_end_date), how='left',
                           on=['user_id', 'sku_id'])
        actions = pd.merge(actions, get_product_cate8_statistical(start_days, train_end_date, 0), how='left',
                           on='sku_id')
        for i in (5, 30):
            start_days = datetime.strptime(train_end_date, '%Y-%m-%d') - timedelta(days=i)
            start_days = start_days.strftime('%Y-%m-%d')
            actions = pd.merge(actions, get_product_cate8_statistical(start_days, train_end_date, i), how='left',
                               on='sku_id')
        actions = actions.fillna(0)
        actions['cart_or_not'] = actions['ui_last_del_cart'] - actions['ui_last_cart']  # 加够天数 - 删够天数 >0 说明不大会买， < 0 会买
        actions['cart_or_not'] = actions['cart_or_not'].map(lambda x: x <= 0)
        actions['cart_or_not'] = actions['cart_or_not'].astype(int)


        pickle.dump(actions, open(dump_path, 'w'))

    users = actions[['user_id', 'sku_id']].copy()

    del actions['user_id']
    del actions['sku_id']
    del actions['sex_0.0']
    del actions['sex_1.0']
    del actions['sex_2.0']
    print("ui test shape:", actions.shape)
    return (users, actions)






#下采样
def subsampe(ratio, user_index, training_data, label, random_seed,names):
    dump_path = './cache/subsample_' + str(ratio) + "_" + str(random_seed) + ".pkl"
    if os.path.exists(dump_path):
        with open(dump_path) as f:
            user_index = pickle.load(f)
            X_train = pickle.load(f)
            y_train =pickle.load(f)
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
        X_train_nobuy.loc[sam_nobuy_index], y_train_nobuy.loc[sam_nobuy_index], user_index_nobuy.loc[sam_nobuy_index])
        X_train = pd.concat([X_train_buy, X_train_nobuy], axis=0, ignore_index=True)
        y_train = pd.concat([y_train_buy, y_train_nobuy], axis=0, ignore_index=True)
        user_index = pd.concat([user_index_buy, user_index_nobuy], axis=0, ignore_index=True)
        X_train, y_train, user_index = shuffle(X_train, y_train, user_index, random_state=random_seed)
        # reset index
        user_index, X_train, y_train = (user_index.reset_index(drop=True),X_train.reset_index(drop=True), y_train.reset_index(drop=True))
        with open(dump_path, 'w') as f:
            pickle.dump(user_index, f)
            pickle.dump(X_train, f)
            pickle.dump(y_train, f)

    return user_index, X_train, y_train





if __name__ == '__main__':
    # 训练集
    train_start_date = '2016-02-06'
    train_end_date = '2016-04-11'
    test_start_date = '2016-04-11'
    test_end_date = '2016-04-16'

    # 线下测试集
    cv_train_start_date = '2016-02-01'
    cv_train_end_date = '2016-04-06'
    cv_test_start_date = '2016-04-06'
    cv_test_end_date = '2016-04-11'

    # 预测集
    sub_start_date = '2016-02-06'
    sub_end_date = '2016-04-16'

    get_reg_day(train_start_date, train_end_date)
    get_reg_day(cv_train_start_date, cv_train_end_date)
    get_reg_day(sub_start_date, sub_end_date)
    get_register_first_buy_interval_day(train_start_date, train_end_date)
    get_register_first_buy_interval_day(cv_train_start_date, cv_train_end_date)
    get_register_first_buy_interval_day(sub_start_date, sub_end_date)


