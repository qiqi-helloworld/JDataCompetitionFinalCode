#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "QIQI"
import pandas as pd


def show_common_res_ui(user_res, ui_res, top_user_accont, top_ui_account):
    merge_dat = pd.merge(user_res.head(top_user_accont), ui_res.head(top_ui_account), how='inner', on=['user_id', 'sku_id'])
    print("Top"+ str(top_user_accont)+"user result+ top" + str(top_ui_account) +"ui result common number:",  merge_dat.shape)
    return merge_dat

def show_common_res(user_res, ui_res, top_user_accont, top_ui_account):
    merge_dat = pd.merge(user_res.head(top_user_accont), ui_res.head(top_ui_account), how='inner', on=['user_id'])
    print("Top"+ str(top_user_accont)+"user result+ top" + str(top_ui_account) +"ui result common number:",  merge_dat.shape)
    return merge_dat


ui_data_path = "./cache/final_sub_prob.csv"

ui_dat = pd.read_csv(ui_data_path)
ui_dat = ui_dat[['user_id', 'sku_id']].astype(int)
ui_dat.to_csv("./cache/sub.csv", index=False, index_label=False)