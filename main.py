from fastapi import FastAPI, Request
from typing import Union
import pandas as pd
import numpy as np
import py_eureka_client.eureka_client as eureka_client

from statsmodels.tsa.api import VAR

from scipy.stats import pearsonr

app = FastAPI()

eureka_client.init(eureka_server="http://localhost:8761/eureka",
                                app_name="cleaning-api",
                                instance_port=8082)

@app.get("/cleaning-api/test")
async def test():
    return {"message":"test"}

@app.post("/cleaning-api/missing-value/find")
async def find_mv(request: Request):
    req = await request.json()
    req_df = pd.json_normalize(req['data'])
    na_df = req_df[req_df.isna().any(axis=1)]
    na_df = na_df.replace({np.nan:None})
    return df_to_response(req['column'], na_df)


@app.post("/cleaning-api/missing-value/run")
async def run_mv(request: Request, method: int, idx_col: Union[str, None] = None):
    req = await request.json()
    req_df = pd.json_normalize(req['data'])
    na_df = req_df[req_df.isna().any(axis=1)]
    na_df = na_df.replace({np.nan: None})
    if(method == 0):
        column_df = pd.json_normalize(req['column'])
        num_type_col = column_df.where(column_df['type'] == 'DOUBLE' or column_df['type'] == 'INT').dropna()
        if (idx_col in num_type_col):
            num_type_col.remove(idx_col)
        data_col_list = list(num_type_col['name'])
        null_index_dict = {}
        for name in num_type_col['name']:
            li = list(req_df[name][req_df[name].isnull()].index)
            if (len(li) != 0):
                null_index_dict[name] = li
        origin_df = req_df.copy()
        origin_df.index = req_df[idx_col]
        ema_df = req_df.copy()
        ema_df.index = req_df[idx_col]

        # denoise
        com = 50
        for name in data_col_list:
            ema = pd.DataFrame(origin_df[name]).ewm(com).mean()
            ema_df[name + '_denoised'] = ema[name]
        denoised_column_list = []
        for e in data_col_list:
            denoised_column_list.append(e + '_denoised')
        cor_dict = {}
        for i in range(0, len(denoised_column_list)):
            cor_dict[denoised_column_list[i]] = {}
        for i in range(0, len(denoised_column_list)):
            for j in range(i + 1, len(denoised_column_list)):
                cor = pearsonr(ema_df[denoised_column_list[i]], ema_df[denoised_column_list[j]])[0]
                cor_dict[denoised_column_list[i]][denoised_column_list[j]] = cor
                cor_dict[denoised_column_list[j]][denoised_column_list[i]] = cor
        res_df = req_df.copy()
        for curr_col in null_index_dict:
            lim = 0.3
            ar = 100
            curr_col_denoised = curr_col + '_denoised'
            curr_dict = cor_dict[curr_col_denoised]
            train_col = []
            train_col.append(curr_col_denoised)
            for key in curr_dict:
                if (abs(curr_dict[key]) > lim):
                    train_col.append(key)
            train_df = ema_df[train_col]
            forecasting_model = VAR(train_df)
            results = forecasting_model.fit(ar)
            for idx in null_index_dict[curr_col]:
                y = train_df.values[idx - ar:idx]
                test = ema_df.iloc[idx:idx+1,:]
                forecast_res = pd.DataFrame(results.forecast(y=y, steps=1), index=test.index, columns=[train_col])
                res_df.loc[idx, curr_col] = round(float(forecast_res[curr_col_denoised].iloc[0]),2)
        na_df = res_df.iloc[list(na_df.index)]
        save = df_to_response(req['column'], res_df)
        run = df_to_response(req['column'], na_df)
        res = {}
        res['save'] = save
        res['run'] = run
        return res
    elif(method==1):
        interpolate_df = req_df.fillna(req_df.interpolate())
        na_df = interpolate_df.iloc[list(na_df.index)]
        save = df_to_response(req['column'], interpolate_df)
        run = df_to_response(req['column'], na_df)
        res = {}
        res['save'] = save
        res['run'] = run
        return res

    else:
        save = df_to_response(req['column'], req_df.dropna(axis=0))
        run = df_to_response(req['column'], na_df.dropna(axis=0))
        res = {}
        res['save'] = save
        res['run'] = run
        return res

def df_to_response(column, df):
    type_dict = {}
    for col in column:
        type_dict[col['name']] = col['type']
    data = list()
    for i in df.index:
        obj = {}
        for col in column:
            if (df.loc[i, col['name']] is None):
                obj[col['name']] = None
            elif (type_dict[col['name']] == "INT"):
                obj[col['name']] = int(df.loc[i, col['name']])
            elif (type_dict[col['name']] == "DOUBLE"):
                obj[col['name']] = float(df.loc[i, col['name']])
            else:
                obj[col['name']] = df.loc[i, col['name']]
        data.append(obj)

    res = {}
    res['column'] = column
    res['data'] = data
    return res

@app.post("/cleaning-api/pearson-correlation")
async def pearson_correlation(request: Request):
    req = await request.json()
    req_df = pd.json_normalize(req['data'])
    req_df = req_df.dropna(axis=0)
    column_df = pd.json_normalize(req['column'])
    num_type_col = column_df.where(column_df['type'] == 'DOUBLE').dropna()
    data_col_list = list(num_type_col['name'])
    cor_dict = {}
    for i in range(0, len(data_col_list)):
        cor_dict[data_col_list[i]] = {}
    for i in range(0, len(data_col_list)):
        for j in range(i + 1, len(data_col_list)):
            cor = round(pearsonr(req_df[data_col_list[i]], req_df[data_col_list[j]])[0],3)
            cor_dict[data_col_list[i]][data_col_list[j]] = cor
            cor_dict[data_col_list[j]][data_col_list[i]] = cor
    return cor_dict