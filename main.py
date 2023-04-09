from fastapi import FastAPI, Request, HTTPException

import py_eureka_client.eureka_client as eureka_client

import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from api import missing_value

app = FastAPI()

app.include_router(missing_value.router)

eureka_client.init(eureka_server="http://localhost:8761/eureka",
                                app_name="preprocessing-api",
                                instance_port=8083)




@app.post("/preprocessing-api/pearson-correlation")
async def pearson_correlation(request: Request):
    req = await request.json()
    req_df = pd.json_normalize(req['data'])
    new_col = list()
    for c in req_df.columns:
        new_col.append(c.replace("value.", ""))
    req_df.columns = new_col
    req_df = req_df.fillna(req_df.interpolate())
    req_df = req_df.fillna(method="ffill")
    req_df = req_df.fillna(method="bfill")
    column_df = pd.json_normalize(req['column'])
    column_list = list(column_df['name'])
    cor_dict = {}
    for i in range(0, len(column_list)):
        cor_dict[column_list[i]] = {}
    for i in range(0, len(column_list)):
        for j in range(i + 1, len(column_list)):
            cor = round(pearsonr(req_df[column_list[i]], req_df[column_list[j]])[0],3)
            cor_dict[column_list[i]][column_list[j]] = cor
            cor_dict[column_list[j]][column_list[i]] = cor
    return cor_dict

@app.post("/preprocessing-api/std")
async def std(request: Request):
    req = await request.json()
    req_df = pd.json_normalize(req['data'])
    new_col = list()
    for c in req_df.columns:
        new_col.append(c.replace("value.", ""))
    req_df.columns = new_col
    column_df = pd.json_normalize(req['column'])
    column_list = list(column_df['name'])
    std_dict = {}
    std = req_df.std()
    for i in range(0, len(column_list)):
        if (column_list[i] not in std.index):
            std[column_list[i]] = 0.0
        std_dict[column_list[i]] = round(std[column_list[i]],3)
    return std_dict

@app.post("/preprocessing-api/mean")
async def mean(request: Request):
    req = await request.json()
    req_df = pd.json_normalize(req['data'])
    new_col = list()
    for c in req_df.columns:
        new_col.append(c.replace("value.", ""))
    req_df.columns = new_col
    column_df = pd.json_normalize(req['column'])
    column_list = list(column_df['name'])
    mean_dict = {}
    mean = req_df.mean()
    for i in range(0, len(column_list)):
        if (column_list[i] not in mean.index):
            mean[column_list[i]] = 0.0
        mean_dict[column_list[i]] = round(mean[column_list[i]],3)
    return mean_dict


@app.post("/preprocessing-api/visualize")
async def visualize(request: Request):
    req = await request.json()
    date_time_column = req['dateTimeColumn']
    req_df = pd.json_normalize(req['data'])
    req_df = req_df.replace({np.nan: None})
    column_df = pd.json_normalize(req['column'])
    column_list = list(column_df['name'])
    req_dict = {}
    if len(req_df.columns) == 0:
        req_dict[date_time_column] = list()
        for i in range(0, len(column_list)):
            req_dict[column_list[i]] = list()
    else:
        new_col = list()
        new_col.append(date_time_column)
        for i in range(1, len(req_df.columns)):
            new_col.append(req_df.columns[i].replace("value.", ""))
        req_df.columns = new_col
        req_dict[date_time_column] = list(req_df[date_time_column])
        for i in range(0, len(column_list)):
            req_dict[column_list[i]] = list(req_df[column_list[i]])
    return req_dict


@app.post("/preprocessing-api/denoise")
async def delete_missing_value(request: Request, com: int, datasetId: int):
    req = await request.json()
    date_time_column = req['dateTimeColumn']
    if len(req['denoiseColumn']) == 0:
        raise HTTPException(status_code=400, detail="column required")
    denoise_col_df = pd.json_normalize(req['denoiseColumn'])
    req_df = pd.json_normalize(req['data'])
    if len(req_df) == 0:
        raise HTTPException(status_code=400, detail="empty data")
    new_col = list()
    new_col.append(date_time_column)
    for i in range(1, len(req_df.columns)):
        new_col.append(req_df.columns[i].replace("value.", ""))
    denoise_col_list = list(denoise_col_df['name'])
    origin_df = req_df.copy()
    req_df.columns = new_col
    req_df = req_df.fillna(method="ffill")
    req_df = req_df.fillna(method="bfill")
    ema_df = pd.DataFrame()
    ema_df[date_time_column] = req_df[date_time_column]
    for name in denoise_col_list:
        ema = pd.DataFrame(req_df[name]).ewm(com).mean()
        ema_df[name] = ema[name].round(3)
    column_df = pd.json_normalize(req['column'])
    column_list = list(column_df['name'])
    for name in column_list:
        if name not in denoise_col_list:
            ema_df[name] = origin_df["value."+name]

    ema_df = ema_df.replace({np.nan: None})

    res = to_response(date_time_column, req['column'], ema_df)
    res['denoiseColumn'] = req['denoiseColumn']
    return res


def to_response(date_time_column, column, df):
    data = list()
    for i in df.index:
        obj = {}
        value = {}
        for col in column:
            if (df.loc[i, col['name']] is None):
                value[col['name']] = None
            else:
                value[col['name']] = float(df.loc[i, col['name']])
        obj['date'] = df.loc[i, date_time_column]
        obj['value'] = value
        data.append(obj)
    res = {}
    res['dateTimeColumn'] = date_time_column
    res['column'] = column
    res['data'] = data
    return res