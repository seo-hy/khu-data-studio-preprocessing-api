from fastapi import FastAPI, Request, HTTPException
import pandas as pd
import numpy as np
# import py_eureka_client.eureka_client as eureka_client
from fastapi.middleware.cors import CORSMiddleware
from statsmodels.tsa.api import VAR

from scipy.stats import pearsonr


app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# eureka_client.init(eureka_server="http://localhost:8761/eureka",
#                                 app_name="preprocessing-api",
#                                 instance_port=8083)

@app.get("/preprocessing-api/test")
async def test():
    return {"message":"test"}

@app.post("/preprocessing-api/missing-value/find")
async def find_mv(request: Request):
    req = await request.json()
    date_time_column = req['dateTimeColumn']
    req_df = pd.json_normalize(req['data'])
    if len(req_df) == 0:
        res = {}
        res['dateTimeColumn'] = date_time_column
        res['column'] = req['column']
        res['data'] = list()
        return res
    new_col = list()
    new_col.append(date_time_column)
    for i in range(1, len(req_df.columns)):
        new_col.append(req_df.columns[i].replace("value.", ""))
    req_df.columns = new_col
    na_df = req_df[req_df.isna().any(axis=1)]
    na_df = na_df.replace({np.nan:None})
    return to_response(date_time_column,req['column'], na_df)


@app.post("/preprocessing-api/missing-value/delete")
async def delete_missing_value(request: Request):
    req = await request.json()
    date_time_column = req['dateTimeColumn']
    req_df = pd.json_normalize(req['data'])
    if len(req_df) == 0:
        raise HTTPException(status_code=400, detail="empty data")
    new_col = list()
    new_col.append(date_time_column)
    for i in range(1, len(req_df.columns)):
        new_col.append(req_df.columns[i].replace("value.", ""))
    req_df.columns = new_col
    na_df = req_df[req_df.isna().any(axis=1)]
    res = {}
    res['deleteDate'] = list(na_df['created_at'])
    res['run'] = to_response(date_time_column, req['column'], na_df.dropna(axis=0))
    return res

@app.post("/preprocessing-api/missing-value/interpolate")
async def interpolate(request: Request):
    req = await request.json()
    date_time_column = req['dateTimeColumn']
    req_df = pd.json_normalize(req['data'])
    if len(req_df) == 0:
        raise HTTPException(status_code=400, detail="empty data")
    new_col = list()
    new_col.append(date_time_column)
    for i in range(1, len(req_df.columns)):
        new_col.append(req_df.columns[i].replace("value.", ""))
    req_df.columns = new_col

    na_df = req_df[req_df.isna().any(axis=1)]
    interpolate_df = req_df.fillna(req_df.interpolate())
    interpolate_df = interpolate_df.fillna(method="ffill")
    interpolate_df = interpolate_df.fillna(method="bfill")
    interpolate_df = interpolate_df.round(3)
    na_df = interpolate_df.iloc[list(na_df.index)]
    return to_response(date_time_column,req['column'], na_df)

@app.post("/preprocessing-api/missing-value/predict")
async def predict(request: Request):
    req = await request.json()
    date_time_column = req['dateTimeColumn']
    req_df = pd.json_normalize(req['data'])
    if len(req_df) == 0:
        raise HTTPException(status_code=400, detail="empty data")
    new_col = list()
    new_col.append(date_time_column)
    for i in range(1, len(req_df.columns)):
        new_col.append(req_df.columns[i].replace("value.", ""))
    req_df.columns = new_col

    na_df = req_df[req_df.isna().any(axis=1)]
    column_df = pd.json_normalize(req['column'])
    column_list = list(column_df['name'])

    null_index_dict = {}
    for name in column_list:
        li = list(req_df[name][req_df[name].isnull()].index)
        if (len(li) != 0):
            null_index_dict[name] = li

    origin_df = req_df.copy()
    origin_df.index = req_df[date_time_column]
    ema_df = req_df.copy()
    ema_df.index = req_df[date_time_column]
    # denoise
    com = 100
    for name in column_list:
        ema = pd.DataFrame(origin_df[name]).ewm(com).mean()
        ema_df[name + '_denoised'] = ema[name]
    denoised_column_list = []
    for e in column_list:
        denoised_column_list.append(e + '_denoised')
    ema_df = ema_df.fillna(method="ffill")
    ema_df = ema_df.fillna(method="bfill")
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
        if(len(train_df) < ar):
            raise HTTPException(status_code=400, detail="The length of data must be at least 100.")
        try:
            forecasting_model = VAR(train_df)
            results = forecasting_model.fit(ar)
        except:
            raise HTTPException(status_code=400, detail="data contains one or more constant columns.")

        for idx in null_index_dict[curr_col]:
            y = train_df.values[idx - ar:idx]
            if idx-ar < 0:
                res_df.loc[idx, curr_col] = round(ema_df.loc[res_df.loc[idx, date_time_column], curr_col_denoised],3)
            else:
                test = ema_df.iloc[idx:idx + 1, :]
                forecast_res = pd.DataFrame(results.forecast(y=y, steps=1), index=test.index, columns=[train_col])
                res_df.loc[idx, curr_col] = round(float(forecast_res[curr_col_denoised].iloc[0]), 3)

    res_df = res_df.iloc[list(na_df.index)]
    return to_response(date_time_column,req['column'], res_df)


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


