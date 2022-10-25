from fastapi import FastAPI, Request, Body
import pandas as pd
import json
import numpy as np
import py_eureka_client.eureka_client as eureka_client

app = FastAPI()

eureka_client.init(eureka_server="http://localhost:8761/eureka",
                                app_name="cleaning-api",
                                instance_port=8082)

@app.post("/cleaning-api/missing-value/find")
async def test(request: Request):
    req = await request.json()
    req_df = pd.json_normalize(req['data'])
    na_df = req_df[req_df.isna().any(axis=1)]
    na_df = na_df.replace({np.nan:None})

    type_dict = {}
    for col in req['column']:
        type_dict[col['name']] = col['type']
    data = list()
    for i in na_df.index:
        obj = {}
        for col in req['column']:
            if(na_df.loc[i, col['name']] is None):
                obj[col['name']] = None
            elif(type_dict[col['name']] == "INT"):
                obj[col['name']] = int(na_df.loc[i, col['name']])
            elif (type_dict[col['name']] == "DOUBLE"):
                obj[col['name']] = float(na_df.loc[i, col['name']])
            else:
                obj[col['name']] = na_df.loc[i, col['name']]
        data.append(obj)

    res = {}
    res['column'] = req['column']
    res['data'] = data
    return res


@app.post("/cleaning-api/missing-value/process")
async def test(request: Request, method: int):
    req = await request.json()
    req_df = pd.json_normalize(req['data'])
    na_df = req_df[req_df.isna().any(axis=1)]
    na_df = na_df.replace({np.nan: None})
    print(method)
    if(method == 0):
        pass
    elif(method==1):
        interpolate_df = req_df.fillna(req_df.interpolate())
        na_df = interpolate_df.iloc[list(na_df.index)]
    else:
        na_df = na_df.dropna(axis=0)

    type_dict = {}
    for col in req['column']:
        type_dict[col['name']] = col['type']
    data = list()
    for i in na_df.index:
        obj = {}
        for col in req['column']:
            if (na_df.loc[i, col['name']] is None):
                obj[col['name']] = None
            elif (type_dict[col['name']] == "INT"):
                obj[col['name']] = int(na_df.loc[i, col['name']])
            elif (type_dict[col['name']] == "DOUBLE"):
                obj[col['name']] = float(na_df.loc[i, col['name']])
            else:
                obj[col['name']] = na_df.loc[i, col['name']]
        data.append(obj)

    res = {}
    res['column'] = req['column']
    res['data'] = data
    return res
