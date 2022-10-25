from fastapi import FastAPI, Request
import pandas as pd
import json
import numpy as np
import py_eureka_client.eureka_client as eureka_client

app = FastAPI()

eureka_client.init(eureka_server="http://localhost:8761/eureka",
                                app_name="cleaning-api",
                                instance_port=8082)

@app.get("/cleaning-api")
async def root():
    return {"message": "test"}
