import gradio as gr
import pandas as pd
import pickle
import numpy as np
with open('stock_model.pkl','rb') as file :
    model = pickle.load(file)


def predict(date,Stock_1_today,Stock_1_yesterday,Stock_1_two_days_ago,Stock_1_mae5,Stock_2_yesterday,Stock_3_yesterday,Stock_4_yesterday,Stock_5_yesterday):
    date = pd.to_datetime(date)
    month = date.month
    day = date.day
    day_of_week = date.dayofweek
    day_of_year = date.dayofyear
    portfolio_mean = np.mean([Stock_1_today,Stock_2_yesterday,Stock_3_yesterday,Stock_4_yesterday,Stock_5_yesterday])
    portfolio_std = np.std([Stock_1_today,Stock_2_yesterday,Stock_3_yesterday,Stock_4_yesterday,Stock_5_yesterday])
    features = pd.DataFrame([[
        month,
        day,
        day_of_week,
        day_of_year,
        Stock_1_yesterday,
        Stock_1_two_days_ago,
        Stock_1_mae5,
        Stock_2_yesterday,
        Stock_3_yesterday,
        Stock_4_yesterday,
        Stock_5_yesterday,
        portfolio_mean,
        portfolio_std
    ]],
    columns=['month','day','day_of_week','day_of_year','Stock_1_lag_1','Stock_1_lag_2','Stock_1_mae5','Stock_2_lag_1','Stock_3_lag_1','Stock_4_lag_1','Stock_5_lag_1'
        ,'portfolio_mean','portfolio_std'
    
    
    ])

    pred = model.predict(features)[0]
    change = pred - Stock_1_today
    direction = 'increased' if change >0 else 'decreased'
    return f'tomorrow the price will be {direction}'

app = gr.Interface(
    fn = predict,
    inputs=[
        gr.Textbox(label='todays date',placeholder='YYYY-MM-DD',value='2025-05-23'),
        gr.Number(label='Stock_1 current price'),
        gr.Number(label='Stock_1 yesterday price'),
        gr.Number(label='Stock_1 two days ago price'),
        gr.Number(label='Stock_1 last 5 days mean'),
        gr.Number(label='Stock_2 yesterday price'),
        gr.Number(label='Stock_3 yesterday price'),
        gr.Number(label='Stock_4 yesterday price'),
        gr.Number(label='Stock_5 yesterday price'),
    ],
    outputs=gr.Markdown(),
    title='stock price prediction',


)
print('starting gradio app')
app.launch(share=True,inbrowser=True)