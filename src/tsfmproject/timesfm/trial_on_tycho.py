import timesfm
import pandas as pd
import matplotlib.pyplot as plt


if __name__ == "__main__":
    tfm = timesfm.TimesFm(hparams=timesfm.TimesFmHparams(
                            backend="gpu",
                            per_core_batch_size=32,
                            horizon_len=32,
                        ),
                        checkpoint=timesfm.TimesFmCheckpoint(
                            huggingface_repo_id="google/timesfm-1.0-200m"),)
    # # get the data
    # df = pd.read_csv("/nethome/sli999/data/Tycho/dengue_laos.csv")
    # df = df[df['SourceName'] == 'Laos Dengue Surveillance System']
    # # preprocess the data
    # df = df[['Admin1ISO', 'PeriodStartDate', 'CountValue']]
    # df.columns = ['unique_id', 'ds', 'y']
    
    # df['ds'] = pd.to_datetime(df['ds'])

    df = pd.read_csv("/nethome/sli999/data/Tycho/timesfm_US_covid_pivot.csv")
    df['ds'] = pd.to_datetime(df['ds'])
    # sort by unique_id and ds
    # df = df.sort_values(by=['unique_id', 'ds'])

    # iteratively using 2 months to forecast 1 month
    # from 2020-01-03 to 2021-07-31
    # data used for forecasting: 2020-01-03 to 2021-05-31
    # evaluation: 2020-03-01 to 2021-07-31
    forecasts = df
    evaluations = df

    RMSEs = []
    SMAPEs = []

    for i in range(0, 17):
        # Forecasting
        # 2 month data, starting from the ith month, to forecast the i+3 month
        if i < 7:
            forecast = forecasts[(forecasts['ds'] >= '2020-0' + str(i+1) + '-01') & 
                                 (forecasts['ds'] < '2020-0' + str(i+3) + '-01')]
            if i == 6:
                evaluation = evaluations[(evaluations['ds'] >= '2020-0' + str(i+3) + '-01') & 
                                    (evaluations['ds'] < '2020-' + str(i+4) + '-01')]
            else:
                evaluation = evaluations[(evaluations['ds'] >= '2020-0' + str(i+3) + '-01') & 
                                     (evaluations['ds'] < '2020-' + str(i+4) + '-01')]
        elif i < 9:
            forecast = forecasts[(forecasts['ds'] >= '2020-0' + str(i+1) + '-01') & 
                                 (forecasts['ds'] < '2020-' + str(i+3) + '-01')]
            evaluation = evaluations[(evaluations['ds'] >= '2020-' + str(i+3) + '-01') &
                                (evaluations['ds'] < '2020-' + str(i+4) + '-01')]
        elif i == 9:
            forecast = forecasts[(forecasts['ds'] >= '2020-' + str(i+1) + '-01') & 
                                 (forecasts['ds'] < '2020-' + str(i+3) + '-01')]
            evaluation = evaluations[(evaluations['ds'] >= '2020-' + str(i+3) + '-01') &
                                (evaluations['ds'] < '2021-' + str(i-8) + '-01')]
        elif i < 12:
            forecast = forecasts[(forecasts['ds'] >= '2020-' + str(i+1) + '-01') & 
                                (forecasts['ds'] < '2021-0' + str(i-9) + '-01')]
            evaluation = evaluations[(evaluations['ds'] >= '2021-0' + str(i-9) + '-01') &
                                (evaluations['ds'] < '2021-0' + str(i-8) + '-01')]
        else:
            forecast = forecasts[(forecasts['ds'] >= '2021-0' + str(i-11) + '-01') & 
                                (forecasts['ds'] < '2021-0' + str(i-9) + '-01')]
            evaluation = evaluations[(evaluations['ds'] >= '2021-0' + str(i-9) + '-01') &
                                (evaluations['ds'] < '2021-0' + str(i-8) + '-01')]
        
        forecast = forecast.melt(id_vars=['ds'], var_name='unique_id', value_name='y')
        evaluation = evaluation.melt(id_vars=['ds'], var_name='unique_id', value_name='y')
        # print(len(forecast))
        # print(len(evaluation))
        forecast_df = tfm.forecast_on_df(
            inputs=forecast,
            freq="D",  # daily frequency
            value_name="y",
            num_jobs=1,
        )
        forecast_df = forecast_df[['ds', 'unique_id', 'timesfm']]
        forecast_df.columns = ['ds', 'unique_id', 'y']

        # let float prediction to closest integer, remain float type
        # Also, set negative prediction to 0
        forecast_df['y'] = forecast_df['y'].apply(lambda x: round(x))
        forecast_df['y'] = forecast_df['y'].apply(lambda x: 0 if x < 0 else x)

        forecast_df_pivot = forecast_df.pivot(index='ds', columns='unique_id', values='y')
        evaluation_pivot = evaluation.pivot(index='ds', columns='unique_id', values='y')
        forecast_df_pivot = forecast_df_pivot[:len(evaluation_pivot)]
        

        # calculate the RMSE and MAPE, for 1, 2, 3 weeks and 1 month
        rmse1 = ((forecast_df_pivot[:7]-evaluation_pivot[:7])**2).mean().mean()**0.5
        rmse2 = ((forecast_df_pivot[:14]-evaluation_pivot[:14])**2).mean().mean()**0.5
        rmse3 = ((forecast_df_pivot[:21]-evaluation_pivot[:21])**2).mean().mean()**0.5
        rmse4 = ((forecast_df_pivot-evaluation_pivot)**2).mean().mean()**0.5
        smape1 = (abs(forecast_df_pivot[:7]-evaluation_pivot[:7])/((abs(forecast_df_pivot[:7])+abs(evaluation_pivot[:7]))/2 + 1e-7)).mean().mean()
        smape2 = (abs(forecast_df_pivot[:14]-evaluation_pivot[:14])/((abs(forecast_df_pivot[:14])+abs(evaluation_pivot[:14]))/2 + 1e-7)).mean().mean()
        smape3 = (abs(forecast_df_pivot[:21]-evaluation_pivot[:21])/((abs(forecast_df_pivot[:21])+abs(evaluation_pivot[:21]))/2 + 1e-7)).mean().mean()
        smape4 = (abs(forecast_df_pivot-evaluation_pivot)/((abs(forecast_df_pivot)+abs(evaluation_pivot))/2 + 1e-7)).mean().mean()

        

        RMSEs.append([rmse1, rmse2, rmse3, rmse4])
        SMAPEs.append([smape1, smape2, smape3, smape4])
    

    # print(forecast_df)
    # print(evaluation)

    # draw the RMSE and MAPE
    months = ['20-01', '20-02', '20-03', '20-04', '20-05', '20-06',
             '20-07', '20-08', '20-09', '20-10', '20-11', '20-12',
             '21-01', '21-02', '21-03', '21-04', '21-05']
    for i in range(4):
        plt.figure(figsize=(10, 8))
        plt.plot(months, [rmse[i] for rmse in RMSEs], label='RMSE')
        plt.xlabel("Month")
        plt.ylabel("RMSE")
        plt.xticks(rotation=45)
        plt.legend()
        plt.title("RMSE for " + str(i+1) + "week")
        plt.savefig("plot_RMSE" + str(i+1) + ".png")
        plt.figure(figsize=(10, 8))
        plt.plot(months, [smape[i] for smape in SMAPEs], label='SMAPE')
        plt.xlabel("Month")
        plt.ylabel("SMAPE")
        plt.xticks(rotation=45)
        plt.legend()
        plt.title("SMAPE for " + str(i+1) + "week")
        plt.savefig("plot_SMAPE" + str(i+1) + ".png")
    

    
    

