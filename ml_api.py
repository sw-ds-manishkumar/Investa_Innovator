from flask import Flask, request, jsonify
from prophet import Prophet
from sklearn.model_selection import train_test_split
import pandas as pd
import yfinance as yf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import datetime, timedelta
import json

stock_symbols = ['TCS.NS', 'RELIANCE.NS', 'HDFCBANK.NS']

# Fetch historical stock data for multiple symbols
def fetch_stock_data(symbols):
    data = []
    for stock_symbol in stock_symbols:
        stock = yf.Ticker(stock_symbol)
        stock_details = stock.info

        # Get the 52-week low and high
        #fiftyTwoWeekLow = stock_details["fiftyTwoWeekLow"]
        #fiftyTwoWeekHigh = stock_details["fiftyTwoWeekHigh"]

        # Get the average price for the past 6 months
        start_date = (datetime.today() - timedelta(days=180)).strftime('%Y-%m-%d')
        end_date = datetime.today().strftime('%Y-%m-%d')
        price_data = stock.history(start=start_date, end=end_date)
        average_price = price_data["Close"].mean()

        # Create a DataFrame for the stock details
        stock_df = pd.DataFrame({
            "Symbol": stock_symbol,
            "fiftyTwoWeekLow": stock_details["fiftyTwoWeekLow"],
            "fiftyTwoWeekHigh": stock_details["fiftyTwoWeekHigh"],
            "marketCap":stock_details["marketCap"],
            'longName':stock_details["longName"],
            "Open": price_data["Open"],
            "High": price_data["High"],
            "Low": price_data["Low"],
            "Close": price_data["Close"],
            "Volume": price_data["Volume"],
            "uuid":stock_details["uuid"],
            "currency":stock_details["currency"],
            "exchange":stock_details["exchange"],
            "quoteType":stock_details["quoteType"],
            "shortName":stock_details["shortName"],
            "firstTradeDateEpochUtc":stock_details["firstTradeDateEpochUtc"],
            "timeZoneFullName":stock_details["timeZoneFullName"],
            "messageBoardId":stock_details["messageBoardId"],
            "financialCurrency":stock_details["financialCurrency"],
            "currentPrice":stock_details["currentPrice"],
        })

        # Append the stock DataFrame to the data list
        data.append(stock_df)

    # Concatenate all stock DataFrames into a single DataFrame
    return pd.concat(data)

# Train Prophet model for each stock
def train_prophet_models(data):
    models = {}
    train_data_symbol = []
    test_data_symbol = []
    for symbol in data['Symbol'].unique():
        data_ = data[data['Symbol'] == symbol]
        train_data_ = data_[['Date', "Close"]]
        train_data_.rename(columns={'Date': 'ds', 'Close': 'y'}, inplace=True)
        train_data, test_data = train_test_split(train_data_, test_size=0.1, random_state=0, shuffle=False)
        train_data['Symbol'] = symbol
        train_data_symbol.append(train_data)
        test_data['Symbol'] = symbol
        test_data_symbol.append(test_data)
        model = Prophet()
        model_fit = model.fit(train_data)
        models[symbol] = model_fit
    return models, pd.concat(train_data_symbol), pd.concat(test_data_symbol)



##############
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Get the user information from the request
    user_info = request.get_json()

    # Extract the username and age from user_info
    income = user_info.get('income')
    duration = user_info.get('duration')
    
    #model
    # Example usage
    #stock_symbols = ['TCS.NS', 'RELIANCE.NS', 'HDFCBANK.NS']
    #start_date = "2019-01-01"
    #end_date = "2022-12-31"
    forecast_steps = 10
 
    # Fetch historical stock data for multiple symbols
    data = fetch_stock_data(stock_symbols)
    data =data.reset_index()
    data['Date'] = data['Date'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
 
 
    #fetch last 10 rows for each symbol
    #current_date_details_ = data.groupby('Symbol').tail(10)
    current_date_details_ = data.groupby('Symbol').tail(1)
 
    models, train_data, test_data = train_prophet_models(data)
 
 
 
    #test on date range-----------
 
    #forecast with start date and end date
    # Forecast future values
    start_date = pd.Timestamp(train_data.iloc[-1]['ds'])
    #end_date = start_date + pd.DateOffset(days=100)
 
    #date_range = pd.date_range(start=start_date, end=end_date)
 
    date_range = pd.date_range(start=start_date, periods=forecast_steps)
 
    # Make predictions for the specified date range
    forecasts_ = {}
    for symbol, model in models.items():
        future = pd.DataFrame({'ds': date_range})
        forecast = model.predict(future)
        #forecasts[symbol] = forecast.tail(forecast_steps)[['ds', 'yhat']]
        forecasts_[symbol] = forecast[['ds', 'yhat']]
 
    # Plot the forecast
    #fig = model.plot(forecast)
 
    # Access forecasted values
    forecast_values= forecast[['ds', 'yhat']]
 
 
 
    # Convert the dictionary to a dataframe--------
    df = pd.concat([pd.DataFrame(rows) for symbol_keys, rows in forecasts_.items()], keys=forecasts_.keys())
 
    # Given multi-index dataframe
    multi_index_df = pd.DataFrame(df
    , index=pd.MultiIndex.from_tuples(df.index, names=['Symbol', 'Index_']))
 
    # Convert the multi-index dataframe to a simple dataframe
    df = multi_index_df.reset_index()
    df.drop("Index_", axis=1,inplace=True)
 
 
 
 
    ####---dynamic-------
 
    df_ = df.copy()
 
    #stock_symbols = ['TCS.NS', 'RELIANCE.NS', 'HDFCBANK.NS']
 
    details = current_date_details_.copy()
 
    cols = ['fiftyTwoWeekLow', 'fiftyTwoWeekHigh', 'marketCap',
           'longName', 'Volume', 'uuid',
           'currency', 'exchange', 'quoteType', 'shortName',
           'firstTradeDateEpochUtc', 'timeZoneFullName', 'messageBoardId',
           'financialCurrency', 'currentPrice']
 
    concatenated_dfs = []
 
    for symbol in stock_symbols:
        df_symbol = df_[df_['Symbol'] == symbol]
        df_details_symbol = details[details['Symbol'] == symbol]
        df_details = df_details_symbol[cols]
        
        repeated_df = pd.concat([df_details] * len(df_symbol), ignore_index=True)
        df_con = pd.concat([df_symbol.reset_index(drop=True), repeated_df], axis=1)
        
        concatenated_dfs.append(df_con)
 
    # Concatenate all the dataframes together
    final_df = pd.concat(concatenated_dfs, ignore_index=True)
 
    # Print the final dataframe
    final_df = final_df.rename(columns={"ds": "Date", "yhat": "predictedClose"})
 
    final_df['Date'] = final_df['Date'].dt.strftime('%Y-%m-%d')
    print(final_df.head())
 
    # Convert DataFrame to dictionary
    data_dict = final_df.to_dict(orient='records')
 
    # Convert dictionary to JSON
    json_data = json.dumps(data_dict)
    print("xxxxxxxxxxxxxx", income, duration)
    print("yyyyyyyyyy", json_data)
    return json_data

if __name__ == '__main__':

    app.run(host="192.168.4.126", port=8001, debug=False)


# =============================================================================
# 
# ##############
# app = Flask(__name__)
# 
# @app.route('/predict', methods=['POST'])
# def predict():
#     # Get the user information from the request
#     user_info = request.get_json()
# 
#     # Extract the username and age from user_info
#     username = user_info.get('username')
#     age = user_info.get('age')
# 
#     # Perform machine learning processing on user information
#     # Replace this with your actual machine learning model code
#     # ...
# 
#     # Create a dummy response for demonstration purposes
#     prediction = 'some_prediction'
#     confidence = 0.85
# 
#     response = {
#         'prediction': prediction,
#         'confidence': confidence,
#         "username":username
#     }
#     print("ffffffffff", username)
#     return jsonify(response)
# 
# if __name__ == '__main__':
# 
#     app.run(host="192.168.4.126", port=8001, debug=False)
# 
# =============================================================================
