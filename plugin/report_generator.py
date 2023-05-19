from flask import Flask, request, jsonify
import yfinance as yf
import matplotlib.pyplot as plt
import io
from tabulate import tabulate
import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
from datetime import timedelta
from tabulate import tabulate

app = Flask(__name__)

def get_financial_data(api_key, function, ticker):
    base_url = "https://www.alphavantage.co/query"
    params = {
        "function": function,
        "symbol": ticker,
        "apikey": api_key,
    }
    response = requests.get(base_url, params=params)
    return response.json()

def filter_last_five_years(data):
    if 'annualReports' not in data:
        return {}

    current_year = datetime.now().year
    five_years_ago = current_year - 5
    filtered_data = [report for report in data['annualReports'] if int(report['fiscalDateEnding'].split("-")[0]) >= five_years_ago]
    return filtered_data

# Example usage

api_key = '6G6DT6CCRO8UWZ39'



def display_valuation(income_statement_data, balance_sheet_data, cash_flow_data):
# Reverse the data
    income_statement_data = income_statement_data.iloc[::-1]
    balance_sheet_data = balance_sheet_data.iloc[::-1]
    cash_flow_data = cash_flow_data.iloc[::-1]

    headers = ["Metric"] + [date.split("-")[0] for date in income_statement_data['fiscalDateEnding']] + [f"Forecast {i+1}" for i in range(2)]

    # Define the metrics
    metrics = ["Total Revenue", "Gross Profit", "Operating Income", "Net Income"]
    balance_sheet_metrics = ["Total Assets", "Total Liabilities", "Current Assets", "Non-Current Assets", "Current Liabilities", "Non-Current Liabilities"]
    cash_flow_metrics = ["Operating Cash Flow", "Cash Flow from Financing", "Cash Flow from Investment", "Free Cash Flow"]

    # Mapping dictionary for accessing values from the reports
    metric_to_key = {"Total Revenue": "totalRevenue", "Gross Profit": "grossProfit", "Operating Income": "operatingIncome", "Net Income": "netIncome"}
    balance_sheet_metric_to_key = {"Total Assets": "totalAssets", "Total Liabilities": "totalLiabilities", "Current Assets": "totalCurrentAssets", "Non-Current Assets": "totalNonCurrentAssets", "Current Liabilities": "totalCurrentLiabilities", "Non-Current Liabilities": "totalNonCurrentLiabilities"}
    cash_flow_metric_to_key = {"Operating Cash Flow": "operatingCashflow", "Cash Flow from Financing": "cashflowFromFinancing", "Cash Flow from Investment": "cashflowFromInvestment"}

    # Initialize past_data, balance_sheet_past_data, and cash_flow_past_data
    past_data = {metric: [] for metric in metrics}
    balance_sheet_past_data = {metric: [] for metric in balance_sheet_metrics}
    cash_flow_past_data = {metric: [] for metric in cash_flow_metrics}

    for index, report in income_statement_data.iterrows():
        for metric in metrics:
            past_data[metric].append(float(report[metric_to_key[metric]]))

    for index, report in balance_sheet_data.iterrows():
        for metric in balance_sheet_metrics:
            key = balance_sheet_metric_to_key[metric]
            value = float(report[key]) if key in report else 0
            balance_sheet_past_data[metric].append(value)

    for index, report in cash_flow_data.iterrows():
        for metric in cash_flow_metrics[:-1]:  # Exclude "Free Cash Flow"
            cash_flow_past_data[metric].append(float(report[cash_flow_metric_to_key[metric]]))


    # Calculate Free Cash Flow
    cash_flow_past_data["Free Cash Flow"] = [cash_flow_past_data["Operating Cash Flow"][i] + cash_flow_past_data["Cash Flow from Financing"][i] + cash_flow_past_data["Cash Flow from Investment"][i] for i in range(len(cash_flow_past_data["Operating Cash Flow"]))]

    # Calculate forecasted future valuations using CAGR
    for metric in metrics:
        num_years = len(past_data[metric]) - 1
        start_value = past_data[metric][0] if past_data[metric] else 0
        end_value = past_data[metric][-1] if metric in past_data and past_data[metric] else None
        if start_value == 0:
            cagr = 0
        else:
            cagr = (end_value / start_value) ** (1 / num_years) - 1

        forecasted_values = [end_value * (1 + cagr) ** (i + 1) for i in range(2)] if end_value is not None else [None] * 2
        past_data[metric].extend(forecasted_values)

    # Calculate forecasted future valuations using CAGR for the balance sheet
    for metric in balance_sheet_metrics:
        num_years = len(balance_sheet_past_data[metric]) - 1
        start_value = balance_sheet_past_data[metric][0] if balance_sheet_past_data[metric] else 0
        end_value = balance_sheet_past_data[metric][-1] if balance_sheet_past_data[metric] else None
        if start_value == 0:
            cagr = 0
        else:
            cagr = (end_value / start_value) ** (1 / num_years) - 1

        forecasted_values = [end_value * (1 + cagr) ** (i + 1) for i in range(2)]if end_value is not None else [None] * 2
        balance_sheet_past_data[metric].extend(forecasted_values)

    # Calculate forecasted future valuations using CAGR for the cash flow statement
    for metric in cash_flow_metrics:
        num_years = len(cash_flow_past_data[metric]) - 1
        start_value = cash_flow_past_data[metric][0] if cash_flow_past_data[metric] else 0
        end_value = cash_flow_past_data[metric][-1] if cash_flow_past_data[metric] else None
        if start_value == 0:
            cagr = 0
        else:
            cagr = (end_value / start_value) ** (1 / num_years) - 1

        forecasted_values = [end_value * (1 + cagr) ** (i + 1) for i in range(2)]if end_value is not None else [None] * 2
        cash_flow_past_data[metric].extend(forecasted_values)

    # Transpose the income statement data
    transposed_data = [[metric] + values for metric, values in past_data.items()]
    income_statement_df = pd.DataFrame(transposed_data, columns=headers)
    st.table(income_statement_df)

    # Transpose the balance sheet data
    transposed_balance_sheet_data = [[metric] + values for metric, values in balance_sheet_past_data.items()]
    balance_sheet_df = pd.DataFrame(transposed_balance_sheet_data, columns=headers)
    st.table(balance_sheet_df)

    # Transpose the cash flow statement data
    transposed_cash_flow_data = [[metric] + values for metric, values in cash_flow_past_data.items()]
    cash_flow_df = pd.DataFrame(transposed_cash_flow_data, columns=headers)
    st.table(cash_flow_df)
    print(income_statement_data)
    # Convert the dataframes to a string representation
    income_statement_str = income_statement_df.to_string(index=False)
    balance_sheet_str = balance_sheet_df.to_string(index=False)
    cash_flow_str = cash_flow_df.to_string(index=False)
    # Convert the dataframes to a dictionary representation
    income_statement_dict = income_statement_df.to_dict(orient='list')
    balance_sheet_dict = balance_sheet_df.to_dict(orient='list')
    cash_flow_dict = cash_flow_df.to_dict(orient='list')

    # Return a dictionary containing the three tables
    return {
        'income_statement': income_statement_dict,
        'balance_sheet': balance_sheet_dict,
        'cash_flow': cash_flow_dict,
    }

def generate_table_response(table_data, title):
    # Generate a response for each row in the table
    rows = [f"{metric}: {values}" for metric, values in table_data.items()]

    # Combine the rows and return
    return f"{title}:\n" + '\n'.join(rows)

import yfinance as yf
import pandas as pd


def get_price_change(ticker):
    # Get the data for the stock
    stock = yf.Ticker(ticker)
    stock_data = stock.history(period='1y')
    # Get the data for the S&P 500
    sp500 = yf.Ticker('^GSPC')
    sp500_data = sp500.history(period='1y')

    # Calculate the price change
    stock_data['Cumulative Price Change'] = stock_data['Close'].pct_change().cumsum()
    sp500_data['Cumulative Price Change'] = sp500_data['Close'].pct_change().cumsum()

    # Merge the two datasets
    data = pd.DataFrame()
    data[ticker] = stock_data['Cumulative Price Change']
    data['S&P 500'] = sp500_data['Cumulative Price Change']

    return data


# Company Overview
def company_overview(ticker,google_mosaic_messages,yahoo_mosaic_messages):
    company_overview_messages = [
    {"role": "system", "content": "You are a hedge fund investment analyst."},
    {"role": "user", "content": f"Provide an company overview for {ticker}, using the following data {google_mosaic_messages}{yahoo_mosaic_messages}, write only company overview, do not write other things other than that"},
    ]   

    return company_overview_messages,model

# Industry Analysis
def industry_analysis(ticker,google_mosaic_messages,yahoo_mosaic_messages):
    industry_analysis_messages = [
    {"role": "system", "content": "You are a hedge fund investment analyst."},
    {"role": "user", "content": f"Provide an industry analysis for {ticker}, using the following data{google_mosaic_messages}{yahoo_mosaic_messages},write only industry analysis, do not write other things other than that"},

    ]

    return industry_analysis_messages

# Valuation

def calculate_financial_ratios(income_statement_data, balance_sheet_data):
    def to_float(value):
        return 0.0 if value is None or value == 'None' else float(value)

    current_assets = to_float(balance_sheet_data.iloc[-1]['totalCurrentAssets'])
    current_liabilities = to_float(balance_sheet_data.iloc[-1]['totalCurrentLiabilities'])

    inventory = to_float(balance_sheet_data.iloc[-1]['inventory'])
    quick_ratio = (current_assets - inventory) / current_liabilities if current_liabilities != 0 else None

    long_term_debt = to_float(balance_sheet_data.iloc[-1]['longTermDebt'])
    short_term_debt = to_float(balance_sheet_data.iloc[-1]['shortTermDebt'])
    total_debt = long_term_debt + short_term_debt

    total_assets = to_float(balance_sheet_data.iloc[-1]['totalAssets'])
    debt_ratio = total_debt / total_assets if total_assets != 0 else None

    ebit = to_float(income_statement_data.iloc[-1]['operatingIncome'])
    interest_expense = to_float(income_statement_data.iloc[-1]['interestExpense'])
    interest_coverage_ratio = ebit / interest_expense if interest_expense != 0 else None

    return {
        "Quick Ratio": quick_ratio,
        "Debt Ratio": debt_ratio,
        "Interest Coverage Ratio": interest_coverage_ratio,
    }


# Financial Analysis
def financial_analysis(ticker, income_statement_data, balance_sheet_data,  calculate_financial_ratios):
    financial_ratios = calculate_financial_ratios( income_statement_data, balance_sheet_data)
    
    financial_analysis_messages = [
        {"role": "system", "content": "You are a hedge fund investment analyst."},
        {"role": "user", "content": f"Perform a financial analysis on {ticker} financial ratios using data from the calculated financial ratios: {financial_ratios}, write only financial analysis, do not write other things other than that"},
    ]

    return financial_analysis_messages


# Investment Thesis
def investment_thesis(ticker,google_mosaic_messages,yahoo_mosaic_messages):
    investment_thesis_messages = [
    {"role": "system", "content": "You are a hedge fund investment analyst."},
    {"role": "user", "content": f"Provide an investment thesis for {ticker}, using the following data{google_mosaic_messages}{yahoo_mosaic_messages},write only investment thesis, do not write other things other than that"},    
    ]

    return    investment_thesis_messages


# Risk Analysis
def risk_analysis(ticker,google_mosaic_messages,yahoo_mosaic_messages):
    risk_analysis_messages = [
    {"role": "system", "content": "You are a hedge fund investment analyst."},
    {"role": "user", "content": f"Perform a risk analysis on {ticker}, using the following data {google_mosaic_messages}{yahoo_mosaic_messages},write only risk analysis, do not write other things other than that"},
    
    ]
    return    risk_analysis_messages

# SWOT Analysis    
def SWOT_analysis(ticker,google_mosaic_messages,yahoo_mosaic_messages):
    SWOT_analysis_messages = [
    {"role": "system", "content": "You are a hedge fund investment analyst."},
    {"role": "user", "content": f"Perform a SWOT analysis on {ticker}, using the following data {google_mosaic_messages}{yahoo_mosaic_messages}, write only SWOT analysis, do not write other things other than that"},
    
    ]
    return     SWOT_analysis_messages

# Investment Recommendations
def investment_recommendations_messages(ticker,google_mosaic_messages,yahoo_mosaic_messages):
    investment_recommendations_messages = [
    {"role": "system", "content": "You are a hedge fund investment analyst."},
    {"role": "user", "content": f"Provide investment recommendations for {ticker}, using the following data {google_mosaic_messages}{yahoo_mosaic_messages},write only Investment Recommendations, do not write other things other than that"},
    ]
    return    investment_recommendations_messages

def scrape_google_news(keyword, num_articles=9):
    url = f'https://news.google.com/rss/search?q={keyword}&hl=en-US&gl=US&ceid=US:en'
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'xml')
    
    news_data = []
    for item in soup.find_all('item')[:num_articles]:
        news_item = {
            'title': item.title.text,
            'link': item.link.text,
            'source': item.source.text
        }
        news_data.append(news_item)
    
    return news_data

def scrape_yahoo_finance_news(keyword, num_articles=9):
    url = f'https://finance.yahoo.com/quote/{keyword}/news?p={keyword}'
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    news_data = []
    for item in soup.find_all('li', class_='js-stream-content')[:num_articles]:
        news_item = {
            'title': item.find('h3').text,
            'link': item.find('a')['href'],
            'source': 'Yahoo Finance'
        }
        news_data.append(news_item)
    
    return news_data


def apply_mosaic_theory_yahoo_finance(keyword,model):
    news_data = scrape_yahoo_finance_news(keyword)
    yahoo_mosaic_messages=[
        {"role": "system", "content": "You are a hedge fund investment analyst."},
        {"role": "user", "content": f"Perform a mosaic analysis on {keyword} using the following news data: {news_data}"},
    ]

    return yahoo_mosaic_messages


def apply_mosaic_theory_google(keyword):
    news_data = scrape_google_news(keyword)
    google_mosaic_messages=[
        {"role": "system", "content": "You are a hedge fund investment analyst."},
        {"role": "user", "content": f"Perform a mosaic analysis on {keyword} using the following news data: {news_data}"},
    ]

    return google_mosaic_messages

@app.route('/api', methods=['POST'])
def generate_report():
    # Extract the ticker symbol from the request data
    data = request.get_json()
    ticker_symbol = data['ticker_symbol']

    # Get the data for the stock and the S&P 500
    stock = yf.Ticker(ticker_symbol)
    sp500 = yf.Ticker('^GSPC')

    # Calculate the cumulative price change for the past year
    stock_data = stock.history(period='1y')['Close'].pct_change().cumsum()
    sp500_data = sp500.history(period='1y')['Close'].pct_change().cumsum()

    # Plot the price change
    plt.figure(figsize=(10, 6))
    plt.plot(stock_data, label=ticker_symbol)
    plt.plot(sp500_data, label='S&P 500')
    plt.legend()
    plt.title('Price Change Over the Past Year')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Price Change')

    # Save the plot to a BytesIO object
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    company_overview(ticker_symbol, google_mosaic_messages, yahoo_mosaic_messages)
    industry_analysis(ticker_symbol, google_mosaic_messages, yahoo_mosaic_messages)
    income_statement_data, balance_sheet_data, cash_flow_data = get_financial_data(ticker_symbol)
    display_valuation(income_statement_data, balance_sheet_data, cash_flow_data)
    financial_analysis(ticker, income_statement_data, balance_sheet_data,  calculate_financial_ratios)
   
    investment_thesis(ticker,google_mosaic_messages,yahoo_mosaic_messages)

    risk_analysis(ticker,google_mosaic_messages,yahoo_mosaic_messages)
 
    SWOT_analysis(ticker,google_mosaic_messages,yahoo_mosaic_messages)

    investment_recommendations_messages(ticker,google_mosaic_messages,yahoo_mosaic_messages)
    # Encode the image in base64 and return it in the response
    image = base64.b64encode(buf.read()).decode('utf-8')
    return jsonify({'image': image})

if __name__ == '__main__':
    app.run(port=8000)
