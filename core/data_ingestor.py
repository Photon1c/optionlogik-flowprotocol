from data_loader import load_stock_data, load_option_chain_data, get_most_recent_option_date

def ingest_data(ticker: str, date: str = None, stock_base="F:/inputs/stocks", option_base="F:/inputs/options/log"):
    stock_df = load_stock_data(ticker, base_dir=stock_base)
    option_df = load_option_chain_data(ticker, date=date, base_dir=option_base)
    latest_date = date or get_most_recent_option_date(ticker, base_dir=option_base)

    return {
        "ticker": ticker.upper(),
        "stock_data": stock_df,
        "option_data": option_df,
        "date_used": latest_date
    }
