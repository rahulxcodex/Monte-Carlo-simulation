from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import numpy as np

# Initialize the FastAPI app
app = FastAPI(title="Stock Data API")

# Add CORS Middleware to allow your HTML frontend to make requests to this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, replace "*" with your HTML site's URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/stock/{ticker}")
def get_stock_data(ticker: str):
    """
    Fetches 1 year of historical data for a given ticker,
    and returns current price, expected return, and volatility.
    """
    try:
        # Fetch data using yfinance
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1y")
        
        if hist.empty or len(hist) < 50:
            raise HTTPException(status_code=404, detail="Not enough historical data found for this ticker.")
        
        # Extract closing prices and current price
        closes = hist['Close'].dropna()
        current_price = float(closes.iloc[-1])
        
        # Calculate daily log returns: ln(Price_t / Price_t-1)
        log_returns = np.log(closes / closes.shift(1)).dropna()
        
        # Calculate annualized expected return and volatility (assuming 252 trading days)
        mean_return = float(log_returns.mean() * 252)
        volatility = float(log_returns.std() * np.sqrt(252))
        
        # Return structured JSON
        return {
            "ticker": ticker,
            "current_price": current_price,
            "expected_return": mean_return,
            "volatility": volatility
        }
    except Exception as e:
        # Catch yfinance errors and return them cleanly
        raise HTTPException(status_code=500, detail=f"Server error processing {ticker}: {str(e)}")
