from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import numpy as np
import time
from typing import Literal

app = FastAPI(title="Monte Carlo Stock API", version="4.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

TRADING_DAYS = 252
RISK_FREE_RATE = 0.05
CACHE_TTL_SECONDS = 600  # 10 minutes — price/history doesn't need refetching every click
EWMA_LAMBDA = 0.94       # RiskMetrics standard decay factor

POPULAR = {
    "apple":"AAPL",
    "microsoft":"MSFT",
    "google":"GOOGL",
    "alphabet":"GOOGL",
    "amazon":"AMZN",
    "nvidia":"NVDA",
    "meta":"META",
    "tesla":"TSLA",
    "reliance":"RELIANCE.NS",
    "tcs":"TCS.NS",
    "infosys":"INFY.NS",
    "hdfc":"HDFCBANK.NS",
    "icici":"ICICIBANK.NS",
    "sbi":"SBIN.NS",
    "wipro":"WIPRO.NS",
    "itc":"ITC.NS"
}

_cache:dict={}

def resolve_symbol(q:str)->str:
    q=q.strip()
    if "." in q or q.isupper():
        return q.upper()
    return POPULAR.get(q.lower(), q.upper())

def get_history_and_info(symbol:str):
    """Fetch (and cache) 2y history + info for a symbol, so repeated requests
    for the same ticker within CACHE_TTL_SECONDS don't re-hit Yahoo Finance."""
    now=time.time()
    entry=_cache.get(symbol)
    if entry and (now-entry["time"])<CACHE_TTL_SECONDS:
        return entry["hist"], entry["info"]

    ticker=yf.Ticker(symbol)
    hist=ticker.history(period="2y", auto_adjust=True)
    try:
        info=ticker.info or {}
    except Exception:
        info={}

    if not hist.empty and len(hist)>=100:
        _cache[symbol]={"time":now,"hist":hist,"info":info}

    return hist, info

def ewma_daily_vol(returns:np.ndarray, lam:float=EWMA_LAMBDA)->float:
    """Exponentially-weighted volatility: recent days count more than old
    ones, so the estimate reacts to a recent calm/turbulent stretch instead
    of blending 2 years into one flat number."""
    var=float(np.var(returns))  # seed with the plain sample variance
    for r in returns:
        var=lam*var+(1-lam)*r*r
    return float(np.sqrt(var))

@app.get("/")
def home():
    return {"status":"ok"}

@app.get("/api/search")
def search(q:str=Query(..., min_length=1)):
    ql=q.lower()
    results=[]
    for name,ticker in POPULAR.items():
        if ql in name:
            results.append({"name":name.title(),"ticker":ticker})
    return results[:10]

@app.get("/api/stock/{query}")
def stock(query:str,
          investment:float=100000,
          horizon:int=252,
          simulations:int=5000,
          model:Literal["gbm","bootstrap"]="gbm"):
    symbol=resolve_symbol(query)
    horizon=max(1,min(horizon,756))
    simulations=max(100,min(simulations,100000))

    hist,info=get_history_and_info(symbol)
    if hist.empty or len(hist)<100:
        raise HTTPException(status_code=404, detail="No historical data found.")

    closes=hist["Close"].dropna()
    log_returns=np.log(closes/closes.shift(1)).dropna()
    returns_arr=log_returns.values

    daily_mu=float(log_returns.mean())
    daily_sigma_flat=float(log_returns.std())
    daily_sigma_ewma=ewma_daily_vol(returns_arr)

    mu=daily_mu*TRADING_DAYS
    sigma_flat=daily_sigma_flat*np.sqrt(TRADING_DAYS)
    sigma=daily_sigma_ewma*np.sqrt(TRADING_DAYS)  # EWMA vol is the "headline" volatility now

    s0=float(closes.iloc[-1])
    dt=1/TRADING_DAYS

    # ---- generate daily log-return shocks for every path ----
    if model=="bootstrap":
        # Resample actual historical daily log returns (with replacement).
        # This preserves the real distribution's fat tails and skew instead
        # of assuming returns are normally distributed.
        shocks=np.random.choice(returns_arr, size=(horizon,simulations), replace=True)
    else:
        # Classic GBM with EWMA volatility, drawn via antithetic variates —
        # for every random draw z we also use -z, which cancels first-order
        # sampling noise and gives a cleaner mean for the same sim count.
        half=simulations//2
        z_half=np.random.normal(size=(horizon,half))
        z=np.concatenate([z_half,-z_half],axis=1)
        if simulations%2==1:
            z=np.concatenate([z,np.random.normal(size=(horizon,1))],axis=1)
        drift=(daily_mu-0.5*daily_sigma_ewma**2)*dt*TRADING_DAYS
        vol=daily_sigma_ewma*np.sqrt(dt*TRADING_DAYS)
        shocks=drift+vol*z

    paths=np.zeros((horizon+1,simulations))
    paths[0]=s0
    for t in range(1,horizon+1):
        paths[t]=paths[t-1]*np.exp(shocks[t-1])

    final_prices=paths[-1]
    portfolio=investment*(final_prices/s0)

    var95=float(np.percentile(portfolio,5)-investment)
    var99=float(np.percentile(portfolio,1)-investment)
    cvar95=float(portfolio[portfolio<=np.percentile(portfolio,5)].mean()-investment)

    sharpe=(mu-RISK_FREE_RATE)/sigma if sigma else 0

    # ---- confidence interval on the *mean* portfolio estimate ----
    portfolio_se=float(portfolio.std(ddof=1)/np.sqrt(simulations))
    portfolio_mean=float(portfolio.mean())

    # ---- max drawdown distribution ----
    # For each simulated path, the worst peak-to-trough % decline it ever
    # experiences — not just where it ends up. A path that dips 40% at
    # month 4 and fully recovers by month 12 looks identical to one that
    # never dipped if you only look at the final price; this doesn't miss it.
    running_max=np.maximum.accumulate(paths,axis=0)
    drawdown_pct=(running_max-paths)/running_max*100
    max_dd_per_path=drawdown_pct.max(axis=0)
    dd_hist_counts,dd_bins=np.histogram(max_dd_per_path,bins=25)

    drawdown={
        "percentiles":{
            "p50":float(np.percentile(max_dd_per_path,50)),
            "p75":float(np.percentile(max_dd_per_path,75)),
            "p90":float(np.percentile(max_dd_per_path,90)),
            "p95":float(np.percentile(max_dd_per_path,95)),
            "worst":float(max_dd_per_path.max())
        },
        "histogram":{
            "bins":dd_bins[:-1].tolist(),
            "counts":dd_hist_counts.tolist()
        }
    }

    percentiles={
        "p5":float(np.percentile(final_prices,5)),
        "p25":float(np.percentile(final_prices,25)),
        "p50":float(np.percentile(final_prices,50)),
        "p75":float(np.percentile(final_prices,75)),
        "p95":float(np.percentile(final_prices,95))
    }

    fan=[]
    for i in range(paths.shape[0]):
        fan.append({
            "day":i,
            "p5":float(np.percentile(paths[i],5)),
            "p25":float(np.percentile(paths[i],25)),
            "p50":float(np.percentile(paths[i],50)),
            "p75":float(np.percentile(paths[i],75)),
            "p95":float(np.percentile(paths[i],95))
        })

    sample_paths=paths[:,:min(100,simulations)].T.tolist()

    hist_counts, bins=np.histogram(final_prices,bins=30)

    return {
        "ticker":symbol,
        "company":info.get("longName",symbol),
        "sector":info.get("sector","N/A"),
        "industry":info.get("industry","N/A"),
        "market_cap":info.get("marketCap"),
        "currency":info.get("currency"),
        "current_price":s0,
        "open":info.get("open"),
        "previous_close":info.get("previousClose"),
        "day_high":info.get("dayHigh"),
        "day_low":info.get("dayLow"),
        "volume":info.get("volume"),
        "model":model,
        "expected_return":mu,
        "volatility":sigma,
        "volatility_historical":sigma_flat,
        "sharpe_ratio":sharpe,
        "investment":investment,
        "expected_portfolio":portfolio_mean,
        "portfolio_std_error":portfolio_se,
        "portfolio_ci95_low":portfolio_mean-1.96*portfolio_se,
        "portfolio_ci95_high":portfolio_mean+1.96*portfolio_se,
        "best_case":float(portfolio.max()),
        "worst_case":float(portfolio.min()),
        "probability_profit":float((portfolio>investment).mean()*100),
        "var95":var95,
        "var99":var99,
        "cvar95":cvar95,
        "drawdown":drawdown,
        "fan_chart":fan,
        "sample_paths":sample_paths,
        "histogram":{
            "bins":bins[:-1].tolist(),
            "counts":hist_counts.tolist()
        },
        "percentiles":percentiles
    }
