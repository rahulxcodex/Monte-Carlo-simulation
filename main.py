from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import numpy as np
import pandas as pd
import requests
import io
import time
import threading
from collections import Counter
from typing import Literal

app = FastAPI(title="Monte Carlo Stock API", version="5.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

TRADING_DAYS = 252
RISK_FREE_RATE = 0.05
CACHE_TTL_SECONDS = 600        # 10 minutes — per-ticker price/history cache
TICKER_CACHE_TTL_SECONDS = 24*3600   # 24 hours — full searchable ticker list cache
EWMA_LAMBDA = 0.94              # RiskMetrics standard decay factor

# Merton jump-diffusion defaults (fixed, documented assumptions — not fitted
# per-ticker; a real quant desk would calibrate these from options data or
# historical jump detection, which is out of scope for a free hobby tool)
JUMP_LAMBDA_PER_YEAR = 3.0   # avg number of "jump" events per year
JUMP_MEAN = -0.01            # average jump size (log-return), slight crash bias
JUMP_STD = 0.05              # jump size volatility

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
    "itc":"ITC.NS",
    "nifty 50":"^NSEI",
    "nifty50":"^NSEI",
    "sensex":"^BSESN",
    "sensex 30":"^BSESN",
    "bitcoin":"BTC-USD",
    "ethereum":"ETH-USD",
}

INDEX_ALIASES = {
    "nifty50":"^NSEI","nifty 50":"^NSEI","nifty":"^NSEI","niftyfifty":"^NSEI",
    "sensex30":"^BSESN","sensex 30":"^BSESN","sensex":"^BSESN","bsesensex":"^BSESN"
}
SYMBOL_DISPLAY_NAMES = {"^NSEI":"Nifty 50","^BSESN":"Sensex 30"}

_price_cache:dict={}

_ticker_cache={"data":[], "nifty50":[], "last_built":0.0}
_ticker_cache_lock=threading.Lock()

def resolve_symbol(q:str)->str:
    q=q.strip()
    ql=q.lower()
    if ql in INDEX_ALIASES:
        return INDEX_ALIASES[ql]
    if "." in q or q.isupper():
        return q.upper()
    return POPULAR.get(ql, q.upper())

def get_history_and_info(symbol:str):
    """Fetch (and cache) 2y history + info for a symbol, so repeated requests
    for the same ticker within CACHE_TTL_SECONDS don't re-hit Yahoo Finance."""
    now=time.time()
    entry=_price_cache.get(symbol)
    if entry and (now-entry["time"])<CACHE_TTL_SECONDS:
        return entry["hist"], entry["info"]

    ticker=yf.Ticker(symbol)
    hist=ticker.history(period="2y", auto_adjust=True)
    try:
        info=ticker.info or {}
    except Exception:
        info={}

    if not hist.empty and len(hist)>=100:
        _price_cache[symbol]={"time":now,"hist":hist,"info":info}

    return hist, info

def ewma_daily_vol(returns:np.ndarray, lam:float=EWMA_LAMBDA)->float:
    """Exponentially-weighted volatility: recent days count more than old
    ones, so the estimate reacts to a recent calm/turbulent stretch instead
    of blending 2 years into one flat number."""
    var=float(np.var(returns))
    for r in returns:
        var=lam*var+(1-lam)*r*r
    return float(np.sqrt(var))

# ---------------------------------------------------------------------------
# Ticker cache: NSE full equity list + official Nifty 50 constituents +
# top crypto by market cap, all from free/official sources, refreshed daily.
# Fetches are best-effort — if any source fails, the others still populate
# the cache, and a previous successful cache is kept rather than wiped.
# ---------------------------------------------------------------------------

_NSE_HEADERS={
    "User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language":"en-US,en;q=0.9",
    "Referer":"https://www.nseindia.com/market-data/live-equity-market"
}

def _fetch_nse_csv(path:str)->str:
    with requests.Session() as s:
        s.headers.update(_NSE_HEADERS)
        s.get("https://www.nseindia.com", timeout=8)  # warm-up: sets cookies NSE requires
        r=s.get(f"https://nsearchives.nseindia.com/content/{path}", timeout=15)
        r.raise_for_status()
        return r.text

def _build_nse_equity_list()->list:
    text=_fetch_nse_csv("equities/EQUITY_L.csv")
    df=pd.read_csv(io.StringIO(text))
    out=[]
    for _,row in df.iterrows():
        sym=str(row.get("SYMBOL","")).strip()
        name=str(row.get("NAME OF COMPANY","")).strip()
        if not sym or not name:
            continue
        out.append({"name":name,"ticker":f"{sym}.NS","exchange":"NSE","type":"stock"})
    return out

def _build_nifty50_list()->list:
    text=_fetch_nse_csv("indices/ind_nifty50list.csv")
    df=pd.read_csv(io.StringIO(text))
    out=[]
    for _,row in df.iterrows():
        sym=str(row.get("Symbol","")).strip()
        name=str(row.get("Company Name","")).strip()
        industry=str(row.get("Industry","Other")).strip()
        if not sym or not name:
            continue
        out.append({"name":name,"ticker":f"{sym}.NS","industry":industry,"exchange":"NSE","type":"index_constituent"})
    return out

def _build_crypto_list(limit:int=250)->list:
    r=requests.get(
        "https://api.coingecko.com/api/v3/coins/markets",
        params={"vs_currency":"usd","order":"market_cap_desc","per_page":limit,"page":1},
        headers={"User-Agent":"Mozilla/5.0"},
        timeout=15
    )
    r.raise_for_status()
    out=[]
    for c in r.json():
        sym=str(c.get("symbol","")).upper().strip()
        name=c.get("name") or sym
        if not sym:
            continue
        out.append({"name":name,"ticker":f"{sym}-USD","exchange":"CRYPTO","type":"crypto"})
    return out

def refresh_ticker_cache(force:bool=False):
    now=time.time()
    if not force and _ticker_cache["data"] and (now-_ticker_cache["last_built"])<TICKER_CACHE_TTL_SECONDS:
        return
    if not _ticker_cache_lock.acquire(blocking=False):
        return  # a refresh is already running elsewhere
    try:
        now=time.time()
        if not force and _ticker_cache["data"] and (now-_ticker_cache["last_built"])<TICKER_CACHE_TTL_SECONDS:
            return

        combined=[]
        nifty50=[]

        try:
            combined.extend(_build_nse_equity_list())
        except Exception as e:
            print(f"[ticker cache] NSE equity list failed: {e}")

        try:
            nifty50=_build_nifty50_list()
            combined.extend(nifty50)
        except Exception as e:
            print(f"[ticker cache] Nifty 50 list failed: {e}")

        try:
            combined.extend(_build_crypto_list())
        except Exception as e:
            print(f"[ticker cache] Crypto list failed: {e}")

        # curated fallbacks + index aliases always included, so search/quick
        # picks work even if every live source above failed
        for name,ticker in POPULAR.items():
            exch="NSE" if ticker.endswith(".NS") else ("CRYPTO" if ticker.endswith("-USD") else "US")
            combined.append({"name":name.title(),"ticker":ticker,"exchange":exch,"type":"index" if ticker.startswith("^") else "stock"})

        if not combined:
            return  # every source failed — keep whatever cache we already had

        seen=set()
        deduped=[]
        for e in combined:
            if e["ticker"] in seen:
                continue
            seen.add(e["ticker"])
            deduped.append(e)

        _ticker_cache["data"]=deduped
        if nifty50:
            _ticker_cache["nifty50"]=nifty50
        _ticker_cache["last_built"]=now
        print(f"[ticker cache] refreshed: {len(deduped)} tickers, {len(_ticker_cache['nifty50'])} Nifty50 constituents")
    finally:
        _ticker_cache_lock.release()

def kick_off_refresh():
    threading.Thread(target=refresh_ticker_cache, daemon=True).start()

@app.on_event("startup")
def _on_startup():
    kick_off_refresh()

@app.get("/")
def home():
    return {"status":"ok","tickers_cached":len(_ticker_cache["data"])}

@app.get("/api/search")
def search(q:str=Query(..., min_length=1)):
    kick_off_refresh()  # no-op if already fresh; otherwise refreshes in the background
    ql=q.lower()
    source=_ticker_cache["data"] or [
        {"name":n.title(),"ticker":t} for n,t in POPULAR.items()
    ]
    results=[]
    for e in source:
        if ql in e["name"].lower() or ql in e["ticker"].lower():
            results.append({"name":e["name"],"ticker":e["ticker"]})
        if len(results)>=15:
            break
    return results

@app.get("/api/index/nifty50-breakdown")
def nifty50_breakdown():
    kick_off_refresh()
    constituents=_ticker_cache["nifty50"]
    if not constituents:
        raise HTTPException(status_code=503, detail="Nifty 50 constituent list isn't cached yet — try again in a few seconds.")
    industry_counts=Counter(c.get("industry","Other") for c in constituents)
    return {
        "count":len(constituents),
        "constituents":constituents,
        "by_industry":[{"industry":k,"count":v} for k,v in sorted(industry_counts.items(), key=lambda x:-x[1])]
    }

@app.get("/api/stock/{query}")
def stock(query:str,
          investment:float=100000,
          horizon:int=252,
          simulations:int=5000,
          model:Literal["gbm","bootstrap","jump"]="gbm"):
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
    sigma=daily_sigma_ewma*np.sqrt(TRADING_DAYS)

    s0=float(closes.iloc[-1])
    dt=1/TRADING_DAYS

    if model=="bootstrap":
        # Resample actual historical daily log returns (with replacement).
        # Formula: r_t drawn from the empirical distribution of past returns
        # instead of an assumed parametric distribution — preserves real
        # fat tails and skew.
        shocks=np.random.choice(returns_arr, size=(horizon,simulations), replace=True)

    elif model=="jump":
        # Merton jump-diffusion: continuous GBM diffusion plus a Poisson-
        # timed jump term.
        # dS/S = (mu - lambda*k)dt + sigma*dW + dJ,  J ~ compound Poisson
        # Discretized per day:
        # S(t+1) = S(t) * exp[(mu - 0.5*sigma^2 - lambda*k)*dt + sigma*sqrt(dt)*Z + N*Y]
        # where N ~ Bernoulli(lambda*dt) (jump indicator this day),
        # Y ~ Normal(jump_mean, jump_std) (jump size when N=1), and
        # k = E[e^Y - 1] (compensator, keeps drift centered on mu).
        half=simulations//2
        z_half=np.random.normal(size=(horizon,half))
        z=np.concatenate([z_half,-z_half],axis=1)
        if simulations%2==1:
            z=np.concatenate([z,np.random.normal(size=(horizon,1))],axis=1)

        diffusion_drift=(daily_mu-0.5*daily_sigma_ewma**2)*dt*TRADING_DAYS
        diffusion_vol=daily_sigma_ewma*np.sqrt(dt*TRADING_DAYS)

        p_jump=JUMP_LAMBDA_PER_YEAR*dt
        jump_occurs=np.random.random(size=(horizon,simulations))<p_jump
        jump_size=np.random.normal(JUMP_MEAN,JUMP_STD,size=(horizon,simulations))*jump_occurs
        k=np.exp(JUMP_MEAN+0.5*JUMP_STD**2)-1
        compensator=-JUMP_LAMBDA_PER_YEAR*k*dt

        shocks=diffusion_drift+compensator+diffusion_vol*z+jump_size

    else:
        # Classic GBM with EWMA volatility, drawn via antithetic variates.
        # Formula: S(t+1) = S(t) * exp[(mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z],  Z ~ N(0,1)
        # For every Z we also simulate -Z (antithetic pairing), which
        # cancels first-order sampling noise for a cleaner mean estimate.
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

    portfolio_se=float(portfolio.std(ddof=1)/np.sqrt(simulations))
    portfolio_mean=float(portfolio.mean())

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
        "histogram":{"bins":dd_bins[:-1].tolist(),"counts":dd_hist_counts.tolist()}
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

    company_name=info.get("longName") or info.get("shortName") or SYMBOL_DISPLAY_NAMES.get(symbol, symbol)

    return {
        "ticker":symbol,
        "company":company_name,
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
        "histogram":{"bins":bins[:-1].tolist(),"counts":hist_counts.tolist()},
        "percentiles":percentiles
    }
