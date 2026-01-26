import streamlit as st
import pandas as pd
import numpy as np
import requests
from io import BytesIO
from zipfile import ZipFile
import statsmodels.api as sm
from sklearn.covariance import LedoitWolf
from scipy.optimize import minimize
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
import hashlib
import json
import io
import base64

# Nota: docx potrebbe non essere installato in tutti gli ambienti standard
try:
    from docx import Document
except ImportError:
    pass

# ---------------------------------------------------------
# CONFIGURAZIONE PAGINA
# ---------------------------------------------------------
st.set_page_config(
    layout="wide", 
    page_title="Institutional Portfolio System | Pro",
    page_icon="🏛️",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# 🔒 PASSWORD PROTECTION SYSTEM
# ---------------------------------------------------------
def check_password():
    """Ritorna True se l'utente ha inserito la password corretta."""
    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False

    if st.session_state["password_correct"]:
        return True

    st.markdown("""
    <style>
        .stTextInput > div > div > input { text-align: center; }
        .block-container { padding-top: 5rem; }
    </style>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<h2 style='text-align: center; color: #111827;'>🔒 Accesso Istituzionale</h2>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #6b7280;'>Inserire le credenziali per accedere al modello.</p>", unsafe_allow_html=True)
        
        pwd = st.text_input("Password", type="password", key="password_input", label_visibility="collapsed")
        
        if pwd:
            try:
                correct_pwd = st.secrets["PASSWORD"]
            except (FileNotFoundError, KeyError):
                correct_pwd = "admin" 

            if pwd == correct_pwd:
                st.session_state["password_correct"] = True
                st.rerun()
            else:
                st.error("⛔ Credenziali non valide")
                
    return False

if not check_password():
    st.stop()

# ---------------------------------------------------------
# CSS AVANZATO (LIGHT MODE - ALTO CONTRASTO)
# ---------------------------------------------------------
st.markdown("""
<style>
    .stApp { background-color: #f0f2f6; color: #000000; font-family: 'Roboto', 'Helvetica Neue', sans-serif; }
    [data-testid="stSidebar"] { background-color: #ffffff; border-right: 1px solid #d1d5db; }
    [data-testid="stSidebar"] * { color: #000000 !important; }
    h1, h2, h3 { color: #111827 !important; font-weight: 700; letter-spacing: -0.5px; }
    h1 { border-bottom: 2px solid #000000; padding-bottom: 15px; margin-bottom: 25px; font-size: 2.5rem; }
    div[data-testid="metric-container"] { background-color: #ffffff; border: 1px solid #e5e7eb; padding: 15px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }
    .stButton > button { background-color: #2563eb; color: #ffffff; border: none; border-radius: 4px; font-weight: 600; text-transform: uppercase; transition: all 0.3s; }
    .stButton > button:hover { background-color: #1d4ed8; color: #ffffff; }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; background-color: transparent; }
    .stTabs [data-baseweb="tab"] { background-color: #e5e7eb; border-radius: 4px 4px 0 0; color: #4b5563; border: 1px solid #d1d5db; font-weight: 600; }
    .stTabs [aria-selected="true"] { background-color: #ffffff; color: #2563eb !important; border: 1px solid #d1d5db; border-bottom: 2px solid #2563eb; }
    .stAlert { background-color: #eff6ff; border: 1px solid #bfdbfe; color: #1e3a8a; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 0. LOGICA ASSET CLASS & REGIME DETECTION (INTEGRATA)
# ---------------------------------------------------------

def auto_classify_assets_initial(asset_names):
    """Classificazione euristica iniziale (First Guess)."""
    groups = {"BONDS": [], "EQUITY": [], "COMMODITIES": []}
    bond_keywords = ['BOND', 'GOV', 'CORP', 'HY', 'T-BILL', 'TREASURY', 'OBL', 'YIELD', 'AGG', 'MON', 'FI', 'BTP', 'BUND']
    comm_keywords = ['COMM', 'GOLD', 'OIL', 'CMD', 'METAL', 'ENERGY', 'AGRI', 'GLD', 'SLV', 'ETC']
    
    for asset in asset_names:
        u_asset = asset.upper()
        if any(k in u_asset for k in comm_keywords):
            groups["COMMODITIES"].append(asset)
        elif any(k in u_asset for k in bond_keywords):
            groups["BONDS"].append(asset)
        else:
            groups["EQUITY"].append(asset)
    return groups

def detect_market_regime(returns, equity_assets, lookback_trend=6, lookback_vol_short=3, lookback_vol_long=24):
    """Rileva il regime di mercato basandosi su Trend e Turbolenza degli asset Equity."""
    if not equity_assets:
        return "NEUTRAL", 0, 0
    
    if len(returns) < lookback_vol_long:
        return "NEUTRAL", 0, 0

    # Indice sintetico equi-pesato degli asset equity
    eq_returns = returns[equity_assets].mean(axis=1)
    eq_index = (1 + eq_returns).cumprod()
    
    # 1. Calcolo Trend (SMA vs Prezzo attuale)
    sma_long = eq_index.rolling(window=12).mean().iloc[-1]
    sma_short = eq_index.rolling(window=3).mean().iloc[-1]
    trend_score = (sma_short / sma_long) - 1 
    
    # 2. Calcolo Turbolenza (Volatilità recente vs Storica)
    vol_short = eq_returns.rolling(window=lookback_vol_short).std().iloc[-1]
    vol_long = eq_returns.rolling(window=lookback_vol_long).std().iloc[-1]
    turbulence_score = vol_short / vol_long if vol_long > 0 else 1.0
    
    # 3. Mapping dei Regimi
    if trend_score > 0 and turbulence_score < 1.1:
        regime = "ESPANSIONE" 
    elif trend_score > 0 and turbulence_score >= 1.1:
        regime = "INFLAZIONE / SURRISCALDAMENTO"
    elif trend_score <= 0 and turbulence_score >= 1.1:
        regime = "RECESSIONE"
    else:
        regime = "RALLENTAMENTO" 
        
    return regime, trend_score, turbulence_score

def get_regime_constraints(regime, max_equity_base, max_bond_base):
    """Definisce come cambiano i vincoli per ogni regime."""
    constraints = {
        "equity_max": max_equity_base,
        "bond_min": 0.0,
        "comm_min": 0.0,
        "vol_mult": 1.0
    }
    
    if regime == "RECESSIONE":
        constraints["equity_max"] = max_equity_base * 0.5 
        constraints["bond_min"] = 0.30 
        constraints["vol_mult"] = 0.7 
    elif regime == "ESPANSIONE":
        constraints["equity_max"] = min(1.0, max_equity_base * 1.2)
        constraints["vol_mult"] = 1.1
    elif regime == "INFLAZIONE / SURRISCALDAMENTO":
        constraints["comm_min"] = 0.15 
        constraints["equity_max"] = max_equity_base * 0.9 
    elif regime == "RALLENTAMENTO":
        constraints["equity_max"] = max_equity_base * 0.8
        constraints["bond_min"] = 0.15
        constraints["vol_mult"] = 0.9
        
    return constraints

# SCENARI DI STRESS PRE-DEFINITI
STRESS_SCENARIOS = {
    "Lehman 2008": {
        "start_date": "2008-09-01", "end_date": "2009-02-28", "description": "Crisi finanziaria globale",
        "shock_multiplier": {"BONDS": 0.9, "EQUITY": 0.5, "COMMODITIES": 0.6}
    },
    "COVID-19 2020": {
        "start_date": "2020-02-01", "end_date": "2020-04-30", "description": "Pandemia globale",
        "shock_multiplier": {"BONDS": 1.1, "EQUITY": 0.6, "COMMODITIES": 0.4}
    },
    "Inflation Shock 2022": {
        "start_date": "2022-01-01", "end_date": "2022-10-31", "description": "Picco inflazione",
        "shock_multiplier": {"BONDS": 0.8, "EQUITY": 0.8, "COMMODITIES": 1.2}
    }
}

class TransactionCostModel:
    def __init__(self): self.base_cost = 0.0010
    def estimate_cost(self, turnover_vector, asset_names, market_volatility=0.15):
        total_cost = np.full_like(turnover_vector, self.base_cost)
        liquidity = 7 
        market_impact = turnover_vector * (1 / liquidity) * market_volatility * 0.3
        total_cost += market_impact
        return total_cost

class DynamicConfidenceCalculator:
    @staticmethod
    def calculate(ff5_models, min_conf=0.1, max_conf=1.0):
        confidences = {}
        for asset, model in ff5_models.items():
            if model is None:
                confidences[asset] = min_conf
                continue
            r2_norm = min(1.0, model.rsquared * 2.0)
            confidences[asset] = np.clip(0.5 * r2_norm + 0.5, min_conf, max_conf)
        return pd.Series(confidences)

# ---------------------------------------------------------
# 1. GESTIONE DATI
# ---------------------------------------------------------

def process_uploaded_data(uploaded_file):
    try:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, sep=None, engine='python', index_col=0, parse_dates=True, dayfirst=True)
        # Pulizia numeri
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')
        
        df_monthly = df.resample('ME').last()
        returns = df_monthly.pct_change().dropna(how='all').fillna(0)
        # Winsorization
        lower = returns.quantile(0.01)
        upper = returns.quantile(0.99)
        returns = returns.clip(lower, upper, axis=1)
        return returns, df_monthly
    except Exception as e:
        return None, str(e)

@st.cache_data(ttl=3600, show_spinner="Downloading FF5 factors...")
def download_ff5_factors():
    url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_CSV.zip"
    try:
        r = requests.get(url, timeout=15)
        if r.status_code != 200: raise Exception("Connection Error")
        z = ZipFile(BytesIO(r.content))
        csv_file = [f for f in z.namelist() if f.endswith('.csv') and 'Factors' in f][0]
        with z.open(csv_file) as f:
            df = pd.read_csv(f, skiprows=3)
            df = df.rename(columns={'Mkt-RF':'MKT_RF','SMB':'SMB','HML':'HML','RMW':'RMW','CMA':'CMA','RF':'RF'})
            df['Date'] = pd.to_datetime(df['Unnamed: 0'].astype(str), format='%Y%m', errors='coerce')
            df = df.dropna(subset=['Date']).set_index('Date').resample('ME').last()
            return df[['MKT_RF','SMB','HML','RMW','CMA','RF']].astype(float) / 100
    except Exception as e:
        st.error(f"Errore download FF5: {e}")
        return None

@st.cache_data(ttl=86400)
def get_ff5_data_cutoff(cutoff_date):
    ff5_full = download_ff5_factors()
    if ff5_full is not None:
        return ff5_full.loc[:cutoff_date]
    return None

def calculate_ff5_views(returns, ff5_data, window=60, return_models=False):
    returns_clean = returns.fillna(0)
    aligned = pd.concat([returns_clean, ff5_data], axis=1, join='inner').dropna()
    if len(aligned) < 24: return None, None, "Dati insufficienti.", None
    
    data_window = aligned.iloc[-window:]
    X = sm.add_constant(data_window[['MKT_RF','SMB','HML','RMW','CMA']])
    
    views = {}
    betas_dict = {}
    models_dict = {}
    
    mom_scores = {}
    for asset in returns.columns:
        if len(returns_clean) >= 13:
            mom_scores[asset] = (1 + returns_clean[asset].iloc[-13:-1]).prod() - 1
        else:
            mom_scores[asset] = 0.0
            
    for asset in returns.columns:
        try:
            model = sm.OLS(data_window[asset] - data_window['RF'], X).fit()
            factors_mean = data_window[['MKT_RF','SMB','HML','RMW','CMA']].mean()
            exp_ret_ff = data_window['RF'].mean() + model.params['const'] + (model.params[['MKT_RF','SMB','HML','RMW','CMA']] * factors_mean).sum()
            
            mom_adjustment = mom_scores.get(asset, 0) * 0.05 
            views[asset] = exp_ret_ff + mom_adjustment
            betas_dict[asset] = model.params[['MKT_RF','SMB','HML','RMW','CMA']]
            models_dict[asset] = model
        except:
            views[asset] = data_window[asset].mean()
            betas_dict[asset] = pd.Series(0, index=['MKT_RF','SMB','HML','RMW','CMA'])
            models_dict[asset] = None
    
    if return_models:
        return pd.Series(views), pd.DataFrame(betas_dict).T, None, models_dict
    else:
        return pd.Series(views), pd.DataFrame(betas_dict).T, None

def robust_covariance_lw(returns):
    lw = LedoitWolf()
    lw.fit(returns.fillna(0))
    return pd.DataFrame(lw.covariance_, index=returns.columns, columns=returns.columns)

def black_litterman_solver(cov_matrix, market_prior, ff5_views, conf_id=0.5, asset_confidences=None):
    tau = 0.05
    P = np.eye(len(cov_matrix))
    Q = ff5_views.values.reshape(-1, 1)
    Pi = market_prior.values.reshape(-1, 1)
    
    if asset_confidences is not None:
        conf = asset_confidences.values.reshape(-1, 1)
        uncertainty = (1 - conf) / (conf + 1e-6)
    else:
        uncertainty = (1 - conf_id) / (conf_id + 1e-6)
    
    Omega = np.diag(np.diag(cov_matrix.values)) * tau * uncertainty
    
    inv_tau_cov = np.linalg.inv(tau * cov_matrix.values)
    try:
        inv_omega = np.linalg.inv(Omega)
    except:
        inv_omega = np.linalg.inv(Omega + np.eye(len(Omega)) * 1e-6)
    
    M_left = np.linalg.inv(inv_tau_cov + inv_omega)
    M_right = np.dot(inv_tau_cov, Pi) + np.dot(inv_omega, Q)
    return pd.Series(np.dot(M_left, M_right).flatten(), index=cov_matrix.index)

# --- NEW OPTIMIZER (Replaces optimize_line to handle MIN constraints) ---
def optimize_portfolio(mu, cov, target_vol, risk_free, min_w, max_w, group_constraints, groups):
    # MU deve essere una Series per mantenere l'indice corretto
    num_assets = len(mu)
    def objective(w): return -np.dot(w, mu.values)
    
    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
        {'type': 'eq', 'fun': lambda x: np.sqrt(np.dot(x.T, np.dot(cov, x))) * np.sqrt(12) - target_vol}
    ]
    
    # Vincoli Gruppo Max
    for group_name, max_val in group_constraints.get('max', {}).items():
        assets = groups.get(group_name, [])
        indices = [i for i, col in enumerate(mu.index) if col in assets]
        if indices:
            constraints.append({'type': 'ineq', 'fun': lambda x, idx=indices, m=max_val: m - np.sum(x[idx])})

    # Vincoli Gruppo Min
    for group_name, min_val in group_constraints.get('min', {}).items():
        assets = groups.get(group_name, [])
        indices = [i for i, col in enumerate(mu.index) if col in assets]
        if indices:
            constraints.append({'type': 'ineq', 'fun': lambda x, idx=indices, m=min_val: np.sum(x[idx]) - m})

    bounds = tuple((min_w, max_w) for _ in range(num_assets))
    init_guess = np.full(num_assets, 1/num_assets)
    
    try:
        res = minimize(objective, init_guess, bounds=bounds, constraints=constraints, method='SLSQP')
        return res.x
    except:
        return init_guess

# --- COMPARATIVE BACKTEST ENGINE ---
def run_comparative_backtest(returns, ff5_full, groups, vol_target, max_eq_base, max_bond_base, min_w, max_w):
    start_idx = 36
    rebalance_freq = 3 
    dates = returns.index[start_idx::rebalance_freq]
    
    ts_strat = [100.0]
    ts_tact = [100.0]
    ts_dates = [returns.index[start_idx-1]]
    regime_history = []
    
    wealth_strat = 100.0
    wealth_tact = 100.0
    
    progress_bar = st.progress(0)
    
    for i, date in enumerate(dates):
        progress_bar.progress((i + 1) / len(dates))
        
        past_returns = returns.loc[:date]
        ff5_slice = ff5_full.loc[:date]
        
        regime, _, _ = detect_market_regime(past_returns, groups["EQUITY"])
        regime_history.append({'Date': date, 'Regime': regime})
        
        views, _, _, models = calculate_ff5_views(past_returns, ff5_slice, return_models=True)
        if views is None: continue
        
        cov = robust_covariance_lw(past_returns)
        conf = DynamicConfidenceCalculator.calculate(models)
        bl_ret = black_litterman_solver(cov, past_returns.mean(), views, asset_confidences=conf)
        
        # Strategic
        cons_strat = {'max': {'EQUITY': max_eq_base, 'BONDS': max_bond_base, 'COMMODITIES': 0.3}, 'min': {}}
        w_strat = optimize_portfolio(bl_ret, cov.values, vol_target, 0.02, min_w, max_w, cons_strat, groups)
        
        # Tactical
        regime_params = get_regime_constraints(regime, max_eq_base, max_bond_base)
        cons_tact = {
            'max': {'EQUITY': regime_params['equity_max'], 'BONDS': 1.0, 'COMMODITIES': 1.0},
            'min': {'BONDS': regime_params['bond_min'], 'COMMODITIES': regime_params.get('comm_min', 0)}
        }
        vol_tact = vol_target * regime_params['vol_mult']
        w_tact = optimize_portfolio(bl_ret, cov.values, vol_tact, 0.02, min_w, max_w, cons_tact, groups)
        
        if i < len(dates) - 1:
            next_date = dates[i+1]
            period_ret = returns.loc[date:next_date].iloc[1:] 
            for d in period_ret.index:
                r_day = period_ret.loc[d].values
                wealth_strat *= (1 + np.dot(w_strat, r_day))
                ts_strat.append(wealth_strat)
                wealth_tact *= (1 + np.dot(w_tact, r_day))
                ts_tact.append(wealth_tact)
                ts_dates.append(d)

    progress_bar.empty()
    return pd.DataFrame({'Strategico': ts_strat, 'Tattico': ts_tact}, index=ts_dates), pd.DataFrame(regime_history)

# --- STANDARD BACKTEST (Updated for optimize_portfolio) ---
def run_walk_forward_backtest(returns, ff5_full, vol_ranges, group_limits_base, equity_limits_per_line, min_w, max_w, tx_cost, view_window, conf_level, stability_penalty, asset_groups):
    start_idx = max(36, view_window)
    returns = returns.fillna(0)
    rebalance_dates = returns.index[start_idx:][::12] 
    wf_results = {f"Linea {i+1}": [] for i in range(6)}
    
    current_weights = {f"Linea {i+1}": np.zeros(len(returns.columns)) for i in range(6)}
    last_rebal_weights = current_weights.copy()
    tx_model = TransactionCostModel()
    
    progress_bar = st.progress(0)
    
    for t_idx in range(start_idx, len(returns)):
        curr_date = returns.index[t_idx]
        progress_bar.progress((t_idx - start_idx + 1) / (len(returns) - start_idx))
        
        if curr_date in rebalance_dates:
            past_returns = returns.iloc[:t_idx]
            ff5_available = get_ff5_data_cutoff(past_returns.index[-1])
            
            if ff5_available is not None and len(ff5_available) >= 24:
                views_t, _, _, models_t = calculate_ff5_views(past_returns, ff5_available, window=view_window, return_models=True)
                if views_t is not None:
                    conf_calc = DynamicConfidenceCalculator()
                    asset_confidences = conf_calc.calculate(models_t)
                    cov_t = robust_covariance_lw(past_returns)
                    bl_post_t = black_litterman_solver(cov_t, past_returns.mean(), views_t, conf_id=conf_level, asset_confidences=asset_confidences)
                    
                    for i, (v_min, v_max) in enumerate(vol_ranges):
                        line_name = f"Linea {i+1}"
                        # Convert old structure to new structure
                        eq_lim = equity_limits_per_line.get(line_name, 1.0)
                        cons_grp = {
                            'max': {'EQUITY': eq_lim, 'BONDS': group_limits_base['BONDS'], 'COMMODITIES': group_limits_base['COMMODITIES']},
                            'min': {}
                        }
                        # Using v_min as target vol for simplicity in backtest standard
                        new_w = optimize_portfolio(bl_post_t, cov_t.values, v_min, 0.02, min_w, max_w, cons_grp, asset_groups)
                        current_weights[line_name] = new_w
        
        month_ret_vector = returns.iloc[t_idx].values
        for line_name in wf_results.keys():
            w_start = current_weights[line_name]
            gross_ret = np.dot(w_start, month_ret_vector)
            cost = 0.0
            if curr_date in rebalance_dates:
                 market_vol = 0.15
                 cost_vector = tx_model.estimate_cost(np.abs(w_start - last_rebal_weights[line_name]), returns.columns, market_vol)
                 cost = np.sum(cost_vector)
                 last_rebal_weights[line_name] = w_start
            wf_results[line_name].append(gross_ret - cost)
    
    progress_bar.empty()
    dates = returns.index[start_idx:]
    min_len = min(len(dates), len(list(wf_results.values())[0]))
    return pd.DataFrame({k: v[:min_len] for k,v in wf_results.items()}, index=dates[:min_len]), None, {}

def run_stress_tests(weights_dict, historical_returns, asset_groups, scenarios=STRESS_SCENARIOS):
    stress_results = []
    for scenario_name, scenario in scenarios.items():
        try:
            start_date = pd.to_datetime(scenario['start_date'])
            end_date = pd.to_datetime(scenario['end_date'])
            scenario_returns = historical_returns.loc[start_date:end_date].copy()
            if len(scenario_returns) == 0: continue
            
            for group, assets in asset_groups.items():
                multiplier = scenario.get('shock_multiplier', {}).get(group, 1.0)
                group_cols = [col for col in scenario_returns.columns if col in assets]
                if len(group_cols) > 0:
                    scenario_returns[group_cols] *= multiplier
            
            for line_name, weights in weights_dict.items():
                w = weights.values if isinstance(weights, pd.Series) else weights
                scenario_perf = scenario_returns.dot(w)
                total_return = (1 + scenario_perf).prod() - 1
                stress_results.append({'Scenario': scenario_name, 'Linea': line_name, 'Rendimento Totale': total_return})
        except: continue
    return pd.DataFrame(stress_results)

def generate_institutional_report(results_dict):
    report_data = {}
    if 'strategic_allocation' in results_dict:
        strat = results_dict['strategic_allocation']
        report_data['Executive_Summary'] = {
            'Data_Analisi': datetime.now().strftime('%Y-%m-%d'),
            'Numero_Asset': len(strat.get('weights_df', pd.DataFrame()).columns),
        }
    return report_data

def style_plotly_chart(fig, title="", height=None):
    fig.update_layout(template="plotly_white", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(243,244,246,0.5)',
        font=dict(family="Roboto, sans-serif", size=12, color="#000000"),
        title=dict(text=f"<b>{title}</b>", font=dict(size=20, color="#111827")),
        colorway=['#1e40af', '#047857', '#b91c1c'], margin=dict(l=40, r=40, t=60, b=40))
    if height: fig.update_layout(height=height)
    return fig

# ---------------------------------------------------------
# UI APP STREAMLIT MIGLIORATA
# ---------------------------------------------------------

st.title("🏛️ Institutional Portfolio System (Pro Suite - Dynamic)")
st.markdown("---")

# Sidebar
st.sidebar.header("⚙️ Configurazione Avanzata")
horizon_labels = {36: "36 Mesi (Tattico)", 60: "60 Mesi (Ciclo)", 120: "120 Mesi (Strutturale)"}
view_horizon = st.sidebar.selectbox("Orizzonte Views", [36, 60, 120], index=1, format_func=lambda x: horizon_labels.get(x, str(x)))
use_dynamic_conf = st.sidebar.checkbox("Usa Confidence Dinamica", value=True)
conf_level = st.sidebar.slider("Confidenza Base", 0.0, 1.0, 0.5)

st.sidebar.markdown("---")
st.sidebar.header("💸 Costi & Rischio")
tx_cost_bps = st.sidebar.number_input("Costi Transazione (bps)", 0, 50, 10) / 10000
stability_penalty = st.sidebar.slider("Penalità Turnover", 0.0, 0.1, 0.01, 0.001)
enable_stress_tests = st.sidebar.checkbox("Attiva Stress Tests", value=True)

st.sidebar.markdown("---")
st.sidebar.header("⚖️ Micro-Vincoli")
c1, c2 = st.sidebar.columns(2)
with c1: user_min_weight = st.number_input("Min % Asset", 0.0, 5.0, 0.0, 0.5) / 100
with c2: user_max_weight = st.number_input("Max % Asset", 10.0, 100.0, 30.0, 5.0) / 100

st.sidebar.markdown("---")
# Definizione range massimi fissi (Ceilings)
vol_max_fixed = [0.025, 0.050, 0.075, 0.100, 0.125, 0.150]
min_vol_constraints = []
with st.sidebar.expander("📉 Vincoli Volatilità Minima"):
    for i in range(6):
        val = st.number_input(f"Min Vol. Linea {i+1}", min_value=0.0, max_value=1.0, value=0.0, step=0.005, format="%.3f")
        min_vol_constraints.append(val)

st.sidebar.markdown("---")
st.sidebar.header("🌍 Macro-Vincoli")
max_bonds = st.sidebar.slider("Max BONDS", 0.0, 1.0, 0.80, 0.05)
max_comm = st.sidebar.slider("Max COMMODITIES", 0.0, 1.0, 0.15, 0.05)
max_equity_base = st.sidebar.slider("Max EQUITY Base", 0.0, 1.0, 0.70)

equity_limits = {}
default_limits = [0.10, 0.20, 0.40, 0.60, 0.80, 1.00]
for i in range(1, 7):
    equity_limits[f"Linea {i}"] = default_limits[i-1]

# Main interface
with st.container():
    st.markdown("### 📂 Data Import")
    uploaded_file = st.file_uploader("Carica CSV (Serie Storiche)", type='csv')

if uploaded_file:
    returns_monthly_raw, prices_monthly = process_uploaded_data(uploaded_file)
    if returns_monthly_raw is None: st.error(f"Errore lettura: {prices_monthly}"); st.stop()
    
    # PULIZIA NOMI COLONNE
    returns_monthly_raw.columns = returns_monthly_raw.columns.str.strip().str.upper()
    returns_monthly = returns_monthly_raw.fillna(0)
    
    st.success(f"Dati caricati: {len(returns_monthly.columns)} asset, {len(returns_monthly)} mesi.")
    
    # --- 1. CLASSIFICAZIONE MANUALE ASSISTITA ---
    st.markdown("### 🏷️ Mappatura Asset Class")
    with st.expander("🔍 Verifica e Correggi la classificazione degli asset", expanded=True):
        col_sel1, col_sel2, col_sel3 = st.columns(3)
        initial_groups = auto_classify_assets_initial(returns_monthly.columns)
        all_assets = list(returns_monthly.columns)
        
        with col_sel1:
            sel_equity = st.multiselect("EQUITY (Rischio)", all_assets, default=initial_groups["EQUITY"])
        with col_sel2:
            sel_bonds = st.multiselect("BONDS (Difesa)", all_assets, default=initial_groups["BONDS"])
        with col_sel3:
            sel_comm = st.multiselect("COMMODITIES (Reale)", all_assets, default=initial_groups["COMMODITIES"])
        
        detected_groups = {"EQUITY": sel_equity, "BONDS": sel_bonds, "COMMODITIES": sel_comm}

    ff5_df = download_ff5_factors()
    
    if ff5_df is not None:
        with st.spinner("Calcolo Asset Allocation Strategica..."):
            ff5_views, betas_df, err, ff5_models = calculate_ff5_views(returns_monthly, ff5_df, window=view_horizon, return_models=True)
            if err: st.error(err); st.stop()
            
            cov_lw = robust_covariance_lw(returns_monthly)
            asset_confidences = DynamicConfidenceCalculator.calculate(ff5_models) if use_dynamic_conf and ff5_models else None
            bl_posterior = black_litterman_solver(cov_lw, returns_monthly.mean(), ff5_views, conf_id=conf_level, asset_confidences=asset_confidences)
            
            # COSTRUZIONE DINAMICA RANGE VOLATILITA'
            vol_ranges = []
            for i in range(6): vol_ranges.append((min_vol_constraints[i], vol_max_fixed[i]))
            
            line_labels = ["Linea 1", "Linea 2", "Linea 3", "Linea 4", "Linea 5", "Linea 6"]
            rf_rate = ff5_df['RF'].iloc[-1] * 12 if len(ff5_df) > 0 else 0.02
            
            results, weights_list, weights_dict = [], [], {}
            group_limits_base = {"BONDS": max_bonds, "COMMODITIES": max_comm}
            
            for name, (v_min, v_max) in zip(line_labels, vol_ranges):
                eq_lim = equity_limits.get(name, 1.0)
                cons_grp = {
                    'max': {'EQUITY': eq_lim, 'BONDS': max_bonds, 'COMMODITIES': max_comm},
                    'min': {}
                }
                
                # USA optimize_portfolio INVECE DI optimize_line
                w = optimize_portfolio(bl_posterior, cov_lw.values, v_min, rf_rate/12, 
                                       user_min_weight, user_max_weight, cons_grp, detected_groups)
                
                ret_ann = np.sum(w * bl_posterior.values) * 12
                vol_ann = np.sqrt(np.dot(w.T, np.dot(cov_lw.values, w))) * np.sqrt(12)
                sharpe = (ret_ann - rf_rate) / vol_ann if vol_ann > 0 else 0
                
                w_series = pd.Series(w, index=returns_monthly.columns, name=name)
                weights_list.append(w_series)
                weights_dict[name] = w_series
                
                # Calcola pesi aggregati per display
                g_w = {g: w_series[assets].sum() for g, assets in detected_groups.items()}
                results.append({"Linea": name, "Rendimento": ret_ann, "Volatilità": vol_ann, "Sharpe": sharpe, 
                                "EQUITY": g_w.get('EQUITY', 0), "BONDS": g_w.get('BONDS', 0), "COMM": g_w.get('COMMODITIES', 0)})
                
            metrics_df = pd.DataFrame(results).set_index("Linea")
            
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["📊 STRATEGIC", "🌪️ DYNAMIC REGIME", "⏮️ BACKTEST COMP", "📈 BACKTEST STD", "🧠 MODELS", "🔥 STRESS", "📑 REPORT"])
        
        with tab1:
            st.markdown("### 🎯 Asset Allocation Ottimale (Strategica)")
            c1, c2 = st.columns([2, 1])
            with c1:
                # TABELLA DETTAGLIATA
                st.table(metrics_df.style.format({
                    "Rendimento": "{:.2%}", "Volatilità": "{:.2%}", "Sharpe": "{:.2f}",
                    "EQUITY": "{:.1%}", "BONDS": "{:.1%}", "COMM": "{:.1%}"
                }).background_gradient(cmap="Greens", subset=["Rendimento", "Sharpe"]))
            with c2:
                plot_df = metrics_df[["BONDS", "EQUITY", "COMM"]].reset_index().melt(id_vars="Linea", var_name="Category", value_name="Weight")
                fig = style_plotly_chart(px.bar(plot_df, x="Linea", y="Weight", color="Category", text_auto=".1%"), "Macro Allocation", 400)
                st.plotly_chart(fig, use_container_width=True)
            
            st.dataframe(pd.DataFrame(weights_list).T.style.format("{:.1%}").background_gradient(cmap="Greens", axis=None), height=400, use_container_width=True)

        with tab2:
            st.markdown("### 🧭 Adattamento al Ciclo Economico")
            regime, trend, turb = detect_market_regime(returns_monthly, detected_groups["EQUITY"])
            
            c_info, c_plot = st.columns([1, 2])
            with c_info:
                st.markdown(f"**Stato Attuale:** {regime}")
                st.metric("Trend Score", f"{trend:.2%}", help="Trend Positivo > 0")
                st.metric("Turbulence Score", f"{turb:.2f}", help="Alta Volatilità > 1.1")
                
                activate_regime = st.checkbox("✅ ADEGUARE AL REGIME", value=False)
                if activate_regime:
                    regime_params = get_regime_constraints(regime, max_equity_base, max_bonds)
                    st.markdown("#### Nuovi Vincoli:")
                    st.write(f"- Equity Max: {regime_params['equity_max']:.1%}")
                    st.write(f"- Bond Min: {regime_params.get('bond_min', 0):.1%}")
            
            with c_plot:
                fig_quad = go.Figure()
                fig_quad.add_shape(type="rect", x0=0, y0=0, x1=0.5, y1=1.1, fillcolor="rgba(0, 255, 0, 0.1)", line_width=0, layer="below")
                fig_quad.add_shape(type="rect", x0=-0.5, y0=1.1, x1=0.5, y1=3.0, fillcolor="rgba(255, 0, 0, 0.1)", line_width=0, layer="below")
                fig_quad.add_shape(type="rect", x0=-0.5, y0=0, x1=0, y1=1.1, fillcolor="rgba(0, 0, 255, 0.1)", line_width=0, layer="below")
                fig_quad.add_trace(go.Scatter(x=[trend], y=[turb], mode='markers+text', marker=dict(size=15, color='black'), text=["TU SEI QUI"], textposition="top center"))
                fig_quad.update_layout(title="Mappa Regimi (Why?)", xaxis_title="Trend", yaxis_title="Turbulence", xaxis=dict(range=[-0.3, 0.3]), yaxis=dict(range=[0.5, 2.0]))
                st.plotly_chart(fig_quad, use_container_width=True)

            if activate_regime:
                 tactical_metrics = []
                 for v_tgt, name in zip([0.03, 0.05, 0.08, 0.10, 0.12, 0.15], line_labels): # Usa range approssimati
                     rp = get_regime_constraints(regime, max_equity_base, max_bonds)
                     cons_tact = {'max': {'EQUITY': rp['equity_max'], 'BONDS': 1.0, 'COMMODITIES': 1.0}, 'min': {'BONDS': rp.get('bond_min', 0), 'COMMODITIES': rp.get('comm_min', 0)}}
                     w_tac = optimize_portfolio(bl_posterior, cov_lw.values, v_tgt * rp['vol_mult'], rf_rate/12, user_min_weight, user_max_weight, cons_tact, detected_groups)
                     ret_t = np.dot(w_tac, bl_posterior.values) * 12
                     vol_t = np.sqrt(np.dot(w_tac.T, np.dot(cov_lw.values, w_tac))) * np.sqrt(12)
                     tactical_metrics.append([name, ret_t, vol_t])
                 st.dataframe(pd.DataFrame(tactical_metrics, columns=["Linea", "Rendimento (Tact)", "Volatilità (Tact)"]).set_index("Linea"), use_container_width=True)

        with tab3:
            st.markdown("### ⏮️ Verità Storica: Strategico vs Tattico")
            if st.button("AVVIA BACKTEST COMPARATIVO"):
                with st.spinner("Elaborazione scenari..."):
                    df_wealth, df_regimes = run_comparative_backtest(returns_monthly, ff5_df, detected_groups, 0.08, max_equity_base, max_bonds, user_min_weight, user_max_weight)
                    
                    cum_ret = df_wealth.iloc[-1] / 100 - 1
                    dd = (df_wealth / df_wealth.cummax() - 1).min()
                    
                    c1, c2 = st.columns(2)
                    c1.metric("Totale Strategico", f"{cum_ret['Strategico']:.2%}", f"MaxDD: {dd['Strategico']:.2%}")
                    c2.metric("Totale Tattico", f"{cum_ret['Tattico']:.2%}", f"MaxDD: {dd['Tattico']:.2%}")
                    
                    st.plotly_chart(px.line(df_wealth, title="Crescita Capitale"), use_container_width=True)
                    st.plotly_chart(px.scatter(df_regimes, x='Date', y='Regime', color='Regime', title="Storico Regimi"), use_container_width=True)

        with tab4:
            st.markdown("### 📈 Walk-Forward Standard")
            with st.spinner("Backtest..."):
                wf_df, wf_err, _ = run_walk_forward_backtest(
                    returns_monthly, ff5_df, vol_ranges, group_limits_base, equity_limits,
                    user_min_weight, user_max_weight, tx_cost_bps, view_horizon, conf_level, stability_penalty, detected_groups
                )
            if wf_err: st.error(wf_err)
            else:
                nav_df = (1 + wf_df).cumprod() * 100
                st.plotly_chart(style_plotly_chart(px.line(nav_df), "Crescita Capitale"), use_container_width=True)

        with tab5:
            st.markdown("### 🧠 Analisi Fattori")
            st.dataframe(betas_df.style.background_gradient(cmap="RdBu", vmin=-1, vmax=1).format("{:.2f}"), use_container_width=True)

        with tab6:
            st.markdown("### 🔥 Stress Testing")
            if enable_stress_tests:
                stress_results = run_stress_tests(weights_dict, returns_monthly, detected_groups)
                if not stress_results.empty:
                    stress_pivot = stress_results.pivot_table(index='Scenario', columns='Linea', values='Rendimento Totale', aggfunc='mean')
                    st.dataframe(stress_pivot.style.format("{:.2%}").background_gradient(cmap="RdYlGn", vmin=-0.3, vmax=0.1), use_container_width=True)

        with tab7:
            st.markdown("### 📑 Report")
            report_data = generate_institutional_report({'strategic_allocation': {'weights_df': pd.DataFrame(weights_list).T}})
            st.table(pd.DataFrame([report_data.get('Executive_Summary', {})]).T)

    else: st.warning("Errore fattori FF5.")
else:
    st.markdown("""<div style='text-align: center; padding: 50px;'><h2 style='color: #000000;'>Benvenuto in Institutional Portfolio System</h2><p style='color: #374151;'>Carica un file CSV con serie storiche di <b>qualsiasi asset</b> per iniziare.</p></div>""", unsafe_allow_html=True)
