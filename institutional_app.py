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
    /* Main Background e Font */
    .stApp {
        background-color: #f0f2f6; /* Grigio Chiaro Istituzionale */
        color: #000000; /* Testo Nero */
        font-family: 'Roboto', 'Helvetica Neue', sans-serif;
    }
    
    /* --------------------------
       SIDEBAR STYLING
       -------------------------- */
    [data-testid="stSidebar"] {
        background-color: #ffffff; /* Bianco Puro */
        border-right: 1px solid #d1d5db; /* Bordo grigio sottile */
    }
    
    /* Forza COLORE NERO su TUTTI gli elementi della sidebar */
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3, 
    [data-testid="stSidebar"] label, 
    [data-testid="stSidebar"] span, 
    [data-testid="stSidebar"] p, 
    [data-testid="stSidebar"] div,
    [data-testid="stSidebar"] .stMarkdown {
        color: #000000 !important;
    }
    
    /* Input Fields nella Sidebar */
    [data-testid="stSidebar"] .stNumberInput input, 
    [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] div {
        color: #000000 !important;
        background-color: #f9fafb !important;
        border-color: #d1d5db !important;
    }
    
    /* --------------------------
       MAIN CONTENT STYLING
       -------------------------- */
    
    /* Headers */
    h1, h2, h3 {
        color: #111827 !important; /* Quasi nero per morbidezza */
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    h1 { border-bottom: 2px solid #000000; padding-bottom: 15px; margin-bottom: 25px; font-size: 2.5rem; }
    h2 { font-size: 1.8rem; margin-top: 25px; color: #1f2937; border-left: 4px solid #2563eb; padding-left: 10px; }
    h3 { font-size: 1.3rem; color: #374151; }
    
    /* Metric Cards Custom (Light Mode) */
    div[data-testid="metric-container"] {
        background-color: #ffffff;
        border: 1px solid #e5e7eb;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    label[data-testid="stMetricLabel"] {
        color: #4b5563 !important; /* Grigio scuro */
        font-weight: 600 !important;
    }
    div[data-testid="stMetricValue"] {
        color: #000000 !important;
        font-weight: 900 !important;
        font-size: 2rem !important;
    }
    
    /* Tables/Dataframes */
    [data-testid="stDataFrame"], [data-testid="stTable"] {
        background-color: #ffffff;
        border: 1px solid #d1d5db;
        color: #000000;
    }
    
    /* Buttons */
    .stButton > button {
        background-color: #2563eb; /* Blu Istituzionale */
        color: #ffffff;
        border: none;
        border-radius: 4px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        background-color: #1d4ed8;
        color: #ffffff;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #e5e7eb;
        border-radius: 4px 4px 0 0;
        color: #4b5563;
        border: 1px solid #d1d5db;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ffffff;
        color: #2563eb !important;
        border: 1px solid #d1d5db;
        border-bottom: 2px solid #2563eb;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #ffffff;
        color: #000000 !important;
        border: 1px solid #e5e7eb;
        font-weight: 600;
    }
    
    /* Info/Warning Boxes */
    .stAlert {
        background-color: #eff6ff;
        border: 1px solid #bfdbfe;
        color: #1e3a8a;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 0. LOGICA DINAMICA ASSET CLASS (AUTO-CLASSIFIER)
# ---------------------------------------------------------

def auto_classify_assets(asset_names):
    """
    Classifica automaticamente gli asset in gruppi basandosi sul nome.
    Keyword matching euristico.
    """
    groups = {
        "BONDS": [],
        "EQUITY": [],
        "COMMODITIES": []
    }
    
    # Keywords per categorizzazione
    bond_keywords = ['BOND', 'GOV', 'CORP', 'HY', 'T-BILL', 'TREASURY', 'OBL', 'YIELD', 'AGG', 'MON', 'FI']
    comm_keywords = ['COMM', 'GOLD', 'OIL', 'CMD', 'METAL', 'ENERGY', 'AGRI']
    # Equity è il "residuale" o keyword specifiche
    eq_keywords = ['EQ', 'STOCK', 'AZ', 'SPX', 'MSCI', 'NDX', 'EM', 'PAC', 'US', 'EU']
    
    for asset in asset_names:
        u_asset = asset.upper()
        
        # Priorità: Commodities -> Bonds -> Equity (Default)
        if any(k in u_asset for k in comm_keywords):
            groups["COMMODITIES"].append(asset)
        elif any(k in u_asset for k in bond_keywords):
            groups["BONDS"].append(asset)
        else:
            # Se non è Bond o Comm, assumiamo sia Equity/Risk Asset
            groups["EQUITY"].append(asset)
            
    return groups

# SCENARI DI STRESS PRE-DEFINITI (Generici)
STRESS_SCENARIOS = {
    "Lehman 2008": {
        "start_date": "2008-09-01", "end_date": "2009-02-28",
        "description": "Crisi finanziaria globale",
        "shock_multiplier": {"BONDS": 0.9, "EQUITY": 0.5, "COMMODITIES": 0.6}
    },
    "COVID-19 2020": {
        "start_date": "2020-02-01", "end_date": "2020-04-30",
        "description": "Pandemia globale",
        "shock_multiplier": {"BONDS": 1.1, "EQUITY": 0.6, "COMMODITIES": 0.4}
    },
    "Inflation Shock 2022": {
        "start_date": "2022-01-01", "end_date": "2022-10-31",
        "description": "Picco inflazione e rialzo tassi",
        "shock_multiplier": {"BONDS": 0.8, "EQUITY": 0.8, "COMMODITIES": 1.2}
    }
}

class TransactionCostModel:
    """Modello avanzato costi transazione (Generic)"""
    def __init__(self, current_aum=100000000):
        self.current_aum = current_aum
        self.base_cost = 0.0010  # 10 bps base
        
    def estimate_cost(self, turnover_vector, asset_names, market_volatility=0.15):
        # Base cost
        total_cost = np.full_like(turnover_vector, self.base_cost)
        # Market impact generico (assumendo liquidità media score=7 per tutti se ignoti)
        liquidity = 7 
        market_impact = turnover_vector * (1 / liquidity) * market_volatility * 0.3
        total_cost += market_impact
        return total_cost

class DynamicConfidenceCalculator:
    """Calcola confidenza dinamica basata su qualità modello"""
    @staticmethod
    def calculate(ff5_models, min_conf=0.1, max_conf=1.0):
        confidences = {}
        for asset, model in ff5_models.items():
            if model is None:
                confidences[asset] = min_conf
                continue
            r2_norm = min(1.0, model.rsquared * 1.5)
            if hasattr(model, 'tvalues'):
                t_stats = model.tvalues.abs().mean()
                t_norm = min(1.0, t_stats / 2)
            else:
                t_norm = 0.5
            stability_score = 0.7 
            conf = 0.4 * r2_norm + 0.3 * t_norm + 0.3 * stability_score
            confidences[asset] = np.clip(conf, min_conf, max_conf)
        return pd.Series(confidences)

# ---------------------------------------------------------
# 1. GESTIONE DATI (MODIFICATA PER ASSET VARIABILI)
# ---------------------------------------------------------

def handle_missing_returns(returns, max_consecutive_na=3):
    returns_filled = returns.copy()
    returns_filled = returns_filled.ffill(limit=max_consecutive_na)
    # Fallback a 0
    returns_filled = returns_filled.fillna(0)
    return returns_filled

def winsorize_returns(returns, lower_percentile=1, upper_percentile=99):
    returns_winsorized = returns.copy()
    for asset in returns.columns:
        if len(returns[asset].dropna()) > 0:
            lower_bound = np.percentile(returns[asset].dropna(), lower_percentile)
            upper_bound = np.percentile(returns[asset].dropna(), upper_percentile)
            returns_winsorized[asset] = returns[asset].clip(lower_bound, upper_bound)
    return returns_winsorized

def detect_regime_shifts(returns, window=24):
    rolling_vol = returns.rolling(window).std()
    vol_shock = (rolling_vol / rolling_vol.shift(window) - 1) > 1.0
    # Use first asset as proxy for corr if enough data
    if len(returns.columns) > 0:
        rolling_corr = returns.rolling(window).corr(returns.iloc[:, 0]).iloc[:, 0]
        corr_shock = rolling_corr.diff().abs() > 0.3
    else:
        corr_shock = pd.Series(0, index=returns.index)
        
    regime_df = pd.DataFrame({
        'volatility_shock': vol_shock.mean(axis=1),
        'correlation_break': corr_shock,
        'high_vol_regime': rolling_vol.mean(axis=1) > rolling_vol.mean(axis=1).quantile(0.8)
    }, index=returns.index)
    return regime_df

def validate_inputs(returns, min_months=24, max_missing=0.3):
    validation_issues = []
    if len(returns) < min_months:
        validation_issues.append(f"Dati insufficienti: {len(returns)} mesi (< {min_months})")
    missing_pct = returns.isna().sum() / len(returns)
    high_missing = missing_pct[missing_pct > max_missing]
    if len(high_missing) > 0:
        validation_issues.append(f"Asset con >{max_missing:.0%} dati mancanti: {list(high_missing.index)}")
    return validation_issues

def process_uploaded_data(uploaded_file):
    try:
        uploaded_file.seek(0)
        for sep in [';', ',', '\t']:
            try:
                df = pd.read_csv(uploaded_file, sep=sep, index_col=0, parse_dates=True, dayfirst=True, encoding='utf-8')
                if df.shape[1] >= 1: # Almeno 1 asset
                    break
            except:
                uploaded_file.seek(0)
                continue
        
        # Pulizia dati: converte tutto in numeri
        for col in df.columns:
            if df[col].dtype == object: 
                df[col] = df[col].astype(str).str.replace('.', '', regex=False)
                df[col] = df[col].str.replace(',', '.', regex=False)
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df_monthly = df.resample('ME').last() 
        returns = df_monthly.pct_change()
        returns = handle_missing_returns(returns)
        returns = winsorize_returns(returns)
        
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
            df = df.dropna()
            df['Date'] = pd.to_datetime(df['Unnamed: 0'].astype(str), format='%Y%m', errors='coerce')
            df = df.dropna(subset=['Date']).set_index('Date')
            cols = ['MKT_RF','SMB','HML','RMW','CMA','RF']
            df = df[cols].astype(float) / 100 
            df = df.resample('ME').last()
            return df
    except Exception as e:
        st.error(f"Errore download FF5: {e}")
        return None

@st.cache_data(ttl=86400, show_spinner=False)
def get_ff5_data_cutoff(cutoff_date):
    ff5_full = download_ff5_factors()
    if ff5_full is not None:
        return ff5_full.loc[:cutoff_date]
    return None

# --- CORE CALCULATION WITH MOMENTUM (CORRETTO) ---
def calculate_ff5_views(returns, ff5_data, window=60, return_models=False):
    returns_clean = returns.fillna(0)
    aligned = pd.concat([returns_clean, ff5_data], axis=1, join='inner').dropna()
    if len(aligned) < 24: return None, None, "Dati insufficienti.", None
    
    data_window = aligned.iloc[-window:] if len(aligned) > window else aligned
    X = sm.add_constant(data_window[['MKT_RF','SMB','HML','RMW','CMA']])
    rf_mean = data_window['RF'].mean()
    factors_mean = data_window[['MKT_RF','SMB','HML','RMW','CMA']].mean()
    
    views = {}
    betas_dict = {}
    models_dict = {}
    
    # === FIX MOMENTUM CALCULATION (Logica 12-1 Month Cumulative) ===
    momentum_scores = {}
    for asset in returns.columns:
        try:
            if len(returns_clean) >= 13:
                # Seleziona finestra temporale t-13 a t-1
                window_returns = returns_clean[asset].iloc[-13:-1]
                # Calcola rendimento composto: (1+r1)*(1+r2)... - 1
                cumulative_return = (1 + window_returns).prod() - 1
                momentum_scores[asset] = cumulative_return
            else:
                momentum_scores[asset] = 0.0
        except:
            momentum_scores[asset] = 0.0
    # ===============================================================
    
    for asset in returns.columns:
        try:
            model = sm.OLS(data_window[asset] - data_window['RF'], X).fit()
            # Rendimento Base (Fama-French)
            exp_ret_ff = rf_mean + model.params['const'] + (model.params[['MKT_RF','SMB','HML','RMW','CMA']] * factors_mean).sum()
            
            # Aggiustamento Momentum (Tilt tattico)
            # Applichiamo un "Tilt": 5% del momentum annuale viene aggiunto alla view
            asset_mom = momentum_scores.get(asset, 0)
            mom_adjustment = asset_mom * 0.05 
            
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

def black_litterman_solver(cov_matrix, market_prior, ff5_views, tau=0.05, conf_id=0.5, asset_confidences=None):
    P = np.eye(len(cov_matrix))
    Q = ff5_views.values.reshape(-1, 1)
    Pi = market_prior.values.reshape(-1, 1)
    cov_values = cov_matrix.values
    
    if asset_confidences is not None:
        confidences = asset_confidences.values.reshape(-1, 1)
        uncertainty_scalar = (1 - confidences) / (confidences + 1e-6)
    else:
        uncertainty_scalar = (1 - conf_id) / (conf_id + 1e-6)
    
    Omega_base = np.dot(np.dot(P, (tau * cov_values)), P.T)
    Omega = Omega_base * uncertainty_scalar
    
    if np.isscalar(conf_id) and conf_id < 0.01: 
        Omega = np.diag(np.diag(cov_values)) * 1000
    
    inv_tau_cov = np.linalg.inv(tau * cov_values)
    try:
        inv_omega = np.linalg.inv(Omega)
    except np.linalg.LinAlgError:
        inv_omega = np.linalg.inv(Omega + np.eye(len(Omega)) * 1e-6)
    
    M_left = np.linalg.inv(inv_tau_cov + np.dot(np.dot(P.T, inv_omega), P))
    M_right = np.dot(inv_tau_cov, Pi) + np.dot(np.dot(P.T, inv_omega), Q)
    posterior = np.dot(M_left, M_right)
    
    return pd.Series(posterior.flatten(), index=cov_matrix.index)

def run_stress_tests(weights_dict, historical_returns, asset_groups, scenarios=STRESS_SCENARIOS):
    stress_results = []
    for scenario_name, scenario in scenarios.items():
        try:
            start_date = pd.to_datetime(scenario['start_date'])
            end_date = pd.to_datetime(scenario['end_date'])
            scenario_returns = historical_returns.loc[start_date:end_date].copy()
            if len(scenario_returns) == 0: continue
            
            # Applica shock per gruppo (Usando gruppi dinamici passati come argomento)
            for group, assets in asset_groups.items():
                multiplier = scenario.get('shock_multiplier', {}).get(group, 1.0)
                group_cols = [col for col in scenario_returns.columns if col in assets]
                if len(group_cols) > 0:
                    scenario_returns[group_cols] *= multiplier
            
            for line_name, weights in weights_dict.items():
                w = weights.values if isinstance(weights, pd.Series) else weights
                scenario_perf = scenario_returns.dot(w)
                total_return = (1 + scenario_perf).prod() - 1
                wealth = (1 + scenario_perf).cumprod()
                drawdown = (wealth / wealth.cummax() - 1).min()
                stress_results.append({
                    'Scenario': scenario_name, 'Linea': line_name,
                    'Durata (mesi)': len(scenario_returns), 'Rendimento Totale': total_return,
                    'Max Drawdown': drawdown, 'Descrizione': scenario['description']
                })
        except Exception as e:
            continue
    return pd.DataFrame(stress_results)

def optimize_line(mu, cov, asset_names, target_vol_min=None, target_vol_max=None, risk_free=0.0, 
                  min_weight=0.0, max_weight=1.0, group_limits=None, stability_penalty=0.0,
                  previous_weights=None, asset_groups=None):
    num_assets = len(mu)
    def objective(weights, mu, prev_w):
        base_return = -np.sum(mu * weights)
        turnover_penalty = 0
        if prev_w is not None and stability_penalty > 0:
            turnover_penalty = stability_penalty * np.sum(np.abs(weights - prev_w))
        return base_return + turnover_penalty
    
    args = (mu, previous_weights) if previous_weights is not None else (mu, None)
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
    if target_vol_max:
        constraints.append({'type': 'ineq', 'fun': lambda x: target_vol_max - np.sqrt(np.dot(x.T, np.dot(cov, x))) * np.sqrt(12)})
    if target_vol_min:
        constraints.append({'type': 'ineq', 'fun': lambda x: np.sqrt(np.dot(x.T, np.dot(cov, x))) * np.sqrt(12) - target_vol_min})
    
    # Vincoli gruppo DINAMICI
    if group_limits and asset_groups:
        for group, limit_max in group_limits.items():
            group_assets = asset_groups.get(group, [])
            indices = [i for i, name in enumerate(asset_names) if name in group_assets]
            if indices:
                constraints.append({'type': 'ineq', 'fun': lambda x, idx=indices, l=limit_max: l - np.sum(x[idx])})
    
    bounds = tuple((min_weight, max_weight) for _ in range(num_assets))
    init_guess = num_assets * [1. / num_assets,]
    try:
        result = minimize(objective, init_guess, args=args, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 1000, 'ftol': 1e-9})
        return result.x
    except:
        return np.full(num_assets, 1.0/num_assets)

def sensitivity_monte_carlo(mu_base, cov_base, asset_names, n_sim=500):
    allocations = []
    for sim in range(n_sim):
        mu_perturbed = mu_base * np.random.normal(1, 0.15, len(mu_base))
        pert_matrix = np.random.normal(1, 0.1, cov_base.shape)
        cov_perturbed = cov_base * (pert_matrix + pert_matrix.T) / 2
        cov_perturbed = (cov_perturbed + cov_perturbed.T) / 2
        min_eig = np.min(np.real(np.linalg.eigvals(cov_perturbed)))
        if min_eig < 0: cov_perturbed -= min_eig * np.eye(*cov_perturbed.shape)
        try:
            w_opt = optimize_line(mu_perturbed, cov_perturbed, asset_names, min_weight=0.01, max_weight=0.3)
            allocations.append(w_opt)
        except: continue
    if allocations:
        return pd.Series(np.array(allocations).std(axis=0), index=asset_names)
    return pd.Series(0, index=asset_names)

def run_walk_forward_backtest(returns, ff5_full, vol_ranges, group_limits_base, 
                             equity_limits_per_line, min_w, max_w, tx_cost, 
                             view_window, conf_level, stability_penalty=0.01, asset_groups=None):
    start_idx = max(36, view_window)
    returns = returns.fillna(0)
    if len(returns) <= start_idx: return None, "Storia troppo breve.", None
    
    rebalance_dates = returns.index[start_idx:][::12] 
    wf_results = {f"Linea {i+1}": [] for i in range(6)}
    current_weights = {f"Linea {i+1}": np.zeros(len(returns.columns)) for i in range(6)}
    last_rebal_weights = current_weights.copy()
    turnover_history = {f"Linea {i+1}": [] for i in range(6)}
    tx_model = TransactionCostModel() # Generic, no scores needed
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for t_idx in range(start_idx, len(returns)):
        curr_date = returns.index[t_idx]
        progress = (t_idx - start_idx + 1) / (len(returns) - start_idx)
        progress_bar.progress(progress)
        
        if curr_date in rebalance_dates:
            status_text.text(f"Rebalancing: {curr_date.date()}...")
            past_returns = returns.iloc[:t_idx]
            ff5_available = get_ff5_data_cutoff(past_returns.index[-1])
            
            if ff5_available is not None and len(ff5_available) >= 24:
                views_t, _, _, models_t = calculate_ff5_views(past_returns, ff5_available, window=view_window, return_models=True)
                if views_t is not None:
                    conf_calc = DynamicConfidenceCalculator()
                    asset_confidences = conf_calc.calculate(models_t)
                    cov_t = robust_covariance_lw(past_returns)
                    prior_t = past_returns.mean()
                    bl_post_t = black_litterman_solver(cov_t, prior_t, views_t, conf_id=conf_level, asset_confidences=asset_confidences)
                    rf_t = ff5_available['RF'].iloc[-1] * 12 if len(ff5_available) > 0 else 0.02
                    
                    for i, (v_min, v_max) in enumerate(vol_ranges):
                        line_name = f"Linea {i+1}"
                        curr_groups = group_limits_base.copy()
                        curr_groups["EQUITY"] = equity_limits_per_line.get(line_name, 1.0)
                        
                        # Fix per l'uso dei vincoli utente nel backtest:
                        # Assicuriamoci che v_min venga passato correttamente
                        # Qui vol_ranges contiene le tuple (min_input_utente, max_fisso)
                        new_w = optimize_line(bl_post_t.values, cov_t.values, returns.columns,
                                                target_vol_min=v_min, # Usa il valore dalla tupla
                                                target_vol_max=v_max, 
                                                risk_free=rf_t/12, 
                                                min_weight=min_w, max_weight=max_w, group_limits=curr_groups, 
                                                stability_penalty=stability_penalty, previous_weights=current_weights[line_name],
                                                asset_groups=asset_groups) # Pass dynamic groups
                        current_weights[line_name] = new_w
        
        month_ret_vector = returns.iloc[t_idx].values
        for line_name in wf_results.keys():
            w_start = current_weights[line_name]
            gross_ret = np.dot(w_start, month_ret_vector)
            cost = 0.0
            if curr_date in rebalance_dates:
                turnover = np.sum(np.abs(w_start - last_rebal_weights[line_name]))
                turnover_history[line_name].append(turnover)
                market_vol = 0.15 # Default
                if t_idx > 0: market_vol = returns.iloc[t_idx-view_window:t_idx].std().mean() * np.sqrt(12)
                cost_vector = tx_model.estimate_cost(np.abs(w_start - last_rebal_weights[line_name]), returns.columns, market_vol)
                cost = np.sum(cost_vector)
                last_rebal_weights[line_name] = w_start
            wf_results[line_name].append(gross_ret - cost)
    
    progress_bar.empty()
    status_text.empty()
    dates = returns.index[start_idx:]
    min_len = min(len(dates), len(list(wf_results.values())[0]))
    df_res = pd.DataFrame({k: v[:min_len] for k,v in wf_results.items()}, index=dates[:min_len])
    avg_turnover = {k: np.mean(v) if v else 0 for k, v in turnover_history.items()}
    return df_res, None, avg_turnover

def generate_institutional_report(results_dict):
    report_data = {}
    if 'strategic_allocation' in results_dict:
        strat = results_dict['strategic_allocation']
        report_data['Executive_Summary'] = {
            'Data_Analisi': datetime.now().strftime('%Y-%m-%d'),
            'Numero_Asset': len(strat.get('weights', [])),
            'Sharpe_Ratio_Medio': np.mean([l.get('Sharpe', 0) for l in strat.get('lines', [])]),
        }
    if 'risk_metrics' in results_dict: report_data['Risk_Metrics'] = results_dict['risk_metrics']
    return report_data

def check_portfolio_alerts(weights_dict):
    alerts = []
    for line_name, weights in weights_dict.items():
        w = weights.values if isinstance(weights, pd.Series) else weights
        max_weight = np.max(w)
        if max_weight > 0.25: alerts.append({'Linea': line_name, 'Tipo': 'Concentrazione', 'Messaggio': f"Asset > 25%", 'Severità': 'High'})
    return alerts

def style_plotly_chart(fig, title="", height=None):
    institutional_palette = ['#1e40af', '#047857', '#b91c1c', '#b45309', '#4b5563', '#7c3aed', '#0891b2', '#be185d']
    fig.update_layout(template="plotly_white", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(243,244,246,0.5)',
        font=dict(family="Roboto, sans-serif", size=12, color="#000000"),
        title=dict(text=f"<b>{title}</b>", font=dict(size=20, color="#111827"), x=0.05, xanchor='left'),
        colorway=institutional_palette, margin=dict(l=40, r=40, t=60, b=40),
        xaxis=dict(showgrid=True, gridcolor='#e5e7eb'), yaxis=dict(showgrid=True, gridcolor='#e5e7eb'))
    if height: fig.update_layout(height=height)
    return fig

# ---------------------------------------------------------
# UI APP STREAMLIT MIGLIORATA
# ---------------------------------------------------------

st.title("🏛️ Institutional Portfolio System (Pro Suite - Dynamic)")
st.markdown("---")

# Sidebar
st.sidebar.header("⚙️ Configurazione Avanzata")
horizon_labels = {36: "36 Mesi (Tattico)", 60: "60 Mesi (Ciclo)", 120: "120 Mesi (Strutturale)", 180: "180 Mesi (Secolare)"}
view_horizon = st.sidebar.selectbox("Orizzonte Views", [36, 60, 120, 180], index=1, format_func=lambda x: horizon_labels.get(x, str(x)))
use_dynamic_conf = st.sidebar.checkbox("Usa Confidence Dinamica", value=True)
conf_level = st.sidebar.slider("Confidenza Base", 0.0, 1.0, 0.5) if use_dynamic_conf else st.sidebar.slider("Confidenza Modello", 0.0, 1.0, 0.5)

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

# NUOVA SEZIONE: VINCOLI VOLATILITÀ MINIMA
st.sidebar.markdown("---")
st.sidebar.info("ℹ️ Volatilità Massima: Impostata per default")
# Definizione range massimi fissi (Ceilings)
vol_max_fixed = [0.025, 0.050, 0.075, 0.100, 0.125, 0.150]
# Lista per raccogliere i nuovi minimi
min_vol_constraints = []

with st.sidebar.expander("📉 Vincoli Volatilità Minima"):
    for i in range(6):
        # Default calcolato come (Max - 2.5%), ma modificabile
        default_min = 0.0
        # Input utente
        val = st.number_input(f"Min Vol. Linea {i+1}", min_value=0.0, max_value=1.0, value=default_min, step=0.005, format="%.3f")
        min_vol_constraints.append(val)

st.sidebar.markdown("---")
st.sidebar.header("🌍 Macro-Vincoli (Auto-Detected)")
max_bonds = st.sidebar.slider("Max BONDS (Se rilevati)", 0.0, 1.0, 0.80, 0.05)
max_comm = st.sidebar.slider("Max COMMODITIES (Se rilevati)", 0.0, 1.0, 0.15, 0.05)

st.sidebar.markdown("---")
st.sidebar.header("🪜 Scalini Equity")
equity_limits = {}
default_limits = [0.10, 0.20, 0.40, 0.60, 0.80, 1.00]
for i in range(1, 7):
    with st.sidebar.expander(f"Vincolo Linea {i}"):
        limit = st.slider(f"Max Eq.", 0.0, 1.0, default_limits[i-1], 0.05, key=f"eq_lim_{i}")
        equity_limits[f"Linea {i}"] = limit

# Main interface
with st.container():
    st.markdown("### 📂 Data Import")
    col_up_1, col_up_2 = st.columns([1, 2])
    with col_up_1:
        uploaded_file = st.file_uploader("Carica CSV (Serie Storiche)", type='csv')

if uploaded_file:
    returns_monthly_raw, prices_monthly = process_uploaded_data(uploaded_file)
    if returns_monthly_raw is None: st.error(f"Errore lettura: {prices_monthly}"); st.stop()
    
    # PULIZIA NOMI COLONNE
    returns_monthly_raw.columns = returns_monthly_raw.columns.str.strip().str.upper()
    returns_monthly = returns_monthly_raw.fillna(0) # Fix per date continuity
    
    # AUTO-CLASSIFICAZIONE ASSET
    detected_groups = auto_classify_assets(returns_monthly.columns)
    
    # Validazione dati
    validation_issues = validate_inputs(returns_monthly)
    if validation_issues:
        with st.expander("⚠️ Issues di Validazione", expanded=True):
            for issue in validation_issues[:3]: st.error(issue)
    
    with col_up_2:
        c1, c2, c3 = st.columns(3)
        c1.metric("Asset Rilevati", f"{len(returns_monthly.columns)}")
        c2.metric("Storico", f"{len(returns_monthly)} Mesi")
        c3.metric("Data Inizio", f"{returns_monthly.index[0].strftime('%b %Y')}")
        
        # -------------------------------------------------------------
        # MODIFICA RICHIESTA: Menu a tendina per spostamento Asset Class
        # -------------------------------------------------------------
        with st.expander("⚙️ Gestione e Modifica Mapping Asset Class (Menu a Tendina)"):
            st.info("Verifica e sposta gli asset nella categoria corretta se l'auto-rilevamento ha fallito.")
            
            # 1. Creiamo una mappa piatta (Asset -> Categoria) dallo stato attuale
            current_mapping = {}
            for cat, assets in detected_groups.items():
                for a in assets: current_mapping[a] = cat
            
            # 2. Interfaccia UI per la modifica (3 Colonne per ordine)
            new_groups_builder = {"BONDS": [], "EQUITY": [], "COMMODITIES": []}
            cols = st.columns(3) 
            sorted_assets = sorted(returns_monthly.columns)
            
            for i, asset in enumerate(sorted_assets):
                col = cols[i % 3] # Distribuisce gli asset su 3 colonne
                default_cat = current_mapping.get(asset, "EQUITY")
                
                with col:
                    # IL MENU A TENDINA RICHIESTO
                    chosen_cat = st.selectbox(
                        label=f"{asset}", 
                        options=["EQUITY", "BONDS", "COMMODITIES"], 
                        index=["EQUITY", "BONDS", "COMMODITIES"].index(default_cat),
                        key=f"map_select_{asset}"
                    )
                    new_groups_builder[chosen_cat].append(asset)
            
            # 3. SOVRASCRIVIAMO detected_groups con le scelte dell'utente
            # In questo modo tutto il codice successivo userà la mappa corretta
            detected_groups = new_groups_builder
        # -------------------------------------------------------------

    ff5_df = download_ff5_factors()
    
    if ff5_df is not None:
        with st.spinner("Calcolo Asset Allocation Strategica..."):
            ff5_views, betas_df, err, ff5_models = calculate_ff5_views(returns_monthly, ff5_df, window=view_horizon, return_models=True)
            if err: st.error(err); st.stop()
            
            cov_lw = robust_covariance_lw(returns_monthly)
            asset_confidences = DynamicConfidenceCalculator.calculate(ff5_models) if use_dynamic_conf and ff5_models else None
            bl_posterior = black_litterman_solver(cov_lw, returns_monthly.mean(), ff5_views, conf_id=conf_level, asset_confidences=asset_confidences)
            
            # COSTRUZIONE DINAMICA RANGE VOLATILITA'
            # Combina i minimi inseriti dall'utente con i massimi fissi
            vol_ranges = []
            for i in range(6):
                vol_ranges.append((min_vol_constraints[i], vol_max_fixed[i]))
            
            line_labels = ["Linea 1", "Linea 2", "Linea 3", "Linea 4", "Linea 5", "Linea 6"]
            rf_rate = ff5_df['RF'].iloc[-1] * 12 if len(ff5_df) > 0 else 0.02
            
            results, weights_list, weights_dict = [], [], {}
            group_limits_base = {"BONDS": max_bonds, "COMMODITIES": max_comm}
            
            for name, (v_min, v_max) in zip(line_labels, vol_ranges):
                curr_groups = group_limits_base.copy()
                curr_groups["EQUITY"] = equity_limits.get(name, 1.0)
                
                # QUI PASSIAMO IL NUOVO v_min (che arriva dalla sidebar)
                w = optimize_line(bl_posterior.values, cov_lw.values, returns_monthly.columns,
                                target_vol_min=v_min, # Usa input utente
                                target_vol_max=v_max, 
                                risk_free=rf_rate/12, 
                                min_weight=user_min_weight, max_weight=user_max_weight, 
                                group_limits=curr_groups, stability_penalty=stability_penalty,
                                asset_groups=detected_groups) 
                
                ret_ann = np.sum(w * bl_posterior.values) * 12
                vol_ann = np.sqrt(np.dot(w.T, np.dot(cov_lw.values, w))) * np.sqrt(12)
                sharpe = (ret_ann - rf_rate) / vol_ann if vol_ann > 0 else 0
                
                w_series = pd.Series(w, index=returns_monthly.columns, name=name)
                weights_list.append(w_series)
                weights_dict[name] = w_series
                
                # Calcola pesi aggregati per display
                g_w = {g: w_series[assets].sum() for g, assets in detected_groups.items()}
                results.append({"Linea": name, "Rendimento": ret_ann, "Volatilità": vol_ann, "Sharpe": sharpe, 
                                "BONDS": g_w.get('BONDS', 0), "EQUITY": g_w.get('EQUITY', 0), "COMM": g_w.get('COMMODITIES', 0)})
                
            metrics_df = pd.DataFrame(results).set_index("Linea")
            with st.spinner("Analisi sensitività..."):
                sensitivity_stability = sensitivity_monte_carlo(bl_posterior.values, cov_lw.values, returns_monthly.columns, n_sim=200)
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 STRATEGIC", "📈 BACKTEST", "🧠 MODELS", "🔥 STRESS", "📑 REPORT"])
        
        with tab1:
            st.markdown("### 🎯 Asset Allocation Ottimale")
            c1, c2 = st.columns([2, 1])
            with c1:
                st.table(metrics_df.style.format({
                    "Rendimento": "{:.2%}", "Volatilità": "{:.2%}", "Sharpe": "{:.2f}",
                    "BONDS": "{:.1%}", "EQUITY": "{:.1%}", "COMM": "{:.1%}"
                }).background_gradient(cmap="Greens", subset=["Rendimento", "Sharpe"]))
            with c2:
                sensitivity_df = pd.DataFrame({'Asset': returns_monthly.columns, 'Stability Score': 1 - (sensitivity_stability / sensitivity_stability.max())}).sort_values('Stability Score', ascending=False)
                fig_sens = style_plotly_chart(px.bar(sensitivity_df, x='Asset', y='Stability Score', color='Stability Score', color_continuous_scale='RdYlGn'), "", 350)
                fig_sens.update_layout(xaxis_title=None, yaxis_title=None, coloraxis_showscale=False)
                st.plotly_chart(fig_sens, use_container_width=True)
            
            st.markdown("#### Frontiera Efficiente")
            fig_frontier = px.scatter(metrics_df, x="Volatilità", y="Rendimento", color="Sharpe", size=[15]*len(metrics_df), text=metrics_df.index, labels={"Volatilità": "Volatilità (Rischio)", "Rendimento": "Rendimento Atteso"}, color_continuous_scale="Bluered_r")
            sorted_metrics = metrics_df.sort_values("Volatilità")
            fig_frontier.add_trace(go.Scatter(x=sorted_metrics["Volatilità"], y=sorted_metrics["Rendimento"], mode='lines', line=dict(color='gray', dash='dash'), name='Frontiera'))
            fig_frontier = style_plotly_chart(fig_frontier, "", 500)
            fig_frontier.update_traces(textposition='top center', hovertemplate="<b>%{text}</b><br>Rendimento: %{y:.2%}<br>Volatilità: %{x:.2%}")
            fig_frontier.update_layout(xaxis=dict(tickformat=".1%"), yaxis=dict(tickformat=".1%"))
            st.plotly_chart(fig_frontier, use_container_width=True)
            
            c1, c2 = st.columns(2)
            with c1:
                plot_df = metrics_df[["BONDS", "EQUITY", "COMM"]].reset_index().melt(id_vars="Linea", var_name="Category", value_name="Weight")
                fig = style_plotly_chart(px.bar(plot_df, x="Linea", y="Weight", color="Category", text_auto=".1%"), "Macro Allocation", 400)
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                st.dataframe(pd.DataFrame(weights_list).T.style.format("{:.1%}").background_gradient(cmap="Greens", axis=None), height=400, use_container_width=True)

        with tab2:
            st.markdown("### 🕰️ Simulazione Walk-Forward")
            with st.spinner("Backtest in corso..."):
                wf_df, wf_err, avg_turnover = run_walk_forward_backtest(
                    returns_monthly, ff5_df, vol_ranges, group_limits_base, equity_limits,
                    user_min_weight, user_max_weight, tx_cost_bps, view_horizon, conf_level, stability_penalty,
                    asset_groups=detected_groups # Usa gruppi dinamici
                )
            if wf_err: st.error(wf_err)
            else:
                nav_df = (1 + wf_df).cumprod() * 100
                roll_max = nav_df.cummax()
                drawdown = (nav_df - roll_max) / roll_max
                max_dd = drawdown.min()
                n_years = (wf_df.index[-1] - wf_df.index[0]).days / 365.25
                cagr = (nav_df.iloc[-1] / 100) ** (1/n_years) - 1
                vol_real = wf_df.std() * np.sqrt(12)
                
                perf_table = pd.DataFrame({"Linea": list(wf_df.columns), "CAGR": [cagr[c] for c in wf_df.columns], "Volatilità": [vol_real[c] for c in wf_df.columns], "Max DD": [max_dd[c] for c in wf_df.columns]}).set_index("Linea")
                st.table(perf_table.style.format("{:.2%}").background_gradient(cmap="RdYlGn", subset=["CAGR"]))
                fig_bt = style_plotly_chart(px.line(nav_df, labels={"value": "Base 100", "variable": ""}), "Crescita Capitale", 500)
                st.plotly_chart(fig_bt, use_container_width=True)
                with st.expander("📉 Analisi Drawdown"):
                    st.plotly_chart(style_plotly_chart(px.line(drawdown), "Drawdown", 300), use_container_width=True)

        with tab3:
            st.markdown("### 🧠 Analisi Fattori")
            c1, c2 = st.columns([3, 2])
            with c1: st.dataframe(betas_df.style.background_gradient(cmap="RdBu", vmin=-1, vmax=1).format("{:.2f}"), use_container_width=True)
            with c2: 
                if use_dynamic_conf:
                    conf_df = pd.DataFrame({'Asset': asset_confidences.index, 'Confidence': asset_confidences.values}).sort_values('Confidence', ascending=False)
                    fig = style_plotly_chart(px.bar(conf_df, x='Asset', y='Confidence', color='Confidence', color_continuous_scale='Viridis'), "", 350)
                    fig.update_layout(xaxis_title=None, coloraxis_showscale=False)
                    st.plotly_chart(fig, use_container_width=True)

        with tab4:
            st.markdown("### 🔥 Stress Testing")
            if enable_stress_tests:
                with st.spinner("Stress test..."):
                    stress_results = run_stress_tests(weights_dict, returns_monthly, detected_groups) # Passa gruppi dinamici
                if not stress_results.empty:
                    stress_pivot = stress_results.pivot_table(index='Scenario', columns='Linea', values='Rendimento Totale', aggfunc='mean')
                    st.dataframe(stress_pivot.style.format("{:.2%}").background_gradient(cmap="RdYlGn", vmin=-0.3, vmax=0.1), use_container_width=True)
                    fig_stress = style_plotly_chart(px.bar(stress_results, x='Scenario', y='Rendimento Totale', color='Linea', barmode='group'), "Performance Crisi", 450)
                    st.plotly_chart(fig_stress, use_container_width=True)
            else: st.info("Abilita Stress Tests nella sidebar.")

        with tab5:
            st.markdown("### 📑 Report")
            report_data = generate_institutional_report({'strategic_allocation': {'lines': results, 'weights_df': pd.DataFrame(weights_list).T}, 'risk_metrics': perf_table if 'perf_table' in locals() else {}})
            exec_summary = pd.DataFrame([report_data.get('Executive_Summary', {})]).T
            exec_summary.columns = ['Valore']
            st.table(exec_summary)

    else: st.warning("Errore fattori FF5.")
else:
    st.markdown("""<div style='text-align: center; padding: 50px;'><h2 style='color: #000000;'>Benvenuto in Institutional Portfolio System</h2><p style='color: #374151;'>Carica un file CSV con serie storiche di <b>qualsiasi asset</b> per iniziare.</p></div>""", unsafe_allow_html=True)
