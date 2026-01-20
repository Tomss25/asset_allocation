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
# CONFIGURAZIONE PAGINA E CSS AVANZATO (LIGHT MODE)
# ---------------------------------------------------------
st.set_page_config(
    layout="wide", 
    page_title="Institutional Portfolio System | Pro",
    page_icon="🏛️",
    initial_sidebar_state="expanded"
)



# --- PASSWORD PROTECTION ---
def check_password():
    """Ritorna True se l'utente ha inserito la password corretta."""
    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False

    if st.session_state["password_correct"]:
        return True

    st.markdown("""
    <style>
        .stTextInput > div > div > input { text-align: center; }
    </style>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.write("")
        st.markdown("<h2 style='text-align: center;'>🔒 Accesso Istituzionale</h2>", unsafe_allow_html=True)
        pwd = st.text_input("Inserisci Password", type="password", key="password_input")
        
        if pwd:
            # Cerca la password nei "Secrets" di Streamlit Cloud
            if pwd == st.secrets["PASSWORD"]:
                st.session_state["password_correct"] = True
                st.rerun()
            else:
                st.error("⛔ Password Errata")
    return False

if not check_password():
    st.stop() # 🛑 BLOCCA L'ESECUZIONE SE PASSWORD SBAGLIATA

# ... (qui inizia st.markdown con il CSS e il resto del codice) ...

# CSS PERSONALIZZATO LIGHT MODE (Alto Contrasto)
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
# LOGICA FUNZIONALE (INVARIATA)
# ---------------------------------------------------------

# LISTA UFFICIALE DEI 15 ASSET
OFFICIAL_ASSETS = [
    "MON EU", "EU GOV ST", "EU CORP ST", "EU GOV", "EU CORP",
    "GL GOV", "GL CORP", "EM BOND", "EU HY", "US HY",
    "AZ EU", "AZ US", "AZ PAC", "AZ EM", "COMM"
]

# MAPPING CATEGORIE (MACRO ASSET CLASS)
ASSET_GROUPS = {
    "BONDS": ["MON EU", "EU GOV ST", "EU CORP ST", "EU GOV", "EU CORP", 
              "GL GOV", "GL CORP", "EM BOND", "EU HY", "US HY"],
    "EQUITY": ["AZ EU", "AZ US", "AZ PAC", "AZ EM"],
    "COMMODITIES": ["COMM"]
}

# SCORE LIQUIDITÀ PER ASSET (1=illiquido, 10=liquido)
LIQUIDITY_SCORES = {
    "MON EU": 9, "EU GOV ST": 8, "EU CORP ST": 7, "EU GOV": 9, "EU CORP": 7,
    "GL GOV": 8, "GL CORP": 7, "EM BOND": 6, "EU HY": 5, "US HY": 5,
    "AZ EU": 8, "AZ US": 9, "AZ PAC": 7, "AZ EM": 6, "COMM": 8
}

# SCENARI DI STRESS PRE-DEFINITI
STRESS_SCENARIOS = {
    "Lehman 2008": {
        "start_date": "2008-09-01",
        "end_date": "2009-02-28",
        "description": "Crisi finanziaria globale",
        "shock_multiplier": {
            "BONDS": 0.8,  # Riduzione rendimenti
            "EQUITY": 0.4, # Crollo azionario
            "COMMODITIES": 0.6
        }
    },
    "COVID-19 2020": {
        "start_date": "2020-02-01",
        "end_date": "2020-04-30",
        "description": "Pandemia globale",
        "shock_multiplier": {
            "BONDS": 1.2,  # Flight to quality
            "EQUITY": 0.5, # Crollo iniziale
            "COMMODITIES": 0.3
        }
    },
    "Inflation Shock 2022": {
        "start_date": "2022-01-01",
        "end_date": "2022-10-31",
        "description": "Picco inflazione e rialzo tassi",
        "shock_multiplier": {
            "BONDS": 0.6,  # Crollo bond
            "EQUITY": 0.8, # Correzione
            "COMMODITIES": 1.1
        }
    }
}

class TransactionCostModel:
    """Modello avanzato costi transazione"""
    def __init__(self, liquidity_scores, current_aum=100000000):
        self.liquidity_scores = liquidity_scores
        self.current_aum = current_aum
        self.base_cost = 0.0010  # 10 bps base
        
    def estimate_cost(self, turnover_vector, asset_names, market_volatility=0.15):
        """
        Costi = CostoFisso + ImpactoMercato + CostoOpportunità
        """
        # Base cost
        total_cost = np.full_like(turnover_vector, self.base_cost)
        
        # Market impact (inversamente proporzionale a liquidità)
        for i, asset in enumerate(asset_names):
            liquidity = self.liquidity_scores.get(asset, 5)
            market_impact = turnover_vector[i] * (1 / liquidity) * market_volatility * 0.3
            total_cost[i] += market_impact
        
        # Opportunity cost per grosse dimensioni
        large_trade_mask = turnover_vector > 0.05
        total_cost[large_trade_mask] += 0.002
        
        return total_cost

class DynamicConfidenceCalculator:
    """Calcola confidenza dinamica basata su qualità modello"""
    @staticmethod
    def calculate(ff5_models, min_conf=0.1, max_conf=1.0):
        confidences = {}
        for asset, model in ff5_models.items():
            # R² normalizzato (0-1)
            r2_norm = min(1.0, model.rsquared * 1.5)
            
            # t-stat media (valore assoluto)
            if hasattr(model, 'tvalues'):
                t_stats = model.tvalues.abs().mean()
                t_norm = min(1.0, t_stats / 2)
            else:
                t_norm = 0.5
            
            # Stabilità parametri (se disponibile)
            stability_score = 0.7  # Placeholder per analisi rolling
            
            # Confidenza combinata
            conf = 0.4 * r2_norm + 0.3 * t_norm + 0.3 * stability_score
            confidences[asset] = np.clip(conf, min_conf, max_conf)
        
        return pd.Series(confidences)

def handle_missing_returns(returns, max_consecutive_na=3):
    """
    Gestione intelligente missing values
    """
    returns_filled = returns.copy()
    
    # 1. Forward fill per buchi consecutivi limitati
    returns_filled = returns_filled.ffill(limit=max_consecutive_na)
    
    # 2. Imputazione per gruppo asset
    for asset in returns.columns:
        if returns_filled[asset].isna().any():
            # Trova gruppo dell'asset
            for group, assets in ASSET_GROUPS.items():
                if asset in assets:
                    # Prendi altri asset dello stesso gruppo
                    group_assets = [a for a in assets if a in returns.columns and a != asset]
                    if len(group_assets) > 0:
                        # Imputa con media del gruppo
                        group_mean = returns_filled[group_assets].mean(axis=1)
                        returns_filled[asset] = returns_filled[asset].fillna(group_mean)
                    break
    
    # 3. Se ancora NaN, usa 0 (ultima risorsa)
    returns_filled = returns_filled.fillna(0)
    
    return returns_filled

def winsorize_returns(returns, lower_percentile=1, upper_percentile=99):
    """
    Taglia outlier preservando struttura distribuzione
    """
    returns_winsorized = returns.copy()
    
    for asset in returns.columns:
        lower_bound = np.percentile(returns[asset].dropna(), lower_percentile)
        upper_bound = np.percentile(returns[asset].dropna(), upper_percentile)
        returns_winsorized[asset] = returns[asset].clip(lower_bound, upper_bound)
    
    return returns_winsorized

def detect_regime_shifts(returns, window=24):
    """
    Identifica cambiamenti di regime di mercato
    """
    rolling_vol = returns.rolling(window).std()
    vol_shock = (rolling_vol / rolling_vol.shift(window) - 1) > 1.0
    
    # Calcola correlazioni rolling
    rolling_corr = returns.rolling(window).corr(returns.iloc[:, 0]).iloc[:, 0]
    corr_shock = rolling_corr.diff().abs() > 0.3
    
    regime_df = pd.DataFrame({
        'volatility_shock': vol_shock.mean(axis=1),
        'correlation_break': corr_shock,
        'high_vol_regime': rolling_vol.mean(axis=1) > rolling_vol.mean(axis=1).quantile(0.8)
    }, index=returns.index)
    
    return regime_df

def validate_inputs(returns, min_months=24, max_missing=0.3):
    """
    Validazione robusta dei dati in input
    """
    validation_issues = []
    
    # 1. Lunghezza minima
    if len(returns) < min_months:
        validation_issues.append(f"Dati insufficienti: {len(returns)} mesi (< {min_months})")
    
    # 2. Missing values eccessivi
    missing_pct = returns.isna().sum() / len(returns)
    high_missing = missing_pct[missing_pct > max_missing]
    if len(high_missing) > 0:
        validation_issues.append(f"Asset con >{max_missing:.0%} dati mancanti: {list(high_missing.index)}")
    
    # 3. Check correlazioni estreme
    corr_matrix = returns.corr()
    extreme_pairs = []
    for i in range(len(corr_matrix)):
        for j in range(i+1, len(corr_matrix)):
            if abs(corr_matrix.iloc[i, j]) > 0.95:
                extreme_pairs.append(f"{corr_matrix.index[i]}-{corr_matrix.columns[j]}: {corr_matrix.iloc[i, j]:.2f}")
    
    if extreme_pairs:
        validation_issues.append(f"Correlazioni estreme (>0.95): {extreme_pairs[:3]}")  # Mostra prime 3
    
    # 4. Check varianza zero o quasi
    zero_var_assets = returns.var()[returns.var() < 1e-6].index.tolist()
    if zero_var_assets:
        validation_issues.append(f"Asset con varianza quasi zero: {zero_var_assets}")
    
    return validation_issues

def process_uploaded_data(uploaded_file):
    """Versione migliorata con validazione e gestione dati"""
    try:
        uploaded_file.seek(0)
        
        # Prova diversi separatori
        for sep in [';', ',', '\t']:
            try:
                df = pd.read_csv(uploaded_file, sep=sep, index_col=0, parse_dates=True, dayfirst=True, encoding='utf-8')
                if df.shape[1] >= 2:
                    break
            except:
                uploaded_file.seek(0)
                continue
        
        # Pulizia dati
        for col in df.columns:
            if df[col].dtype == object: 
                df[col] = df[col].astype(str).str.replace('.', '', regex=False)
                df[col] = df[col].str.replace(',', '.', regex=False)
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Resample mensile
        df_monthly = df.resample('ME').last() 
        
        # Calcola returns con gestione missing migliorata
        returns = df_monthly.pct_change()
        
        # Gestione missing values avanzata
        returns = handle_missing_returns(returns)
        
        # Winsorizzazione per outlier
        returns = winsorize_returns(returns)
        
        return returns, df_monthly

    except Exception as e:
        return None, str(e)

@st.cache_data(ttl=3600, show_spinner="Downloading FF5 factors...")
def download_ff5_factors():
    """Versione originale mantenuta"""
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
    """FF5 dati fino a cutoff date (per evitare look-ahead bias)"""
    ff5_full = download_ff5_factors()
    if ff5_full is not None:
        return ff5_full.loc[:cutoff_date]
    return None

def calculate_ff5_views(returns, ff5_data, window=60, return_models=False):
    """
    Versione migliorata che ritorna anche modelli per confidence
    """
    returns_clean = returns.fillna(0)
    aligned = pd.concat([returns_clean, ff5_data], axis=1, join='inner').dropna()
    
    if len(aligned) < 24: 
        return None, None, "Dati insufficienti (Min 24 mesi).", None
    
    data_window = aligned.iloc[-window:] if len(aligned) > window else aligned
    
    X = sm.add_constant(data_window[['MKT_RF','SMB','HML','RMW','CMA']])
    rf_mean = data_window['RF'].mean()
    factors_mean = data_window[['MKT_RF','SMB','HML','RMW','CMA']].mean()
    
    views = {}
    betas_dict = {}
    models_dict = {}
    
    for asset in returns.columns:
        try:
            model = sm.OLS(data_window[asset] - data_window['RF'], X).fit()
            exp_ret = rf_mean + model.params['const'] + (model.params[['MKT_RF','SMB','HML','RMW','CMA']] * factors_mean).sum()
            views[asset] = exp_ret
            betas_dict[asset] = model.params[['MKT_RF','SMB','HML','RMW','CMA']]
            models_dict[asset] = model
        except:
            # Fallback a media semplice se regressione fallisce
            views[asset] = data_window[asset].mean()
            betas_dict[asset] = pd.Series(0, index=['MKT_RF','SMB','HML','RMW','CMA'])
            models_dict[asset] = None
    
    if return_models:
        return pd.Series(views), pd.DataFrame(betas_dict).T, None, models_dict
    else:
        return pd.Series(views), pd.DataFrame(betas_dict).T, None

def robust_covariance_lw(returns):
    """Versione originale mantenuta"""
    lw = LedoitWolf()
    lw.fit(returns.fillna(0))
    return pd.DataFrame(lw.covariance_, index=returns.columns, columns=returns.columns)

def black_litterman_solver(cov_matrix, market_prior, ff5_views, tau=0.05, conf_id=0.5, 
                           asset_confidences=None):
    """
    Versione migliorata con confidence per asset
    """
    P = np.eye(len(cov_matrix))
    Q = ff5_views.values.reshape(-1, 1)
    Pi = market_prior.values.reshape(-1, 1)
    cov_values = cov_matrix.values
    
    # Omega con confidence per asset
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

def calculate_stress_metrics(weights, historical_returns):
    """Versione originale mantenuta"""
    port_history = historical_returns.fillna(0).dot(weights)
    wealth_index = (1 + port_history).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdown = (wealth_index - previous_peaks) / previous_peaks
    max_dd = drawdown.min()
    try:
        yearly_rets = (1 + port_history).resample('YE').prod() - 1
    except:
        yearly_rets = (1 + port_history).resample('Y').prod() - 1
    worst_year = yearly_rets.min()
    var_95 = np.percentile(port_history, 5) 
    cvar_95 = port_history[port_history <= var_95].mean()
    return max_dd, worst_year, var_95, cvar_95

def run_stress_tests(weights_dict, historical_returns, scenarios=STRESS_SCENARIOS):
    """
    Stress testing parametrico su scenari predefiniti
    """
    stress_results = []
    
    for scenario_name, scenario in scenarios.items():
        try:
            # Estrai periodo scenario
            start_date = pd.to_datetime(scenario['start_date'])
            end_date = pd.to_datetime(scenario['end_date'])
            
            scenario_returns = historical_returns.loc[start_date:end_date].copy()
            
            if len(scenario_returns) == 0:
                continue
            
            # Applica shock per gruppo
            for group, assets in ASSET_GROUPS.items():
                multiplier = scenario.get('shock_multiplier', {}).get(group, 1.0)
                group_cols = [col for col in scenario_returns.columns if col in assets]
                if len(group_cols) > 0:
                    scenario_returns[group_cols] *= multiplier
            
            # Calcola performance per ogni linea
            for line_name, weights in weights_dict.items():
                if isinstance(weights, pd.Series):
                    w = weights.values
                else:
                    w = weights
                
                # Performance durante lo scenario
                scenario_perf = scenario_returns.dot(w)
                total_return = (1 + scenario_perf).prod() - 1
                
                # Drawdown massimo durante scenario
                wealth = (1 + scenario_perf).cumprod()
                drawdown = (wealth / wealth.cummax() - 1).min()
                
                stress_results.append({
                    'Scenario': scenario_name,
                    'Linea': line_name,
                    'Durata (mesi)': len(scenario_returns),
                    'Rendimento Totale': total_return,
                    'Max Drawdown': drawdown,
                    'Descrizione': scenario['description']
                })
                
        except Exception as e:
            st.warning(f"Errore nello scenario {scenario_name}: {str(e)}")
    
    return pd.DataFrame(stress_results)

def optimize_line(mu, cov, asset_names, target_vol_min=None, target_vol_max=None, risk_free=0.0, 
                  min_weight=0.0, max_weight=1.0, group_limits=None, stability_penalty=0.0,
                  previous_weights=None):
    """
    Versione migliorata con stability penalty
    """
    num_assets = len(mu)
    
    def objective(weights, mu, prev_w):
        # Return negativo (da massimizzare)
        base_return = -np.sum(mu * weights)
        
        # Penalità per turnover se previous_weights fornito
        turnover_penalty = 0
        if prev_w is not None and stability_penalty > 0:
            turnover = np.sum(np.abs(weights - prev_w))
            turnover_penalty = stability_penalty * turnover
        
        return base_return + turnover_penalty
    
    args = (mu, previous_weights) if previous_weights is not None else (mu, None)
    
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
    
    # Vincoli volatilità
    if target_vol_max:
        constraints.append({'type': 'ineq', 'fun': lambda x: target_vol_max - np.sqrt(np.dot(x.T, np.dot(cov, x))) * np.sqrt(12)})
    if target_vol_min:
        constraints.append({'type': 'ineq', 'fun': lambda x: np.sqrt(np.dot(x.T, np.dot(cov, x))) * np.sqrt(12) - target_vol_min})
    
    # Vincoli gruppo
    if group_limits:
        for group, limit_max in group_limits.items():
            group_assets = ASSET_GROUPS.get(group, [])
            indices = [i for i, name in enumerate(asset_names) if name in group_assets]
            if indices:
                constraints.append({'type': 'ineq', 'fun': lambda x, idx=indices, l=limit_max: l - np.sum(x[idx])})
    
    bounds = tuple((min_weight, max_weight) for _ in range(num_assets))
    init_guess = num_assets * [1. / num_assets,]
    
    try:
        result = minimize(objective, init_guess, args=args, method='SLSQP', 
                         bounds=bounds, constraints=constraints,
                         options={'maxiter': 1000, 'ftol': 1e-9})
        return result.x
    except Exception as e:
        st.warning(f"Ottimizzazione fallita: {e}, usando pesi equal weight")
        return np.full(num_assets, 1.0/num_assets)

def sensitivity_monte_carlo(mu_base, cov_base, asset_names, n_sim=500):
    """
    Sensitivity analysis via Monte Carlo
    """
    allocations = []
    
    for sim in range(n_sim):
        # Perturba expected returns
        mu_perturbed = mu_base * np.random.normal(1, 0.15, len(mu_base))
        
        # Perturba covarianza
        pert_matrix = np.random.normal(1, 0.1, cov_base.shape)
        cov_perturbed = cov_base * (pert_matrix + pert_matrix.T) / 2
        
        # Simmetrizza e rendi definita positiva
        cov_perturbed = (cov_perturbed + cov_perturbed.T) / 2
        min_eig = np.min(np.real(np.linalg.eigvals(cov_perturbed)))
        if min_eig < 0:
            cov_perturbed -= min_eig * np.eye(*cov_perturbed.shape)
        
        # Ottimizza
        try:
            w_opt = optimize_line(mu_perturbed, cov_perturbed, asset_names,
                                 min_weight=0.01, max_weight=0.3)
            allocations.append(w_opt)
        except:
            continue
    
    if allocations:
        allocations_array = np.array(allocations)
        stability = pd.Series(allocations_array.std(axis=0), index=asset_names)
        return stability
    else:
        return pd.Series(0, index=asset_names)

def run_walk_forward_backtest(returns, ff5_full, vol_ranges, group_limits_base, 
                             equity_limits_per_line, min_w, max_w, tx_cost, 
                             view_window, conf_level, stability_penalty=0.01):
    """
    Versione migliorata con:
    1. No look-ahead bias per FF5
    2. Stability penalty
    3. Transaction cost migliorato
    """
    start_idx = max(36, view_window)
    returns = returns.fillna(0)
    
    if len(returns) <= start_idx: 
        return None, "Storia troppo breve per Walk-Forward.", None
    
    # Valida dati
    validation = validate_inputs(returns.iloc[:start_idx])
    if validation:
        st.warning(f"Warning validazione: {validation[:2]}")
    
    rebalance_dates = returns.index[start_idx:][::12] 
    wf_results = {f"Linea {i+1}": [] for i in range(6)}
    current_weights = {f"Linea {i+1}": np.zeros(len(returns.columns)) for i in range(6)}
    last_rebal_weights = current_weights.copy()
    turnover_history = {f"Linea {i+1}": [] for i in range(6)}
    
    # Inizializza transaction cost model
    tx_model = TransactionCostModel(LIQUIDITY_SCORES)
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for t_idx in range(start_idx, len(returns)):
        curr_date = returns.index[t_idx]
        
        # Update progress
        progress = (t_idx - start_idx + 1) / (len(returns) - start_idx)
        progress_bar.progress(progress)
        
        if curr_date in rebalance_dates:
            status_text.text(f"Rebalancing: {curr_date.date()}...")
            past_returns = returns.iloc[:t_idx]
            
            # FF5 solo fino a data disponibile (no look-ahead)
            ff5_available = get_ff5_data_cutoff(past_returns.index[-1])
            
            if ff5_available is not None and len(ff5_available) >= 24:
                views_t, _, _, models_t = calculate_ff5_views(past_returns, ff5_available, 
                                                            window=view_window, return_models=True)
                
                if views_t is not None:
                    # Calcola confidence dinamica
                    confidence_calc = DynamicConfidenceCalculator()
                    asset_confidences = confidence_calc.calculate(models_t)
                    
                    cov_t = robust_covariance_lw(past_returns)
                    prior_t = past_returns.mean()
                    
                    # Black-Litterman con confidence per asset
                    bl_post_t = black_litterman_solver(cov_t, prior_t, views_t, 
                                                      conf_id=conf_level,
                                                      asset_confidences=asset_confidences)
                    
                    rf_t = ff5_available['RF'].iloc[-1] * 12 if len(ff5_available) > 0 else 0.02
                    
                    for i, (v_min, v_max) in enumerate(vol_ranges):
                        line_name = f"Linea {i+1}"
                        curr_eq_lim = equity_limits_per_line.get(line_name, 1.0)
                        curr_groups = group_limits_base.copy()
                        curr_groups["EQUITY"] = curr_eq_lim
                        
                        # Ottimizza con stability penalty
                        new_w = optimize_line(bl_post_t.values, cov_t.values, returns.columns,
                                            target_vol_min=v_min, target_vol_max=v_max,
                                            risk_free=rf_t/12, min_weight=min_w, max_weight=max_w,
                                            group_limits=curr_groups, 
                                            stability_penalty=stability_penalty,
                                            previous_weights=current_weights[line_name])
                        
                        current_weights[line_name] = new_w
        
        month_ret_vector = returns.iloc[t_idx].values
        
        for line_name in wf_results.keys():
            w_start = current_weights[line_name]
            gross_ret = np.dot(w_start, month_ret_vector)
            
            # Calcola costi transazione avanzati
            cost = 0.0
            if curr_date in rebalance_dates:
                turnover = np.sum(np.abs(w_start - last_rebal_weights[line_name]))
                turnover_history[line_name].append(turnover)
                
                # Transaction cost avanzato
                if t_idx > 0:
                    market_vol = returns.iloc[t_idx-view_window:t_idx].std().mean() * np.sqrt(12)
                else:
                    market_vol = 0.15
                
                cost_vector = tx_model.estimate_cost(
                    np.abs(w_start - last_rebal_weights[line_name]),
                    returns.columns,
                    market_vol
                )
                cost = np.sum(cost_vector)
                
                last_rebal_weights[line_name] = w_start
            
            net_ret = gross_ret - cost
            wf_results[line_name].append(net_ret)
    
    progress_bar.empty()
    status_text.empty()
    
    dates = returns.index[start_idx:]
    min_len = min(len(dates), len(list(wf_results.values())[0]))
    df_res = pd.DataFrame({k: v[:min_len] for k,v in wf_results.items()}, index=dates[:min_len])
    
    # Calcola turnover medio
    avg_turnover = {k: np.mean(v) if v else 0 for k, v in turnover_history.items()}
    
    return df_res, None, avg_turnover

def generate_institutional_report(results_dict, template_type="executive"):
    """
    Genera report istituzionale in formato DataFrame o HTML
    """
    report_data = {}
    
    # Executive Summary
    if 'strategic_allocation' in results_dict:
        strat = results_dict['strategic_allocation']
        report_data['Executive_Summary'] = {
            'Data_Analisi': datetime.now().strftime('%Y-%m-%d'),
            'Numero_Asset': len(strat.get('weights', [])),
            'Periodo_Analisi': f"{results_dict.get('start_date', 'N/A')} - {results_dict.get('end_date', 'N/A')}",
            'Sharpe_Ratio_Medio': np.mean([l.get('Sharpe', 0) for l in strat.get('lines', [])]),
            'Max_Drawdown_Peggiore': np.min([l.get('Max_DD', 0) for l in strat.get('lines', [])])
        }
    
    # Risk Metrics
    if 'risk_metrics' in results_dict:
        report_data['Risk_Metrics'] = results_dict['risk_metrics']
    
    # Strategic Allocation
    if 'strategic_allocation' in results_dict:
        report_data['Strategic_Allocation'] = results_dict['strategic_allocation'].get('weights_df', pd.DataFrame())
    
    # Stress Test Results
    if 'stress_tests' in results_dict:
        report_data['Stress_Tests'] = results_dict['stress_tests']
    
    # Sensitivity Analysis
    if 'sensitivity' in results_dict:
        report_data['Sensitivity_Analysis'] = {
            'Stability_Scores': results_dict['sensitivity'].get('stability', {}),
            'Key_Drivers': results_dict['sensitivity'].get('drivers', [])
        }
    
    return report_data

def check_portfolio_alerts(weights_dict, market_conditions=None):
    """
    Sistema di alert per portafoglio
    """
    alerts = []
    
    for line_name, weights in weights_dict.items():
        if isinstance(weights, pd.Series):
            w = weights.values
            asset_names = weights.index
        else:
            w = weights
            asset_names = range(len(w))
        
        # Alert concentrazione
        max_weight = np.max(w)
        max_asset = asset_names[np.argmax(w)]
        if max_weight > 0.25:
            alerts.append({
                'Linea': line_name,
                'Tipo': 'Concentrazione',
                'Messaggio': f"{max_asset}: {max_weight:.1%} (>25%)",
                'Severità': 'High'
            })
        
        # Alert diversificazione
        effective_n = 1 / np.sum(w**2)
        if effective_n < 5:
            alerts.append({
                'Linea': line_name,
                'Tipo': 'Diversificazione',
                'Messaggio': f"N effettivo: {effective_n:.1f} (<5)",
                'Severità': 'Medium'
            })
        
        # Alert rischio cambio (se applicabile)
        em_assets = [name for name in asset_names if 'EM' in str(name)]
        em_exposure = np.sum([w[i] for i, name in enumerate(asset_names) if name in em_assets])
        if em_exposure > 0.15:
            alerts.append({
                'Linea': line_name,
                'Tipo': 'Rischio EM',
                'Messaggio': f"Exposure EM: {em_exposure:.1%} (>15%)",
                'Severità': 'Medium'
            })
    
    return alerts

# ---------------------------------------------------------
# FUNZIONE DI AIUTO PER LO STILE DEI GRAFICI (LIGHT MODE)
# ---------------------------------------------------------
def style_plotly_chart(fig, title="", height=None):
    """Applica il tema istituzionale 'Light Mode' ai grafici Plotly"""
    
    # Palette professionale per sfondo chiaro (Colori più scuri e saturi)
    institutional_palette = [
        '#1e40af', # Blu Scuro (Primary)
        '#047857', # Verde Smeraldo (Positive)
        '#b91c1c', # Rosso Rubino (Negative)
        '#b45309', # Ambra Scuro (Highlight)
        '#4b5563', # Grigio Antracite (Neutral)
        '#7c3aed', # Viola Reale
        '#0891b2', # Ciano Scuro
        '#be185d'  # Magenta Scuro
    ]
    
    fig.update_layout(
        template="plotly_white", # Base bianca
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(243,244,246,0.5)', # Grigio chiarissimo per area plot
        font=dict(family="Roboto, sans-serif", size=12, color="#000000"), # Testo NERO
        title=dict(
            text=f"<b>{title}</b>",
            font=dict(size=20, color="#111827"), # Titolo quasi nero
            x=0.05,
            xanchor='left'
        ),
        colorway=institutional_palette,
        margin=dict(l=40, r=40, t=60, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='#e5e7eb',
            borderwidth=1,
            font=dict(color="#374151")
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor='#e5e7eb', # Griglia grigio chiaro
            zeroline=True,
            zerolinecolor='#9ca3af',
            tickfont=dict(color='#374151'),
            title_font=dict(color='#111827')
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='#e5e7eb',
            zeroline=True,
            zerolinecolor='#9ca3af',
            tickfont=dict(color='#374151'),
            title_font=dict(color='#111827')
        )
    )
    if height:
        fig.update_layout(height=height)
    return fig

# ---------------------------------------------------------
# UI APP STREAMLIT MIGLIORATA
# ---------------------------------------------------------

st.title("🏛️ Institutional Portfolio System (Pro Suite)")
st.markdown("---")

# Sidebar avanzata
st.sidebar.header("⚙️ Configurazione Avanzata")

# Configurazione base
horizon_labels = {36: "36 Mesi (Tattico)", 60: "60 Mesi (Ciclo)", 
                  120: "120 Mesi (Strutturale)", 180: "180 Mesi (Secolare)"}
view_horizon = st.sidebar.selectbox("Orizzonte Views", [36, 60, 120, 180], 
                                    index=1, format_func=lambda x: horizon_labels.get(x, str(x)))

# Dynamic confidence toggle
use_dynamic_conf = st.sidebar.checkbox("Usa Confidence Dinamica", value=True,
                                       help="Confidenza basata su R² e t-stat dei modelli FF5")

if use_dynamic_conf:
    conf_level = st.sidebar.slider("Confidenza Base", 0.0, 1.0, 0.5,
                                   help="Livello base di confidenza, aggiustato dinamicamente per asset")
else:
    conf_level = st.sidebar.slider("Confidenza Modello", 0.0, 1.0, 0.5)

st.sidebar.markdown("---")
st.sidebar.header("💸 Costi & Rischio Avanzati")

# Transaction cost avanzato
tx_cost_bps = st.sidebar.number_input("Costi Transazione Base (bps)", 0, 50, 10,
                                      help="Costo medio per ogni operazione.") / 10000

# Stability penalty
stability_penalty = st.sidebar.slider("Penalità Turnover", 0.0, 0.1, 0.01, 0.001,
                                      help="Penalizza cambiamenti bruschi nell'allocazione")

# Stress test toggle
enable_stress_tests = st.sidebar.checkbox("Attiva Stress Tests", value=True)

st.sidebar.markdown("---")
st.sidebar.header("⚖️ Micro-Vincoli")
col_s1, col_s2 = st.sidebar.columns(2)
with col_s1: 
    user_min_weight = st.number_input("Min % Asset", 0.0, 5.0, 0.0, 0.5) / 100
with col_s2: 
    user_max_weight = st.number_input("Max % Asset", 10.0, 100.0, 30.0, 5.0) / 100

st.sidebar.markdown("---")
st.sidebar.header("🌍 Macro-Vincoli")
max_bonds = st.sidebar.slider("Max BONDS", 0.0, 1.0, 0.80, 0.05)
max_comm = st.sidebar.slider("Max COMMODITIES", 0.0, 1.0, 0.15, 0.05)

st.sidebar.markdown("---")
st.sidebar.header("🪜 Scalini Equity")
equity_limits = {}
default_limits = [0.10, 0.20, 0.40, 0.60, 0.80, 1.00]
for i in range(1, 7):
    line_key = f"Linea {i}"
    with st.sidebar.expander(f"Vincolo {line_key}"):
        limit = st.slider(f"Max Eq.", 0.0, 1.0, default_limits[i-1], 0.05, key=f"eq_lim_{i}")
        equity_limits[line_key] = limit

# Main interface
# Usiamo un container visuale per l'upload
with st.container():
    st.markdown("### 📂 Data Import")
    col_up_1, col_up_2 = st.columns([1, 2])
    with col_up_1:
        uploaded_file = st.file_uploader("Carica CSV (Weekly o Monthly)", type='csv')

if uploaded_file:
    # A. DATI CON VALIDAZIONE
    returns_monthly_raw, prices_monthly = process_uploaded_data(uploaded_file)
    if returns_monthly_raw is None: 
        st.error(f"Errore lettura: {prices_monthly}")
        st.stop()
    
    returns_monthly_raw.columns = returns_monthly_raw.columns.str.strip().str.upper()
    available_assets = [c for c in OFFICIAL_ASSETS if c in returns_monthly_raw.columns]
    if len(available_assets) < len(OFFICIAL_ASSETS):
        missing = set(OFFICIAL_ASSETS) - set(available_assets)
        st.warning(f"⚠️ Mancano questi asset nel CSV: {missing}")
    
    returns_monthly = returns_monthly_raw[available_assets]
    
    # Validazione dati
    with st.spinner("Validazione dati in corso..."):
        validation_issues = validate_inputs(returns_monthly)
        if validation_issues:
            with st.expander("⚠️ Issues di Validazione", expanded=True):
                for issue in validation_issues[:3]:  # Mostra prime 3
                    st.error(issue)
            if st.checkbox("Procedi nonostante gli warnings"):
                pass
            else:
                st.stop()
    
    # Regime detection
    regime_df = detect_regime_shifts(returns_monthly)
    
    with col_up_2:
        # KPI Cards per i dati
        c1, c2, c3 = st.columns(3)
        c1.metric("Storico Dati", f"{len(returns_monthly)} Mesi")
        c2.metric("Asset Class", f"{len(returns_monthly.columns)}")
        c3.metric("Data Inizio", f"{returns_monthly.index[0].strftime('%b %Y')}")
        
        st.success(f"✅ Dati caricati e validati con successo.")
        
        # Mostra regime detection se necessario
        if regime_df['high_vol_regime'].any():
            high_vol_periods = regime_df[regime_df['high_vol_regime']].index
            st.info(f"⚠️ Rilevati {len(high_vol_periods)} periodi di alta volatilità nei dati storici.")
    
    # B. FATTORI FF5
    ff5_df = download_ff5_factors()
    
    if ff5_df is not None:
        # C. CALCOLI STRATEGICI
        with st.spinner("Calcolo Asset Allocation Strategica (Oggi)..."):
            # Calcola views con modelli per confidence
            ff5_views, betas_df, err, ff5_models = calculate_ff5_views(
                returns_monthly, ff5_df, window=view_horizon, return_models=True
            )
            
            if err: 
                st.error(err)
                st.stop()
            
            cov_lw = robust_covariance_lw(returns_monthly)
            
            # Dynamic confidence se richiesto
            if use_dynamic_conf and ff5_models:
                confidence_calc = DynamicConfidenceCalculator()
                asset_confidences = confidence_calc.calculate(ff5_models)
            else:
                asset_confidences = None
            
            bl_posterior = black_litterman_solver(cov_lw, returns_monthly.mean(), 
                                                 ff5_views, conf_id=conf_level,
                                                 asset_confidences=asset_confidences)
            
            vol_ranges = [(0.000, 0.025), (0.025, 0.050), (0.050, 0.075), 
                          (0.075, 0.100), (0.100, 0.125), (0.125, 0.150)]
            line_labels = ["Linea 1", "Linea 2", "Linea 3", "Linea 4", "Linea 5", "Linea 6"]
            rf_rate = ff5_df['RF'].iloc[-1] * 12 if len(ff5_df) > 0 else 0.02
            
            results, weights_list, weights_dict = [], [], {}
            group_limits_base = {"BONDS": max_bonds, "COMMODITIES": max_comm}
            
            for name, (v_min, v_max) in zip(line_labels, vol_ranges):
                curr_max_equity = equity_limits.get(name, 1.0)
                curr_groups = group_limits_base.copy()
                curr_groups["EQUITY"] = curr_max_equity
                
                w = optimize_line(bl_posterior.values, cov_lw.values, returns_monthly.columns,
                                target_vol_min=v_min, target_vol_max=v_max, 
                                risk_free=rf_rate/12, min_weight=user_min_weight, 
                                max_weight=user_max_weight, group_limits=curr_groups,
                                stability_penalty=stability_penalty)
                
                ret_ann = np.sum(w * bl_posterior.values) * 12
                vol_ann = np.sqrt(np.dot(w.T, np.dot(cov_lw.values, w))) * np.sqrt(12)
                sharpe = (ret_ann - rf_rate) / vol_ann if vol_ann > 0 else 0
                
                w_series = pd.Series(w, index=returns_monthly.columns)
                w_series.name = name
                weights_list.append(w_series)
                weights_dict[name] = w_series
                
                group_w = {k: w_series[v].sum() for k, v in ASSET_GROUPS.items()}
                results.append({
                    "Linea": name, "Rendimento": ret_ann, "Volatilità": vol_ann, "Sharpe": sharpe,
                    "BONDS": group_w["BONDS"], "EQUITY": group_w["EQUITY"], "COMM": group_w["COMMODITIES"]
                })
                
            metrics_df = pd.DataFrame(results).set_index("Linea")
            
            # Sensitivity analysis
            with st.spinner("Analisi di sensitività in corso..."):
                sensitivity_stability = sensitivity_monte_carlo(
                    bl_posterior.values, cov_lw.values, returns_monthly.columns, n_sim=200
                )
        
        # F. VISUALIZZAZIONE MIGLIORATA
        # Separazione logica netta tra le sezioni
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 STRATEGIC ALLOCATION", 
            "📈 WALK-FORWARD ANALYSIS", 
            "🧠 FACTOR MODELS",
            "🔥 STRESS TESTS",
            "📑 REPORTING"
        ])
        
        # --- TAB 1: ALLOCAZIONE ---
        with tab1:
            st.markdown("### 🎯 Asset Allocation Ottimale (Current View)")
            
            # Alert system box
            alerts = check_portfolio_alerts(weights_dict)
            if alerts:
                with st.expander("🚨 Avvisi di Portafoglio (Concentrazione / Diversificazione)", expanded=True):
                    for alert in alerts[:5]:  # Mostra prime 5
                        if alert['Severità'] == 'High':
                            st.error(f"**{alert['Linea']}**: {alert['Messaggio']}")
                        else:
                            st.warning(f"**{alert['Linea']}**: {alert['Messaggio']}")
            
            col_met_1, col_met_2 = st.columns([2, 1])
            
            with col_met_1:
                # Tabella metriche stilizzata
                st.markdown("#### Metriche di Rischio/Rendimento")
                st.table(metrics_df.style.format({
                    "Rendimento": "{:.2%}", "Volatilità": "{:.2%}", "Sharpe": "{:.2f}",
                    "BONDS": "{:.1%}", "EQUITY": "{:.1%}", "COMM": "{:.1%}"
                }).background_gradient(cmap="Greens", subset=["Rendimento", "Sharpe"]))
            
            with col_met_2:
                # Sensitivity Chart
                st.markdown("#### Robustezza (Stability Score)")
                st.caption("Misura la resistenza dell'allocazione a shock nei parametri. Alto (Verde) = Asset Stabile.")
                sensitivity_df = pd.DataFrame({
                    'Asset': returns_monthly.columns,
                    'Stability Score': 1 - (sensitivity_stability / sensitivity_stability.max())
                }).sort_values('Stability Score', ascending=False)
                
                fig_sens = px.bar(sensitivity_df, x='Asset', y='Stability Score',
                                 color='Stability Score', color_continuous_scale='RdYlGn')
                fig_sens = style_plotly_chart(fig_sens, "", height=350)
                # Togli assi e label superflui per pulizia
                fig_sens.update_layout(xaxis_title=None, yaxis_title=None, coloraxis_showscale=False)
                st.plotly_chart(fig_sens, use_container_width=True)
            
            # -------------------------------------------------------------
            # NUOVO GRAFICO: FRONTIERA EFFICIENTE (PROPOSTA TARGET)
            # -------------------------------------------------------------
            st.markdown("---")
            st.markdown("#### Frontiera Efficiente: Proposta Target")
            
            # Creazione Scatter Plot Rischio-Rendimento
            fig_frontier = px.scatter(
                metrics_df, 
                x="Volatilità", 
                y="Rendimento",
                color="Sharpe",
                size=[15]*len(metrics_df), # Dimensione fissa dei punti
                text=metrics_df.index, # Etichetta (Linea 1, Linea 2...)
                title="Frontiera Efficiente (Rischio vs Rendimento Atteso)",
                labels={"Volatilità": "Volatilità Attesa (Rischio)", "Rendimento": "Rendimento Atteso"},
                color_continuous_scale="Bluered_r" # Rosso basso sharpe, Blu alto sharpe
            )
            
            # Aggiungi una linea che collega i punti (Interpolazione visiva della frontiera)
            # Ordiniamo per volatilità per disegnare la linea correttamente
            sorted_metrics = metrics_df.sort_values("Volatilità")
            fig_frontier.add_trace(go.Scatter(
                x=sorted_metrics["Volatilità"], 
                y=sorted_metrics["Rendimento"],
                mode='lines',
                line=dict(color='gray', dash='dash'),
                name='Frontiera'
            ))

            # Styling del grafico della frontiera
            fig_frontier = style_plotly_chart(fig_frontier, "Proposta Target: Rendimento vs Volatilità", height=500)
            
            # Miglioramenti tooltip e assi
            fig_frontier.update_traces(
                textposition='top center',
                hovertemplate="<b>%{text}</b><br>Rendimento: %{y:.2%}<br>Volatilità: %{x:.2%}<br>Sharpe: %{marker.color:.2f}"
            )
            fig_frontier.update_layout(
                xaxis=dict(tickformat=".1%"),
                yaxis=dict(tickformat=".1%")
            )
            
            st.plotly_chart(fig_frontier, use_container_width=True)
            st.markdown("---")
            # -------------------------------------------------------------
            
            c1, c2 = st.columns(2)
            with c1:
                # Stacked Bar Chart
                plot_df = metrics_df[["BONDS", "EQUITY", "COMM"]].reset_index().melt(
                    id_vars="Linea", var_name="Category", value_name="Weight"
                )
                fig = px.bar(plot_df, x="Linea", y="Weight", color="Category", 
                             text_auto=".1%")
                fig = style_plotly_chart(fig, "Macro Allocation per Linea", height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with c2:
                # Heatmap Pesi
                st.markdown("#### Dettaglio Pesi (Micro Allocation)")
                st.dataframe(pd.DataFrame(weights_list).T.style.format("{:.1%}").background_gradient(cmap="Greens", axis=None), 
                            height=400, use_container_width=True)
        
        # --- TAB 2: BACKTEST ---
        with tab2:
            st.markdown("### 🕰️ Simulazione Walk-Forward (Out-of-Sample)")
            
            # KPI Box per parametri backtest
            with st.container():
                cols = st.columns(4)
                cols[0].info(f"**No Look-Ahead**: Attivo")
                cols[1].info(f"**Transaction Cost**: Dinamico")
                cols[2].info(f"**Penalty Turnover**: {stability_penalty}")
                cols[3].info(f"**Dynamic Conf**: {'ON' if use_dynamic_conf else 'OFF'}")
            
            with st.spinner("Esecuzione simulazione storica avanzata..."):
                wf_df, wf_err, avg_turnover = run_walk_forward_backtest(
                    returns_monthly, ff5_df, vol_ranges, group_limits_base, equity_limits,
                    user_min_weight, user_max_weight, tx_cost_bps, view_horizon, 
                    conf_level, stability_penalty
                )
            
            if wf_err:
                st.error(wf_err)
            else:
                nav_df = (1 + wf_df).cumprod() * 100
                roll_max = nav_df.cummax()
                drawdown = (nav_df - roll_max) / roll_max
                max_dd = drawdown.min()
                
                n_years = (wf_df.index[-1] - wf_df.index[0]).days / 365.25
                cagr = (nav_df.iloc[-1] / 100) ** (1/n_years) - 1
                vol_real = wf_df.std() * np.sqrt(12)
                
                # Tabella Performance Realizzata
                st.markdown("#### Performance Realizzata (Net of Costs)")
                perf_table = pd.DataFrame({
                    "Linea": list(wf_df.columns),
                    "CAGR (Netto)": [cagr[col] for col in wf_df.columns],
                    "Volatilità": [vol_real[col] for col in wf_df.columns],
                    "Max Drawdown": [max_dd[col] for col in wf_df.columns],
                    "Turnover Medio": [avg_turnover.get(col, 0) for col in wf_df.columns]
                }).set_index("Linea")
                
                st.table(perf_table.style.format({
                    "CAGR (Netto)": "{:.2%}",
                    "Volatilità": "{:.2%}",
                    "Max Drawdown": "{:.2%}",
                    "Turnover Medio": "{:.2%}"
                }).background_gradient(cmap="RdYlGn", subset=["CAGR (Netto)"]))
                
                # Grafico NAV
                fig_bt = px.line(nav_df, labels={"value": "Valore (Base 100)", "variable": ""})
                fig_bt = style_plotly_chart(fig_bt, "Crescita del Capitale Investito", height=500)
                st.plotly_chart(fig_bt, use_container_width=True)
                
                # Grafico Drawdown
                with st.expander("📉 Analisi Drawdown"):
                    fig_dd = px.line(drawdown, labels={"value": "Drawdown %", "variable": ""})
                    fig_dd = style_plotly_chart(fig_dd, "Profondità Drawdown Storico", height=300)
                    st.plotly_chart(fig_dd, use_container_width=True)
                
                # Rendimenti Annuali
                st.markdown("#### Rendimenti Annuali")
                try:
                    yearly_ret = nav_df.resample('YE').last().pct_change().dropna()
                except:
                    yearly_ret = nav_df.resample('Y').last().pct_change().dropna()
                
                yearly_ret.index = yearly_ret.index.year
                st.dataframe(yearly_ret.style.format("{:.2%}").background_gradient(
                    cmap="RdYlGn", vmin=-0.15, vmax=0.15), use_container_width=True)
        
        # --- TAB 3: FATTORI ---
        with tab3:
            st.markdown("### 🧠 Analisi Fattoriale (Fama-French 5)")
            st.info(f"Modello calcolato su finestra mobile di {view_horizon} mesi.")
            
            c1, c2 = st.columns([3, 2])
            
            with c1:
                st.markdown("#### Sensibilità ai Fattori (Betas)")
                st.dataframe(betas_df.style.background_gradient(cmap="RdBu", vmin=-1, vmax=1).format("{:.2f}"), 
                            use_container_width=True)
            
            with c2:
                if use_dynamic_conf and asset_confidences is not None:
                    st.markdown("#### Affidabilità del Modello (Confidence)")
                    st.caption("Qualità statistica (R², t-stat) della regressione Fama-French. Alto = View attendibile.")
                    conf_df = pd.DataFrame({
                        'Asset': asset_confidences.index,
                        'Confidence': asset_confidences.values
                    }).sort_values('Confidence', ascending=False)
                    
                    fig_conf = px.bar(conf_df, x='Asset', y='Confidence', 
                                     color='Confidence', color_continuous_scale='Viridis')
                    fig_conf = style_plotly_chart(fig_conf, "", height=350)
                    fig_conf.update_layout(xaxis_title=None, coloraxis_showscale=False)
                    st.plotly_chart(fig_conf, use_container_width=True)
            
            st.markdown("#### Confronto Viste: Storico vs Modello")
            view_comparison = pd.DataFrame({
                "Storico (Prior)": returns_monthly.mean() * 12,
                "Fama-French (View)": ff5_views * 12,
                "Black-Litterman (Final)": bl_posterior * 12
            })
            
            # Grafico a barre raggruppate per confronto views
            fig_views = px.bar(view_comparison, barmode='group')
            fig_views = style_plotly_chart(fig_views, "Rendimenti Attesi Annualizzati", height=400)
            st.plotly_chart(fig_views, use_container_width=True)
        
        # --- TAB 4: STRESS TEST ---
        with tab4:
            st.markdown("### 🔥 Stress Testing Parametrico")
            
            if enable_stress_tests:
                with st.spinner("Simulazione scenari di crisi..."):
                    stress_results = run_stress_tests(weights_dict, returns_monthly)
                
                if not stress_results.empty:
                    # Pivot table per visualizzazione pulita
                    stress_pivot = stress_results.pivot_table(
                        index='Scenario', 
                        columns='Linea', 
                        values='Rendimento Totale',
                        aggfunc='mean'
                    )
                    
                    st.markdown("#### Impatto sul Rendimento Totale")
                    st.dataframe(stress_pivot.style.format("{:.2%}").background_gradient(
                        cmap="RdYlGn", vmin=-0.3, vmax=0.1), use_container_width=True)
                    
                    # Grafico Visuale
                    fig_stress = px.bar(stress_results, x='Scenario', y='Rendimento Totale',
                                       color='Linea', barmode='group',
                                       hover_data=['Descrizione', 'Durata (mesi)'])
                    fig_stress = style_plotly_chart(fig_stress, "Performance in Scenari di Crisi", height=450)
                    st.plotly_chart(fig_stress, use_container_width=True)
                    
                    # Dettagli
                    with st.expander("Dettagli Scenari"):
                        st.write(stress_results[['Scenario', 'Descrizione', 'Durata (mesi)']].drop_duplicates())
                else:
                    st.warning("Nessun dato disponibile per gli scenari di stress selezionati.")
            else:
                st.info("Abilita gli Stress Test dalla sidebar per visualizzare questa analisi.")
        
        # --- TAB 5: REPORTING ---
        with tab5:
            st.markdown("### 📑 Reportistica Istituzionale")
            
            # Genera dati report
            report_data = generate_institutional_report({
                'strategic_allocation': {
                    'lines': results,
                    'weights_df': pd.DataFrame(weights_list).T,
                    'weights_dict': weights_dict
                },
                'risk_metrics': perf_table if 'perf_table' in locals() else {},
                'stress_tests': stress_results.to_dict('records') if 'stress_results' in locals() else [],
                'sensitivity': {
                    'stability': sensitivity_stability.to_dict(),
                    'drivers': sensitivity_df.nlargest(5, 'Stability Score')['Asset'].tolist()
                },
                'start_date': returns_monthly.index[0].date(),
                'end_date': returns_monthly.index[-1].date()
            })
            
            col_rep_1, col_rep_2 = st.columns([1, 1])
            
            with col_rep_1:
                st.markdown("#### Executive Summary")
                exec_summary = pd.DataFrame([report_data.get('Executive_Summary', {})]).T
                exec_summary.columns = ['Valore']
                st.table(exec_summary)
            
            with col_rep_2:
                st.markdown("#### Export Data")
                st.info("I dati sottostanti sono pronti per l'esportazione in formati standard.")
                # Esempio bottone dummy
                st.button("📥 Scarica PDF (Mockup)", key="pdf_dl")
                st.button("📊 Scarica Excel (Raw Data)", key="xls_dl")
            
            st.markdown("#### Risk Metrics Overview")
            if 'Risk_Metrics' in report_data and not report_data['Risk_Metrics'].empty:
                st.dataframe(report_data['Risk_Metrics'], use_container_width=True)

    else: 
        st.warning("⚠️ Impossibile scaricare i fattori Fama-French. Verifica la connessione internet.")
else: 
    # Welcome Screen
    st.markdown("""
    <div style='text-align: center; padding: 50px;'>
        <h2 style='color: #000000;'>Benvenuto in Institutional Portfolio System</h2>
        <p style='color: #374151;'>Carica un file CSV con i dati di mercato per iniziare l'analisi.</p>
        <p style='font-size: 0.8em; color: #6b7280;'>Formato richiesto: Date in prima colonna, Prezzi/Indici nelle colonne successive.</p>
    </div>
    """, unsafe_allow_html=True)