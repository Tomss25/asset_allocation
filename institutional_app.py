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
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

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
    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False
    if st.session_state["password_correct"]:
        return True
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<h2 style='text-align: center;'>🔒 Accesso Istituzionale</h2>", unsafe_allow_html=True)
        pwd = st.text_input("Password", type="password", key="password_input")
        if pwd:
            # Fallback a "admin" se non configurato in secrets
            correct_pwd = st.secrets.get("PASSWORD", "admin") 
            if pwd == correct_pwd:
                st.session_state["password_correct"] = True
                st.rerun()
            else:
                st.error("⛔ Credenziali non valide")
    return False

if not check_password():
    st.stop()

# ---------------------------------------------------------
# CSS AVANZATO (LIGHT MODE)
# ---------------------------------------------------------
st.markdown("""
<style>
    .stApp { background-color: #f8f9fa; color: #212529; font-family: 'Roboto', sans-serif; }
    [data-testid="stSidebar"] { background-color: #ffffff; border-right: 1px solid #dee2e6; }
    h1, h2, h3 { color: #0f172a !important; font-weight: 700; }
    h1 { border-bottom: 2px solid #0f172a; padding-bottom: 15px; }
    .stButton > button { background-color: #2563eb; color: #ffffff; border-radius: 4px; font-weight: 600; }
    div[data-testid="metric-container"] { background-color: #ffffff; border: 1px solid #e9ecef; padding: 10px; border-radius: 6px; box-shadow: 0 1px 2px rgba(0,0,0,0.05); }
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] { background-color: #e9ecef; border-radius: 4px 4px 0 0; font-weight: 600; }
    .stTabs [aria-selected="true"] { background-color: #ffffff; color: #2563eb !important; border-top: 2px solid #2563eb; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 0. LOGICHE CORE & REGIME DETECTION
# ---------------------------------------------------------

def auto_classify_assets_initial(asset_names):
    """Classificazione euristica iniziale basata sul nome."""
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

def detect_market_regime(returns, equity_assets, lookback_trend=6, lookback_vol_short=3, lookback_vol_long=24):
    """
    Rileva il regime di mercato basandosi su Trend e Turbolenza degli asset Equity.
    Restituisce regime e metriche numeriche.
    """
    if not equity_assets:
        return "NEUTRAL", 0, 0
    
    # Slice temporale sicuro
    if len(returns) < lookback_vol_long:
        return "NEUTRAL", 0, 0

    # Indice sintetico equi-pesato degli asset equity
    eq_returns = returns[equity_assets].mean(axis=1)
    eq_index = (1 + eq_returns).cumprod()
    
    # 1. Calcolo Trend (SMA vs Prezzo attuale)
    sma_long = eq_index.rolling(window=12).mean().iloc[-1]
    sma_short = eq_index.rolling(window=3).mean().iloc[-1]
    
    # Trend Score: > 0 Trend positivo, < 0 Trend negativo
    trend_score = (sma_short / sma_long) - 1 
    
    # 2. Calcolo Turbolenza (Volatilità recente vs Storica)
    vol_short = eq_returns.rolling(window=lookback_vol_short).std().iloc[-1]
    vol_long = eq_returns.rolling(window=lookback_vol_long).std().iloc[-1]
    
    turbulence_score = vol_short / vol_long if vol_long > 0 else 1.0
    
    # 3. MAPPING DEI REGIMI
    if trend_score > 0 and turbulence_score < 1.1:
        regime = "ESPANSIONE" 
    elif trend_score > 0 and turbulence_score >= 1.1:
        regime = "INFLAZIONE / SURRISCALDAMENTO"
    elif trend_score <= 0 and turbulence_score >= 1.1:
        regime = "RECESSIONE"
    else:
        regime = "RALLENTAMENTO" # Trend negativo ma volatilità bassa
        
    return regime, trend_score, turbulence_score

def get_regime_constraints(regime, max_equity_base, max_bond_base):
    """
    Definisce come cambiano i vincoli per ogni regime.
    """
    constraints = {
        "equity_max": max_equity_base,
        "bond_min": 0.0,
        "comm_min": 0.0,
        "vol_mult": 1.0
    }
    
    if regime == "RECESSIONE":
        # Taglio Equity drastico, Rifugio in Bonds
        constraints["equity_max"] = max_equity_base * 0.5 
        constraints["bond_min"] = 0.30 
        constraints["vol_mult"] = 0.7 
        
    elif regime == "ESPANSIONE":
        # Aumenta Equity, accetta più vol
        constraints["equity_max"] = min(1.0, max_equity_base * 1.2)
        constraints["vol_mult"] = 1.1
        
    elif regime == "INFLAZIONE / SURRISCALDAMENTO":
        # Forza Asset Reali
        constraints["comm_min"] = 0.15 
        constraints["equity_max"] = max_equity_base * 0.9 
        
    elif regime == "RALLENTAMENTO":
        # Difensivo ma non panico
        constraints["equity_max"] = max_equity_base * 0.8
        constraints["bond_min"] = 0.15
        constraints["vol_mult"] = 0.9
        
    return constraints

# ---------------------------------------------------------
# DATA PROCESSING & MODELS
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
        
        # Winsorization (1% - 99%)
        lower = returns.quantile(0.01)
        upper = returns.quantile(0.99)
        returns = returns.clip(lower, upper, axis=1)
        return returns
    except Exception as e:
        return None

@st.cache_data(ttl=3600)
def download_ff5_factors():
    url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_CSV.zip"
    try:
        r = requests.get(url, timeout=10)
        z = ZipFile(BytesIO(r.content))
        csv_name = [f for f in z.namelist() if 'Factors' in f and f.endswith('.csv')][0]
        df = pd.read_csv(z.open(csv_name), skiprows=3)
        df = df.rename(columns={'Mkt-RF':'MKT_RF','SMB':'SMB','HML':'HML','RMW':'RMW','CMA':'CMA','RF':'RF'})
        df['Date'] = pd.to_datetime(df['Unnamed: 0'].astype(str), format='%Y%m', errors='coerce')
        df = df.dropna(subset=['Date']).set_index('Date').resample('ME').last()
        return df[['MKT_RF','SMB','HML','RMW','CMA','RF']].astype(float) / 100
    except:
        return None

def calculate_ff5_views(returns, ff5_data, window=60):
    aligned = pd.concat([returns, ff5_data], axis=1, join='inner').dropna()
    if len(aligned) < 24: return None, None, {}
    
    data = aligned.iloc[-window:]
    X = sm.add_constant(data[['MKT_RF','SMB','HML','RMW','CMA']])
    
    views = {}
    models = {}
    
    # Momentum Calculation (12-1 months)
    mom_scores = {}
    if len(returns) > 13:
        mom_raw = (1 + returns.iloc[-13:-1]).prod() - 1
        mom_scores = mom_raw.to_dict()
    
    for asset in returns.columns:
        try:
            model = sm.OLS(data[asset] - data['RF'], X).fit()
            models[asset] = model
            factors_mean = data[['MKT_RF','SMB','HML','RMW','CMA']].mean()
            # Rendimento atteso base Fama-French
            exp_ret = data['RF'].mean() + model.params['const'] + (model.params[['MKT_RF','SMB','HML','RMW','CMA']] * factors_mean).sum()
            # Momentum Tilt
            tilt = mom_scores.get(asset, 0) * 0.05
            views[asset] = exp_ret + tilt
        except:
            views[asset] = data[asset].mean()
            models[asset] = None
            
    return pd.Series(views), models

def robust_covariance(returns):
    lw = LedoitWolf()
    lw.fit(returns)
    return pd.DataFrame(lw.covariance_, index=returns.columns, columns=returns.columns)

def black_litterman(cov, prior, views, confidences, conf_level=0.5):
    tau = 0.05
    P = np.eye(len(cov))
    Q = views.values
    Pi = prior.values
    uncertainty = (1 - confidences.values) / (confidences.values + 1e-6)
    Omega = np.diag(np.diag(cov.values)) * tau * uncertainty
    
    inv_tau_cov = np.linalg.inv(tau * cov.values)
    inv_omega = np.linalg.inv(Omega + np.eye(len(Omega))*1e-9)
    
    M_left = np.linalg.inv(inv_tau_cov + inv_omega)
    M_right = np.dot(inv_tau_cov, Pi) + np.dot(inv_omega, Q)
    
    post_ret = np.dot(M_left, M_right)
    return pd.Series(post_ret, index=cov.columns)

def optimize_portfolio(mu, cov, target_vol, risk_free, min_w, max_w, group_constraints, groups):
    # FIX: mu deve essere una Series per avere .index
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

# ---------------------------------------------------------
# COMPARATIVE BACKTEST ENGINE
# ---------------------------------------------------------
def run_comparative_backtest(returns, ff5_full, groups, vol_target, max_eq_base, max_bond_base, min_w, max_w):
    """
    Esegue un backtest walk-forward confrontando Strategico vs Tattico.
    """
    start_idx = 36
    rebalance_freq = 3 # Trimestrale
    
    dates = returns.index[start_idx::rebalance_freq]
    
    # Init Tracking
    n_assets = len(returns.columns)
    wealth_strat = 100.0
    wealth_tact = 100.0
    
    ts_strat = [100.0]
    ts_tact = [100.0]
    ts_dates = [returns.index[start_idx-1]]
    regime_history = []
    
    progress_bar = st.progress(0)
    
    for i, date in enumerate(dates):
        progress = (i + 1) / len(dates)
        progress_bar.progress(progress)
        
        # 1. Historical Data Slice
        past_returns = returns.loc[:date]
        ff5_slice = ff5_full.loc[:date]
        
        # 2. Market Regime Detection (No Lookahead)
        regime, _, _ = detect_market_regime(past_returns, groups["EQUITY"])
        regime_history.append({'Date': date, 'Regime': regime})
        
        # 3. Calculate Inputs (BL)
        views, models = calculate_ff5_views(past_returns, ff5_slice)
        if views is None: continue
        
        cov = robust_covariance(past_returns)
        conf = DynamicConfidenceCalculator.calculate(models)
        bl_ret = black_litterman(cov, past_returns.mean(), views, conf)
        
        # 4. Optimize STRATEGIC (Static Constraints)
        cons_strat = {
            'max': {'EQUITY': max_eq_base, 'BONDS': max_bond_base, 'COMMODITIES': 0.3},
            'min': {}
        }
        # FIX: Passo bl_ret (Series) e NON bl_ret.values
        w_strat = optimize_portfolio(bl_ret, cov.values, vol_target, 0.02, min_w, max_w, cons_strat, groups)
        
        # 5. Optimize TACTICAL (Dynamic Constraints)
        regime_params = get_regime_constraints(regime, max_eq_base, max_bond_base)
        cons_tact = {
            'max': {'EQUITY': regime_params['equity_max'], 'BONDS': 1.0, 'COMMODITIES': 1.0},
            'min': {'BONDS': regime_params['bond_min'], 'COMMODITIES': regime_params.get('comm_min', 0)}
        }
        vol_tact = vol_target * regime_params['vol_mult']
        # FIX: Passo bl_ret (Series) e NON bl_ret.values
        w_tact = optimize_portfolio(bl_ret, cov.values, vol_tact, 0.02, min_w, max_w, cons_tact, groups)
        
        # 6. Performance Simulation (Next Period)
        if i < len(dates) - 1:
            next_date = dates[i+1]
            period_ret = returns.loc[date:next_date].iloc[1:] # Exclude start date
            
            for d in period_ret.index:
                r_day = period_ret.loc[d].values
                
                # Strategico
                ret_s = np.dot(w_strat, r_day)
                wealth_strat *= (1 + ret_s)
                ts_strat.append(wealth_strat)
                
                # Tattico
                ret_t = np.dot(w_tact, r_day)
                wealth_tact *= (1 + ret_t)
                ts_tact.append(wealth_tact)
                
                ts_dates.append(d)

    progress_bar.empty()
    
    df_perf = pd.DataFrame({
        'Strategico': ts_strat,
        'Tattico': ts_tact
    }, index=ts_dates)
    
    return df_perf, pd.DataFrame(regime_history)

# ---------------------------------------------------------
# UI STREAMLIT
# ---------------------------------------------------------

st.title("🏛️ Institutional Portfolio System | Dynamic Regime")
st.markdown("---")

# SIDEBAR CONFIG
st.sidebar.header("⚙️ Configurazione Strategica")
view_horizon = st.sidebar.selectbox("Orizzonte Views", [36, 60, 120], index=1)
conf_level = st.sidebar.slider("Confidenza Modello (BL)", 0.0, 1.0, 0.6)

st.sidebar.markdown("---")
st.sidebar.header("⚖️ Vincoli Base")
user_min_w = st.sidebar.number_input("Min % Asset", 0.0, 0.10, 0.0, 0.01)
user_max_w = st.sidebar.number_input("Max % Asset", 0.10, 1.0, 0.30, 0.05)
max_equity_base = st.sidebar.slider("Max EQUITY Base", 0.0, 1.0, 0.70)
max_bonds_base = st.sidebar.slider("Max BONDS Base", 0.0, 1.0, 0.80)

# UPLOAD
uploaded_file = st.file_uploader("📂 Carica Serie Storiche (CSV)", type='csv')

if uploaded_file:
    returns = process_uploaded_data(uploaded_file)
    
    if returns is not None:
        st.success(f"Dati caricati: {len(returns.columns)} asset, {len(returns)} mesi.")
        
        # --- 1. CLASSIFICAZIONE MANUALE ASSISTITA ---
        st.markdown("### 🏷️ Mappatura Asset Class")
        with st.expander("🔍 Verifica e Correggi la classificazione", expanded=True):
            col_sel1, col_sel2, col_sel3 = st.columns(3)
            initial_groups = auto_classify_assets_initial(returns.columns)
            all_assets = list(returns.columns)
            
            with col_sel1:
                sel_equity = st.multiselect("EQUITY (Rischio)", all_assets, default=initial_groups["EQUITY"])
            with col_sel2:
                sel_bonds = st.multiselect("BONDS (Difesa)", all_assets, default=initial_groups["BONDS"])
            with col_sel3:
                sel_comm = st.multiselect("COMMODITIES (Reale)", all_assets, default=initial_groups["COMMODITIES"])
            
            final_groups = {"EQUITY": sel_equity, "BONDS": sel_bonds, "COMMODITIES": sel_comm}

        # --- CALCOLO MODELLO ---
        ff5_data = download_ff5_factors()
        
        if ff5_data is not None:
            views, models = calculate_ff5_views(returns, ff5_data, window=view_horizon)
            confidences = DynamicConfidenceCalculator.calculate(models)
            cov_matrix = robust_covariance(returns)
            
            if views is not None:
                bl_returns = black_litterman(cov_matrix, returns.mean(), views, confidences, conf_level)
                
                # Setup Linee
                vol_targets = [0.03, 0.05, 0.08, 0.10, 0.12, 0.15]
                line_names = [f"Linea {i+1}" for i in range(6)]
                
                # --- STRATEGIC OPTIMIZATION ---
                strategic_weights = {}
                strategic_metrics = []
                
                for v_tgt, name in zip(vol_targets, line_names):
                    grp_con = {'max': {'EQUITY': max_equity_base, 'BONDS': max_bonds_base, 'COMMODITIES': 0.3}, 'min': {}}
                    # FIX: Passo bl_returns (Series) per mantenere l'indice
                    w = optimize_portfolio(bl_returns, cov_matrix.values, v_tgt, 0.02, user_min_w, user_max_w, grp_con, final_groups)
                    strategic_weights[name] = pd.Series(w, index=returns.columns)
                    ret = np.dot(w, bl_returns.values) * 12
                    vol = np.sqrt(np.dot(w.T, np.dot(cov_matrix.values, w))) * np.sqrt(12)
                    strategic_metrics.append([name, ret, vol, (ret-0.02)/vol])

                df_strat = pd.DataFrame(strategic_metrics, columns=["Linea", "Rendimento", "Volatilità", "Sharpe"]).set_index("Linea")

                # --- UI TABS ---
                tab1, tab2, tab3 = st.tabs(["📊 STRATEGIC ALLOCATION", "🌪️ DYNAMIC REGIME", "⏮️ BACKTEST COMPARATIVO"])
                
                with tab1:
                    st.markdown("### Asset Allocation Strategica (Base)")
                    c1, c2 = st.columns([1, 2])
                    with c1: st.table(df_strat.style.format("{:.2%}", subset=["Rendimento", "Volatilità"]).format("{:.2f}", subset=["Sharpe"]))
                    with c2: st.plotly_chart(px.bar(pd.DataFrame(strategic_weights).T, barmode='stack', title="Composizione Portafoglio"), use_container_width=True)

                with tab2:
                    st.markdown("### 🧭 Adattamento al Ciclo Economico")
                    
                    # RILEVAMENTO
                    regime, trend, turb = detect_market_regime(returns, final_groups["EQUITY"])
                    
                    # VISUALIZZAZIONE "WHY" - QUADRANTE
                    c_info, c_plot = st.columns([1, 2])
                    
                    with c_info:
                        st.markdown(f"**Stato Attuale:** {regime}")
                        st.metric("Trend Score (Equity)", f"{trend:.2%}", help="Media Mobile 3m vs 12m. >0 è Positivo.")
                        st.metric("Turbulence Score", f"{turb:.2f}", help="Volatilità Recente / Storica. >1.1 è Alta.")
                        
                        activate_regime = st.checkbox("✅ ADEGUARE AL REGIME", value=False)
                        
                        if activate_regime:
                            regime_params = get_regime_constraints(regime, max_equity_base, max_bonds_base)
                            st.markdown("#### Nuovi Vincoli:")
                            st.write(f"- Equity Max: {regime_params['equity_max']:.1%}")
                            st.write(f"- Bond Min: {regime_params.get('bond_min', 0):.1%}")
                            st.write(f"- Volatility Target: x{regime_params['vol_mult']}")

                    with c_plot:
                        # CREAZIONE QUADRANTE
                        fig_quad = go.Figure()
                        
                        # Sfondo Quadranti
                        fig_quad.add_shape(type="rect", x0=0, y0=0, x1=0.5, y1=1.1, fillcolor="rgba(0, 255, 0, 0.1)", line_width=0, layer="below") # Espansione
                        fig_quad.add_shape(type="rect", x0=-0.5, y0=1.1, x1=0.5, y1=3.0, fillcolor="rgba(255, 0, 0, 0.1)", line_width=0, layer="below") # Recessione/Inflazione
                        fig_quad.add_shape(type="rect", x0=-0.5, y0=0, x1=0, y1=1.1, fillcolor="rgba(0, 0, 255, 0.1)", line_width=0, layer="below") # Rallentamento
                        
                        # Punto Attuale
                        fig_quad.add_trace(go.Scatter(
                            x=[trend], y=[turb], mode='markers+text',
                            marker=dict(size=15, color='black'),
                            text=["TU SEI QUI"], textposition="top center",
                            name="Stato Attuale"
                        ))
                        
                        # Linee di soglia
                        fig_quad.add_hline(y=1.1, line_dash="dash", annotation_text="Soglia Turbolenza (1.1)")
                        fig_quad.add_vline(x=0.0, line_dash="dash", annotation_text="Inversione Trend")
                        
                        fig_quad.update_layout(
                            title="Mappa dei Regimi (Why?)",
                            xaxis_title="Trend Score (Momentum)",
                            yaxis_title="Turbulence Score (Rischio)",
                            xaxis=dict(range=[-0.3, 0.3]),
                            yaxis=dict(range=[0.5, 2.0]),
                            height=400
                        )
                        st.plotly_chart(fig_quad, use_container_width=True)

                    if activate_regime:
                        tactical_metrics = []
                        for v_tgt, name in zip(vol_targets, line_names):
                            rp = get_regime_constraints(regime, max_equity_base, max_bonds_base)
                            cons_tact = {
                                'max': {'EQUITY': rp['equity_max'], 'BONDS': 1.0, 'COMMODITIES': 1.0},
                                'min': {'BONDS': rp.get('bond_min', 0), 'COMMODITIES': rp.get('comm_min', 0)}
                            }
                            # FIX: Passo bl_returns (Series) per mantenere l'indice
                            w_tac = optimize_portfolio(bl_returns, cov_matrix.values, v_tgt * rp['vol_mult'], 0.02, user_min_w, user_max_w, cons_tact, final_groups)
                            ret_t = np.dot(w_tac, bl_returns.values) * 12
                            vol_t = np.sqrt(np.dot(w_tac.T, np.dot(cov_matrix.values, w_tac))) * np.sqrt(12)
                            tactical_metrics.append([name, ret_t, vol_t])
                        
                        st.dataframe(pd.DataFrame(tactical_metrics, columns=["Linea", "Rendimento (Tact)", "Volatilità (Tact)"]).set_index("Linea"), use_container_width=True)

                with tab3:
                    st.markdown("### ⏮️ Verità Storica: Strategico vs Tattico")
                    st.info("Simulazione Walk-Forward (Out-of-sample). Il Tattico decide il regime mese per mese senza conoscere il futuro.")
                    
                    if st.button("AVVIA BACKTEST COMPARATIVO"):
                        with st.spinner("Elaborazione scenari storici..."):
                            linea_test_idx = 3 # Usa Linea 4 (Media Volatilità) per il test
                            vol_test = vol_targets[linea_test_idx]
                            
                            df_wealth, df_regimes = run_comparative_backtest(returns, ff5_data, final_groups, vol_test, max_equity_base, max_bonds_base, user_min_w, user_max_w)
                            
                            # Metrics Finale
                            cum_ret = df_wealth.iloc[-1] / 100 - 1
                            dd = (df_wealth / df_wealth.cummax() - 1).min()
                            
                            c1, c2, c3 = st.columns(3)
                            c1.metric("Totale Strategico", f"{cum_ret['Strategico']:.2%}", f"MaxDD: {dd['Strategico']:.2%}")
                            c2.metric("Totale Tattico", f"{cum_ret['Tattico']:.2%}", f"MaxDD: {dd['Tattico']:.2%}")
                            
                            # Plot Wealth
                            fig_bt = px.line(df_wealth, title=f"Crescita Capitale (Base 100) - {line_names[linea_test_idx]}")
                            st.plotly_chart(fig_bt, use_container_width=True)
                            
                            # Plot Regimes History
                            st.markdown("#### Cronologia Regimi Rilevati")
                            fig_reg = px.scatter(df_regimes, x='Date', y='Regime', color='Regime', title="Storico Decisioni Tattiche")
                            st.plotly_chart(fig_reg, use_container_width=True)

    else:
        st.error("Errore download fattori.")
