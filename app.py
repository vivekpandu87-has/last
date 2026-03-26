"""
Smart Cricket Pod — Analytics Dashboard
3 files only: app.py + cricket_pod_survey_data.csv + requirements.txt
Models trained on startup. No .pkl files needed.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import io
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge, LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix,
                              roc_curve, mean_squared_error, r2_score,
                              mean_absolute_error, silhouette_score)
from sklearn.model_selection import train_test_split
from scipy import stats
from itertools import combinations

# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Cricket Pod — Analytics",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded",
)

PRIMARY   = "#1D9E75"
SECONDARY = "#7F77DD"
ACCENT    = "#EF9F27"
DANGER    = "#D85A30"
CLUSTER_COLORS = [PRIMARY, SECONDARY, ACCENT, DANGER, "#5DCAA5", "#F0997B", "#85B7EB"]

st.markdown(f"""
<style>
[data-testid="stSidebar"]{{background-color:#0f0f1a}}
[data-testid="stSidebar"] *{{color:#e0e0e0 !important}}
.metric-card{{background:linear-gradient(135deg,#1a1a2e 0%,#16213e 100%);
  border:1px solid {PRIMARY}44;border-radius:12px;padding:1.2rem 1.4rem;text-align:center;color:white}}
.metric-card .val{{font-size:2rem;font-weight:700;color:{PRIMARY}}}
.metric-card .lbl{{font-size:0.78rem;color:#aaa;margin-top:4px}}
.sec{{font-size:1.05rem;font-weight:600;border-left:4px solid {PRIMARY};
  padding-left:0.7rem;margin:1.2rem 0 0.8rem;color:inherit}}
.ibox{{background:#1a1a2e;border-left:3px solid {ACCENT};padding:0.8rem 1rem;
  border-radius:0 8px 8px 0;font-size:0.85rem;color:#ddd;margin:0.5rem 0}}
div[data-testid="metric-container"]{{background:#1a1a2e;border:1px solid #333;
  border-radius:10px;padding:0.5rem 1rem}}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# ORDINAL MAPS
# ══════════════════════════════════════════════════════════════════════════════
AGE_ORD    = {"Under 15":1,"15-18":2,"19-25":3,"26-35":4,"36-50":5,"50+":6}
CITY_ORD   = {"Rural":1,"Tier 3":2,"Tier 2":3,"Tier 1":4,"Metro":5}
INC_ORD    = {"Below 20K":1,"20K-40K":2,"40K-75K":3,"75K-150K":4,"Above 150K":5}
EDU_ORD    = {"Up to 10th":1,"12th/Diploma":2,"Bachelors":3,"Masters+":4}
PDAYS_ORD  = {"0":0,"1-2":1,"3-4":2,"5-6":3,"Daily":4}
ROLE_ORD   = {"Not interested":0,"Fan only":1,"Occasional":2,"Regular":3,"Competitive":4,"Coach":3}
SPEND_ORD  = {"0":0,"1-500":1,"501-1000":2,"1001-2500":3,"2501-5000":4,"Above 5000":5}
MEM_ORD    = {"Would not subscribe":0,"Up to 499":1,"500-999":2,"1000-1999":3,"2000-3000":4,"Above 3000":5}
DIG_ORD    = {"0":0,"1-200":1,"201-500":2,"501-1000":3,"Above 1000":4}
FD_ORD     = {"Rarely":1,"1-2/week":2,"3-4/week":3,"Daily":4}
TECH_ORD   = {"Tech avoider":0,"Laggard":1,"Late majority":2,"Early majority":3,"Early adopter":4}
DIST_ORD   = {"Within 1km":1,"Up to 3km":2,"Up to 5km":3,"Up to 10km":4,"Any distance":5}
FC_ORD     = {"Not interested":0,"Aware not using":1,"Occasional":2,"Active":3}
GENDER_ORD = {"Male":0,"Female":1,"Other/PNS":2}
PSM_TC_ORD = {"Below 50":1,"50-99":2,"100-149":3,"150-199":4,"200-249":5}
PSM_R_ORD  = {"100-149":1,"150-199":2,"200-249":3,"250-299":4,"300-349":5}
PSM_E_ORD  = {"200-299":1,"300-399":2,"400-499":3,"500-599":4,"600+":5}
PSM_TE_ORD = {"300-399":1,"400-499":2,"500-599":3,"600-799":4,"800+":5}
PSM_R_MID  = {"100-149":125,"150-199":175,"200-249":225,"250-299":275,"300-349":325}
PSM_TC_MID = {"Below 50":50,"50-99":75,"100-149":125,"150-199":175,"200-249":225}
PSM_E_MID  = {"200-299":250,"300-399":350,"400-499":450,"500-599":550,"600+":650}
PSM_TE_MID = {"300-399":350,"400-499":450,"500-599":550,"600-799":700,"800+":900}

MULTI_SELECT_COLS = [
    "use_academy","use_boxcricket","use_bowlingmachine","use_homenet","use_videoanalysis","use_mobilegame","use_gym",
    "disc_freetrial","disc_buy5get1","disc_student","disc_family","disc_offpeak","disc_referral","disc_academy","disc_corporate",
    "feat_ai","feat_bowlingmachine","feat_batspeed","feat_footwork","feat_videoreplay","feat_leaderboard","feat_progressreport","feat_appbooking",
    "brand_mrf","brand_sg","brand_ss","brand_kookaburra","brand_graynicolls","brand_adidasnike","brand_decathlon","brand_nopref",
    "addon_smartbat","addon_wearables","addon_aicoaching","addon_highlights","addon_fitness","addon_merch",
    "stream_hotstar","stream_jiocinema","stream_netflix","stream_prime","stream_youtube","stream_sonyliv",
    "act_gym","act_yoga","act_othersport","act_swimming","act_videogaming","act_running",
    "past_boxcricket","past_trampoline","past_vr","past_bowling","past_gokarting","past_fitclass","past_academy","past_golf",
    "frust_nodata","frust_coachattention","frust_timing","frust_crowded","frust_distance","frust_cost","frust_equipment","frust_notracking",
    "bar_price","bar_location","bar_humancoach","bar_aidistrust","bar_time","bar_notserious","bar_academy","bar_safety","bar_social",
    "hh_self","hh_child","hh_spouse","hh_sibling","hh_parent",
]

# Reduced ARM basket — only the most business-meaningful columns (prevents hang)
ARM_COLS = [
    "feat_ai","feat_bowlingmachine","feat_batspeed","feat_videoreplay","feat_progressreport","feat_leaderboard",
    "addon_smartbat","addon_wearables","addon_aicoaching","addon_highlights","addon_fitness",
    "use_academy","use_boxcricket","use_gym","use_videoanalysis",
    "past_boxcricket","past_trampoline","past_vr","past_fitclass","past_academy",
    "disc_freetrial","disc_student","disc_family","disc_referral","disc_corporate",
    "frust_nodata","frust_notracking","frust_coachattention","frust_timing",
    "bar_price","bar_aidistrust","bar_humancoach","bar_location",
    "stream_hotstar","stream_youtube","stream_netflix",
    "act_gym","act_othersport","act_videogaming","act_running",
]

CLUSTERING_FEATURES    = ["age_num","income_num","city_num","role_num","practice_num","data_importance",
                           "pod_interest","spend_num","tech_num","dist_num","nps_score","digital_num","fd_num",
                           "addon_count","past_exp_count","barrier_count","feat_count"]
CLASSIFICATION_FEATURES= ["age_num","gender_num","city_num","income_num","edu_num","role_num","practice_num",
                           "data_importance","pod_interest","spend_num","mem_num","digital_num","fd_num",
                           "tech_num","dist_num","nps_score","addon_count","feat_count","past_exp_count",
                           "barrier_count","frust_count","past_boxcricket","past_trampoline","past_vr",
                           "feat_ai","feat_bowlingmachine","feat_progressreport","bar_aidistrust",
                           "bar_price","bar_location","bar_notserious","use_academy","use_boxcricket"]
REGRESSION_FEATURES    = ["age_num","income_num","city_num","role_num","practice_num","data_importance",
                           "pod_interest","spend_num","tech_num","digital_num","nps_score","addon_count",
                           "feat_count","past_exp_count","frust_count","barrier_count","mem_num","dist_num"]

# ══════════════════════════════════════════════════════════════════════════════
# ENCODING
# ══════════════════════════════════════════════════════════════════════════════
def encode(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["age_num"]      = d["age_group"].map(AGE_ORD).fillna(3)
    d["city_num"]     = d["city_tier"].map(CITY_ORD).fillna(3)
    d["income_num"]   = d["income_bracket"].map(INC_ORD).fillna(3)
    d["edu_num"]      = d["education"].map(EDU_ORD).fillna(2)
    d["role_num"]     = d["cricket_role"].map(ROLE_ORD).fillna(1)
    d["practice_num"] = d["practice_days"].map(PDAYS_ORD).fillna(1)
    d["spend_num"]    = d["monthly_rec_spend"].map(SPEND_ORD).fillna(2)
    d["mem_num"]      = d["membership_wtp"].map(MEM_ORD).fillna(1)
    d["digital_num"]  = d["digital_spend"].map(DIG_ORD).fillna(1)
    d["fd_num"]       = d["food_delivery_freq"].map(FD_ORD).fillna(2)
    d["tech_num"]     = d["tech_adoption"].map(TECH_ORD).fillna(2)
    d["dist_num"]     = d["distance_tolerance"].map(DIST_ORD).fillna(3)
    d["gender_num"]   = d["gender"].map(GENDER_ORD).fillna(0)
    d["fc_num"]       = d["fantasy_cricket"].map(FC_ORD).fillna(1)
    d["psm_tc_num"]   = d["psm_too_cheap"].map(PSM_TC_ORD).fillna(3)
    d["psm_r_num"]    = d["psm_reasonable"].map(PSM_R_ORD).fillna(3)
    d["psm_e_num"]    = d["psm_expensive"].map(PSM_E_ORD).fillna(3)
    d["psm_te_num"]   = d["psm_too_expensive"].map(PSM_TE_ORD).fillna(3)

    addon_cols   = ["addon_smartbat","addon_wearables","addon_aicoaching","addon_highlights","addon_fitness","addon_merch"]
    feat_cols    = ["feat_ai","feat_bowlingmachine","feat_batspeed","feat_footwork","feat_videoreplay","feat_leaderboard","feat_progressreport","feat_appbooking"]
    past_cols    = ["past_boxcricket","past_trampoline","past_vr","past_bowling","past_gokarting","past_fitclass","past_academy","past_golf"]
    barrier_cols = ["bar_price","bar_location","bar_humancoach","bar_aidistrust","bar_time","bar_notserious","bar_academy","bar_safety","bar_social"]
    frust_cols   = ["frust_nodata","frust_coachattention","frust_timing","frust_crowded","frust_distance","frust_cost","frust_equipment","frust_notracking"]

    for clist, cname in [(addon_cols,"addon_count"),(feat_cols,"feat_count"),(past_cols,"past_exp_count"),
                         (barrier_cols,"barrier_count"),(frust_cols,"frust_count")]:
        avail = [c for c in clist if c in d.columns]
        d[cname] = d[avail].fillna(0).sum(axis=1)
    return d

def feat(df_enc, flist):
    avail = [c for c in flist if c in df_enc.columns]
    return df_enc[avail].fillna(df_enc[avail].median())

# ══════════════════════════════════════════════════════════════════════════════
# FAST APRIORI (pairs only — max_len=2 for speed without mlxtend)
# ══════════════════════════════════════════════════════════════════════════════
def fast_arm(df: pd.DataFrame, arm_cols, min_support=0.05, min_confidence=0.50, min_lift=1.2):
    """Fast pair-only association rules. Runs in seconds on 2000 rows."""
    avail = [c for c in arm_cols if c in df.columns]
    basket = df[avail].fillna(0).astype(bool)
    n = len(basket)
    arr = basket.values
    cols = basket.columns.tolist()

    # Single item supports
    sup1 = {i: arr[:, i].sum() / n for i in range(len(cols))}
    freq1 = {i: s for i, s in sup1.items() if s >= min_support}

    rows = []
    freq1_list = sorted(freq1.keys())
    for i in range(len(freq1_list)):
        for j in range(i + 1, len(freq1_list)):
            ci, cj = freq1_list[i], freq1_list[j]
            sup_ij = (arr[:, ci] & arr[:, cj]).sum() / n
            if sup_ij < min_support:
                continue
            for ant_idx, con_idx in [(ci, cj), (cj, ci)]:
                sup_a = sup1[ant_idx]
                sup_b = sup1[con_idx]
                if sup_a == 0 or sup_b == 0:
                    continue
                conf = sup_ij / sup_a
                lift = conf / sup_b
                if conf >= min_confidence and lift >= min_lift:
                    rows.append({
                        "antecedents": cols[ant_idx],
                        "consequents": cols[con_idx],
                        "support":     round(sup_ij, 4),
                        "confidence":  round(conf, 4),
                        "lift":        round(lift, 4),
                    })

    if not rows:
        return pd.DataFrame(columns=["antecedents","consequents","support","confidence","lift"])
    return pd.DataFrame(rows).sort_values("lift", ascending=False).reset_index(drop=True)

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOAD
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data
def load_data(uploaded_bytes=None):
    if uploaded_bytes is not None:
        return pd.read_csv(io.BytesIO(uploaded_bytes))
    try:
        return pd.read_csv("cricket_pod_survey_data.csv")
    except FileNotFoundError:
        return None

# ══════════════════════════════════════════════════════════════════════════════
# MODEL TRAINING  — @st.cache_resource so it runs ONCE per session
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def train_all(_df):
    df = _df.copy()
    results = {}
    df_enc = encode(df)

    # ── CLUSTERING ────────────────────────────────────────────────────────────
    X_c = feat(df_enc, CLUSTERING_FEATURES)
    sc_c = StandardScaler()
    Xcs = sc_c.fit_transform(X_c)

    inertias, sils = [], []
    for k in range(2, 9):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labs = km.fit_predict(Xcs)
        inertias.append(km.inertia_)
        sils.append(silhouette_score(Xcs, labs))

    best_k = 5
    km_f = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    cl   = km_f.fit_predict(Xcs)
    df_enc["cluster"] = cl

    profiles, persona_map = [], {}
    for c in range(best_k):
        mask = df_enc["cluster"] == c
        sub  = df_enc[mask]
        prof = {
            "cluster": c, "size": int(mask.sum()),
            "avg_income":      float(sub["income_num"].mean()),
            "avg_role":        float(sub["role_num"].mean()),
            "avg_pod_interest":float(sub["pod_interest"].mean()),
            "avg_spend":       float(sub["realistic_monthly_spend"].mean()) if "realistic_monthly_spend" in sub else 0,
            "avg_age":         float(sub["age_num"].mean()),
            "conversion_rate": float((sub["pod_conversion_binary"]==1).mean()) if "pod_conversion_binary" in sub else 0,
        }
        profiles.append(prof)
        if   prof["avg_role"] >= 3.5 and prof["avg_income"] <= 2.5: name = "Rising Star"
        elif prof["avg_role"] >= 3.0 and prof["avg_income"] >= 3.5: name = "Elite Competitor"
        elif prof["avg_income"] >= 4.0 and prof["avg_role"] <= 2.0: name = "Corporate Cricket Fan"
        elif prof["avg_pod_interest"] <= 2.5:                        name = "Sceptic / Disengaged"
        else:                                                         name = "Recreational Player"
        persona_map[c] = name

    results["clustering"] = {"inertias":inertias,"silhouettes":sils,"best_k":best_k,
                              "profiles":profiles,"persona_map":persona_map}

    # ── CLASSIFICATION ────────────────────────────────────────────────────────
    vmask  = df_enc["pod_conversion_binary"].notna()
    df_clf = df_enc[vmask].copy()
    Xcr    = feat(df_clf, CLASSIFICATION_FEATURES)
    y_clf  = df_clf["pod_conversion_binary"].astype(int)
    sc_f   = StandardScaler()
    Xcs_f  = sc_f.fit_transform(Xcr)
    Xtr,Xte,ytr,yte = train_test_split(Xcs_f, y_clf, test_size=0.2, random_state=42, stratify=y_clf)

    rf = RandomForestClassifier(n_estimators=200, max_depth=10, class_weight="balanced", random_state=42)
    rf.fit(Xtr, ytr)
    rp = rf.predict(Xte); rprob = rf.predict_proba(Xte)[:,1]
    fpr,tpr,_ = roc_curve(yte, rprob)

    lrc = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    lrc.fit(Xtr, ytr)
    lp = lrc.predict(Xte); lprob = lrc.predict_proba(Xte)[:,1]
    fpr_l,tpr_l,_ = roc_curve(yte, lprob)

    fi = pd.Series(rf.feature_importances_, index=Xcr.columns).sort_values(ascending=False)

    results["classification"] = {
        "rf":{"acc":accuracy_score(yte,rp),"prec":precision_score(yte,rp),
              "rec":recall_score(yte,rp),"f1":f1_score(yte,rp),
              "auc":roc_auc_score(yte,rprob),"cm":confusion_matrix(yte,rp).tolist(),
              "fpr":fpr.tolist(),"tpr":tpr.tolist()},
        "lr":{"acc":accuracy_score(yte,lp),"prec":precision_score(yte,lp),
              "rec":recall_score(yte,lp),"f1":f1_score(yte,lp),
              "auc":roc_auc_score(yte,lprob),"cm":confusion_matrix(yte,lp).tolist(),
              "fpr":fpr_l.tolist(),"tpr":tpr_l.tolist()},
        "feat_imp":fi.head(20).to_dict(),
        "y_test":yte.tolist(),"rf_prob":rprob.tolist(),
    }

    # ── ASSOCIATION RULES (fast pairs only) ───────────────────────────────────
    rules = fast_arm(df, ARM_COLS, min_support=0.05, min_confidence=0.50, min_lift=1.2)
    results["association"] = {"rules": rules}

    # ── REGRESSION ────────────────────────────────────────────────────────────
    Xrr  = feat(df_enc, REGRESSION_FEATURES)
    y_r  = df_enc["realistic_monthly_spend"].fillna(df_enc["realistic_monthly_spend"].median())
    sc_r = StandardScaler()
    Xrrs = sc_r.fit_transform(Xrr)
    Xrtr,Xrte,yrtr,yrte = train_test_split(Xrrs, y_r, test_size=0.2, random_state=42)

    ridge  = Ridge(alpha=1.0);    ridge.fit(Xrtr,yrtr);  rp2 = ridge.predict(Xrte)
    lr_reg = LinearRegression();  lr_reg.fit(Xrtr,yrtr); lrp = lr_reg.predict(Xrte)
    ci = pd.Series(np.abs(ridge.coef_), index=Xrr.columns).sort_values(ascending=False)

    results["regression"] = {
        "ridge":{"r2":r2_score(yrte,rp2),"rmse":np.sqrt(mean_squared_error(yrte,rp2)),
                 "mae":mean_absolute_error(yrte,rp2),"y_test":yrte.tolist(),"y_pred":rp2.tolist()},
        "lr":   {"r2":r2_score(yrte,lrp),"rmse":np.sqrt(mean_squared_error(yrte,lrp)),
                 "mae":mean_absolute_error(yrte,lrp)},
        "coef_imp":ci.head(15).to_dict(),
    }

    mdl = {
        "kmeans":km_f,"scaler_clust":sc_c,"cluster_features":X_c.columns.tolist(),
        "cluster_profiles":profiles,"persona_map":persona_map,
        "rf_classifier":rf,"lr_classifier":lrc,"scaler_clf":sc_f,
        "clf_features":Xcr.columns.tolist(),
        "ridge_regressor":ridge,"lr_regressor":lr_reg,"scaler_reg":sc_r,
        "reg_features":Xrr.columns.tolist(),
        "assoc_rules":rules,"all_results":results,
    }
    df_enc["persona"] = df_enc["cluster"].map(persona_map)
    return mdl, df_enc

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR + LOAD
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🏏 Smart Cricket Pod")
    st.markdown("**Data-Driven Analytics Platform**")
    st.markdown("---")
    uf = st.file_uploader("📂 Upload survey CSV (optional)", type=["csv"],
                          help="Upload if cricket_pod_survey_data.csv is not in the same folder")
    st.markdown("---")
    page = st.radio("Navigate", [
        "🏠  Home",
        "📊  Descriptive Analysis",
        "🔍  Diagnostic Analysis",
        "🎯  Classification",
        "👥  Clustering — Personas",
        "🔗  Association Rules",
        "📈  Regression — Spend Forecast",
        "🚀  New Customer Predictor",
    ])

ub = uf.read() if uf else None
df = load_data(ub)

if df is None:
    st.error("❌ CSV not found. Upload `cricket_pod_survey_data.csv` via the sidebar.")
    st.stop()

with st.spinner("🏏 Training models… (30–60 sec first load, then cached)"):
    models, df_enc = train_all(df)

with st.sidebar:
    st.markdown("---")
    st.caption(f"Dataset: {len(df):,} rows · {df.shape[1]} cols")
    st.caption("v2.0 — Smart Cricket Pod")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: HOME
# ══════════════════════════════════════════════════════════════════════════════
def page_home():
    st.title("🏏 Smart Cricket Pod — Analytics Dashboard")
    st.markdown("#### Data-Driven Decision Making for India's First AI-Powered Cricket Pod Network")
    st.markdown("---")

    total      = len(df)
    interested = int((df["pod_conversion_binary"]==1).sum())
    not_int    = int((df["pod_conversion_binary"]==0).sum())
    maybe      = int(df["pod_conversion_binary"].isna().sum())
    conv_rate  = interested/(interested+not_int)*100
    avg_spend  = df["realistic_monthly_spend"].mean()
    avg_nps    = df["nps_score"].mean()

    c1,c2,c3,c4,c5,c6 = st.columns(6)
    def kpi(col, val, lbl, color=PRIMARY):
        col.markdown(f'<div class="metric-card"><div class="val" style="color:{color}">{val}</div>'
                     f'<div class="lbl">{lbl}</div></div>', unsafe_allow_html=True)
    kpi(c1,f"{total:,}","Total Respondents")
    kpi(c2,f"{interested:,}","Interested (Label=1)")
    kpi(c3,f"{conv_rate:.1f}%","Conversion Rate",ACCENT)
    kpi(c4,f"₹{avg_spend:,.0f}","Avg Monthly Spend",SECONDARY)
    kpi(c5,f"{avg_nps:.1f}/10","Avg NPS Score","#E74C3C" if avg_nps<6 else PRIMARY)
    kpi(c6,f"{maybe:,}","Maybe (Undecided)",DANGER)
    st.markdown("<br>", unsafe_allow_html=True)

    col1,col2,col3 = st.columns([1.2,1,1])
    with col1:
        st.markdown('<div class="sec">Conversion Signal</div>', unsafe_allow_html=True)
        cc = df["pod_conversion"].value_counts()
        cm = {"Yes - definitely":PRIMARY,"Yes - if price right":"#5DCAA5","Maybe":ACCENT,"Unlikely":"#F0997B","No":DANGER}
        fig = go.Figure(go.Bar(x=cc.values,y=cc.index,orientation="h",
                               marker_color=[cm.get(l,"#888") for l in cc.index],
                               text=[f"{v:,}" for v in cc.values],textposition="outside"))
        fig.update_layout(height=280,margin=dict(l=0,r=40,t=10,b=10),
                          plot_bgcolor="rgba(0,0,0,0)",paper_bgcolor="rgba(0,0,0,0)",
                          xaxis=dict(showgrid=False,color="#aaa"),yaxis=dict(color="#ccc"),font=dict(color="#ccc"))
        st.plotly_chart(fig,use_container_width=True)
    with col2:
        st.markdown('<div class="sec">True Segments</div>', unsafe_allow_html=True)
        sc = df["true_segment"].value_counts()
        fig2 = go.Figure(go.Pie(labels=sc.index,values=sc.values,hole=0.45,
                                marker_colors=[PRIMARY,SECONDARY,ACCENT,DANGER,"#888"],textinfo="percent"))
        fig2.update_layout(height=280,margin=dict(l=0,r=0,t=10,b=10),paper_bgcolor="rgba(0,0,0,0)",
                           legend=dict(font=dict(color="#ccc",size=10)),font=dict(color="#ccc"))
        st.plotly_chart(fig2,use_container_width=True)
    with col3:
        st.markdown('<div class="sec">City Tier</div>', unsafe_allow_html=True)
        ctc = df["city_tier"].value_counts()
        fig3 = go.Figure(go.Bar(x=ctc.index,y=ctc.values,marker_color=SECONDARY,
                                text=ctc.values,textposition="outside"))
        fig3.update_layout(height=280,margin=dict(l=0,r=0,t=10,b=30),
                           plot_bgcolor="rgba(0,0,0,0)",paper_bgcolor="rgba(0,0,0,0)",
                           xaxis=dict(color="#ccc"),yaxis=dict(showgrid=False,color="#aaa"),font=dict(color="#ccc"))
        st.plotly_chart(fig3,use_container_width=True)

    st.markdown('<div class="sec">Dataset Health</div>', unsafe_allow_html=True)
    ca,cb = st.columns(2)
    with ca:
        nc = df.isnull().sum(); np_ = (nc/len(df)*100).round(2)
        nd = pd.DataFrame({"Missing":nc,"Missing %":np_})
        nd = nd[nd["Missing"]>0].sort_values("Missing %",ascending=False)
        st.dataframe(nd.head(15),use_container_width=True,height=260)
    with cb:
        sd = {"Total Rows":len(df),"Total Columns":df.shape[1],
              "Numeric Cols":int(df.select_dtypes(include=np.number).shape[1]),
              "Categorical Cols":int(df.select_dtypes(include="object").shape[1]),
              "Binary Cols":int((df.nunique()==2).sum()),
              "Duplicate Rows":int(df.duplicated().sum()),
              "Complete Rows":int(df.dropna().shape[0])}
        st.dataframe(pd.DataFrame.from_dict(sd,orient="index",columns=["Value"]),
                     use_container_width=True,height=260)

    st.markdown("---")
    st.markdown("### Navigation Guide")
    guide=[("📊 Descriptive","Demographics, PSM price curves, barrier analysis"),
           ("🔍 Diagnostic","Correlations, chi-square, ANOVA, frustration heatmap"),
           ("🎯 Classification","RF + LR, Accuracy/Precision/Recall/F1/ROC-AUC, feature importance"),
           ("👥 Clustering","K-Means personas, elbow, silhouette, radar charts"),
           ("🔗 Association Rules","Apriori: support, confidence, lift — bundle discovery"),
           ("📈 Regression","Ridge regression, residuals, revenue forecaster"),
           ("🚀 Predictor","Upload new leads → score conversion + spend + offer")]
    gcols = st.columns(4)
    for i,(t,d) in enumerate(guide):
        gcols[i%4].markdown(f'<div style="background:#1a1a2e;border:1px solid #333;border-radius:10px;'
                             f'padding:0.8rem;margin-bottom:8px;min-height:90px">'
                             f'<div style="font-weight:600;font-size:0.85rem;color:{PRIMARY};margin-bottom:4px">{t}</div>'
                             f'<div style="font-size:0.75rem;color:#aaa;line-height:1.4">{d}</div></div>',
                             unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DESCRIPTIVE
# ══════════════════════════════════════════════════════════════════════════════
def page_descriptive():
    st.title("📊 Descriptive Analysis")
    st.markdown("Customer landscape — demographics, behaviour, and pricing signals.")
    st.markdown("---")

    st.markdown('<div class="sec">Demographics</div>', unsafe_allow_html=True)
    c1,c2,c3 = st.columns(3)
    def bar(col,title,x,y,color,h=280,xlab=None,ylab=None,horizontal=False):
        if horizontal:
            fig=go.Figure(go.Bar(x=y,y=x,orientation="h",marker_color=color,text=y,textposition="outside"))
        else:
            fig=go.Figure(go.Bar(x=x,y=y,marker_color=color,text=y,textposition="outside"))
        fig.update_layout(title=title,height=h,plot_bgcolor="rgba(0,0,0,0)",paper_bgcolor="rgba(0,0,0,0)",
                          xaxis=dict(color="#ccc",title=xlab,showgrid=False),
                          yaxis=dict(color="#aaa",title=ylab,showgrid=False),
                          font=dict(color="#ccc"),margin=dict(t=35,b=10,l=0,r=40))
        col.plotly_chart(fig,use_container_width=True)

    age_c=df["age_group"].value_counts().reindex([o for o in ["Under 15","15-18","19-25","26-35","36-50","50+"] if o in df["age_group"].values])
    bar(c1,"Age distribution",age_c.index,age_c.values,PRIMARY)
    gen_c=df["gender"].value_counts()
    fig=go.Figure(go.Pie(labels=gen_c.index,values=gen_c.values,hole=0.4,marker_colors=[SECONDARY,PRIMARY,ACCENT]))
    fig.update_layout(title="Gender split",height=280,paper_bgcolor="rgba(0,0,0,0)",
                      legend=dict(font=dict(color="#ccc",size=11)),font=dict(color="#ccc"),margin=dict(t=35,b=10,l=0,r=0))
    c2.plotly_chart(fig,use_container_width=True)
    inc_c=df["income_bracket"].value_counts().reindex([o for o in ["Below 20K","20K-40K","40K-75K","75K-150K","Above 150K"] if o in df["income_bracket"].values])
    bar(c3,"Income bracket",inc_c.index,inc_c.values,ACCENT,horizontal=True)

    c4,c5,c6=st.columns(3)
    role_c=df["cricket_role"].value_counts()
    fig=go.Figure(go.Pie(labels=role_c.index,values=role_c.values,hole=0.4,
                         marker_colors=[PRIMARY,SECONDARY,ACCENT,DANGER,"#888","#5DCAA5"]))
    fig.update_layout(title="Cricket role",height=280,paper_bgcolor="rgba(0,0,0,0)",
                      legend=dict(font=dict(color="#ccc",size=10)),font=dict(color="#ccc"),margin=dict(t=35,b=10,l=0,r=0))
    c4.plotly_chart(fig,use_container_width=True)
    occ_c=df["occupation"].value_counts().head(8)
    bar(c5,"Occupation",occ_c.index,occ_c.values,SECONDARY,horizontal=True)
    city_c=df["city_tier"].value_counts()
    bar(c6,"City tier",city_c.index,city_c.values,DANGER)

    st.markdown('<div class="sec">Cricket Behaviour & Feature Interest</div>', unsafe_allow_html=True)
    c7,c8=st.columns(2)
    pd_c=df["practice_days"].value_counts().reindex([o for o in ["0","1-2","3-4","5-6","Daily"] if o in df["practice_days"].values])
    fig=go.Figure(go.Bar(x=pd_c.index,y=pd_c.values,
                         marker_color=[PRIMARY if v in ["3-4","5-6","Daily"] else "#888" for v in pd_c.index],
                         text=pd_c.values,textposition="outside"))
    fig.update_layout(title="Practice days/week",height=260,plot_bgcolor="rgba(0,0,0,0)",paper_bgcolor="rgba(0,0,0,0)",
                      xaxis=dict(color="#ccc"),yaxis=dict(showgrid=False,color="#aaa"),
                      font=dict(color="#ccc"),margin=dict(t=35,b=10,l=0,r=0))
    c7.plotly_chart(fig,use_container_width=True)
    fc=["feat_ai","feat_bowlingmachine","feat_batspeed","feat_footwork","feat_videoreplay","feat_leaderboard","feat_progressreport","feat_appbooking"]
    fl=["AI Analysis","Bowling Machine","Bat Speed","Footwork","Video Replay","Leaderboard","Progress Report","App Booking"]
    av=[c for c in fc if c in df.columns]; la=[fl[fc.index(c)] for c in av]
    fp=df[av].fillna(0).mean()*100
    fig=go.Figure(go.Bar(x=fp.values,y=la,orientation="h",marker_color=PRIMARY,
                         text=[f"{v:.1f}%" for v in fp.values],textposition="outside"))
    fig.update_layout(title="Feature interest (%)",height=280,plot_bgcolor="rgba(0,0,0,0)",paper_bgcolor="rgba(0,0,0,0)",
                      xaxis=dict(range=[0,100],showgrid=False,color="#aaa"),yaxis=dict(color="#ccc"),
                      font=dict(color="#ccc"),margin=dict(t=35,b=10,l=0,r=60))
    c8.plotly_chart(fig,use_container_width=True)

    st.markdown('<div class="sec">Barriers to Adoption</div>', unsafe_allow_html=True)
    bc=["bar_price","bar_location","bar_humancoach","bar_aidistrust","bar_time","bar_notserious","bar_academy","bar_safety","bar_social"]
    bl=["Too expensive","No pod nearby","Prefer human coach","AI distrust","No time","Not serious","Already in academy","Safety","Want friends"]
    ab=[c for c in bc if c in df.columns]; lb=[bl[bc.index(c)] for c in ab]
    bp=pd.Series(df[ab].fillna(0).mean().values*100,index=lb).sort_values(ascending=True)
    fig=go.Figure(go.Bar(x=bp.values,y=bp.index,orientation="h",
                         marker_color=[DANGER if v>30 else ACCENT if v>20 else "#888" for v in bp.values],
                         text=[f"{v:.1f}%" for v in bp.values],textposition="outside"))
    fig.update_layout(height=320,plot_bgcolor="rgba(0,0,0,0)",paper_bgcolor="rgba(0,0,0,0)",
                      xaxis=dict(range=[0,70],showgrid=False,color="#aaa"),yaxis=dict(color="#ccc"),
                      font=dict(color="#ccc"),margin=dict(t=10,b=10,l=0,r=60))
    st.plotly_chart(fig,use_container_width=True)

    st.markdown('<div class="sec">Van Westendorp Price Sensitivity Meter — Optimal Launch Price</div>', unsafe_allow_html=True)
    dp=df.copy()
    dp["tc"]=dp["psm_too_cheap"].map(PSM_TC_MID)
    dp["r"] =dp["psm_reasonable"].map(PSM_R_MID)
    dp["e"] =dp["psm_expensive"].map(PSM_E_MID)
    dp["te"]=dp["psm_too_expensive"].map(PSM_TE_MID)
    dp=dp.dropna(subset=["tc","r","e","te"]); nv=len(dp)
    prices=np.arange(50,900,10)
    ptc=[(dp["tc"]>=p).sum()/nv*100 for p in prices]
    pr= [(dp["r"] <=p).sum()/nv*100 for p in prices]
    pe= [(dp["e"] <=p).sum()/nv*100 for p in prices]
    pte=[(dp["te"]<=p).sum()/nv*100 for p in prices]
    ra,ea,tca,tea=np.array(pr),np.array(pe),np.array(ptc),np.array(pte)
    opp=int(prices[np.argmin(np.abs(ra-ea))])
    alo_idx=np.where(ra-tca>=0)[0]; alo=int(prices[alo_idx[0]]) if len(alo_idx) else 100
    ahi_idx=np.where(tea-(100-ra)>=0)[0]; ahi=int(prices[ahi_idx[0]]) if len(ahi_idx) else 500
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=prices,y=ptc,name="Too cheap",  line=dict(color="#5DCAA5",width=2,dash="dot")))
    fig.add_trace(go.Scatter(x=prices,y=pr, name="Reasonable",line=dict(color=PRIMARY,width=2.5)))
    fig.add_trace(go.Scatter(x=prices,y=pe, name="Expensive", line=dict(color=ACCENT,width=2,dash="dash")))
    fig.add_trace(go.Scatter(x=prices,y=pte,name="Too expensive",line=dict(color=DANGER,width=2.5)))
    fig.add_vrect(x0=alo,x1=ahi,fillcolor=PRIMARY,opacity=0.08,layer="below",line_width=0)
    fig.add_vline(x=opp,line_color=PRIMARY,line_dash="dash",line_width=2,
                  annotation_text=f"OPP: ₹{opp}",annotation_font_color=PRIMARY,annotation_position="top right")
    fig.update_layout(height=380,plot_bgcolor="rgba(0,0,0,0)",paper_bgcolor="rgba(0,0,0,0)",
                      xaxis=dict(title="Price ₹",color="#ccc",tickprefix="₹",showgrid=True,gridcolor="#333"),
                      yaxis=dict(title="% respondents",color="#aaa",showgrid=True,gridcolor="#333"),
                      legend=dict(font=dict(color="#ccc")),font=dict(color="#ccc"),margin=dict(t=20,b=40,l=0,r=0))
    st.plotly_chart(fig,use_container_width=True)
    p1,p2,p3=st.columns(3)
    p1.metric("Acceptable Price Range",f"₹{alo} – ₹{ahi}")
    p2.metric("Optimal Price Point",   f"₹{opp}")
    p3.metric("Recommended Launch",    f"₹{opp-10} – ₹{opp+15}")
    st.markdown(f'<div class="ibox">📌 <strong>Founder action:</strong> Launch at ₹{opp} per 30-min session. '
                f'Acceptable range ₹{alo}–₹{ahi} supports student discounts and premium weekend slots.</div>',
                unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DIAGNOSTIC
# ══════════════════════════════════════════════════════════════════════════════
def page_diagnostic():
    st.title("🔍 Diagnostic Analysis")
    st.markdown("Why are customers interested — correlations, statistical tests, and frustration signals.")
    st.markdown("---")

    nc=["age_num","income_num","city_num","role_num","practice_num","data_importance","pod_interest",
        "spend_num","tech_num","dist_num","nps_score","digital_num","addon_count","feat_count",
        "past_exp_count","barrier_count","frust_count","pod_conversion_binary"]
    av=[c for c in nc if c in df_enc.columns]
    cd=df_enc[av].apply(pd.to_numeric,errors="coerce").corr().round(2)
    lm={"age_num":"Age","income_num":"Income","city_num":"City","role_num":"Role","practice_num":"Practice",
        "data_importance":"Data Imp","pod_interest":"Pod Interest","spend_num":"Spend","tech_num":"Tech",
        "dist_num":"Distance","nps_score":"NPS","digital_num":"Digital","addon_count":"Addons",
        "feat_count":"Features","past_exp_count":"Past Exp","barrier_count":"Barriers",
        "frust_count":"Frustrations","pod_conversion_binary":"Conversion"}
    tl=[lm.get(c,c) for c in cd.columns]

    st.markdown('<div class="sec">Correlation Heatmap</div>', unsafe_allow_html=True)
    fig=go.Figure(go.Heatmap(z=cd.values,x=tl,y=tl,
                              colorscale=[[0,DANGER],[0.5,"#1a1a2e"],[1,PRIMARY]],
                              zmid=0,zmin=-1,zmax=1,text=cd.values.round(2),texttemplate="%{text}",
                              textfont=dict(size=8),colorbar=dict(tickfont=dict(color="#ccc"))))
    fig.update_layout(height=500,margin=dict(t=10,b=10,l=0,r=0),paper_bgcolor="rgba(0,0,0,0)",
                      xaxis=dict(tickfont=dict(color="#ccc",size=9),tickangle=-45),
                      yaxis=dict(tickfont=dict(color="#ccc",size=9)),font=dict(color="#ccc"))
    st.plotly_chart(fig,use_container_width=True)

    if "pod_conversion_binary" in cd.columns:
        cc2=cd["pod_conversion_binary"].drop("pod_conversion_binary").abs().sort_values(ascending=False)
        t5=cc2.head(5)
        cols5=st.columns(5)
        for i,(f,v) in enumerate(t5.items()):
            cols5[i].metric(lm.get(f,f),f"{v:.3f}")

    st.markdown("---")
    c1,c2=st.columns(2)
    with c1:
        st.markdown('<div class="sec">Avg Reasonable Price by Income</div>', unsafe_allow_html=True)
        sd2=df.copy(); sd2["r"]=sd2["psm_reasonable"].map(PSM_R_MID); sd2["inc"]=sd2["income_bracket"].map(INC_ORD)
        sd2=sd2.dropna(subset=["r","inc"])
        il={1:"<20K",2:"20-40K",3:"40-75K",4:"75-150K",5:">150K"}
        sd2["ilbl"]=sd2["inc"].map(il)
        aw=sd2.groupby("ilbl")["r"].mean().reset_index()
        oi=["<20K","20-40K","40-75K","75-150K",">150K"]
        aw["o"]=aw["ilbl"].map({v:i for i,v in enumerate(oi)})
        aw=aw.sort_values("o")
        fig=go.Figure(go.Bar(x=aw["ilbl"],y=aw["r"],marker_color=PRIMARY,
                             text=[f"₹{v:.0f}" for v in aw["r"]],textposition="outside"))
        fig.update_layout(height=300,plot_bgcolor="rgba(0,0,0,0)",paper_bgcolor="rgba(0,0,0,0)",
                          xaxis=dict(color="#ccc"),yaxis=dict(showgrid=False,color="#aaa",title="₹"),
                          font=dict(color="#ccc"),margin=dict(t=10,b=10,l=0,r=0))
        st.plotly_chart(fig,use_container_width=True)
    with c2:
        st.markdown('<div class="sec">Conversion Rate by City Tier</div>', unsafe_allow_html=True)
        if "city_tier" in df_enc.columns and "pod_conversion_binary" in df_enc.columns:
            cbc=df_enc.groupby("city_tier")["pod_conversion_binary"].mean().dropna()*100
            cbc=cbc.sort_values(ascending=False)
            fig=go.Figure(go.Bar(x=cbc.index,y=cbc.values,
                                 marker_color=[PRIMARY if v==cbc.max() else SECONDARY for v in cbc.values],
                                 text=[f"{v:.1f}%" for v in cbc.values],textposition="outside"))
            fig.update_layout(height=300,plot_bgcolor="rgba(0,0,0,0)",paper_bgcolor="rgba(0,0,0,0)",
                              xaxis=dict(color="#ccc"),yaxis=dict(showgrid=False,color="#aaa",title="%"),
                              font=dict(color="#ccc"),margin=dict(t=10,b=10,l=0,r=0))
            st.plotly_chart(fig,use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="sec">Statistical Tests</div>', unsafe_allow_html=True)
    c3,c4=st.columns(2)
    with c3:
        st.markdown("**Chi-Square: Past Box Cricket vs Conversion**")
        if "past_boxcricket" in df_enc.columns and "pod_conversion_binary" in df_enc.columns:
            td=df_enc[["past_boxcricket","pod_conversion_binary"]].dropna().copy()
            td["past_boxcricket"]=td["past_boxcricket"].fillna(0).astype(int)
            ct=pd.crosstab(td["past_boxcricket"],td["pod_conversion_binary"].astype(int))
            chi2,p,dof,_=stats.chi2_contingency(ct)
            st.dataframe(ct,use_container_width=True)
            st.markdown(f'<div class="ibox">χ² = <strong>{chi2:.2f}</strong> | p = <strong>{p:.4f}</strong> | dof = {dof}<br>'
                        f'{"✅ Significant (p<0.05) — past tech-leisure experience predicts conversion." if p<0.05 else "❌ Not significant."}</div>',
                        unsafe_allow_html=True)
    with c4:
        st.markdown("**ANOVA: Tech Adoption vs Pod Interest**")
        if "tech_adoption" in df_enc.columns:
            grps=[df_enc[df_enc["tech_adoption"]==g]["pod_interest"].dropna().values
                  for g in df_enc["tech_adoption"].unique() if len(df_enc[df_enc["tech_adoption"]==g])>5]
            if len(grps)>=2:
                fs,pv=stats.f_oneway(*grps)
                at=df_enc.groupby("tech_adoption")["pod_interest"].mean().sort_values(ascending=False)
                fig=go.Figure(go.Bar(x=at.index,y=at.values,marker_color=SECONDARY,
                                     text=[f"{v:.2f}" for v in at.values],textposition="outside"))
                fig.update_layout(height=220,plot_bgcolor="rgba(0,0,0,0)",paper_bgcolor="rgba(0,0,0,0)",
                                  xaxis=dict(color="#ccc",tickangle=-20),yaxis=dict(showgrid=False,color="#aaa"),
                                  font=dict(color="#ccc",size=10),margin=dict(t=10,b=60,l=0,r=0))
                st.plotly_chart(fig,use_container_width=True)
                st.markdown(f'<div class="ibox">F = <strong>{fs:.2f}</strong> | p = <strong>{pv:.4f}</strong><br>'
                            f'{"✅ Significant." if pv<0.05 else "❌ Not significant."}</div>',unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="sec">Competitor Frustration → Conversion Lift</div>', unsafe_allow_html=True)
    fcols=["frust_nodata","frust_coachattention","frust_timing","frust_crowded","frust_distance","frust_cost","frust_equipment","frust_notracking"]
    flab=["No data/feedback","Coach inattentive","Bad timing","Too crowded","Too far","High cost","Old equipment","No progress tracking"]
    af=[c for c in fcols if c in df_enc.columns]; laf=[flab[fcols.index(c)] for c in af]
    if af and "pod_conversion_binary" in df_enc.columns:
        rows=[]
        for col2,lbl in zip(af,laf):
            sub=df_enc[[col2,"pod_conversion_binary"]].dropna().copy()
            sub[col2]=sub[col2].fillna(0)
            g0=sub[sub[col2]==0]["pod_conversion_binary"].mean()
            g1=sub[sub[col2]==1]["pod_conversion_binary"].mean()
            rows.append({"Frustration":lbl,"Has frustration":g1,"No frustration":g0,"Lift":g1-g0})
        fd=pd.DataFrame(rows).sort_values("Lift",ascending=False)
        fig=go.Figure()
        fig.add_trace(go.Bar(name="Has frustration",x=fd["Frustration"],y=(fd["Has frustration"]*100).round(1),
                             marker_color=PRIMARY,text=[f"{v:.1f}%" for v in fd["Has frustration"]*100],textposition="outside"))
        fig.add_trace(go.Bar(name="No frustration",x=fd["Frustration"],y=(fd["No frustration"]*100).round(1),marker_color="#444"))
        fig.update_layout(barmode="group",height=340,plot_bgcolor="rgba(0,0,0,0)",paper_bgcolor="rgba(0,0,0,0)",
                          xaxis=dict(color="#ccc",tickangle=-25),yaxis=dict(showgrid=False,color="#aaa",title="Conversion %"),
                          legend=dict(font=dict(color="#ccc")),font=dict(color="#ccc"),margin=dict(t=10,b=80,l=0,r=0))
        st.plotly_chart(fig,use_container_width=True)
        st.markdown('<div class="ibox">📌 Respondents frustrated by "No data/feedback" and "No progress tracking" '
                    'show highest conversion lift — your core marketing message.</div>',unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════════
def page_classification():
    st.title("🎯 Classification — Will This Customer Convert?")
    st.markdown("**Random Forest** vs **Logistic Regression** — Accuracy, Precision, Recall, F1, ROC-AUC, Feature Importance.")
    st.markdown("---")

    res=models.get("all_results",{})
    clf=res.get("classification",{}); rf=clf.get("rf",{}); lr=clf.get("lr",{})
    if not rf: st.error("Classification results not found."); return

    met=["Accuracy","Precision","Recall","F1-Score","ROC-AUC"]
    rv=[rf["acc"],rf["prec"],rf["rec"],rf["f1"],rf["auc"]]
    lv=[lr["acc"],lr["prec"],lr["rec"],lr["f1"],lr["auc"]]

    st.markdown('<div class="sec">Performance Metrics</div>', unsafe_allow_html=True)
    cols5=st.columns(5)
    for i,(m,r2,l2) in enumerate(zip(met,rv,lv)):
        cols5[i].metric(m,f"{r2:.3f}",f"RF vs LR: {r2-l2:+.3f}")

    fig=go.Figure()
    fig.add_trace(go.Bar(name="Random Forest",x=met,y=rv,marker_color=PRIMARY,
                         text=[f"{v:.3f}" for v in rv],textposition="outside"))
    fig.add_trace(go.Bar(name="Logistic Regression",x=met,y=lv,marker_color=SECONDARY,
                         text=[f"{v:.3f}" for v in lv],textposition="outside"))
    fig.update_layout(barmode="group",height=300,plot_bgcolor="rgba(0,0,0,0)",paper_bgcolor="rgba(0,0,0,0)",
                      xaxis=dict(color="#ccc"),yaxis=dict(range=[0,1.15],showgrid=False,color="#aaa"),
                      legend=dict(font=dict(color="#ccc")),font=dict(color="#ccc"),margin=dict(t=10,b=10,l=0,r=0))
    st.plotly_chart(fig,use_container_width=True)
    st.markdown("---")

    c1,c2=st.columns(2)
    with c1:
        st.markdown('<div class="sec">ROC Curve</div>', unsafe_allow_html=True)
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=rf["fpr"],y=rf["tpr"],mode="lines",
                                  name=f"Random Forest (AUC={rf['auc']:.3f})",line=dict(color=PRIMARY,width=2.5)))
        fig.add_trace(go.Scatter(x=lr["fpr"],y=lr["tpr"],mode="lines",
                                  name=f"Logistic Reg (AUC={lr['auc']:.3f})",line=dict(color=SECONDARY,width=2,dash="dash")))
        fig.add_trace(go.Scatter(x=[0,1],y=[0,1],mode="lines",name="Baseline",line=dict(color="#555",width=1,dash="dot")))
        fig.update_layout(height=380,plot_bgcolor="rgba(0,0,0,0)",paper_bgcolor="rgba(0,0,0,0)",
                          xaxis=dict(title="False Positive Rate",color="#ccc",showgrid=True,gridcolor="#333"),
                          yaxis=dict(title="True Positive Rate",color="#aaa",showgrid=True,gridcolor="#333"),
                          legend=dict(font=dict(color="#ccc",size=11)),font=dict(color="#ccc"),margin=dict(t=10,b=40,l=0,r=0))
        st.plotly_chart(fig,use_container_width=True)
    with c2:
        st.markdown("**Confusion Matrix — Random Forest**")
        cm=np.array(rf["cm"]); lcm=["Not Interested","Interested"]
        fig=ff.create_annotated_heatmap(z=cm,x=lcm,y=lcm,colorscale=[[0,"#1a1a2e"],[1,PRIMARY]],
                                         annotation_text=cm.astype(str),showscale=True)
        fig.update_layout(height=300,paper_bgcolor="rgba(0,0,0,0)",font=dict(color="#ccc"),
                          margin=dict(t=30,b=60,l=0,r=0),
                          xaxis=dict(title="Predicted",color="#ccc"),yaxis=dict(title="Actual",color="#ccc"))
        st.plotly_chart(fig,use_container_width=True)
        mt=pd.DataFrame({"Metric":["Accuracy","Precision","Recall","F1-Score","ROC-AUC"],
                          "Random Forest":[f"{rf[k]:.4f}" for k in ["acc","prec","rec","f1","auc"]],
                          "Logistic Reg.":[f"{lr[k]:.4f}" for k in ["acc","prec","rec","f1","auc"]]})
        st.dataframe(mt,use_container_width=True,hide_index=True)

    st.markdown("---")
    st.markdown('<div class="sec">Feature Importance — Random Forest (Top 20)</div>', unsafe_allow_html=True)
    fi=clf.get("feat_imp",{})
    fnm={"role_num":"Cricket Role","pod_interest":"Pod Interest","data_importance":"Data Importance",
         "income_num":"Income","past_exp_count":"Past Exp Count","tech_num":"Tech Adoption",
         "nps_score":"NPS Score","feat_count":"Feature Count","addon_count":"Add-on Count",
         "barrier_count":"Barrier Count","age_num":"Age","city_num":"City Tier","spend_num":"Rec Spend",
         "practice_num":"Practice Days","frust_count":"Frustration Count","bar_aidistrust":"AI Distrust",
         "bar_notserious":"Not Serious","past_boxcricket":"Past Box Cricket","past_vr":"Past VR",
         "feat_ai":"Feature: AI","mem_num":"Membership WTP","dist_num":"Distance","digital_num":"Digital Spend",
         "gender_num":"Gender","edu_num":"Education"}
    if fi:
        fs=pd.Series(fi).sort_values(ascending=True)
        flab=[fnm.get(k,k) for k in fs.index]
        fc3=[PRIMARY if v>=fs.quantile(0.75) else ACCENT if v>=fs.quantile(0.5) else "#555" for v in fs.values]
        fig=go.Figure(go.Bar(x=fs.values,y=flab,orientation="h",marker_color=fc3,
                             text=[f"{v:.4f}" for v in fs.values],textposition="outside"))
        fig.update_layout(height=500,plot_bgcolor="rgba(0,0,0,0)",paper_bgcolor="rgba(0,0,0,0)",
                          xaxis=dict(showgrid=False,color="#aaa",title="Importance"),
                          yaxis=dict(color="#ccc"),font=dict(color="#ccc"),margin=dict(t=10,b=20,l=0,r=80))
        st.plotly_chart(fig,use_container_width=True)
        t3=[fnm.get(k,k) for k in list(pd.Series(fi).sort_values(ascending=False).head(3).index)]
        st.markdown(f'<div class="ibox">📌 <strong>Top 3 predictors:</strong> {t3[0]}, {t3[1]}, {t3[2]}.</div>',
                    unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="sec">Top High-Probability Leads</div>', unsafe_allow_html=True)
    try:
        cf2=models["clf_features"]; sc2=models["scaler_clf"]; rfm=models["rf_classifier"]
        vm=df_enc["pod_conversion_binary"].notna(); ds=df_enc[vm].copy()
        Xs=ds[cf2].fillna(ds[cf2].median())
        pr=rfm.predict_proba(sc2.transform(Xs))[:,1]
        ds["conversion_probability"]=pr
        ds["lead_grade"]=pd.cut(pr,bins=[0,0.45,0.65,0.80,1.0],labels=["Cold","Warm","Hot","Very Hot"])
        dc=["respondent_id","true_segment","city_tier","income_bracket","cricket_role","conversion_probability","lead_grade"]
        ad=[c for c in dc if c in ds.columns]
        tl2=ds.sort_values("conversion_probability",ascending=False)[ad].head(30)
        tl2["conversion_probability"]=tl2["conversion_probability"].round(3)
        st.dataframe(tl2,use_container_width=True,hide_index=True)
        gc=ds["lead_grade"].value_counts()
        ca2,cb2,cc2,cd2=st.columns(4)
        for col2,g,color in zip([ca2,cb2,cc2,cd2],["Very Hot","Hot","Warm","Cold"],[DANGER,ACCENT,PRIMARY,"#888"]):
            col2.markdown(f'<div class="metric-card"><div class="val" style="color:{color}">{int(gc.get(g,0))}</div>'
                          f'<div class="lbl">{g} Leads</div></div>',unsafe_allow_html=True)
    except Exception as e:
        st.warning(f"Could not score leads: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: CLUSTERING
# ══════════════════════════════════════════════════════════════════════════════
DISCOUNT_MAP={
    "Rising Star":          ("Student/U-18 discount + free first session","WhatsApp groups, school coaches"),
    "Elite Competitor":     ("Buy-5-get-1 + AI coaching bundle 15% off",  "Coach network, BCCI academies"),
    "Corporate Cricket Fan":("Corporate group package + employer tie-up",  "LinkedIn, HR managers"),
    "Recreational Player":  ("Weekend off-peak discount + referral offer", "Instagram, Google Maps"),
    "Sceptic / Disengaged": ("Free trial — no commitment needed",          "YouTube retargeting"),
}

def page_clustering():
    st.title("👥 Clustering — Customer Personas")
    st.markdown("K-Means segmentation: elbow curve, silhouette, PCA scatter, persona cards, and radar charts.")
    st.markdown("---")

    res=models.get("all_results",{}); cl=res.get("clustering",{})
    pm=models.get("persona_map",{})
    if not cl: st.error("Clustering results not found."); return

    kv=list(range(2,9))
    c1,c2=st.columns(2)
    with c1:
        st.markdown('<div class="sec">Elbow Curve</div>', unsafe_allow_html=True)
        fig=go.Figure(go.Scatter(x=kv,y=cl["inertias"],mode="lines+markers",
                                  line=dict(color=PRIMARY,width=2.5),marker=dict(color=ACCENT,size=8)))
        fig.add_vline(x=cl["best_k"],line_color=DANGER,line_dash="dash",
                      annotation_text=f"k={cl['best_k']}",annotation_font_color=DANGER)
        fig.update_layout(height=280,plot_bgcolor="rgba(0,0,0,0)",paper_bgcolor="rgba(0,0,0,0)",
                          xaxis=dict(title="k",color="#ccc"),yaxis=dict(title="Inertia",color="#aaa",showgrid=False),
                          font=dict(color="#ccc"),margin=dict(t=35,b=40,l=0,r=0))
        st.plotly_chart(fig,use_container_width=True)
    with c2:
        st.markdown('<div class="sec">Silhouette Score</div>', unsafe_allow_html=True)
        fig=go.Figure(go.Scatter(x=kv,y=cl["silhouettes"],mode="lines+markers",
                                  line=dict(color=SECONDARY,width=2.5),marker=dict(color=ACCENT,size=8)))
        bks=kv[cl["silhouettes"].index(max(cl["silhouettes"]))]
        fig.add_vline(x=bks,line_color=PRIMARY,line_dash="dash",
                      annotation_text=f"Best k={bks}",annotation_font_color=PRIMARY)
        fig.update_layout(height=280,plot_bgcolor="rgba(0,0,0,0)",paper_bgcolor="rgba(0,0,0,0)",
                          xaxis=dict(title="k",color="#ccc"),yaxis=dict(title="Silhouette",color="#aaa",showgrid=False),
                          font=dict(color="#ccc"),margin=dict(t=35,b=40,l=0,r=0))
        st.plotly_chart(fig,use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="sec">PCA 2D Cluster Scatter</div>', unsafe_allow_html=True)
    if "cluster" in df_enc.columns:
        Xc2=feat(df_enc,CLUSTERING_FEATURES)
        Xcs2=models["scaler_clust"].transform(Xc2)
        pca=PCA(n_components=2,random_state=42); pcs=pca.fit_transform(Xcs2)
        pdf=pd.DataFrame({"PC1":pcs[:,0],"PC2":pcs[:,1],"Cluster":df_enc["cluster"].astype(str)})
        fig=go.Figure()
        for ci in sorted(pdf["Cluster"].unique()):
            sub=pdf[pdf["Cluster"]==ci]; per=pm.get(int(ci),f"Cluster {ci}")
            fig.add_trace(go.Scatter(x=sub["PC1"],y=sub["PC2"],mode="markers",name=f"C{ci}: {per}",
                                      marker=dict(color=CLUSTER_COLORS[int(ci)%len(CLUSTER_COLORS)],size=5,opacity=0.65)))
        fig.update_layout(height=400,plot_bgcolor="rgba(0,0,0,0)",paper_bgcolor="rgba(0,0,0,0)",
                          xaxis=dict(title=f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)",color="#ccc",showgrid=True,gridcolor="#333"),
                          yaxis=dict(title=f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)",color="#aaa",showgrid=True,gridcolor="#333"),
                          legend=dict(font=dict(color="#ccc",size=10)),font=dict(color="#ccc"),margin=dict(t=10,b=40,l=0,r=0))
        st.plotly_chart(fig,use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="sec">Customer Personas & Recommended Strategies</div>', unsafe_allow_html=True)
    for prof in models.get("cluster_profiles",[]):
        ci=prof["cluster"]; per=pm.get(ci,f"Cluster {ci}")
        disc,ch=DISCOUNT_MAP.get(per,("Free trial","Social media"))
        color=CLUSTER_COLORS[ci%len(CLUSTER_COLORS)]
        cp=prof.get("conversion_rate",0)*100; asp=prof.get("avg_spend",0)
        st.markdown(f"""<div style="background:#1a1a2e;border:1px solid {color}55;border-left:4px solid {color};
border-radius:12px;padding:1rem 1.2rem;margin-bottom:10px">
<div style="display:flex;justify-content:space-between;margin-bottom:8px">
<div><span style="background:{color}22;color:{color};font-size:0.7rem;font-weight:600;padding:2px 8px;border-radius:10px">CLUSTER {ci}</span>
<span style="font-size:1.05rem;font-weight:600;color:#fff;margin-left:10px">{per}</span></div>
<span style="font-size:0.8rem;color:#aaa">{prof['size']:,} respondents</span></div>
<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:10px">
<div style="text-align:center"><div style="font-size:1.2rem;font-weight:700;color:{color}">{cp:.0f}%</div><div style="font-size:0.72rem;color:#aaa">Conversion</div></div>
<div style="text-align:center"><div style="font-size:1.2rem;font-weight:700;color:{color}">₹{asp:.0f}</div><div style="font-size:0.72rem;color:#aaa">Avg spend/mo</div></div>
<div style="text-align:center"><div style="font-size:1.2rem;font-weight:700;color:{color}">{prof['avg_pod_interest']:.1f}/5</div><div style="font-size:0.72rem;color:#aaa">Pod interest</div></div>
<div style="text-align:center"><div style="font-size:1.2rem;font-weight:700;color:{color}">{prof['avg_income']:.1f}/5</div><div style="font-size:0.72rem;color:#aaa">Income score</div></div>
</div>
<div style="font-size:0.8rem;color:#ccc"><span style="color:{color};font-weight:600">Best offer:</span> {disc}<br>
<span style="color:{color};font-weight:600">Reach via:</span> {ch}</div></div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="sec">Cluster Radar — Feature Profiles</div>', unsafe_allow_html=True)
    rf2=["age_num","income_num","role_num","practice_num","data_importance","pod_interest","spend_num","tech_num"]
    rl=["Age","Income","Cricket Role","Practice","Data Imp","Pod Interest","Spend","Tech"]
    if "cluster" in df_enc.columns:
        ar=[c for c in rf2 if c in df_enc.columns]; al=[rl[rf2.index(c)] for c in ar]
        cm2=df_enc.groupby("cluster")[ar].mean()
        nm=(cm2-cm2.min())/(cm2.max()-cm2.min()+1e-9)
        fig=go.Figure()
        for ci,row in nm.iterrows():
            per=pm.get(int(ci),f"Cluster {ci}"); color=CLUSTER_COLORS[int(ci)%len(CLUSTER_COLORS)]
            vals=row.tolist()+[row.tolist()[0]]; lbls=al+[al[0]]
            fig.add_trace(go.Scatterpolar(r=vals,theta=lbls,fill="toself",name=f"C{ci}: {per}",
                                           line=dict(color=color),fillcolor=color,opacity=0.25))
        fig.update_layout(polar=dict(bgcolor="rgba(0,0,0,0)",radialaxis=dict(visible=True,range=[0,1],color="#666"),
                                      angularaxis=dict(color="#ccc")),
                          paper_bgcolor="rgba(0,0,0,0)",height=420,
                          legend=dict(font=dict(color="#ccc",size=10)),font=dict(color="#ccc"),margin=dict(t=20,b=20,l=0,r=0))
        st.plotly_chart(fig,use_container_width=True)

    if "cluster" in df_enc.columns and "true_segment" in df_enc.columns:
        st.markdown('<div class="sec">Cluster Purity vs True Segment</div>', unsafe_allow_html=True)
        ct=pd.crosstab(df_enc["cluster"].map(lambda x: pm.get(x,str(x))),
                       df_enc["true_segment"],normalize="index").round(3)*100
        st.dataframe(ct.style.background_gradient(cmap="Greens",axis=1),use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ASSOCIATION RULES
# ══════════════════════════════════════════════════════════════════════════════
LBL={
    "feat_ai":"AI Analysis","feat_bowlingmachine":"Bowling Machine","feat_batspeed":"Bat Speed",
    "feat_footwork":"Footwork","feat_videoreplay":"Video Replay","feat_leaderboard":"Leaderboard",
    "feat_progressreport":"Progress Report","feat_appbooking":"App Booking",
    "addon_smartbat":"Smart Bat","addon_wearables":"Wearables","addon_aicoaching":"AI Coaching",
    "addon_highlights":"Video Highlights","addon_fitness":"Fitness Program",
    "use_academy":"Uses Academy","use_boxcricket":"Box Cricket","use_gym":"Gym","use_videoanalysis":"Video Analysis",
    "past_boxcricket":"Paid Box Cricket","past_trampoline":"Trampoline","past_vr":"VR Gaming",
    "past_fitclass":"Fitness Class","past_academy":"Paid Academy",
    "disc_freetrial":"Free Trial","disc_student":"Student Disc","disc_family":"Family Bundle",
    "disc_referral":"Referral Disc","disc_corporate":"Corporate Deal",
    "frust_nodata":"No Data/Feedback","frust_notracking":"No Tracking","frust_coachattention":"Coach Inattentive",
    "frust_timing":"Bad Timing","bar_price":"Barrier:Price","bar_aidistrust":"Barrier:AI Distrust",
    "bar_humancoach":"Wants Human Coach","bar_location":"Barrier:Location",
    "stream_hotstar":"Hotstar","stream_youtube":"YouTube","stream_netflix":"Netflix",
    "act_gym":"Gym","act_othersport":"Other Sport","act_videogaming":"Gaming","act_running":"Running",
}

def _p(s): return LBL.get(str(s).strip(), str(s).strip())

def page_association():
    st.title("🔗 Association Rule Mining")
    st.markdown("What products, features and behaviours go together — **Support, Confidence, Lift**.")
    st.markdown("---")

    rules=models.get("assoc_rules")
    if rules is None or not isinstance(rules,pd.DataFrame) or rules.empty:
        st.warning("No association rules found. Try lowering thresholds."); return

    c1,c2,c3=st.columns(3)
    mn_s=c1.slider("Min Support",   0.01,0.30,0.05,0.01)
    mn_c=c2.slider("Min Confidence",0.30,0.95,0.50,0.05)
    mn_l=c3.slider("Min Lift",      1.0, 5.0, 1.2, 0.1)

    filt=rules[(rules["support"]>=mn_s)&(rules["confidence"]>=mn_c)&(rules["lift"]>=mn_l)].copy()
    filt=filt.sort_values("lift",ascending=False).reset_index(drop=True)
    st.markdown(f"**{len(filt)} rules** match filters.")

    if filt.empty:
        st.info("No rules at current thresholds — lower Min Support or Lift."); return

    st.markdown("---")
    c1b,c2b,c3b,c4b=st.columns(4)
    c1b.metric("Total Rules",    len(filt))
    c2b.metric("Max Lift",       f"{filt['lift'].max():.3f}")
    c3b.metric("Max Confidence", f"{filt['confidence'].max():.3f}")
    c4b.metric("Max Support",    f"{filt['support'].max():.3f}")

    st.markdown('<div class="sec">Top Rules — Ranked by Lift</div>', unsafe_allow_html=True)
    dr=filt.head(30).copy()
    dr["antecedents"]=dr["antecedents"].apply(_p)
    dr["consequents"]=dr["consequents"].apply(_p)
    dr=dr[["antecedents","consequents","support","confidence","lift"]].round(4)
    dr.columns=["IF (Antecedent)","THEN (Consequent)","Support","Confidence","Lift"]
    st.dataframe(dr,use_container_width=True,hide_index=True)

    st.markdown("---")
    st.markdown('<div class="sec">Support vs Confidence — Bubble Size = Lift</div>', unsafe_allow_html=True)
    pd2=filt.head(80).copy()
    pd2["ap"]=pd2["antecedents"].apply(lambda x: _p(x)[:35])
    pd2["cp"]=pd2["consequents"].apply(lambda x: _p(x)[:35])
    fig=go.Figure(go.Scatter(x=pd2["support"],y=pd2["confidence"],mode="markers",
                              marker=dict(size=np.clip(pd2["lift"]*6,6,30),color=pd2["lift"],
                                          colorscale=[[0,"#333"],[0.5,SECONDARY],[1,PRIMARY]],
                                          showscale=True,colorbar=dict(title="Lift",tickfont=dict(color="#ccc")),opacity=0.8),
                              text=[f"IF: {a}<br>THEN: {c}<br>Conf={cf:.3f} | Lift={l:.3f}"
                                    for a,c,cf,l in zip(pd2["ap"],pd2["cp"],pd2["confidence"],pd2["lift"])],
                              hoverinfo="text"))
    fig.update_layout(height=400,plot_bgcolor="rgba(0,0,0,0)",paper_bgcolor="rgba(0,0,0,0)",
                      xaxis=dict(title="Support",color="#ccc",showgrid=True,gridcolor="#333"),
                      yaxis=dict(title="Confidence",color="#aaa",showgrid=True,gridcolor="#333"),
                      font=dict(color="#ccc"),margin=dict(t=10,b=40,l=0,r=0))
    st.plotly_chart(fig,use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="sec">Top 10 Rules by Lift</div>', unsafe_allow_html=True)
    t10=filt.head(10).copy()
    t10["rl"]=[f"{_p(str(a))[:28]}→ {_p(str(c))[:22]}" for a,c in zip(t10["antecedents"],t10["consequents"])]
    ts=t10.sort_values("lift",ascending=True)
    fig=go.Figure()
    fig.add_trace(go.Bar(y=ts["rl"],x=ts["lift"],orientation="h",name="Lift",marker_color=PRIMARY,
                          text=[f"{v:.3f}" for v in ts["lift"]],textposition="outside"))
    fig.add_trace(go.Bar(y=ts["rl"],x=ts["confidence"],orientation="h",name="Confidence",
                          marker_color=SECONDARY,text=[f"{v:.3f}" for v in ts["confidence"]],
                          textposition="outside",visible="legendonly"))
    fig.update_layout(height=400,barmode="overlay",plot_bgcolor="rgba(0,0,0,0)",paper_bgcolor="rgba(0,0,0,0)",
                      xaxis=dict(showgrid=False,color="#aaa",title="Score"),yaxis=dict(color="#ccc"),
                      legend=dict(font=dict(color="#ccc")),font=dict(color="#ccc"),margin=dict(t=10,b=20,l=0,r=80))
    st.plotly_chart(fig,use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="sec">Rule Categories</div>', unsafe_allow_html=True)
    t1,t2,t3=st.tabs(["🏏 Feature → Add-on Bundles","😤 Frustration Signals","🎯 Discount Patterns"])
    with t1:
        fr=filt[filt["antecedents"].str.startswith("feat_",na=False)&filt["consequents"].str.startswith("addon_",na=False)].head(15).copy()
        if len(fr):
            fr["antecedents"]=fr["antecedents"].apply(_p); fr["consequents"]=fr["consequents"].apply(_p)
            fr=fr[["antecedents","consequents","support","confidence","lift"]].round(4)
            fr.columns=["Feature Interest","Bundle Add-on","Support","Confidence","Lift"]
            st.dataframe(fr,use_container_width=True,hide_index=True)
        else: st.info("Lower thresholds to see feature→addon rules.")
    with t2:
        frr=filt[filt["antecedents"].str.startswith("frust_",na=False)].head(15).copy()
        if len(frr):
            frr["antecedents"]=frr["antecedents"].apply(_p); frr["consequents"]=frr["consequents"].apply(_p)
            frr=frr[["antecedents","consequents","support","confidence","lift"]].round(4)
            frr.columns=["Frustration","Associated Pattern","Support","Confidence","Lift"]
            st.dataframe(frr,use_container_width=True,hide_index=True)
        else: st.info("No frustration rules at current thresholds.")
    with t3:
        dr2=filt[filt["antecedents"].str.startswith("disc_",na=False)|filt["consequents"].str.startswith("disc_",na=False)].head(15).copy()
        if len(dr2):
            dr2["antecedents"]=dr2["antecedents"].apply(_p); dr2["consequents"]=dr2["consequents"].apply(_p)
            dr2=dr2[["antecedents","consequents","support","confidence","lift"]].round(4)
            dr2.columns=["Pattern A","Pattern B","Support","Confidence","Lift"]
            st.dataframe(dr2,use_container_width=True,hide_index=True)
        else: st.info("No discount pattern rules at current thresholds.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: REGRESSION
# ══════════════════════════════════════════════════════════════════════════════
CLBL={"income_num":"Income","role_num":"Cricket Role","pod_interest":"Pod Interest",
      "practice_num":"Practice Days","data_importance":"Data Importance","tech_num":"Tech Adoption",
      "past_exp_count":"Past Experiences","addon_count":"Add-on Interest","feat_count":"Feature Count",
      "spend_num":"Current Spend","nps_score":"NPS Score","city_num":"City Tier","age_num":"Age",
      "digital_num":"Digital Spend","mem_num":"Membership WTP","frust_count":"Frustrations",
      "dist_num":"Distance Tolerance","barrier_count":"Barriers"}

def page_regression():
    st.title("📈 Regression — How Much Will They Spend?")
    st.markdown("Predicting monthly spend using **Ridge** and **Linear Regression** + interactive revenue forecaster.")
    st.markdown("---")

    res=models.get("all_results",{}); reg=res.get("regression",{})
    if not reg: st.error("Regression results not found."); return

    r=reg["ridge"]; l=reg["lr"]
    c1,c2,c3,c4,c5,c6=st.columns(6)
    c1.metric("Ridge R²",   f"{r['r2']:.4f}")
    c2.metric("Ridge RMSE", f"₹{r['rmse']:.0f}")
    c3.metric("Ridge MAE",  f"₹{r['mae']:.0f}")
    c4.metric("LinReg R²",  f"{l['r2']:.4f}")
    c5.metric("LinReg RMSE",f"₹{l['rmse']:.0f}")
    c6.metric("LinReg MAE", f"₹{l['mae']:.0f}")
    st.markdown("---")

    co1,co2=st.columns(2)
    with co1:
        st.markdown('<div class="sec">Actual vs Predicted Spend</div>', unsafe_allow_html=True)
        yt=r["y_test"]; yp=r["y_pred"]; mv=max(max(yt),max(yp))
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=yt,y=yp,mode="markers",marker=dict(color=PRIMARY,size=4,opacity=0.5),name="Predictions"))
        fig.add_trace(go.Scatter(x=[0,mv],y=[0,mv],mode="lines",line=dict(color=DANGER,dash="dash",width=1.5),name="Perfect fit"))
        fig.update_layout(height=360,plot_bgcolor="rgba(0,0,0,0)",paper_bgcolor="rgba(0,0,0,0)",
                          xaxis=dict(title="Actual ₹",color="#ccc",showgrid=True,gridcolor="#333"),
                          yaxis=dict(title="Predicted ₹",color="#aaa",showgrid=True,gridcolor="#333"),
                          legend=dict(font=dict(color="#ccc")),font=dict(color="#ccc"),margin=dict(t=10,b=40,l=0,r=0))
        st.plotly_chart(fig,use_container_width=True)
    with co2:
        st.markdown('<div class="sec">Residual Plot</div>', unsafe_allow_html=True)
        res2=np.array(yp)-np.array(yt)
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=yp,y=res2,mode="markers",marker=dict(color=SECONDARY,size=4,opacity=0.5)))
        fig.add_hline(y=0,line_color=DANGER,line_dash="dash",line_width=1.5)
        fig.update_layout(height=360,plot_bgcolor="rgba(0,0,0,0)",paper_bgcolor="rgba(0,0,0,0)",
                          xaxis=dict(title="Predicted ₹",color="#ccc",showgrid=True,gridcolor="#333"),
                          yaxis=dict(title="Residual",color="#aaa",showgrid=True,gridcolor="#333"),
                          font=dict(color="#ccc"),margin=dict(t=10,b=40,l=0,r=0))
        st.plotly_chart(fig,use_container_width=True)

    st.markdown("---")
    ci=reg.get("coef_imp",{})
    if ci:
        st.markdown('<div class="sec">Ridge Feature Coefficients (|abs|)</div>', unsafe_allow_html=True)
        cs=pd.Series(ci).sort_values(ascending=True)
        cl2=[CLBL.get(k,k) for k in cs.index]
        fig=go.Figure(go.Bar(x=cs.values,y=cl2,orientation="h",
                             marker_color=[PRIMARY if v>=cs.quantile(0.7) else ACCENT if v>=cs.quantile(0.4) else "#555" for v in cs.values],
                             text=[f"{v:.2f}" for v in cs.values],textposition="outside"))
        fig.update_layout(height=420,plot_bgcolor="rgba(0,0,0,0)",paper_bgcolor="rgba(0,0,0,0)",
                          xaxis=dict(showgrid=False,color="#aaa",title="|Coeff|"),yaxis=dict(color="#ccc"),
                          font=dict(color="#ccc"),margin=dict(t=10,b=20,l=0,r=80))
        st.plotly_chart(fig,use_container_width=True)

    if "cluster" in df_enc.columns and "realistic_monthly_spend" in df_enc.columns:
        st.markdown("---")
        st.markdown('<div class="sec">Avg Spend by Customer Persona</div>', unsafe_allow_html=True)
        pm2=models.get("persona_map",{})
        sb=df_enc.groupby("cluster")["realistic_monthly_spend"].agg(["mean","std"]).reset_index()
        sb["persona"]=sb["cluster"].map(pm2); sb=sb.sort_values("mean",ascending=False)
        fig=go.Figure()
        for _,row in sb.iterrows():
            col2=CLUSTER_COLORS[int(row["cluster"])%len(CLUSTER_COLORS)]
            fig.add_trace(go.Bar(name=str(row["persona"]),x=[str(row["persona"])],y=[row["mean"]],
                                  error_y=dict(type="data",array=[row["std"]],visible=True,color="#666"),
                                  marker_color=col2,text=f"₹{row['mean']:.0f}",textposition="outside"))
        fig.update_layout(height=320,barmode="group",plot_bgcolor="rgba(0,0,0,0)",paper_bgcolor="rgba(0,0,0,0)",
                          xaxis=dict(color="#ccc"),yaxis=dict(title="Avg ₹/month",showgrid=False,color="#aaa"),
                          legend=dict(font=dict(color="#ccc")),font=dict(color="#ccc"),margin=dict(t=10,b=60,l=0,r=0))
        st.plotly_chart(fig,use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="sec">Interactive Revenue Forecaster</div>', unsafe_allow_html=True)
    f1,f2,f3,f4=st.columns(4)
    np2=f1.slider("Pods",          1, 50,  3)
    sd2=f2.slider("Sessions/pod/day",5, 30, 15)
    ap2=f3.slider("Avg price ₹",  100,500,220)
    oc =f4.slider("Occupancy %",   20,100, 60)
    ms=np2*sd2*26*(oc/100); mr=ms*ap2; ar=mr*12
    m1,m2,m3,m4=st.columns(4)
    m1.metric("Monthly Sessions",  f"{int(ms):,}")
    m2.metric("Monthly Revenue",   f"₹{mr:,.0f}")
    m3.metric("Annual Revenue",    f"₹{ar:,.0f}")
    m4.metric("Revenue/Pod/Month", f"₹{mr/max(np2,1):,.0f}")
    rp3=[mr*(1.02)**m for m in range(12)]
    fig=go.Figure(go.Scatter(x=[f"M{m}" for m in range(1,13)],y=rp3,mode="lines+markers",
                              line=dict(color=PRIMARY,width=2.5),marker=dict(color=ACCENT,size=7),
                              fill="tozeroy",fillcolor=f"{PRIMARY}22"))
    fig.update_layout(title="12-month projection (2% MoM growth)",height=280,
                      plot_bgcolor="rgba(0,0,0,0)",paper_bgcolor="rgba(0,0,0,0)",
                      xaxis=dict(color="#ccc"),yaxis=dict(title="₹",color="#aaa",showgrid=False),
                      font=dict(color="#ccc"),margin=dict(t=40,b=20,l=0,r=0))
    st.plotly_chart(fig,use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: NEW CUSTOMER PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════
def _act(p,per):
    if p>=0.75: return "🔥 HOT — Offer free trial immediately"
    elif p>=0.55: return "⭐ WARM — Send ₹100 first-session voucher"
    elif p>=0.40: return "🔄 NURTURE — Share AI demo video"
    return "📧 COLD — Add to newsletter only"

def _ch(per):
    return {"Rising Star":"WhatsApp/school coaches","Elite Competitor":"Cricket academy/coach referral",
            "Corporate Cricket Fan":"LinkedIn/HR tie-up","Recreational Player":"Instagram/Google Maps",
            "Sceptic / Disengaged":"YouTube retargeting"}.get(str(per),"Social media/digital ads")

def score_df(new_df):
    defaults={"age_group":"19-25","gender":"Male","city_tier":"Metro","occupation":"Salaried private",
              "income_bracket":"40K-75K","education":"Bachelors","cricket_role":"Regular","practice_days":"1-2",
              "data_importance":3,"fantasy_cricket":"Occasional","pod_interest":3,"monthly_rec_spend":"501-1000",
              "psm_too_cheap":"100-149","psm_reasonable":"200-249","psm_expensive":"400-499","psm_too_expensive":"600-799",
              "membership_wtp":"500-999","digital_spend":"201-500","food_delivery_freq":"1-2/week",
              "influence_source":"Self","tech_adoption":"Early majority","distance_tolerance":"Up to 5km",
              "preferred_timeslot":"Evening","nps_score":7}
    for c,v in defaults.items():
        if c not in new_df.columns: new_df[c]=v
    for c in MULTI_SELECT_COLS:
        if c not in new_df.columns: new_df[c]=0
    de=encode(new_df)
    cf=models["clf_features"]; rf2=[f for f in cf if f in de.columns]
    pr=models["rf_classifier"].predict_proba(models["scaler_clf"].transform(de[rf2].fillna(de[rf2].median())))[:,1]
    cc=[f for f in models["cluster_features"] if f in de.columns]
    cl=models["kmeans"].predict(models["scaler_clust"].transform(de[cc].fillna(de[cc].median())))
    pm3=models.get("persona_map",{})
    rr=[f for f in models["reg_features"] if f in de.columns]
    sp=np.clip(models["ridge_regressor"].predict(models["scaler_reg"].transform(de[rr].fillna(de[rr].median()))),0,5000)
    out=new_df.copy()
    out["conversion_probability"]=np.round(pr,3)
    out["lead_grade"]=pd.cut(pr,bins=[0,0.40,0.55,0.75,1.01],labels=["Cold","Warm","Hot","Very Hot"])
    out["persona"]=[pm3.get(c,f"Cluster {c}") for c in cl]
    out["predicted_spend_pm"]=np.round(sp,0)
    out["recommended_action"]=[_act(p,per) for p,per in zip(pr,out["persona"])]
    out["recommended_channel"]=[_ch(per) for per in out["persona"]]
    return out

def page_predictor():
    st.title("🚀 New Customer Predictor")
    st.markdown("Upload new survey responses → get **conversion probability, persona, predicted spend, and recommended action** per lead.")
    st.markdown("---")

    st.markdown('<div class="sec">Step 1 — Download Template</div>', unsafe_allow_html=True)
    sc=["age_group","gender","city_tier","occupation","income_bracket","education","cricket_role",
        "practice_days","data_importance","fantasy_cricket","pod_interest","monthly_rec_spend",
        "psm_too_cheap","psm_reasonable","psm_expensive","psm_too_expensive","membership_wtp",
        "digital_spend","food_delivery_freq","influence_source","tech_adoption","distance_tolerance",
        "preferred_timeslot","nps_score"]
    sr=[["19-25","Male","Metro","Salaried private","40K-75K","Bachelors","Regular","3-4",4,"Active",4,"1001-2500","100-149","200-249","400-499","600-799","1000-1999","201-500","3-4/week","Friends/Teammates","Early majority","Up to 5km","Evening",8],
        ["15-18","Male","Tier 1","School student","Below 20K","Up to 10th","Competitive","5-6",5,"Occasional",5,"501-1000","50-99","150-199","300-399","500-599","500-999","1-200","1-2/week","Parents/Family","Early adopter","Up to 3km","Early morning",9],
        ["26-35","Female","Metro","Professional","75K-150K","Masters+","Fan only","0",2,"Not interested",2,"1001-2500","150-199","250-299","500-599","800+","Would not subscribe","501-1000","1-2/week","Social media","Late majority","Any distance","Evening",6]]
    sd=pd.DataFrame(sr,columns=sc)
    buf=io.StringIO(); sd.to_csv(buf,index=False)
    st.download_button("⬇️ Download sample template CSV",data=buf.getvalue(),
                       file_name="new_customers_template.csv",mime="text/csv")
    st.markdown("---")

    st.markdown('<div class="sec">Step 2 — Upload & Score</div>', unsafe_allow_html=True)
    up=st.file_uploader("Upload CSV with new respondents",type=["csv"])

    if up:
        try:
            nd=pd.read_csv(up)
            st.success(f"✅ Loaded {len(nd):,} rows × {nd.shape[1]} columns")
            st.dataframe(nd.head(5),use_container_width=True)
            with st.spinner("Scoring leads..."):
                sc2=score_df(nd.copy())
            st.markdown("---")
            gc=sc2["lead_grade"].value_counts()
            c1,c2,c3,c4,c5=st.columns(5)
            c1.metric("Total",      len(sc2))
            c2.metric("🔥 Very Hot",int(gc.get("Very Hot",0)))
            c3.metric("⭐ Hot",     int(gc.get("Hot",0)))
            c4.metric("🔄 Warm",    int(gc.get("Warm",0)))
            c5.metric("📧 Cold",    int(gc.get("Cold",0)))
            ca,cb=st.columns(2)
            with ca:
                fig=go.Figure(go.Pie(labels=gc.index,values=gc.values,hole=0.45,
                                      marker_colors=[DANGER,ACCENT,PRIMARY,"#555"]))
                fig.update_layout(title="Lead Grade Distribution",height=280,paper_bgcolor="rgba(0,0,0,0)",
                                  legend=dict(font=dict(color="#ccc")),font=dict(color="#ccc"),margin=dict(t=35,b=10,l=0,r=0))
                st.plotly_chart(fig,use_container_width=True)
            with cb:
                pc=sc2["persona"].value_counts()
                fig=go.Figure(go.Bar(x=pc.values,y=pc.index,orientation="h",marker_color=SECONDARY,
                                      text=pc.values,textposition="outside"))
                fig.update_layout(title="Personas Detected",height=280,plot_bgcolor="rgba(0,0,0,0)",
                                  paper_bgcolor="rgba(0,0,0,0)",xaxis=dict(showgrid=False,color="#aaa"),
                                  yaxis=dict(color="#ccc"),font=dict(color="#ccc"),margin=dict(t=35,b=10,l=0,r=60))
                st.plotly_chart(fig,use_container_width=True)

            score_cols=[c for c in sc2.columns if c in nd.columns or
                        c in ["conversion_probability","lead_grade","persona","predicted_spend_pm","recommended_action","recommended_channel"]]
            st.dataframe(sc2[score_cols],use_container_width=True,hide_index=True)
            ob=io.StringIO(); sc2[score_cols].to_csv(ob,index=False)
            st.download_button("⬇️ Download scored leads CSV",data=ob.getvalue(),
                               file_name="scored_leads.csv",mime="text/csv")
        except Exception as e:
            st.error(f"Error: {e}"); st.exception(e)
    else:
        st.info("👆 Upload a CSV to score new customers.")
        st.markdown('<div class="ibox">📌 <strong>How it works:</strong><br>'
                    '1. CSV encoded with same pipeline as training<br>'
                    '2. Missing columns auto-filled with defaults<br>'
                    '3. Random Forest → conversion probability (0–1)<br>'
                    '4. K-Means → persona cluster<br>'
                    '5. Ridge Regression → predicted monthly spend<br>'
                    '6. Each row gets recommended action + channel</div>',unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# ROUTER
# ══════════════════════════════════════════════════════════════════════════════
if   "Home"           in page: page_home()
elif "Descriptive"    in page: page_descriptive()
elif "Diagnostic"     in page: page_diagnostic()
elif "Classification" in page: page_classification()
elif "Clustering"     in page: page_clustering()
elif "Association"    in page: page_association()
elif "Regression"     in page: page_regression()
elif "Predictor"      in page: page_predictor()
