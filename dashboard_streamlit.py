# -*- coding: utf-8 -*-
"""
DASHBOARD STREAMLIT - KLASTER DERET WAKTU PENJUALAN MINGGUAN (DTW)
Berdasarkan analisis_klaster_dtw.py (preprocessing, min-max, DTW, PAM, HDBSCAN sama persis).

Instalasi : pip install streamlit pandas numpy scikit-learn dtaidistance plotly
Jalankan  : streamlit run dashboard_streamlit.py
"""
import glob
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dtaidistance import dtw
from sklearn.cluster import HDBSCAN

st.set_page_config(page_title="Dashboard Pola Penjualan Mingguan", page_icon="📊", layout="wide")

# ---------------------------------------------------------------- fungsi inti (dari kode asli)
def minmax(A):
    mn, mx = A.min(axis=1, keepdims=True), A.max(axis=1, keepdims=True)
    return (A - mn) / np.where(mx - mn == 0, 1, mx - mn)

def dtw_matrix(A, window):
    A = np.ascontiguousarray(A, dtype=np.double)
    D = dtw.distance_matrix_fast(A, window=window)
    D = np.where(np.isinf(D), 0, D)
    D = np.maximum(D, D.T)
    np.fill_diagonal(D, 0)
    return D

def pam(D, k, max_iter=100):
    m = len(D)
    med = [int(np.argmin(D.sum(1)))]
    for _ in range(k - 1):
        nearest = D[:, med].min(1)
        gain = np.maximum(nearest[:, None] - D, 0).sum(0)
        gain[med] = -1
        med.append(int(np.argmax(gain)))
    med = np.array(med)
    cost = D[:, med].min(1).sum()
    for _ in range(max_iter):
        best = (0.0, None, None)
        for mi in range(k):
            d_other = D[:, np.delete(med, mi)].min(1)
            new_cost = np.minimum(d_other[:, None], D).sum(0)
            new_cost[med] = np.inf
            h = int(np.argmin(new_cost))
            delta = new_cost[h] - cost
            if delta < best[0] - 1e-12:
                best = (delta, mi, h)
        if best[1] is None:
            break
        med[best[1]] = best[2]
        cost += best[0]
    return D[:, med].argmin(1)

def load_data(src):
    df = pd.read_csv(src, sep=None, engine="python")
    if df.shape[1] == 1:
        header = df.columns[0].split(";")
        df = df.iloc[:, 0].astype(str).str.split(";", expand=True)
        df.columns = header
    return df

@st.cache_data(show_spinner="Menghitung jarak DTW...")
def prepare(raw, zero_share, window):
    wk = sorted([c for c in raw.columns if c.startswith("W") and c[1:].isdigit()], key=lambda c: int(c[1:]))
    raw = raw.copy()
    raw[wk] = raw[wk].apply(pd.to_numeric, errors="coerce")
    raw = raw.dropna(subset=wk).drop_duplicates("Product_Code").reset_index(drop=True)
    X_all = raw[wk].values.astype(float)
    reg = (X_all == 0).mean(axis=1) <= zero_share
    X = X_all[reg]
    return wk, raw, X_all, reg, X, dtw_matrix(minmax(X), window)

@st.cache_data(show_spinner="Mengelompokkan produk...")
def cluster(D, method, param):
    if method == "HDBSCAN":
        return HDBSCAN(min_cluster_size=int(param), metric="precomputed").fit_predict(D.copy())
    return pam(D, int(param))

PAL = ["#ec4899", "#3b6cf5", "#8b5cf6", "#06b6d4", "#f472b6", "#60a5fa"]   # pink - biru - ungu - cyan
GREY = "#b4b8d6"
PINK, BLUE = "#ec4899", "#3b6cf5"

import plotly.io as pio
pio.templates["cantik"] = go.layout.Template(layout=dict(
    font=dict(family="Inter, Segoe UI, sans-serif", color="#312e81"), colorway=PAL,
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", hoverlabel=dict(bgcolor="white", font_color="#312e81"),
    xaxis=dict(gridcolor="rgba(120,120,190,.15)", zeroline=False), yaxis=dict(gridcolor="rgba(120,120,190,.15)", zeroline=False)))
pio.templates.default = "cantik"
st.markdown("""<style>
.stApp{background:linear-gradient(135deg,#eef2ff 0%,#fdf2f8 55%,#eef2ff 100%)}
h1{background:linear-gradient(90deg,#3b6cf5,#ec4899);-webkit-background-clip:text;-webkit-text-fill-color:transparent;font-weight:800!important}
h2,h3{color:#312e81!important}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#e0e7ff,#fce7f3)}
[data-testid="stMetric"]{background:rgba(255,255,255,.75);border:1px solid rgba(236,72,153,.25);border-radius:16px;padding:14px 18px;box-shadow:0 6px 18px rgba(59,108,245,.10)}
[data-testid="stMetricValue"],[data-testid="stMetricValue"] *{color:#3b6cf5!important}
[data-testid="stMetricLabel"] *,[data-testid="stMetricDelta"] *{color:#4c4a8a!important}
.stApp p,.stApp label,.stApp li,.stApp h2,.stApp h3,.stApp h4,[data-testid="stCaptionContainer"] *,[data-testid="stWidgetLabel"] *,[data-testid="stSidebar"] p,[data-testid="stSidebar"] label,[data-testid="stSidebar"] h2{color:#312e81!important}
[data-testid="stAlert"] *{color:#312e81!important}
.stDownloadButton button,.stButton button{background:linear-gradient(90deg,#3b6cf5,#ec4899);color:#fff;border:0;border-radius:12px}
[data-testid="stPlotlyChart"]{background:rgba(255,255,255,.7);border-radius:16px;padding:8px;box-shadow:0 6px 18px rgba(59,108,245,.08)}
div[role="radiogroup"]{gap:16px}
div[role="radiogroup"] label{background:rgba(255,255,255,.85);border:2px solid rgba(236,72,153,.35);border-radius:16px;padding:14px 36px;box-shadow:0 4px 14px rgba(59,108,245,.10);cursor:pointer}
div[role="radiogroup"] label p{font-size:1.5rem!important;font-weight:800!important;letter-spacing:.03em}
div[role="radiogroup"] label:has(input:checked){background:linear-gradient(90deg,#3b6cf5,#ec4899);border-color:transparent}
div[role="radiogroup"] label:has(input:checked) p{color:#fff!important}
div[role="radiogroup"] label>div:first-child{display:none}
[data-testid="stWidgetLabel"] p{font-size:1.05rem!important;font-weight:600!important}
</style>""", unsafe_allow_html=True)

# ---------------------------------------------------------------- sidebar
ADMIN = st.query_params.get("admin") == "1"                 # mode admin: tambahkan ?admin=1 di akhir link
DATA_FILE = next(iter(sorted(glob.glob("Sales_transaction*.csv")) or sorted(glob.glob("*.csv"))), None)
NEED_UPLOAD = DATA_FILE is None                              # tidak ada CSV di folder -> perlu unggah
up, zs, win = None, 0.30, 5                                  # nilai bawaan hasil tuning (dipakai pengunjung)
if ADMIN or NEED_UPLOAD:
    with st.sidebar:
        st.header("Pengaturan")
        up = st.file_uploader("Unggah data penjualan (CSV)", type="csv")
        if ADMIN:
            with st.expander("Parameter lanjutan"):
                zs = st.slider("Batas minggu nol produk 'jarang'", 0.05, 0.6, 0.30, 0.05)
                win = st.slider("Lebar pita DTW (minggu)", 1, 10, 5)
else:                                                        # pengunjung: sidebar disembunyikan
    st.markdown("<style>[data-testid='stSidebar'],[data-testid='stSidebarCollapsedControl'],[data-testid='collapsedControl']{display:none}</style>",
                unsafe_allow_html=True)

st.title("📊 Dashboard Pola Penjualan Mingguan")
cA, cB = st.columns([3, 2])
with cA:
    method_label = st.radio("Pilih metode klaster", ["HDBSCAN", "PAM"], horizontal=True,
                            help="PAM memasukkan semua produk ke klaster, dipakai sebagai pembanding.")
method = "HDBSCAN" if method_label == "HDBSCAN" else "PAM"
with cB:
    if ADMIN:
        param = st.slider("Ukuran minimum klaster" if method == "HDBSCAN" else "Jumlah klaster (k)",
                          *((3, 30, 5) if method == "HDBSCAN" else (2, 6, 2)))
    else:
        param = 5 if method == "HDBSCAN" else 2   # nilai terbaik hasil tuning, dikunci

if up is None:
    try:
        raw = load_data(DATA_FILE)
    except Exception:
        st.info("Unggah file CSV (kolom Product_Code, W0..W51) di sidebar untuk memulai.")
        st.stop()
else:
    raw = load_data(up)

wk, raw, X_all, is_reg, X, D = prepare(raw, zs, win)
codes = raw["Product_Code"].values[is_reg]
labels = cluster(D, method, param)

# ---------------------------------------------------------------- tabel produk + penamaan klaster
t = np.arange(len(wk))
df = pd.DataFrame({"Produk": codes, "label": labels,
                   "Rata-rata/minggu": X.mean(1), "Variasi (CV)": X.std(1) / np.where(X.mean(1) == 0, 1, X.mean(1)),
                   "Tren/minggu": [np.polyfit(t, x, 1)[0] for x in X]})
ks = sorted([k for k in np.unique(labels) if k >= 0], key=lambda k: -df.loc[df.label == k, "Rata-rata/minggu"].mean())
def klaster_label(i, n):
    tag = "Produk Terlaris" if i == 0 else ("Produk Kurang Laris" if i == n - 1 else "Produk Cukup Laris")
    return f"Klaster {i + 1} – {tag}"
name_of = {k: klaster_label(i, len(ks)) for i, k in enumerate(ks)}
name_of[-1] = "Produk Pola Beragam"
color_of = {k: PAL[i % len(PAL)] for i, k in enumerate(ks)}
color_of[-1] = GREY
df["Klaster"] = df.label.map(name_of)
order = ks + ([-1] if (labels == -1).any() else [])

# ---------------------------------------------------------------- header + KPI
st.caption(f"Produk dikelompokkan berdasarkan bentuk pola {len(wk)} minggu (jarak DTW) · metode: {method}")

tot = X_all.sum(0)
pk, pi = tot.max(), int(tot.argmax())
drops = tot[2:] - tot[:-2]
di = int(drops.argmin()) + 2
drop_pct = (1 - tot[di] / tot[di - 2]) * 100
c = st.columns(4)
c[0].metric("Total produk", f"{len(raw):,}")
c[1].metric("Produk reguler (diklaster)", f"{is_reg.sum():,}", f"{(~is_reg).sum()} produk jarang dikeluarkan", delta_color="off")
c[2].metric("Puncak mingguan", f"{pk:,.0f}", f"minggu {wk[pi]}", delta_color="off")
c[3].metric("Penurunan terdalam", f"−{(1 - tot.min() / pk) * 100:.0f}%", f"titik terendah {wk[int(tot.argmin())]}", delta_color="off")

# ---------------------------------------------------------------- tren total
st.subheader("Total penjualan per minggu")
f = go.Figure(go.Scatter(x=wk, y=tot, mode="lines", line=dict(width=3, color="#3b6cf5"), fill="tozeroy", fillcolor="rgba(59,108,245,.12)", name="Total"))
f.add_vrect(x0=wk[di - 2], x1=wk[di], fillcolor="#ec4899", opacity=0.15, line_width=0,
            annotation_text="Fluktuasi tajam", annotation_position="top left")
f.update_layout(height=380, margin=dict(l=10, r=10, t=20, b=10), yaxis_title="Unit terjual", hovermode="x unified")
f.update_xaxes(rangeslider_visible=True)
st.plotly_chart(f, width="stretch", theme=None)
st.markdown(f"Total penjualan **turun ±{drop_pct:.0f}%** dalam 2 minggu (**{wk[di-2]}→{wk[di]}**), lalu pulih bertahap ke "
            f"**{tot[-1]:,.0f}** di akhir periode (**{tot[-1] / pk * 100:.0f}% dari puncak**).")

# ---------------------------------------------------------------- klaster
st.subheader("Profil klaster")
cols = st.columns(len(order))
prof = {}
for col, k in zip(cols, order):
    g = df[df.label == k]
    prof[k] = g
    col.markdown(f"<div style='border-left:6px solid {color_of[k]};padding:12px 16px;border-radius:14px;background:rgba(255,255,255,.75);box-shadow:0 6px 18px rgba(59,108,245,.10);color:#312e81'>"
                 f"<b>{name_of[k]}</b><br><span style='font-size:26px;font-weight:700'>{len(g)} produk</span><br>"
                 f"<small>{g['Rata-rata/minggu'].mean():.1f} unit/minggu · CV {g['Variasi (CV)'].mean():.2f}<br>"
                 f"tren {'naik' if g['Tren/minggu'].mean() > 0 else 'turun'} {abs(g['Tren/minggu'].mean()):.2f}/minggu</small></div>",
                 unsafe_allow_html=True)

l, r = st.columns(2)
with l:
    st.markdown("**Pola rata-rata mingguan tiap klaster**")
    f = go.Figure()
    for k in order:
        f.add_scatter(x=wk, y=X[labels == k].mean(0), name=name_of[k], line=dict(color=color_of[k], width=2.5))
    f.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.2), hovermode="x unified")
    st.plotly_chart(f, width="stretch", theme=None)
with r:
    st.markdown("**Rata-rata penjualan vs kestabilan produk** (tiap titik = 1 produk)")
    f = go.Figure()
    for k in order:
        g = df[df.label == k]
        f.add_scatter(x=g["Rata-rata/minggu"], y=g["Variasi (CV)"], mode="markers", name=name_of[k], text=g["Produk"],
                      marker=dict(color=color_of[k], size=8, opacity=0.8, line=dict(width=1, color="white")),
                      hovertemplate="<b>%{text}</b><br>Rata-rata/minggu: %{x:.1f}<br>CV: %{y:.2f}<extra></extra>")
    f.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="Rata-rata unit terjual per minggu (makin ke kanan makin laris)",
                    yaxis_title="Variasi penjualan/CV (makin ke bawah makin stabil)", legend=dict(orientation="h", y=-0.25))
    st.plotly_chart(f, width="stretch", theme=None)

# ---------------------------------------------------------------- produk
st.subheader("Produk teratas per klaster")
pick = st.selectbox("Pilih klaster", [name_of[k] for k in order])
g = df[df.Klaster == pick].sort_values("Rata-rata/minggu", ascending=False).head(15)
cl, cr = st.columns([1, 1.4])
with cl:
    st.dataframe(g[["Produk", "Rata-rata/minggu", "Variasi (CV)", "Tren/minggu"]].round(2), hide_index=True, width="stretch")
with cr:
    prod = st.selectbox("Lihat deret mingguan produk", g["Produk"].tolist())
    s = X[list(codes).index(prod)]
    f = go.Figure(go.Scatter(x=wk, y=s, mode="lines+markers", line=dict(color="#ec4899")))
    f.add_hline(y=s.mean(), line_dash="dot", annotation_text="rata-rata")
    f.update_layout(height=290, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(f, width="stretch", theme=None)

# ---------------------------------------------------------------- cari produk
st.subheader("🔍 Cari produk")
cmap = dict(zip(codes, df["Klaster"]))
allp = pd.DataFrame({"Produk": raw["Product_Code"].astype(str).values, "Rata-rata/minggu": X_all.mean(1),
                     "Variasi (CV)": X_all.std(1) / np.where(X_all.mean(1) == 0, 1, X_all.mean(1)),
                     "Tren/minggu": [np.polyfit(t, x, 1)[0] for x in X_all], "Minggu nol (%)": (X_all == 0).mean(1) * 100})
allp["Klaster"] = allp["Produk"].map(cmap).fillna("Produk Jarang (tidak diklaster)")
q = st.text_input("Ketik kode produk (boleh sebagian, mis. P12)", placeholder="P1")
if q.strip():
    hit = allp[allp["Produk"].str.contains(q.strip(), case=False, regex=False)]
    if hit.empty:
        st.warning(f"Tidak ada produk dengan kode yang mengandung '{q}'.")
    else:
        st.caption(f"{len(hit)} produk ditemukan")
        st.dataframe(hit.round(2), hide_index=True, width="stretch", height=min(300, 40 + 35 * len(hit)))
        one = st.selectbox("Lihat detail produk", hit["Produk"].tolist()) if len(hit) > 1 else hit["Produk"].iloc[0]
        row = allp[allp["Produk"] == one].iloc[0]
        series = X_all[list(allp["Produk"]).index(one)]
        m = st.columns(4)
        m[0].metric("Klaster", row["Klaster"])
        m[1].metric("Rata-rata/minggu", f"{row['Rata-rata/minggu']:.1f}")
        m[2].metric("Variasi (CV)", f"{row['Variasi (CV)']:.2f}")
        m[3].metric("Tren/minggu", f"{row['Tren/minggu']:+.2f}")
        f = go.Figure(go.Scatter(x=wk, y=series, name=one, mode="lines+markers", line=dict(color="#ec4899", width=2.5)))
        if one in cmap:
            kk = [k for k, v in name_of.items() if v == row["Klaster"]]
            if kk:
                f.add_scatter(x=wk, y=X[labels == kk[0]].mean(0), name="Rata-rata klaster", line=dict(color=GREY, dash="dot"))
        f.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(f, width="stretch", theme=None)

# ---------------------------------------------------------------- insight
st.subheader("Insight utama")
top = max(order, key=lambda k: prof[k]["Rata-rata/minggu"].mean())
tp = prof[top]
tr = lambda v: f"**{'naik' if v > 0 else 'turun'} {abs(v):.2f}/minggu**"
ins = [f"**Fluktuasi di pertengahan periode:** total penjualan **turun ±{drop_pct:.0f}%** dalam 2 minggu (**{wk[di-2]}→{wk[di]}**) "
       f"dan belum kembali ke puncak. Pola ini dialami banyak produk sekaligus, jadi penyebabnya kemungkinan **faktor eksternal** (musim, harga, stok, kebijakan).",
       f"**{name_of[top]}:** hanya **{len(tp)} produk**, tetapi rata-rata **{tp['Rata-rata/minggu'].mean():.1f} unit/minggu** dengan variasi {tp['Variasi (CV)'].mean():.2f}; "
       f"tren {tr(tp['Tren/minggu'].mean())} — **prioritaskan menjaga stok** dan pantau arah trennya."]
for k in ks:
    if k != top:
        p_ = prof[k]
        ins.append(f"**{name_of[k]}:** **{len(p_)} produk**, rata-rata **{p_['Rata-rata/minggu'].mean():.1f} unit/minggu**, tren {tr(p_['Tren/minggu'].mean())}.")
if -1 in prof:
    ins.append(f"**{name_of[-1]}:** **{len(prof[-1])} dari {len(df)} produk reguler** tidak masuk klaster padat mana pun (polanya tidak seragam), "
               "sehingga **perlu strategi per produk**. Gunakan metode PAM sebagai pembanding bila ingin setiap produk punya kelompok.")
ins.append(f"**Produk jarang:** **{(~is_reg).sum()} produk ({(~is_reg).mean() * 100:.0f}%)** sering kosong penjualannya — **tinjau kelayakan stok**.")
for i in ins:
    st.markdown("- " + i)

st.download_button("⬇️ Unduh hasil klaster (CSV)", df.drop(columns="label").round(3).to_csv(index=False).encode("utf-8"),
                   "hasil_klaster_dashboard.csv", "text/csv")
