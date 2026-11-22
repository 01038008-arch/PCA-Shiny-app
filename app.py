from shiny import App, ui, render, reactive
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json

DEFAULT_COLORS = ["#e74c3c","#3498db","#2ecc71","#f39c12","#9b59b6",
                  "#1abc9c","#e67e22","#34495e","#e91e63","#00bcd4"]

COLOR_THEMES = {
    "預設":     ["#e74c3c","#3498db","#2ecc71","#f39c12","#9b59b6","#1abc9c","#e67e22","#34495e"],
    "粉色系":   ["#ADADAD","#FF2D2D","#FF79BC","#FF77FF","#BE77FF","#7D7DFF","#46A3FF","#4DFFFF"],
    "自然色系": ["#A8FF24","#F9F900","#FFD306","#FFA042","#FF8040","#B87070","#AFAF61","#6FB7B7"],
    "莫蘭迪":   ["#B87070","#AFAF61","#6FB7B7","#9999CC","#B766AD","#7D7DFF","#FF79BC","#ADADAD"],
    "高對比":   ["#FF0000","#0000FF","#00AA00","#FF8800","#8800FF","#00CCCC","#FF0088","#888800"],
    "學術風":   ["#4878CF","#D65F5F","#6ACC65","#B47CC7","#C4AD66","#77BEDB","#F7910F","#D7301F"],
}

LOADING_CSS = """
<style>
.plot-wrapper { position: relative; min-height: 200px; }
.plot-loading {
    position: absolute; inset: 0; background: rgba(255,255,255,0.85);
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    z-index: 10; border-radius: 8px;
    font-size: 15px; color: #2c3e50; gap: 14px;
}
.spinner {
    width: 48px; height: 48px;
    border: 5px solid #dde; border-top-color: #2980b9;
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
}
@keyframes spin { to { transform: rotate(360deg); } }
</style>
"""

app_ui = ui.page_fluid(
    ui.tags.head(ui.HTML(LOADING_CSS)),
    ui.tags.style("""
        body { font-family: Arial, sans-serif; background: #f8f9fa; }
        .title-box { background: #2c3e50; color: white; padding: 18px 28px; margin-bottom: 20px; }
        .card { background: white; border-radius: 8px; padding: 20px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.08); margin-bottom: 16px; }
        .sec { font-size:14px; font-weight:600; color:#2c3e50;
               margin:10px 0 6px 0; border-left:3px solid #2980b9; padding-left:8px; }
        .color-row { display:flex; align-items:center; gap:8px; margin-bottom:6px; }
        .cprev { width:28px; height:28px; border-radius:4px;
                 border:1px solid #ccc; display:inline-block; flex-shrink:0; }
        table { border-collapse:collapse; width:100%; font-size:13px; }
        th { background:#2c3e50; color:white; padding:8px 12px;
             text-align:left; position:sticky; top:0; }
        td { padding:6px 12px; border-bottom:1px solid #eee; }
        tr:hover { background:#f0f4f8; }
        .table-scroll { max-height:480px; overflow-y:auto; }
        .dl-btn { background:#27ae60; color:white; padding:7px 16px;
                  border-radius:5px; text-decoration:none; font-size:13px; }
    """),

    ui.div({"class":"title-box"},
        ui.h2("🔬 PCA 分析工具", style="margin:0"),
        ui.p("上傳 CSV 或 Excel 檔案，自動繪製 PCA 圖",
             style="margin:4px 0 0 0; opacity:0.8")
    ),

    ui.row(
        # ══ 左側控制欄 ══
        ui.column(3, ui.div({"class":"card"},

            ui.div({"class":"sec"}, "📂 上傳資料"),
            ui.input_file("file","選擇檔案 (CSV / XLSX)",
                accept=[".csv",".xlsx"], placeholder="尚未選擇檔案"),
            ui.output_ui("sheet_selector"),
            ui.hr(),

            ui.div({"class":"sec"}, "⚙️ 基本設定"),
            ui.input_text("group_col","組別欄位名稱", value="Compound"),
            ui.input_numeric("n_components","PCA 主成分數", value=2, min=2, max=10),
            ui.hr(),

            ui.div({"class":"sec"}, "🎨 顏色設定"),
            ui.input_select("color_mode","配色模式",
                choices={"theme":"主題配色","manual":"手動輸入色號"}),
            ui.output_ui("color_theme_ui"),
            ui.output_ui("color_pickers"),

            # 自訂主題管理
            ui.output_ui("custom_theme_ui"),
            ui.hr(),

            ui.div({"class":"sec"}, "📐 圖表大小"),
            ui.input_slider("plot_width","寬度 (px)", min=400, max=1400, value=800, step=50),
            ui.input_slider("plot_height","高度 (px)", min=300, max=900,  value=520, step=50),
            ui.hr(),

            ui.div({"class":"sec"}, "📏 軸刻度間隔"),
            ui.input_numeric("x_tick","X 軸間隔（0=自動）", value=0, min=0),
            ui.input_numeric("y_tick","Y 軸間隔（0=自動）", value=0, min=0),
            ui.hr(),

            ui.div({"class":"sec"}, "🔧 圖表選項"),
            ui.input_checkbox("show_loading_plot","顯示 Loading Plot", value=False),
            ui.input_checkbox("show_ellipse","顯示群組橢圓", value=True),
            ui.input_numeric("point_size","點的大小", value=12, min=5, max=30),
            ui.hr(),

            ui.div({"class":"sec"}, "💾 匯出圖表"),
            ui.p("如需匯出圖片，請使用圖表區塊右上角工具列最左邊的 📷 相機圖示。",
                 style="font-size:12px; color:#555; line-height:1.6;"),
            ui.hr(),

            ui.div({"class":"sec"}, "📊 資料資訊"),
            ui.output_text("data_info"),
        )),

        # ══ 右側主畫面 ══
        ui.column(9,
            ui.navset_tab(
                ui.nav_panel("📈 PCA Score Plot",
                    ui.div({"class":"card"},
                        ui.div({"class":"plot-wrapper"},
                            ui.output_ui("plot_pca")
                        )
                    ),
                    ui.div({"class":"card"},
                        ui.div({"class":"sec"}, "解釋變異量"),
                        ui.output_ui("variance_table")
                    )
                ),
                ui.nav_panel("📉 Scree Plot",
                    ui.div({"class":"card"},
                        ui.div({"class":"plot-wrapper"},
                            ui.output_ui("plot_scree")
                        )
                    )
                ),
                ui.nav_panel("🗃️ Table",
                    ui.navset_tab(
                        ui.nav_panel("原始資料",
                            ui.div({"class":"card"},
                                ui.div({"class":"table-scroll"},
                                    ui.output_ui("table_raw")))
                        ),
                        ui.nav_panel("PCA Scores",
                            ui.div({"class":"card"},
                                ui.div({"class":"table-scroll"},
                                    ui.output_ui("table_scores")))
                        ),
                    )
                ),
            )
        )
    ),

    # 自訂主題資料（隱藏的 JSON 儲存）
    ui.tags.div({"id":"custom_themes_store", "style":"display:none"}, "{}"),
)

# ── 轉圈 HTML ──
SPINNER_HTML = """
<div class="plot-loading">
    <div class="spinner"></div>
    <span>Loading 中，請稍候…</span>
</div>
"""

THEMES_JS = """
<script>
// 自訂主題存在 localStorage
function saveCustomThemes(obj) {
    try { localStorage.setItem('pca_custom_themes', JSON.stringify(obj)); } catch(e){}
}
function loadCustomThemes() {
    try { return JSON.parse(localStorage.getItem('pca_custom_themes') || '{}'); } catch(e){ return {}; }
}
// 色塊即時預覽
document.addEventListener('input', function(e) {
    var id = e.target.id || '';
    if (id.startsWith('color_')) {
        var idx = id.replace('color_','');
        var prev = document.getElementById('preview_' + idx);
        if (prev) prev.style.background = e.target.value.trim();
    }
});
</script>
"""

def server(input, output, session):

    # ── 自訂主題（session 內記憶）──
    custom_themes = reactive.value({})
    # 色號快取：{ group_name: color }
    color_cache = reactive.value({})

    @reactive.calc
    def file_info():
        f = input.file()
        if not f:
            return None
        path = f[0]["datapath"]
        name = f[0]["name"].lower()
        is_excel = name.endswith(".xlsx") or name.endswith(".xls")
        sheets = pd.ExcelFile(path).sheet_names if is_excel else []
        return {"path":path, "is_excel":is_excel, "sheets":sheets}

    @output
    @render.ui
    def sheet_selector():
        info = file_info()
        if not info or not info["is_excel"]:
            return ui.div()
        return ui.div(
            ui.input_select("sheet","選擇工作表", choices=info["sheets"]),
            style="margin-top:10px;")

    @reactive.calc
    def loaded_data():
        info = file_info()
        if not info:
            return None
        try:
            if info["is_excel"]:
                sheet = input.sheet() if input.sheet() else info["sheets"][0]
                return pd.read_excel(info["path"], sheet_name=sheet)
            return pd.read_csv(info["path"])
        except:
            return None

    @reactive.calc
    def unique_groups():
        df = loaded_data()
        if df is None:
            return []
        gc = input.group_col()
        if gc not in df.columns:
            return []
        return list(dict.fromkeys(df[gc].astype(str).tolist()))

    # ── 主題選單（含自訂）──
    @output
    @render.ui
    def color_theme_ui():
        if input.color_mode() != "theme":
            return ui.div()
        all_themes = {**COLOR_THEMES, **custom_themes.get()}
        return ui.input_select("color_theme","選擇主題",
            choices=list(all_themes.keys()), selected="預設")

    # ── 手動色號（帶快取記憶）──
    @output
    @render.ui
    def color_pickers():
        if input.color_mode() != "manual":
            return ui.div()
        groups = unique_groups()
        if not groups:
            return ui.p("請先上傳資料", style="font-size:12px;color:#888;")
        cache = color_cache.get()
        rows = []
        for i, grp in enumerate(groups):
            # 優先用快取，否則預設色
            default = cache.get(grp, DEFAULT_COLORS[i % len(DEFAULT_COLORS)])
            rows.append(ui.div({"class":"color-row"},
                ui.tags.span({"id":f"preview_{i}", "class":"cprev",
                              "style":f"background:{default};"}),
                ui.tags.label(grp, style="min-width:65px;font-size:13px;"),
                ui.input_text(f"color_{i}","", value=default, width="110px"),
            ))
        return ui.div(*rows, ui.HTML(THEMES_JS))

    # ── 自訂主題管理 UI ──
    @output
    @render.ui
    def custom_theme_ui():
        if input.color_mode() != "manual":
            return ui.div()
        ct = custom_themes.get()
        saved_list = (ui.p("（尚無自訂主題）", style="font-size:12px;color:#888;")
                      if not ct else
                      ui.tags.ul(*[
                          ui.tags.li(name, style="font-size:12px;")
                          for name in ct.keys()
                      ], style="padding-left:16px;margin:4px 0;"))
        return ui.div(
            ui.hr(),
            ui.div({"class":"sec"}, "📌 自訂主題"),
            ui.input_text("new_theme_name","主題名稱", placeholder="輸入名稱後按儲存"),
            ui.input_action_button("save_theme","💾 儲存目前配色為主題",
                class_="btn btn-sm btn-outline-primary",
                style="margin-bottom:8px;"),
            ui.p("已儲存的自訂主題：", style="font-size:12px;margin:4px 0 2px 0;"),
            saved_list,
        )

    # ── 儲存自訂主題 ──
    @reactive.effect
    @reactive.event(input.save_theme)
    def _save_theme():
        name = (input.new_theme_name() or "").strip()
        if not name:
            return
        groups = unique_groups()
        colors = []
        for i, grp in enumerate(groups):
            try:
                val = input[f"color_{i}"]()
                colors.append(val.strip() if val else DEFAULT_COLORS[i % len(DEFAULT_COLORS)])
            except:
                colors.append(DEFAULT_COLORS[i % len(DEFAULT_COLORS)])
        ct = dict(custom_themes.get())
        ct[name] = colors
        custom_themes.set(ct)

    # ── 每次手動色號變更時更新快取 ──
    @reactive.effect
    def _update_cache():
        if input.color_mode() != "manual":
            return
        groups = unique_groups()
        cache = {}
        for i, grp in enumerate(groups):
            try:
                val = input[f"color_{i}"]()
                if val and val.strip():
                    cache[grp] = val.strip()
            except:
                pass
        if cache:
            color_cache.set(cache)

    def get_color_map():
        groups = unique_groups()
        all_themes = {**COLOR_THEMES, **custom_themes.get()}
        if input.color_mode() == "theme":
            theme_name = input.color_theme() if input.color_theme() else "預設"
            palette = all_themes.get(theme_name, DEFAULT_COLORS)
        else:
            palette = DEFAULT_COLORS
        color_map = {}
        for i, grp in enumerate(groups):
            if input.color_mode() == "manual":
                try:
                    val = input[f"color_{i}"]()
                    color_map[grp] = val.strip() if val and val.strip() else palette[i % len(palette)]
                except:
                    color_map[grp] = palette[i % len(palette)]
            else:
                color_map[grp] = palette[i % len(palette)]
        return color_map

    # ── PCA 計算（快取：只在資料/工作表/主成分數改變時重算）──
    @reactive.calc
    def run_pca():
        df = loaded_data()
        if df is None:
            return None
        gc = input.group_col()
        if gc not in df.columns:
            return None
        groups = df[gc].astype(str)
        X = df.drop(columns=[gc]).select_dtypes(include=[np.number]).dropna(axis=1)
        X_scaled = StandardScaler().fit_transform(X)
        n_comp = min(input.n_components(), X_scaled.shape[0], X_scaled.shape[1])
        pca = PCA(n_components=n_comp)
        scores = pca.fit_transform(X_scaled)
        pca_full = PCA(n_components=min(X_scaled.shape[0], X_scaled.shape[1]))
        pca_full.fit(X_scaled)
        return {
            "scores": scores,
            "loadings": pca.components_,
            "explained": pca.explained_variance_ratio_ * 100,
            "explained_full": pca_full.explained_variance_ratio_ * 100,
            "groups": groups.tolist(),
            "features": X.columns.tolist(),
        }

    def build_pca_fig():
        result = run_pca()
        if result is None:
            return None
        scores    = result["scores"]
        groups    = result["groups"]
        explained = result["explained"]
        loadings  = result["loadings"]
        features  = result["features"]
        color_map = get_color_map()
        unique_grps = list(dict.fromkeys(groups))

        fig = go.Figure()
        for grp in unique_grps:
            idx = [i for i,g in enumerate(groups) if g==grp]
            x = [scores[i,0] for i in idx]
            y = [scores[i,1] for i in idx]
            fig.add_trace(go.Scatter(
                x=x, y=y, mode="markers", name=grp,
                marker=dict(size=input.point_size(), color=color_map[grp],
                            line=dict(width=1.5, color="white")),
                hovertemplate=f"<b>{grp}</b><br>PC1: %{{x:.3f}}<br>PC2: %{{y:.3f}}<extra></extra>"
            ))
            if input.show_ellipse() and len(idx) >= 3:
                pts = np.array([[scores[i,0],scores[i,1]] for i in idx])
                cov  = np.cov(pts.T)
                mean = pts.mean(axis=0)
                vals, vecs = np.linalg.eigh(cov)
                order = vals.argsort()[::-1]
                vals, vecs = vals[order], vecs[:,order]
                t  = np.linspace(0, 2*np.pi, 100)
                s  = 2.0
                ex = mean[0]+s*np.sqrt(vals[0])*np.cos(t)*vecs[0,0]-s*np.sqrt(vals[1])*np.sin(t)*vecs[0,1]
                ey = mean[1]+s*np.sqrt(vals[0])*np.cos(t)*vecs[1,0]-s*np.sqrt(vals[1])*np.sin(t)*vecs[1,1]
                fig.add_trace(go.Scatter(x=ex, y=ey, mode="lines",
                    line=dict(color=color_map[grp], width=1.5, dash="dash"),
                    showlegend=False, hoverinfo="skip"))

        if input.show_loading_plot():
            mag = np.sqrt(loadings[0]**2+loadings[1]**2)
            top_idx = np.argsort(mag)[-10:]
            sf = max(abs(scores[:,0].max()), abs(scores[:,1].max())) * 0.8
            for i in top_idx:
                lx, ly = loadings[0,i]*sf, loadings[1,i]*sf
                fname = features[i][:18]+"…" if len(features[i])>20 else features[i]
                fig.add_annotation(x=lx, y=ly, text=fname, showarrow=True,
                    arrowhead=2, arrowcolor="#e74c3c",
                    font=dict(size=9, color="#e74c3c"), ax=0, ay=0)

        info = file_info()
        sheet_label = f" — {input.sheet()}" if info and info["is_excel"] and input.sheet() else ""
        xt = input.x_tick() or None
        yt = input.y_tick() or None
        fig.update_layout(
            title=dict(text=f"PCA Score Plot{sheet_label}", font=dict(size=18)),
            xaxis_title=f"PC1 ({explained[0]:.1f}%)",
            yaxis_title=f"PC2 ({explained[1]:.1f}%)",
            legend_title="組別",
            plot_bgcolor="white", paper_bgcolor="white",
            xaxis=dict(showgrid=True, gridcolor="#eee", zeroline=True,
                       zerolinecolor="#ccc", dtick=xt),
            yaxis=dict(showgrid=True, gridcolor="#eee", zeroline=True,
                       zerolinecolor="#ccc", dtick=yt),
            width=input.plot_width(), height=input.plot_height(),
            font=dict(family="Arial")
        )
        return fig

    def build_scree_fig():
        result = run_pca()
        if result is None:
            return None
        exp_full = result["explained_full"]
        show_n   = min(15, len(exp_full))
        x_labels = [f"PC{i+1}" for i in range(show_n)]
        y_vals   = exp_full[:show_n]
        cum      = np.cumsum(y_vals)
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=x_labels, y=y_vals, name="各主成分變異量",
            marker_color="#3498db", opacity=0.8), secondary_y=False)
        fig.add_trace(go.Scatter(x=x_labels, y=cum, name="累積變異量",
            mode="lines+markers", line=dict(color="#e74c3c", width=2),
            marker=dict(size=7)), secondary_y=True)
        xt = input.x_tick() or None
        yt = input.y_tick() or None
        fig.update_layout(
            title=dict(text="Scree Plot", font=dict(size=18)),
            plot_bgcolor="white", paper_bgcolor="white",
            xaxis=dict(showgrid=True, gridcolor="#eee", dtick=xt),
            legend=dict(x=0.6, y=0.4),
            width=input.plot_width(), height=input.plot_height(),
            font=dict(family="Arial")
        )
        fig.update_yaxes(title_text="解釋變異量 (%)", gridcolor="#eee",
                         showgrid=True, secondary_y=False, dtick=yt)
        fig.update_yaxes(title_text="累積變異量 (%)", range=[0,105], secondary_y=True)
        return fig

    # ── 圖表輸出（含轉圈 loading）──
    @output
    @render.ui
    def plot_pca():
        if loaded_data() is None:
            return ui.p("⬅️ 請先上傳資料檔案。",
                        style="color:#888;padding:40px;text-align:center;")
        # 資料載入中時顯示 spinner
        if run_pca() is None:
            return ui.HTML(SPINNER_HTML)
        fig = build_pca_fig()
        if fig is None:
            return ui.HTML(SPINNER_HTML)
        return ui.HTML(fig.to_html(include_plotlyjs="cdn", full_html=False))

    @output
    @render.ui
    def plot_scree():
        if loaded_data() is None:
            return ui.p("⬅️ 請先上傳資料檔案。",
                        style="color:#888;padding:40px;text-align:center;")
        if run_pca() is None:
            return ui.HTML(SPINNER_HTML)
        fig = build_scree_fig()
        if fig is None:
            return ui.HTML(SPINNER_HTML)
        return ui.HTML(fig.to_html(include_plotlyjs="cdn", full_html=False))

    @output
    @render.text
    def data_info():
        df = loaded_data()
        if df is None:
            return "尚未載入資料"
        info = file_info()
        gc   = input.group_col()
        n_groups = df[gc].nunique() if gc in df.columns else "?"
        n_feat   = len(df.columns) - 1
        sheet_info = f"\n工作表：{input.sheet()}" if info and info["is_excel"] and input.sheet() else ""
        return f"樣本數：{len(df)}\n變數數：{n_feat}\n組別數：{n_groups}{sheet_info}"

    @output
    @render.ui
    def variance_table():
        result = run_pca()
        if result is None:
            return ui.p("")
        explained = result["explained"]
        rows = ""
        cum  = 0
        for i, v in enumerate(explained):
            cum += v
            rows += f"<tr><td>PC{i+1}</td><td>{v:.2f}%</td><td>{cum:.2f}%</td></tr>"
        return ui.HTML(f"""
        <table style='max-width:400px'>
            <thead><tr><th>主成分</th><th>解釋變異量</th><th>累積變異量</th></tr></thead>
            <tbody>{rows}</tbody>
        </table>""")

    @output
    @render.ui
    def table_raw():
        df = loaded_data()
        if df is None:
            return ui.p("尚未載入資料", style="color:#888;padding:20px;")
        display_df = df.iloc[:, :50]
        truncated  = len(df.columns) > 50
        header = "".join(f"<th>{c}</th>" for c in display_df.columns)
        rows   = ""
        for _, row in display_df.iterrows():
            cells = "".join(
                f"<td>{v:.4f}</td>" if isinstance(v, float) else f"<td>{v}</td>"
                for v in row)
            rows += f"<tr>{cells}</tr>"
        note = (f"<p style='color:#e67e22;font-size:12px;'>"
                f"⚠️ 僅顯示前 50 欄（共 {len(df.columns)} 欄）</p>") if truncated else ""
        return ui.HTML(f"{note}<table><thead><tr>{header}</tr></thead>"
                       f"<tbody>{rows}</tbody></table>")

    @output
    @render.ui
    def table_scores():
        result = run_pca()
        if result is None:
            return ui.p("尚未載入資料", style="color:#888;padding:20px;")
        scores = result["scores"]
        groups = result["groups"]
        n_comp = scores.shape[1]
        header = ("<th>樣本</th><th>組別</th>" +
                  "".join(f"<th>PC{i+1}</th>" for i in range(n_comp)))
        rows = ""
        for i, (grp, row) in enumerate(zip(groups, scores)):
            cells = (f"<td>Sample {i+1}</td><td>{grp}</td>" +
                     "".join(f"<td>{v:.4f}</td>" for v in row))
            rows += f"<tr>{cells}</tr>"
        return ui.HTML(f"<table><thead><tr>{header}</tr></thead>"
                       f"<tbody>{rows}</tbody></table>")

app = App(app_ui, server)
