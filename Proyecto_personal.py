import streamlit as st 
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from datetime import datetime
from prophet import Prophet
import warnings

# Suprime warnings innecesarios
warnings.filterwarnings("ignore")

st.title("Dividendos, Rentabilidad y Análisis de Acciones en el IBEX 35")

# -------------------------------
# Funciones con caché para datos
# -------------------------------
@st.cache_data
def get_full_data(ticker):
    """
    Descarga y procesa los datos históricos completos para el ticker desde el 2000 hasta hoy.
    Se utiliza caching para evitar múltiples requests.
    """
    end_date = (datetime.today() + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    try:
        data = yf.download(ticker, start="2000-01-01", end=end_date, progress=False)
    except Exception:
        return pd.DataFrame()
    if data.empty:
        return data
    precio_col = "Adj Close" if "Adj Close" in data.columns else "Close"
    data["Precio"] = data[precio_col]
    return data

@st.cache_data
def get_data(ticker, start, end):
    """
    Descarga datos históricos para un ticker en un rango de fechas específico.
    """
    try:
        data = yf.download(ticker, start=start, end=end, progress=False)
    except Exception:
        return pd.DataFrame()
    if data.empty:
        return data
    precio_col = "Adj Close" if "Adj Close" in data.columns else "Close"
    data["Precio"] = data[precio_col]
    return data

@st.cache_data
def get_ticker_info(ticker):
    """
    Obtiene y cachea la información de un ticker.
    """
    try:
        info = yf.Ticker(ticker).info
    except Exception:
        info = {}
    return info

@st.cache_data
def get_rt_data(tickers_str):
    """
    Descarga y cachea los datos en tiempo real para todos los tickers (una única request)
    hasta que se recargue la web.
    """
    try:
        data_rt = yf.download(tickers_str, period='1d', interval='1m', group_by='ticker', progress=False)
        return data_rt
    except Exception:
        return None

# Función para calcular el RSI (14 días por defecto)
def calcular_RSI(data, period=14):
    delta = data["Precio"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Diccionario con algunas empresas del IBEX 35 y sus tickers en Yahoo Finance.
empresas = {
    "Acciona": "ANA.MC",
    "Acciona Energías": "ANE.MC",  
    "Acerinox": "ACX.MC",
    "ACS": "ACS.MC",
    "Aena": "AENA.MC",
    "Amadeus": "AMS.MC",
    "ArcelorMittal": "MTS.MC",       
    "Banco Sabadell": "SAB.MC",
    "Banco Santander": "SAN.MC",
    "Bankinter": "BKT.MC",
    "BBVA": "BBVA.MC",
    "CaixaBank": "CABK.MC",
    "Cellnex": "CLNX.MC",
    "Colonial": "COL.MC",
    "Enagás": "ENG.MC",
    "Endesa": "ELE.MC",
    "Ferrovial": "FER.MC",
    "Fluidra": "FDR.MC",          
    "Grifols": "GRF.MC",
    "IAG": "IAG.MC",
    "Iberdrola": "IBE.MC",
    "Inditex": "ITX.MC",
    "Indra": "IDR.MC",
    "Logista": "LOG.MC",
    "Mapfre": "MAP.MC",
    "Merlin Properties": "MRL.MC",
    "Naturgy": "NTGY.MC",
    "Puig": "PUIG.MC",
    "Redeia": "RED.MC",          
    "Repsol": "REP.MC",
    "Rovi": "ROVI.MC",           
    "Sacyr": "SCYR.MC",           
    "Solaria": "SLR.MC",
    "Telefónica": "TEF.MC",
    "Unicaja": "UNI.MC"
}


# Barra lateral para seleccionar la sección a visualizar
seccion = st.sidebar.selectbox(
    "Selecciona la sección a visualizar:",
    ["Gráfica Histórica", "Rentabilidad Dividendaria",
     "Predicción de Acción", "Análisis Técnico", "Análisis Fundamental",
     "Mejor Momento de Compra", "Comparación de Cierres Diarios Normalizados"]
)



# =====================================================
# Sección 1: Gráfica Histórica
# =====================================================
if seccion == "Gráfica Histórica":
    st.subheader("Evolución Histórica del Precio de una Empresa")
    empresa_sel = st.selectbox("Selecciona una empresa del IBEX 35:", list(empresas.keys()))
    col1, col2 = st.columns(2)
    with col1:
        fecha_inicio = st.date_input("Fecha de inicio",
                                     datetime(datetime.today().year - 25, 1, 1),
                                     min_value=datetime(1900, 1, 1),
                                     max_value=datetime.today())
    with col2:
        fecha_fin = st.date_input("Fecha final",
                                  datetime.today(),
                                  min_value=datetime(1900, 1, 1),
                                  max_value=datetime.today())
    if st.button("Mostrar gráfico", key="hist"):
        ticker_sym = empresas[empresa_sel]
        st.write(f"Descargando datos de **{empresa_sel}** ({ticker_sym}) desde {fecha_inicio} hasta {fecha_fin}...")
        data_hist = get_data(ticker_sym, fecha_inicio.strftime("%Y-%m-%d"), fecha_fin.strftime("%Y-%m-%d"))
        if data_hist.empty:
            st.error("No se han obtenido datos para esta empresa.")
        else:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=data_hist.index,
                y=data_hist["Precio"],
                mode='lines',
                name=f'Valor de {empresa_sel}'
            ))
            fig.update_layout(
                title=f"Acciones de {empresa_sel} entre {fecha_inicio} y {fecha_fin}",
                xaxis_title="Fecha",
                yaxis_title="Precio (€)",
                template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True)

# =====================================================
# Sección 2: Rentabilidad Dividendaria
# =====================================================
elif seccion == "Rentabilidad Dividendaria":
    st.subheader("Rentabilidad Dividendaria en Tiempo Real (basada en dividendos anuales)")
    # Se realiza una única request a yfinance para todos los tickers (almacenada en caché)
    tickers_list = list(empresas.values())
    tickers_str = " ".join(tickers_list)
    data_rt = get_rt_data(tickers_str)
    
    resultados = []
    if data_rt is not None and not data_rt.empty:
        for emp, ticker in empresas.items():
            try:
                if ticker in data_rt:
                    df_ticker = data_rt[ticker]
                    precio_actual = df_ticker['Close'].iloc[-1]
                else:
                    continue
            except Exception:
                continue

            info = get_ticker_info(ticker)
            try:
                dividend = info.get("trailingAnnualDividendRate") or info.get("dividendRate")
                dividend = float(dividend) if dividend is not None else 0.0
            except Exception:
                dividend = 0.0

            try:
                rentabilidad = round((dividend / precio_actual) * 100, 2) if precio_actual > 0 else None
            except Exception:
                rentabilidad = None

            try:
                market_cap = info.get("marketCap")
                market_cap_str = f"{market_cap/1e9:.2f}B €" if market_cap is not None else "N/A"
            except Exception:
                market_cap_str = "N/A"

            try:
                one_year_ago = datetime.today() - pd.DateOffset(years=1)
                hist_year = yf.Ticker(ticker).history(start=one_year_ago, end=datetime.today())
                if not hist_year.empty:
                    precio_year = hist_year['Close'].iloc[0]
                    precio_actual_adj = precio_actual - dividend if dividend > 0 else precio_actual
                    cambio = round(((precio_actual_adj - precio_year) / precio_year) * 100, 2) if precio_year > 0 else None
                else:
                    cambio = None
            except Exception:
                cambio = None

            resultados.append({
                "Empresa": emp,
                "Ticker": ticker,
                "Precio Actual (€/acción)": round(precio_actual, 2) if precio_actual is not None else None,
                "Dividendos Anuales (€/acción)": round(dividend, 2) if dividend is not None else None,
                "Rentabilidad Dividendaria (%)": rentabilidad,
                "Capitalización": market_cap_str,
                "Cambio Último Año (%)": cambio
            })
        if resultados:
            st.dataframe(pd.DataFrame(resultados))
        else:
            st.warning("No se han obtenido datos de rentabilidad.")
    else:
        st.error("No se pudieron obtener datos en tiempo real.")

# =====================================================
# Sección 3: Predicción de Acción
# =====================================================
elif seccion == "Predicción de Acción":
    st.subheader("Predicción de Acción")
    accion_pred = st.selectbox("Selecciona la acción a predecir:", list(empresas.keys()))
    ticker_pred = empresas[accion_pred]

    def preparar_datos_prediccion(ticker, anios=5):
        data_full = get_full_data(ticker)
        if data_full.empty:
            return None
        fecha_lim = datetime.today() - pd.DateOffset(years=anios)
        data_reciente = data_full[data_full.index >= fecha_lim]
        df_prophet = data_reciente.reset_index()[["Date", "Precio"]].dropna()
        df_prophet.rename(columns={"Date": "ds", "Precio": "y"}, inplace=True)
        return df_prophet

    tipo_pred = st.radio("Tipo de predicción:", ["Corto Plazo (30 días)", "Largo Plazo (1 año)"])
    if tipo_pred == "Corto Plazo (30 días)":
        if st.button("Predecir 30 días", key="corto"):
            df_pred = preparar_datos_prediccion(ticker_pred, anios=5)
            if df_pred is None or df_pred.empty:
                st.error("No se pudieron obtener datos para la predicción.")
            else:
                model = Prophet()
                model.fit(df_pred)
                future = model.make_future_dataframe(periods=30)
                forecast = model.predict(future)
                ultimo = df_pred['ds'].max()
                forecast_future = forecast[forecast['ds'] > ultimo]

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=forecast_future['ds'],
                    y=forecast_future['yhat'],
                    mode='lines',
                    name='Predicción'
                ))
                fig.add_trace(go.Scatter(
                    x=df_pred['ds'],
                    y=df_pred['y'],
                    mode='lines',
                    name='Histórico (últimos 5 años)'
                ))
                fig.update_layout(
                    title=f"Predicción a Corto Plazo (30 días) para {accion_pred}",
                    xaxis_title="Fecha",
                    yaxis_title="Precio (€)",
                    template="plotly_white"
                )
                st.plotly_chart(fig, use_container_width=True)
    else:
        if st.button("Predecir 1 año", key="largo"):
            df_pred = preparar_datos_prediccion(ticker_pred, anios=5)
            if df_pred is None or df_pred.empty:
                st.error("No se pudieron obtener datos para la predicción.")
            else:
                model = Prophet()
                model.fit(df_pred)
                future = model.make_future_dataframe(periods=365)
                forecast = model.predict(future)
                ultimo = df_pred['ds'].max()
                forecast_future = forecast[forecast['ds'] > ultimo]

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=forecast_future['ds'],
                    y=forecast_future['yhat'],
                    mode='lines',
                    name='Predicción'
                ))
                fig.add_trace(go.Scatter(
                    x=df_pred['ds'],
                    y=df_pred['y'],
                    mode='lines',
                    name='Histórico (últimos 5 años)'
                ))
                fig.update_layout(
                    title=f"Predicción a Largo Plazo (1 año) para {accion_pred}",
                    xaxis_title="Fecha",
                    yaxis_title="Precio (€)",
                    template="plotly_white"
                )
                st.plotly_chart(fig, use_container_width=True)

# =====================================================
# Sección 4: Análisis Técnico
# =====================================================
elif seccion == "Análisis Técnico":
    st.subheader("Análisis Técnico")
    empresa_tecnico = st.selectbox("Selecciona una empresa para análisis técnico:", list(empresas.keys()))
    ticker_tecnico = empresas[empresa_tecnico]
    fecha_inicio = (datetime.today() - pd.DateOffset(years=1)).strftime("%Y-%m-%d")
    fecha_fin = datetime.today().strftime("%Y-%m-%d")
    data_tecnico = get_data(ticker_tecnico, fecha_inicio, fecha_fin)
    if data_tecnico.empty:
        st.error("No se han obtenido datos para análisis técnico.")
    else:
        data_tecnico["SMA50"] = data_tecnico["Precio"].rolling(window=50).mean()
        data_tecnico["SMA200"] = data_tecnico["Precio"].rolling(window=200).mean()
        data_tecnico["RSI"] = calcular_RSI(data_tecnico)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data_tecnico.index, y=data_tecnico["Precio"],
                                 mode="lines", name="Precio"))
        fig.add_trace(go.Scatter(x=data_tecnico.index, y=data_tecnico["SMA50"],
                                 mode="lines", name="SMA 50"))
        fig.add_trace(go.Scatter(x=data_tecnico.index, y=data_tecnico["SMA200"],
                                 mode="lines", name="SMA 200"))
        fig.update_layout(title=f"Análisis Técnico de {empresa_tecnico}",
                          xaxis_title="Fecha", yaxis_title="Precio (€)",
                          template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
        
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=data_tecnico.index, y=data_tecnico["RSI"],
                                     mode="lines", name="RSI"))
        fig_rsi.update_layout(title=f"RSI de {empresa_tecnico}",
                              xaxis_title="Fecha", yaxis_title="RSI",
                              template="plotly_white")
        st.plotly_chart(fig_rsi, use_container_width=True)

# =====================================================
# Sección 5: Análisis Fundamental
# =====================================================
elif seccion == "Análisis Fundamental":
    st.subheader("Análisis Fundamental")
    empresa_fund = st.selectbox("Selecciona una empresa para análisis fundamental:", list(empresas.keys()))
    ticker_fund = empresas[empresa_fund]
    info = get_ticker_info(ticker_fund)
    if not info:
        st.error("No se pudo obtener información fundamental.")
    else:
        fundamental_data = {
            "Nombre": info.get("longName", "N/A"),
            "Sector": info.get("sector", "N/A"),
            "País": info.get("country", "N/A"),
            "P/E (TTM)": info.get("trailingPE", "N/A"),
            "P/B": info.get("priceToBook", "N/A"),
            "Dividend Yield (%)": round(info.get("dividendYield", 0)*100, 2) if info.get("dividendYield") else "N/A",
            "Beta": info.get("beta", "N/A"),
            "Market Cap": f"{info.get('marketCap', 0)/1e9:.2f}B €" if info.get("marketCap") else "N/A"
        }
        df_fund = pd.DataFrame(fundamental_data.items(), columns=["Indicador", "Valor"])
        st.table(df_fund)
# =====================================================
# Sección 6: Mejor Momento de Compra
# =====================================================
elif seccion == "Mejor Momento de Compra":
    st.subheader("Análisis de Mejor Momento de Compra (últimos 4 años, sin datos del año en curso)")
    empresa_compra = st.selectbox("Selecciona una empresa:", list(empresas.keys()))
    ticker_compra = empresas[empresa_compra]
    # Usamos datos históricos completos cacheados
    data_hist = get_full_data(ticker_compra)
    if data_hist.empty:
        st.error("No se han obtenido datos históricos para esta empresa.")
    else:
        # Filtrar datos de los últimos 4 años
        fecha_lim = datetime.today() - pd.DateOffset(years=4)
        data_hist = data_hist[data_hist.index >= fecha_lim].copy()
        # Excluir el año en curso
        data_hist = data_hist[data_hist.index.year < datetime.today().year]
        
        if data_hist.empty:
            st.error("No hay datos suficientes de los últimos 4 años (excluyendo el año en curso).")
        else:
            # Añadir columnas 'Mes' y 'Año'
            data_hist["Mes"] = data_hist.index.month
            data_hist["Año"] = data_hist.index.year

            # Calcular precio promedio y mínimo por mes (acumulado de los 4 años)
            precio_promedio = data_hist.groupby("Mes")["Precio"].mean()
            precio_minimo = data_hist.groupby("Mes")["Precio"].min()
            
            # Identificar el mes con el precio promedio y mínimo más bajos
            mejor_mes_prom = precio_promedio.idxmin()
            mejor_mes_min = precio_minimo.idxmin()
            
            st.write(f"El mes con el precio promedio más bajo en los últimos 4 años (sin el año en curso) es: **{mejor_mes_prom}**")
            st.write(f"El mes con el precio mínimo más bajo en los últimos 4 años (sin el año en curso) es: **{mejor_mes_min}**")
            
            # Gráfico de precios promedio y mínimos por mes (acumulado)
            fig_prom = go.Figure()
            fig_prom.add_trace(go.Bar(
                x=precio_promedio.index,
                y=precio_promedio.values,
                name="Precio Promedio"
            ))
            fig_prom.update_layout(
                title=f"Precio Promedio por Mes para {empresa_compra} (últimos 4 años, sin {datetime.today().year})",
                xaxis_title="Mes",
                yaxis_title="Precio (€)",
                template="plotly_white"
            )
            st.plotly_chart(fig_prom, use_container_width=True)
            
            fig_min = go.Figure()
            fig_min.add_trace(go.Bar(
                x=precio_minimo.index,
                y=precio_minimo.values,
                name="Precio Mínimo"
            ))
            fig_min.update_layout(
                title=f"Precio Mínimo por Mes para {empresa_compra} (últimos 4 años, sin {datetime.today().year})",
                xaxis_title="Mes",
                yaxis_title="Precio (€)",
                template="plotly_white"
            )
            st.plotly_chart(fig_min, use_container_width=True)
            
            # Pivotear para obtener el precio promedio por mes para cada año
            pivot_df = data_hist.groupby(["Año", "Mes"])["Precio"].mean().unstack("Año")
            
            # Asegurarse de que el índice incluya los 12 meses (1 a 12)
            pivot_df = pivot_df.reindex(range(1, 13))
            
            # Rellenar valores faltantes: backward fill para que el primer mes tenga valor
            pivot_df = pivot_df.fillna(method='bfill')
            
            # Normalizar: para cada columna, usar el primer valor (ahora no NaN) y multiplicar por 100
            pivot_df_norm = pivot_df.apply(lambda col: col / col.iloc[0] * 100, axis=0)
            
            fig_norm = go.Figure()
            for anio in pivot_df_norm.columns:
                fig_norm.add_trace(go.Scatter(
                    x=pivot_df_norm.index,
                    y=pivot_df_norm[anio],
                    mode='lines+markers',
                    name=f"{anio}"
                ))
            fig_norm.update_layout(
                title="Comparación de Crecimiento Relativo (normalizado a 100 en el primer mes)",
                xaxis_title="Mes",
                yaxis_title="Índice Normalizado",
                template="plotly_white"
            )
            st.plotly_chart(fig_norm, use_container_width=True)

# =====================================================
# Sección X: Comparación de Cierres Diarios Normalizados
# =====================================================
elif seccion == "Comparación de Cierres Diarios Normalizados":
    st.subheader("Comparación de Cierres Diarios Normalizados (últimos 4 años, sin el año en curso)")
    empresa_normal = st.selectbox("Selecciona una empresa:", list(empresas.keys()))
    ticker_normal = empresas[empresa_normal]
    
    # Obtener datos históricos completos cacheados
    data_hist = get_full_data(ticker_normal)
    if data_hist.empty:
        st.error("No se han obtenido datos históricos para esta empresa.")
    else:
        # Filtrar para los últimos 4 años y excluir el año en curso
        fecha_lim = datetime.today() - pd.DateOffset(years=4)
        data_hist = data_hist[data_hist.index >= fecha_lim].copy()
        data_hist = data_hist[data_hist.index.year < datetime.today().year]
        
        if data_hist.empty:
            st.error("No hay datos suficientes de los últimos 4 años (excluyendo el año en curso).")
        else:
            # Añadir columna para el año
            data_hist["Año"] = data_hist.index.year
            
            # Crear el gráfico con una línea para cada año
            fig_daily = go.Figure()
            for anio in sorted(data_hist["Año"].unique()):
                df_year = data_hist[data_hist["Año"] == anio].copy()
                if df_year.empty:
                    continue
                # Normalizar: establecer el primer valor disponible de ese año a 100
                first_close = df_year["Precio"].iloc[0]
                df_year["Precio_Normalizado"] = df_year["Precio"] / first_close * 100
                # Guardar la fecha original en formato string para el hover
                df_year["FechaOriginal"] = df_year.index.strftime("%Y-%m-%d")
                # Crear una columna de fecha de referencia usando un año fijo (por ejemplo, 2000)
                df_year["FechaRef"] = pd.to_datetime("2000-" + df_year.index.strftime("%m-%d"))
                fig_daily.add_trace(go.Scatter(
                    x = df_year["FechaRef"],
                    y = df_year["Precio_Normalizado"],
                    mode = "lines+markers",
                    name = str(anio),
                    customdata = df_year["FechaOriginal"],
                    hovertemplate = "Fecha: %{customdata}<br>Cierre Normalizado: %{y:.2f}<extra></extra>"
                ))
            
            # Obtener dividendos del ticker y filtrar para el mismo periodo
            ticker_obj = yf.Ticker(ticker_normal)
            dividendos = ticker_obj.dividends
            # Convertir fecha límite a naïve
            fecha_lim_naive = (datetime.today() - pd.DateOffset(years=4)).replace(tzinfo=None)
            # Convertir el índice de dividendos a naïve para la comparación
            dividendos_filtrados = dividendos[
                (dividendos.index.tz_convert(None) >= fecha_lim_naive) &
                (dividendos.index.tz_convert(None).year < datetime.today().year)
            ]
            
            # Agregar vertical lines y anotaciones para cada dividendo
            for div_date, div_value in dividendos_filtrados.items():
                original_year = div_date.year
                # Convertir la fecha a formato de referencia (año 2000)
                fecha_ref = pd.to_datetime("2000-" + div_date.strftime("%m-%d"))
                fig_daily.add_vline(x=fecha_ref, line_width=1, line_dash="dash", line_color="red")
                fig_daily.add_annotation(
                    x=fecha_ref,
                    y=95,  # Ajusta este valor según el rango de tus datos
                    text=f"Div: {div_value:.2f} ({original_year})",
                    showarrow=True,
                    arrowhead=1,
                    ax=0,
                    ay=-20,
                    font=dict(color="red")
                )
            
            fig_daily.update_layout(
                title="Comparación de Cierres Diarios Normalizados (base 100 en el primer día)\nCon fechas de dividendos marcadas",
                xaxis_title="Fecha (MM-DD)",
                yaxis_title="Cierre Normalizado (base 100)",
                template="plotly_white"
            )
            st.plotly_chart(fig_daily, use_container_width=True)