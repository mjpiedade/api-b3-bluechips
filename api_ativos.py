from fastapi import FastAPI, HTTPException
import yfinance as yf
import pandas as pd
import numpy as np
from cachetools import cached, TTLCache # <-- NOVO IMPORT

app = FastAPI(title="API de Ativos - Dashboard")

ATIVOS_PADRAO = ['BOVA11.SA', 'PETR4.SA', 'VALE3.SA', 'BBAS3.SA', 'BPAC11.SA', 'ITUB4.SA', 'BBDC4.SA', 'WEGE3.SA', 'AMER3.SA']


# --- Funções Auxiliares de Cálculo (Mantidas do seu código original) ---
def _aplicar_calculos(dados: pd.DataFrame, p_sma_20=20, p_std_20=20, p_sma_50=50, p_std_50=50, p_sma_200=200, p_std_200=200, p_vol_200=200) -> pd.DataFrame:
    dados[f'SMA_{p_sma_20}'] = dados['Close'].rolling(window=p_sma_20, min_periods=1).mean()
    dados[f'SMA_{p_sma_50}'] = dados['Close'].rolling(window=p_sma_50, min_periods=1).mean()
    dados[f'SMA_{p_sma_200}'] = dados['Close'].rolling(window=p_sma_200, min_periods=1).mean()
    dados[f'SMA_Volume_{p_vol_200}'] = dados['Volume'].rolling(window=p_vol_200, min_periods=1).mean()
    dados[f'Desvio_Padrao_{p_std_20}'] = dados['Close'].rolling(window=p_std_20, min_periods=1).std()
    dados[f'Desvio_Padrao_{p_std_50}'] = dados['Close'].rolling(window=p_std_50, min_periods=1).std()
    dados[f'Desvio_Padrao_{p_std_200}'] = dados['Close'].rolling(window=p_std_200, min_periods=1).std()
    return dados

def _processar_historico_diario(ticker):
    dados = yf.download(ticker, period='2y', progress=False, auto_adjust=True)
    if isinstance(dados.columns, pd.MultiIndex): dados.columns = dados.columns.get_level_values(0)
    dados = dados.loc[:,~dados.columns.duplicated()]
    if dados.index.tz is not None:
        dados.index = dados.index.tz_convert('America/Sao_Paulo').tz_localize(None)
    if not dados.empty and len(dados) >= 2:
        return _aplicar_calculos(dados)
    return None

def _processar_historico_horario(ticker):
    dados = yf.download(ticker, period='10d', interval='5m', progress=False, auto_adjust=True)
    if isinstance(dados.columns, pd.MultiIndex): dados.columns = dados.columns.get_level_values(0)
    dados = dados.loc[:,~dados.columns.duplicated()]
    if dados.index.tz is not None:
        dados.index = dados.index.tz_convert('America/Sao_Paulo').tz_localize(None)
    
    if not dados.empty and len(dados) >= 2:
        dados = dados.resample('1h', offset='20min').agg({
            'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
        })
        dados = dados.dropna(subset=['Close'])
        return _aplicar_calculos(dados)
    return None

# --- ROTAS DA API ---

@app.get("/")
def home():
    return {"mensagem": "API rodando! Acesse /docs para ver a documentação interativa."}

# --- NOVO: CACHE DA API ---
# Guarda o resultado por 900 segundos (15 minutos). 
# Assim o celular carrega em 0.5 segundos em vez de 15 segundos!
cache_resumo = TTLCache(maxsize=1, ttl=900)

@cached(cache_resumo)
def gerar_resumo_pesado():
    lista_resumo = []
    for ticker in ATIVOS_PADRAO:
        df_d = _processar_historico_diario(ticker)
        df_h = _processar_historico_horario(ticker)
        
        if df_d is None or df_h is None:
            continue
            
        last_row_d = df_d.iloc[-1]
        last_preco = last_row_d['Close'].item() if hasattr(last_row_d['Close'], 'item') else last_row_d['Close']
        
        if len(df_d) >= 2:
            preco_anterior = df_d.iloc[-2]['Close'].item() if hasattr(df_d.iloc[-2]['Close'], 'item') else df_d.iloc[-2]['Close']
            variacao_diaria = (last_preco / preco_anterior) - 1 if preco_anterior != 0 else 0
        else:
            variacao_diaria = 0

        max_dia = last_row_d['High'].item() if hasattr(last_row_d['High'], 'item') else last_row_d['High']
        min_dia = last_row_d['Low'].item() if hasattr(last_row_d['Low'], 'item') else last_row_d['Low']
        
        amp_dia = max_dia - min_dia
        pct_candle_diario = (last_preco - min_dia) / amp_dia if amp_dia != 0 else 0
        
        max_20d = df_d['High'].tail(20).max()
        min_20d = df_d['Low'].tail(20).min()
        amp_20d = max_20d - min_20d
        pct_candle_20d = (last_preco - min_20d) / amp_20d if amp_20d != 0 else 0
        
        sma_20 = last_row_d['SMA_20']
        pct_sma_20 = (last_preco / sma_20) - 1 if sma_20 != 0 else 0

        lista_resumo.append({
            "ativo": ticker,
            "ultimo_preco": float(last_preco),
            "variacao_diaria": float(variacao_diaria),
            "max_dia": float(max_dia),
            "min_dia": float(min_dia),
            "pct_candle_dia": float(pct_candle_diario),
            "max_20d": float(max_20d),
            "min_20d": float(min_20d),
            "pct_candle_20d": float(pct_candle_20d),
            "sma_20": float(sma_20),
            "pct_vs_sma_20": float(pct_sma_20)
        })
    return lista_resumo

@app.get("/resumo")
def obter_resumo_dashboard():
    # A rota agora só chama a função com cache
    return gerar_resumo_pesado()
@app.get("/velas/{ticker}")
def obter_velas_horarias(ticker: str):
    """
    Devolve os dados de Open, High, Low, Close horários de um ativo específico para plotar o gráfico
    """
    # Formata o ticker caso o app envie sem o .SA
    if not ticker.endswith('.SA'):
        ticker += '.SA'
        
    df_h = _processar_historico_horario(ticker)
    
    if df_h is None:
        raise HTTPException(status_code=404, detail="Dados não encontrados para este ativo.")
        
    # Adicionando o Hour_Index e a Pct_Close_Range
    df_h['Hour_Index'] = df_h.groupby(df_h.index.date).cumcount() + 1
    amplitude = df_h['High'] - df_h['Low']
    df_h['Pct_Close_Range'] = np.where(amplitude == 0, 0, (df_h['Close'] - df_h['Low']) / amplitude)
    
    # Substitui os NaN por None (para o JSON aceitar) e reseta o index para a data virar uma coluna
    df_h = df_h.replace({np.nan: None}).reset_index()
    
    # Renomeia a coluna de data para algo padronizado
    df_h = df_h.rename(columns={'Datetime': 'DataHora', 'index': 'DataHora'})
    
    # Converte as datas para string no formato ISO (Padrão de APIs)
    df_h['DataHora'] = df_h['DataHora'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    # Transforma o DataFrame em uma lista de dicionários
    return df_h[['DataHora', 'Open', 'High', 'Low', 'Close', 'SMA_20', 'Hour_Index', 'Pct_Close_Range']].to_dict(orient='records')