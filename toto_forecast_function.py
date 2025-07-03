"""
Funzione personalizzata per generare previsioni con Toto
"""

import os
import sys
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

# Aggiungi il percorso di toto al PYTHONPATH
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
sys.path.insert(0, os.path.join(script_dir, 'toto'))

import numpy as np
import pandas as pd
import torch

# Import locali
try:
    # Prima prova con percorso relativo dalla directory toto
    from model.toto import Toto
    from inference.forecaster import TotoForecaster
    from data.util.dataset import MaskedTimeseries
except ImportError as e1:
    try:
        # Poi prova con percorso completo
        from toto.model.toto import Toto
        from toto.inference.forecaster import TotoForecaster
        from toto.data.util.dataset import MaskedTimeseries
    except ImportError as e2:
        print(f"Errore import 1: {e1}")
        print(f"Errore import 2: {e2}")
        raise ImportError("Impossibile importare i moduli Toto necessari")



def genera_previsione_toto(
    nome_file_dati: str,
    lunghezza_finestra_mobile: int = 96,
    lunghezza_finestra_previsione: Optional[int] = None,
    percorso_modello: Optional[str] = None,
    data_iniziale_previsione: Union[str, datetime] = None,
    lunghezza_previsione: int = None,
    percorso_dati_default: str = "data",
    visualizza_grafico: bool = True
) -> dict:
    """
    Genera previsioni utilizzando il modello Toto.
    
    Parametri:
    ----------
    nome_file_dati : str
        Nome del file di dati (con percorso completo se diverso da quello di default)
    lunghezza_finestra_mobile : int, default=96
        Lunghezza della finestra mobile (contesto storico)
    lunghezza_finestra_previsione : int, optional
        Lunghezza della finestra di previsione (default: uguale a lunghezza_finestra_mobile)
    percorso_modello : str, optional
        Percorso del checkpoint del modello Toto. Se None, usa la convenzione:
        toto/model/{nome_file}_{lunghezza_finestra_mobile}_{lunghezza_finestra_previsione}
    data_iniziale_previsione : str o datetime
        Data iniziale della previsione
    lunghezza_previsione : int
        Numero di passi temporali da prevedere
    percorso_dati_default : str, default="data"
        Percorso di default per i file di dati
    visualizza_grafico : bool, default=True
        Se True, genera automaticamente un grafico delle previsioni
        
    Ritorna:
    --------
    dict
        Dizionario contenente:
        - 'previsioni_medie': array numpy con le previsioni medie
        - 'previsioni_campioni': array numpy con campioni multipli (se richiesti)
        - 'date_previsione': array con le date corrispondenti alle previsioni
        - 'contesto_utilizzato': dati storici utilizzati per la previsione
    """
    
    # Impostazioni default
    if lunghezza_finestra_previsione is None:
        lunghezza_finestra_previsione = lunghezza_finestra_mobile
    
    if lunghezza_previsione is None:
        lunghezza_previsione = lunghezza_finestra_previsione
    
    # Gestione del percorso del modello
    if percorso_modello is None:
        # Per il modello locale venezia, controlla se i parametri sono compatibili
        nome_base = os.path.splitext(os.path.basename(nome_file_dati))[0]
        if (nome_base.lower() == "venezia" and 
            lunghezza_finestra_mobile == 96 and 
            lunghezza_finestra_previsione == 96):
            # Usa il modello venezia fine-tuned disponibile
            percorso_modello = "toto_venezia_simple_finetuned.pt"
            print(f"Utilizzo modello venezia fine-tuned: {percorso_modello}")
        else:
            # Costruisci il nome del modello secondo la convenzione
            nome_modello = f"{nome_base}_{lunghezza_finestra_mobile}_{lunghezza_finestra_previsione}"
            percorso_modello = os.path.join("toto/model", nome_modello)
            print(f"Utilizzo modello predefinito: {percorso_modello}")
    
    # Gestione del percorso del file dati
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    if os.path.sep in nome_file_dati:
        # Il nome file contiene già un percorso
        percorso_file = nome_file_dati
    else:
        # Usa il percorso di default
        percorso_file = os.path.join(percorso_dati_default, nome_file_dati)
    
    # Se il percorso non è assoluto, prova diverse alternative
    if not os.path.isabs(percorso_file):
        possibili_percorsi = [
            percorso_file,  # percorso relativo dalla directory corrente
            os.path.join(script_dir, percorso_file),  # relativo alla directory dello script
            os.path.join(script_dir, "data", nome_file_dati),  # nella directory data/
            os.path.join(script_dir, "toto", "datasets", nome_file_dati),  # nella directory toto/datasets/
            os.path.join(os.getcwd(), percorso_file),  # relativo alla directory di lavoro corrente
        ]
        
        percorso_trovato = None
        for p in possibili_percorsi:
            if os.path.exists(p):
                percorso_trovato = p
                break
        
        if percorso_trovato:
            percorso_file = percorso_trovato
        else:
            raise FileNotFoundError(f"File di dati non trovato. Percorsi provati:\n" + "\n".join(possibili_percorsi))
    
    # Carica i dati
    print(f"Caricamento dati da: {percorso_file}")
    
    # Supporto per diversi formati di file
    if percorso_file.endswith('.csv'):
        dati = pd.read_csv(percorso_file, parse_dates=True, index_col=0)
    elif percorso_file.endswith('.parquet'):
        dati = pd.read_parquet(percorso_file)
    elif percorso_file.endswith('.npy'):
        dati_array = np.load(percorso_file)
        # Assumi che i dati numpy non abbiano indice temporale
        dati = pd.DataFrame(dati_array)
    else:
        raise ValueError(f"Formato file non supportato: {percorso_file}")
    
    # Gestione della data iniziale
    if data_iniziale_previsione is not None:
        if isinstance(data_iniziale_previsione, str):
            data_iniziale_previsione = pd.to_datetime(data_iniziale_previsione)
        
        # Trova l'indice corrispondente alla data
        if isinstance(dati.index, pd.DatetimeIndex):
            try:
                idx_inizio = dati.index.get_loc(data_iniziale_previsione)
            except KeyError:
                raise ValueError(f"Data {data_iniziale_previsione} non trovata nei dati")
        else:
            raise ValueError("I dati devono avere un indice temporale per usare data_iniziale_previsione")
    else:
        # Usa gli ultimi dati disponibili come punto di partenza
        idx_inizio = len(dati) - lunghezza_finestra_mobile
    
    # Estrai il contesto storico
    if idx_inizio < 0:
        raise ValueError(f"Non ci sono abbastanza dati storici. Richiesti: {lunghezza_finestra_mobile}, disponibili: {len(dati)}")
    
    contesto = dati.iloc[idx_inizio:idx_inizio + lunghezza_finestra_mobile]
    
    # Gestione del percorso del modello
    if not os.path.isabs(percorso_modello):
        possibili_modelli = [
            percorso_modello,
            os.path.join(script_dir, percorso_modello),
            os.path.join(os.getcwd(), percorso_modello),
        ]
        
        # Se è il modello venezia, aggiungi il percorso diretto
        if percorso_modello == "toto_venezia_simple_finetuned.pt":
            possibili_modelli.insert(1, os.path.join(script_dir, "toto_venezia_simple_finetuned.pt"))
        
        # Aggiungi anche le varianti con estensioni
        for ext in [".safetensors", ".pt", ".pth", ""]:
            for base_path in possibili_modelli[:]:
                if not base_path.endswith(ext):
                    possibili_modelli.append(base_path + ext)
        
        modello_trovato = None
        for p in possibili_modelli:
            if os.path.exists(p) or (os.path.isdir(p) and os.path.exists(os.path.join(p, "model.safetensors"))):
                modello_trovato = p
                break
        
        if modello_trovato:
            percorso_modello = modello_trovato
        else:
            print(f"AVVISO: Modello predefinito non trovato. Usando il modello fine-tuned disponibile.")
            percorso_modello = os.path.join(script_dir, "toto_venezia_simple_finetuned.pt")
    
    # Carica il modello
    print(f"Caricamento modello da: {percorso_modello}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo utilizzato: {device}")
    
    model = None
    
    # Per il modello venezia, usa l'approccio di quick_test_venezia.py
    if "venezia" in percorso_modello.lower():
        try:
            print("Caricamento modello venezia fine-tuned...")
            model = Toto.from_pretrained('Datadog/Toto-Open-Base-1.0')
            checkpoint = torch.load(percorso_modello, map_location=str(device), weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(device)
            model.eval()
            print("Modello venezia caricato con successo!")
        except Exception as e:
            print(f"Caricamento modello venezia fallito: {e}")
            model = None
    
    # Se il modello venezia non è stato caricato, prova altri metodi
    if model is None:
        try:
            # Prova prima come checkpoint Toto standard
            model = Toto.load_from_checkpoint(percorso_modello, map_location=str(device))
            model = model.to(device)
            model.eval()
        except Exception as e:
            print(f"Caricamento come checkpoint Toto fallito: {e}")
            print("Tentativo di caricamento del modello pre-addestrato da HuggingFace...")
            try:
                model = Toto.from_pretrained('Datadog/Toto-Open-Base-1.0')
                model = model.to(device)
                model.eval()
                print("Modello pre-addestrato caricato con successo!")
            except Exception as e3:
                raise RuntimeError(f"Impossibile caricare qualsiasi modello. Errori: {e}, {e3}")
    
    # Prepara i dati per Toto (seguendo il formato di run_csv_test.py)
    if isinstance(contesto, pd.DataFrame):
        feature_columns = contesto.columns.tolist()
        dati_numpy = contesto.values.T  # Shape: [variates, time_steps]
    else:
        feature_columns = ['serie']
        dati_numpy = contesto.values.reshape(1, -1)  # Shape: [1, time_steps]
    
    # Per il modello venezia, usa il formato di quick_test_venezia.py
    if isinstance(contesto, pd.DataFrame) and len(feature_columns) == 1:
        # Singola variabile - usa il formato di venezia
        serie_temporale = torch.tensor(
            contesto[feature_columns[0]].values,
            dtype=torch.float32
        ).unsqueeze(0).unsqueeze(0).to(device)  # Shape: [1, 1, time_steps]
        
        num_variates = 1  # Definisci num_variates
        context_length = len(contesto)
        
        # Timestamps
        if isinstance(contesto.index, pd.DatetimeIndex):
            contesto_temp = contesto.copy()
            contesto_temp['timestamp_seconds'] = (contesto.index - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
            timestamp_seconds = torch.tensor(
                contesto_temp['timestamp_seconds'].values,
                dtype=torch.int64
            ).unsqueeze(0).unsqueeze(0).to(device)
            interval = 3600  # 1 ora per default
        else:
            timestamp_seconds = torch.arange(len(contesto)).unsqueeze(0).unsqueeze(0).to(device)
            interval = 1
            
        time_interval_seconds = torch.tensor([[interval]], dtype=torch.int).to(device)
        
    else:
        # Formato originale per dati multivariati
        serie_temporale = torch.from_numpy(dati_numpy).float().to(device)
        num_variates, context_length = serie_temporale.shape
        
        # Crea timestamps fittizi se non disponibili
        if isinstance(contesto.index, pd.DatetimeIndex):
            timestamp_seconds = (contesto.index - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
            timestamp_seconds = torch.from_numpy(timestamp_seconds.values).expand((num_variates, context_length)).to(device)
            
            # Calcola intervallo temporale
            if len(contesto) > 1:
                time_diffs = contesto.index.to_series().diff().dt.total_seconds().dropna()
                interval = int(time_diffs.mode()[0]) if len(time_diffs) > 0 else 3600
            else:
                interval = 3600
            time_interval_seconds = torch.full((num_variates,), interval).to(device)
        else:
            # Usa timestamps fittizi
            timestamp_seconds = torch.arange(context_length).expand((num_variates, context_length)).to(device)
            time_interval_seconds = torch.full((num_variates,), 1).to(device)
    
    # Crea l'oggetto MaskedTimeseries (come in run_csv_test.py)
    input_data = MaskedTimeseries(
        series=serie_temporale,
        padding_mask=torch.full_like(serie_temporale, True, dtype=torch.bool),
        id_mask=torch.zeros_like(serie_temporale, dtype=torch.int),
        timestamp_seconds=timestamp_seconds,
        time_interval_seconds=time_interval_seconds
    )
    
    # Crea il forecaster
    forecaster = TotoForecaster(model.model)
    
    # Genera le previsioni
    print(f"Generazione previsioni per {lunghezza_previsione} passi temporali...")
    
    with torch.no_grad():
        # Genera previsioni con campioni multipli per incertezza
        forecast = forecaster.forecast(
            inputs=input_data,
            prediction_length=lunghezza_previsione,
            num_samples=100,  # Genera 100 campioni per quantificare l'incertezza
            samples_per_batch=10,
            use_kv_cache=True
        )
    
    # Estrai i risultati (come in quick_test_venezia.py)
    if forecast.samples is not None:
        # Estrai usando median e quantili come in quick_test_venezia.py
        if forecast.median.dim() > 1 and forecast.median.shape[0] == 1:
            # Formato venezia [1, 1, time_steps]
            previsioni_medie = forecast.median[0, 0].cpu().numpy()
            previsioni_mediana = forecast.median[0, 0].cpu().numpy()
            previsioni_campioni = forecast.samples[0, 0].cpu().numpy()
        else:
            # Formato originale
            previsioni_medie = forecast.mean.cpu().numpy()
            previsioni_mediana = forecast.median.cpu().numpy() if hasattr(forecast, 'median') else None
            previsioni_campioni = forecast.samples.cpu().numpy()
    else:
        previsioni_medie = forecast.mean.cpu().numpy()
        previsioni_mediana = None
        previsioni_campioni = None
    
    # Genera le date di previsione se i dati hanno un indice temporale
    date_previsione = None
    if isinstance(dati.index, pd.DatetimeIndex):
        freq = pd.infer_freq(dati.index)
        ultima_data = contesto.index[-1]
        date_previsione = pd.date_range(
            start=ultima_data + pd.Timedelta(1, freq=freq),
            periods=lunghezza_previsione,
            freq=freq
        )
    
    # Prepara il risultato
    risultato = {
        'previsioni_medie': previsioni_medie,
        'previsioni_campioni': previsioni_campioni,
        'previsione_mediana': previsioni_mediana,
        'deviazione_standard': forecast.std.cpu().numpy() if forecast.samples is not None else None,
        'date_previsione': date_previsione,
        'contesto_utilizzato': contesto,
        'dati_originali': dati,  # Aggiungi i dati originali per calcolo MAE
        'num_variabili': num_variates,
        'lunghezza_previsione': lunghezza_previsione
    }
    
    print("Previsione completata!")
    
    # Visualizza il grafico se richiesto
    if visualizza_grafico:
        visualizza_previsioni(risultato)
    
    return risultato


# Funzione di utilità per visualizzare i risultati
def visualizza_previsioni(risultato: dict, variabile_idx: int = 0):
    """
    Visualizza le previsioni generate.
    
    Parametri:
    ----------
    risultato : dict
        Output della funzione genera_previsione_toto
    variabile_idx : int, default=0
        Indice della variabile da visualizzare (per dati multivariati)
    """
    import matplotlib.pyplot as plt
    
    # Estrai i dati
    contesto = risultato['contesto_utilizzato']
    previsioni_medie = risultato['previsioni_medie']
    date_previsione = risultato['date_previsione']
    
    # Crea il grafico
    plt.figure(figsize=(12, 6))
    
    # Plot del contesto storico
    if isinstance(contesto, pd.DataFrame):
        if contesto.shape[1] > variabile_idx:
            plt.plot(contesto.index, contesto.iloc[:, variabile_idx], 
                    label='Dati storici', color='blue')
        else:
            plt.plot(contesto.index, contesto.iloc[:, 0], 
                    label='Dati storici', color='blue')
    else:
        plt.plot(contesto.index, contesto.values, 
                label='Dati storici', color='blue')
    
    # Plot delle previsioni
    if date_previsione is not None:
        x_axis = date_previsione
    else:
        # Calcola la lunghezza corretta per x_axis
        if previsioni_medie.ndim > 1:
            x_axis = range(previsioni_medie.shape[1])
        else:
            x_axis = range(len(previsioni_medie))
    
    # Gestisci le dimensioni delle previsioni
    if isinstance(previsioni_medie, np.ndarray):
        if previsioni_medie.ndim > 1 and previsioni_medie.shape[0] > variabile_idx:
            y_values = previsioni_medie[variabile_idx]
        else:
            y_values = previsioni_medie.flatten()
    else:
        # È già un array 1D (formato venezia)
        y_values = previsioni_medie
    
    # Calcola il MAE se abbiamo dei valori reali per confronto
    mae_text = ""
    if 'contesto_utilizzato' in risultato and date_previsione is not None:
        # Se abbiamo date, proviamo a trovare valori reali per il periodo di previsione
        try:
            # Cerca nei dati originali i valori reali per il periodo di previsione
            dati_completi = risultato.get('dati_originali', None)
            if dati_completi is not None and isinstance(dati_completi.index, pd.DatetimeIndex):
                # Trova l'overlap tra le date di previsione e i dati reali
                overlap_mask = dati_completi.index.isin(date_previsione)
                if overlap_mask.any():
                    valori_reali = dati_completi.loc[overlap_mask].iloc[:, 0].values[:len(y_values)]
                    if len(valori_reali) == len(y_values):
                        mae = np.mean(np.abs(y_values - valori_reali))
                        mae_text = f" (MAE: {mae:.2f})"
        except:
            pass  # Se non riusciamo a calcolare il MAE, continua senza
    
    plt.plot(x_axis, y_values, 
            label=f'Previsione media{mae_text}', color='red', linestyle='--')
    
    # Aggiungi intervalli di confidenza se disponibili
    if risultato['previsioni_campioni'] is not None:
        # Gestisci le dimensioni dei campioni
        campioni = risultato['previsioni_campioni']
        if campioni.ndim > 2 and campioni.shape[0] > variabile_idx:
            campioni_var = campioni[variabile_idx]
        else:
            # Se è una singola variabile, usa tutti i campioni
            campioni_var = campioni.reshape(campioni.shape[-2], campioni.shape[-1])
        
        percentile_10 = np.percentile(campioni_var, 10, axis=-1)
        percentile_90 = np.percentile(campioni_var, 90, axis=-1)
        
        plt.fill_between(x_axis, percentile_10, percentile_90, 
                        alpha=0.3, color='red', label='Intervallo 80%')
    
    plt.xlabel('Tempo')
    plt.ylabel('Valore')
    
    # Titolo del grafico con MAE se disponibile
    titolo = f'Previsione Serie Temporale - Variabile {variabile_idx}'
    if mae_text:
        titolo = f'Previsione Serie Temporale{mae_text}'
    plt.title(titolo)
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# Funzione per parsare gli argomenti della riga di comando
def parse_arguments():
    """
    Parsa gli argomenti della riga di comando
    """
    parser = argparse.ArgumentParser(
        description='Genera previsioni utilizzando il modello Toto',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        'nome_file_dati',
        nargs='?',
        default='venezia.csv',
        help='Nome del file di dati (con percorso completo se diverso da toto/datasets)'
    )
    
    parser.add_argument(
        '--lunghezza_finestra_mobile',
        type=int,
        default=96,
        help='Lunghezza della finestra mobile (contesto storico)'
    )
    
    parser.add_argument(
        '--lunghezza_finestra_previsione',
        type=int,
        default=None,
        help='Lunghezza della finestra di previsione (default: uguale a finestra mobile)'
    )
    
    parser.add_argument(
        '--percorso_modello',
        type=str,
        default=None,
        help='Percorso del checkpoint del modello (default: toto/model/{nome}_{finestra}_{previsione})'
    )
    
    parser.add_argument(
        '--data_iniziale_previsione',
        type=str,
        default=None,
        help='Data iniziale della previsione (formato: YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--lunghezza_previsione',
        type=int,
        default=None,
        help='Numero di passi temporali da prevedere (default: lunghezza_finestra_previsione)'
    )
    
    parser.add_argument(
        '--percorso_dati_default',
        type=str,
        default='data',
        help='Percorso di default per i file di dati'
    )
    
    parser.add_argument(
        '--no_grafico',
        action='store_true',
        help='Disabilita la visualizzazione del grafico'
    )
    
    return parser.parse_args()


# Funzione main per venezia.csv
def main():
    """
    Funzione principale che esegue la previsione
    """
    # Parsa gli argomenti della riga di comando
    args = parse_arguments()
    
    print(f"=== Previsione Serie Temporale: {args.nome_file_dati} ===\n")
    
    # Converti no_grafico in visualizza_grafico
    visualizza_grafico = not args.no_grafico
    
    try:
        # Genera previsione con i parametri dalla riga di comando
        risultato = genera_previsione_toto(
            nome_file_dati=args.nome_file_dati,
            lunghezza_finestra_mobile=args.lunghezza_finestra_mobile,
            lunghezza_finestra_previsione=args.lunghezza_finestra_previsione,
            percorso_modello=args.percorso_modello,
            data_iniziale_previsione=args.data_iniziale_previsione,
            lunghezza_previsione=args.lunghezza_previsione,
            percorso_dati_default=args.percorso_dati_default,
            visualizza_grafico=visualizza_grafico
        )
        
        # Stampa statistiche riassuntive
        print("\n=== Statistiche Previsione ===")
        print(f"Numero variabili: {risultato['num_variabili']}")
        print(f"Lunghezza previsione: {risultato['lunghezza_previsione']}")
        
        if risultato['previsioni_medie'] is not None:
            media_previsioni = risultato['previsioni_medie'].mean(axis=-1)
            print(f"Media previsioni per variabile: {media_previsioni}")
        
        if risultato['deviazione_standard'] is not None:
            std_media = risultato['deviazione_standard'].mean(axis=-1)
            print(f"Deviazione standard media: {std_media}")
        
        print("\nPrevisione completata con successo!")
        
    except FileNotFoundError:
        print(f"ERRORE: File {args.nome_file_dati} non trovato!")
        print(f"Assicurati che il file sia presente in {args.percorso_dati_default}/{args.nome_file_dati}")
    except Exception as e:
        print(f"ERRORE durante la previsione: {str(e)}")
        import traceback
        traceback.print_exc()


# Esecuzione del main
if __name__ == "__main__":
    main()