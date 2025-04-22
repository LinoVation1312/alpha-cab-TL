"""
Exemple d'acquisition multi-canal avec affichage interactif
avec panneau de paramètres et bouton pour relancer la mesure.
"""

################################################################################
# Panneau de paramètres
################################################################################


PARAMS = {
    # Adresse IP du module LAN-XI utilisé pour les requêtes REST
    "ip": "169.254.254.218",

    # Durée de la capture ou de l'acquisition en secondes
    "duration_seconds": 4.0,

    # Nombre total de canaux d'entrée actifs (parmi les 4 disponibles sur le module)
    "nb_channels": 4,

    # Numéro du canal de sortie utilisé pour le générateur de signal (ex : 1 pour canal 1)
    "generator_output_channel": 1,

    # Type de signal généré : 
    # "sine", "square", "linsweep", "logsweep", "random", "p_random", etc.
    "waveform": "logsweep",

    # Fréquence du signal généré en Hz (valable pour signaux périodiques comme "sine")
    "generator_frequency": 432.0,

    # Gain appliqué à la sortie du générateur (valeur entre 0.0 et < 1.0)
    "generator_gain": 0.200,

    # Offset DC ajouté à la sortie du générateur (entre -1.0 et 1.0)
    "generator_offset": 0.0,

    # Mode flottant (True = flottant, False = masse référencée)
    "generator_floating": False,

    # Gain appliqué au signal d'entrée du générateur (interne à la configuration)
    "input_gain": 0.75,

    # Offset appliqué au signal d'entrée du générateur
    "input_offset": 0.0,

    # Phase initiale du signal (en degrés ou radians selon le système, si applicable)
    "phase": 0.0,

    # Bande passante du bruit (valable pour waveform="random")
    "bandwidth": 10000,

    # Activation du filtre passe-haut (valable pour waveform="random")
    "hp_filter": True,

    # Paramètres spécifiques pour le signal pseudo-aléatoire (p_random)
    "fftlines": 1600,    # Nombre de lignes FFT dans la génération
    "nbseq": 4,          # Nombre de séquences à générer

    # Paramètre de synchronisation (master = référence, slave = asservi à un autre)
    "sync_mode": "master",

    # Rampe d'arrêt en douceur lors de l'arrêt du générateur
    "ramp_down": True,

    # Sensibilités micro (en V/Pa) pour chaque canal actif (attention : valeurs > 0 requises)
    # Exemple : [4189: 42.5385 mV/Pa → 0.0425385 V/Pa]
    "mic_sensitivities": [0.0425385, 0.0432, 0.0435, 0.04358],

    # Paramètres d'affichage du spectre FFT
    "graph_background_color": "lightyellow",      # Couleur de fond principale
    "graph_background2_color": [0, 0.01, 0.01],    # Couleur de fond secondaire (par ex. pour un graphique en 3D)
    "graph_line_color": "blue",                   # Couleur de la courbe FFT
    "graph_title": "Spectre FFT [dB SPL]",   # Titre principal du graphique

    # Limites d'affichage sur les axes (fréquences et niveaux SPL)
    "xlim": [250, 10000],   # Axe des fréquences (Hz)
    "ylim": [-20, 130],     # Axe des niveaux SPL (dB)

    # Lissage de la courbe FFT (valeur plus haute = courbe plus lisse)
    "smooth_qty": 200,

    # --- Sweep parameters for linsweep/logsweep ---
    "start_frequency": 100.0,      # Hz, default sweep start
    "stop_frequency": 10000.0,      # Hz, default sweep stop
    "sweep_direction": 0,          # 0: start->stop->start, 1: start->stop->reverse
    "linsweep_hz_second": 4000.0,  # Hz/s, sweep speed for linsweep
    "logsweep_decades_sec": 2,   # decades/s, sweep speed for logsweep
}

host = "http://" + PARAMS["ip"]

################################################################################
# Importations et fonctions utilitaires
################################################################################
import requests
import socket
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox
from time import sleep
import sys

import HelpFunctions.utility as utility
from openapi.openapi_header import OpenapiHeader
from openapi.openapi_stream import OpenapiStream

def reset_recorder(host):
    """Ferme et réinitialise le recorder pour éviter les erreurs 403"""
    try:
        # Essayer d'abord d'arrêter la mesure et le générateur
        requests.put(host + "/rest/rec/measurements/stop", timeout=2)
        requests.put(host + "/rest/rec/generator/stop", timeout=2)
        # Ensuite fermer le recorder
        requests.put(host + "/rest/rec/close", timeout=2)
        print("Recorder réinitialisé avec succès")
        sleep(1)  # Attendre que le système se stabilise
        return True
    except Exception as e:
        print(f"Erreur lors de la réinitialisation du recorder: {e}")
        sleep(2)  # Attendre plus longtemps en cas d'erreur
        return False

def open_recorder(host):
    try:
        response = requests.put(host + "/rest/rec/open")
        response.raise_for_status()
        return True
    except Exception as e:
        print(f"Erreur lors de l'ouverture du recorder: {e}")
        return False

def get_module_info(host):
    try:
        response = requests.get(host + "/rest/rec/module/info")
        response.raise_for_status()
        module_info = response.json()
        print(f"Module Info: {module_info}")
        return module_info
    except Exception as e:
        print(f"Erreur lors de la récupération des infos du module: {e}")
        return None

def create_recording(host):
    try:
        response = requests.put(host + "/rest/rec/create")
        response.raise_for_status()
        return True
    except Exception as e:
        print(f"Erreur lors de la création de l'enregistrement: {e}")
        reset_recorder(host)
        sys.exit(1)

def get_default_input_setup(host):
    response = requests.get(host + "/rest/rec/channels/input/default")
    response.raise_for_status()
    return response.json()

def configure_input_channels(host, setup):
    # -------------------- Correction --------------------
    # Désactiver TOUS les canaux renvoyés dans le setup
    for idx, channel in enumerate(setup["channels"]):
        channel["enabled"] = False

    # Activation et configuration uniquement des canaux souhaités
    for ch in range(len(setup["channels"])):
        if ch < PARAMS["nb_channels"]:
            setup["channels"][ch]["enabled"] = True
            # Activer le mode CCLD pour les microphones à condensateur
            setup["channels"][ch]["ccld"] = True
            # Configurer la sensibilité uniquement si elle est définie
            if ch < len(PARAMS["mic_sensitivities"]) and PARAMS["mic_sensitivities"][ch] > 0:
                sensitivity_v_pa = PARAMS["mic_sensitivities"][ch]
            else:
                # Valeur par défaut si non définie ou nulle
                sensitivity_v_pa = 1.0
            setup["channels"][ch]["transducer"]["sensitivity"] = sensitivity_v_pa
            setup["channels"][ch]["transducer"]["unit"] = "Pa"
            print(f"Canal {ch}: sensibilité configurée à {sensitivity_v_pa} V/Pa")
        else:
            # Désactiver explicitement les canaux non utilisés
            setup["channels"][ch]["enabled"] = False
    print("Configuration des canaux d'entrée :", setup)
    response = requests.put(host + "/rest/rec/channels/input", json=setup)
    response.raise_for_status()
    # -----------------------------------------------------

def prepare_generator(host):
    """Prépare le générateur en réinitialisant les paramètres"""
    try:
        response = requests.put(
            f"{host}/rest/rec/generator/prepare",
            json={
                "outputs": [
                    {"number": PARAMS["generator_output_channel"]}
                ]
            },
            timeout=2
        )
        response.raise_for_status()
        print(f"Générateur préparé sur le canal {PARAMS['generator_output_channel']}")
        return True
    except requests.exceptions.RequestException as e:
        print(f"Erreur de préparation du générateur: {str(e)}")
        return False
    
def get_default_generator_setup(host):
    """Récupère la configuration par défaut du générateur"""
    try:
        response = requests.get(f"{host}/rest/rec/generator/output/default", timeout=2)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Erreur de récupération de configuration: {str(e)}")
        return None
# https://github.com/hbk-world/open-api-tutorials/blob/master/doc/be187215.pdf pour les infos générateur
 
 
def configure_generator(host, generator_setup):
    """Configure dynamiquement le générateur selon les paramètres utilisateur"""
    try:
        # Accès aux paramètres de sortie du premier canal
        output = generator_setup["outputs"][0]
        
        # Configuration des paramètres de sortie principaux
        output["gain"] = max(0.0, min(0.999, PARAMS.get("generator_gain", 1.0)))
        output["floating"] = PARAMS.get("generator_floating", False)
        
        # Si le paramètre offset existe dans la structure, on le configure
        if "offset" in output:
            output["offset"] = max(-0.999, min(0.999, PARAMS.get("generator_offset", 0.0)))
        
        # Configuration du signal d'entrée (type de signal)
        if len(output["inputs"]) > 0:
            input_signal = output["inputs"][0]
            input_signal["signalType"] = PARAMS["waveform"]
            input_signal["gain"] = max(0.0, min(0.999, PARAMS.get("input_gain", 0.75)))
            
            # Si le paramètre offset existe dans la structure, on le configure
            if "offset" in input_signal:
                input_signal["offset"] = max(-0.999, min(0.999, PARAMS.get("input_offset", 0.0)))
            
            # Paramètres spécifiques selon le type de signal
            if PARAMS["waveform"] in ["sine", "square"]:
                input_signal["frequency"] = PARAMS["generator_frequency"]
                if "phase" in input_signal:
                    input_signal["phase"] = PARAMS.get("phase", 0.0)
            
            # --- Ajout sweep ---
            if PARAMS["waveform"] == "linsweep":
                input_signal["start_frequency"] = PARAMS.get("start_frequency", 100.0)
                input_signal["stop_frequency"] = PARAMS.get("stop_frequency", 5000.0)
                input_signal["phase"] = PARAMS.get("phase", 0.0)
                input_signal["direction"] = PARAMS.get("sweep_direction", 0)
                input_signal["hz_second"] = PARAMS.get("linsweep_hz_second", 1000.0)
            if PARAMS["waveform"] == "logsweep":
                input_signal["start_frequency"] = PARAMS.get("start_frequency", 100.0)
                input_signal["stop_frequency"] = PARAMS.get("stop_frequency", 5000.0)
                input_signal["phase"] = PARAMS.get("phase", 0.0)
                input_signal["direction"] = PARAMS.get("sweep_direction", 0)
                input_signal["decades_sec"] = PARAMS.get("logsweep_decades_sec", 1.0)
            # --- fin sweep ---
            
            # Paramètres spécifiques aux signaux aléatoires
            if PARAMS["waveform"] == "random" and "bandwidth" in input_signal:
                input_signal["bandwidth"] = PARAMS.get("bandwidth", 10000)
                if "hp_filter" in input_signal:
                    input_signal["hp_filter"] = PARAMS.get("hp_filter", True)
            
            # Paramètres spécifiques au signal pseudo-aléatoire
            if PARAMS["waveform"] == "p_random":
                if "fftlines" in input_signal:
                    input_signal["fftlines"] = PARAMS.get("fftlines", 1600)
                if "nbseq" in input_signal:
                    input_signal["nbseq"] = PARAMS.get("nbseq", 4)
        
        print("Configuration du générateur:", generator_setup)
        response = requests.put(f"{host}/rest/rec/generator/output", json=generator_setup, timeout=5)
        response.raise_for_status()
        print(f"Générateur configuré avec {PARAMS['waveform']} à {PARAMS.get('generator_frequency', 'N/A')}Hz")
        return generator_setup
    
    except (KeyError, IndexError) as e:
        print(f"Erreur dans la structure de configuration du générateur: {str(e)}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Erreur de configuration du générateur: {str(e)}")
        return None

def start_generator(host):
    """Démarre la génération du signal avec gestion des erreurs"""
    try:
        response = requests.put(
            f"{host}/rest/rec/generator/start",
            json={
                "outputs": [
                    {
                        "number": PARAMS["generator_output_channel"],
                        "sync": PARAMS.get("sync_mode", "master")
                    }
                ]
            },
            timeout=2
        )
        response.raise_for_status()
        print(f"Génération démarrée sur le canal {PARAMS['generator_output_channel']}")
        return True
    except requests.exceptions.RequestException as e:
        print(f"Erreur de démarrage du générateur: {str(e)}")
        return False

def start_measurement(host):
    response = requests.post(host + "/rest/rec/measurements")
    response.raise_for_status()

def get_streaming_port(host):
    response = requests.get(host + "/rest/rec/destination/socket")
    response.raise_for_status()
    port = response.json()["tcpPort"]
    print(f"Port de streaming : {port}")
    return port

def stop_measurement(host):
    try:
        response = requests.put(host + "/rest/rec/measurements/stop")
        response.raise_for_status()
    except Exception as e:
        print(f"Erreur lors de l'arrêt de la mesure : {e}")

def stop_generator(host):
    """Arrête proprement la génération du signal"""
    try:
        response = requests.put(
            f"{host}/rest/rec/generator/stop",
            json={
                "outputs": [
                    {
                        "number": PARAMS["generator_output_channel"],
                        "ramp_down": PARAMS.get("ramp_down", True)
                    }
                ]
            },
            timeout=2
        )
        response.raise_for_status()
        print("Génération arrêtée avec succès")
        return True
    except requests.exceptions.RequestException as e:
        print(f"Erreur lors de l'arrêt du générateur: {str(e)}")
        return False

def calculate_sample_rate(setup, module_info):
    # Extraction et calcul du taux d'échantillonnage en fonction de la bande passante
    bandwidth_str = setup["channels"][0]["bandwidth"]
    bandwidth_value = eval(bandwidth_str.replace('kHz', '*1000'))
    supported_sample_rates = module_info["supportedSampleRates"]
    sample_rate = min(supported_sample_rates, key=lambda x: abs(x - bandwidth_value * 2))
    print(f"Taux d'échantillonnage utilisé : {sample_rate}")
    return sample_rate

################################################################################
# Acquisition multi-canal et traitement des données
################################################################################
def stream_data(ip, port, sample_rate, duration_seconds, nb_channels):
    total_samples = int(sample_rate * duration_seconds)
    data_arrays = {chan: np.array([]) for chan in range(nb_channels)}
    interpretations = [{} for _ in range(nb_channels)]
     
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((ip, port))
            s.settimeout(10)  # Timeout pour éviter de bloquer indéfiniment
            
            while any(data_arrays[chan].size < total_samples for chan in range(nb_channels)):
                try:
                    header_data = s.recv(28)
                    if not header_data or len(header_data) < 28:
                        print("Données d'en-tête incomplètes, tentative de reconnexion...")
                        continue
                    
                    header = OpenapiHeader.from_bytes(header_data)
                    content_length = header.content_length
                    
                    remaining_data = bytearray()
                    bytes_to_read = content_length
                    
                    while bytes_to_read > 0:
                        chunk = s.recv(min(4096, bytes_to_read))
                        if not chunk:
                            break
                        remaining_data.extend(chunk)
                        bytes_to_read -= len(chunk)
                    
                    if bytes_to_read > 0:
                        print(f"Paquet incomplet: attendu {content_length}, reçu {content_length - bytes_to_read}")
                        continue
                    
                    complete_packet = header_data + remaining_data
                    package = OpenapiStream.from_bytes(complete_packet)
                    
                    if package.header.message_type == OpenapiStream.Header.EMessageType.e_interpretation:
                        for interpretation in package.content.interpretations:
                            channel_idx = interpretation.signal_id - 1
                            if 0 <= channel_idx < nb_channels:
                                interpretations[channel_idx][interpretation.descriptor_type] = interpretation.value
                                if interpretation.descriptor_type == OpenapiStream.Interpretation.EDescriptorType.unit:
                                    print(f"Canal {channel_idx}: unité = {interpretation.value}")
                    
                    if package.header.message_type == OpenapiStream.Header.EMessageType.e_signal_data:
                        for signal in package.content.signals:
                            if signal is not None:
                                chan = signal.signal_id - 1
                                if 0 <= chan < nb_channels:
                                    sensitivity = PARAMS["mic_sensitivities"][chan] if chan < len(PARAMS["mic_sensitivities"]) else 1
                                    values = np.array([x.calc_value for x in signal.values])
                                    data_arrays[chan] = np.append(data_arrays[chan], values)
                                    print(f"Canal {chan}: {len(values)} échantillons reçus, total: {data_arrays[chan].size}")
                
                except Exception as e:
                    print(f"Erreur durant le streaming: {e}")
    
    except Exception as e:
        print(f"Une erreur s'est produite : {e}")
    
    for chan in range(nb_channels):
        print(f"Total échantillons canal {chan}: {data_arrays[chan].size}")
    
    return data_arrays, interpretations

def compute_fft(data, sample_rate, interpretation, chan_idx):
    if len(data) == 0:
        return np.array([]), np.array([])
        
    window = np.hamming(len(data))
    scale_factor = interpretation.get(OpenapiStream.Interpretation.EDescriptorType.scale_factor, 1)
    
    data_volt = (data * scale_factor) / (2 ** 23)
    freq, s_dbfs = utility.dbfft(data_volt, sample_rate, window, ref=1)
    
    # Conversion de dBV à dBSPL
    sensitivity = PARAMS["mic_sensitivities"][chan_idx] if chan_idx < len(PARAMS["mic_sensitivities"]) else 1
    if sensitivity > 0:
        s_dbspl = s_dbfs - 20 * np.log10(sensitivity) + 94 - 27.8
    else:
        s_dbspl = s_dbfs  # Éviter la division par zéro ou log(0)
    
    return freq, s_dbspl


def calculate_energy_average_SPL(spl_arrays):
    """
    Calcule le SPL moyen en énergie à partir d'une liste de courbes SPL en dB.
    
    Pour chaque point fréquentiel de chaque courbe SPL (en dB) :
      - Conversion en pression acoustique quadratique moyenne (p²) :
            p² = (20e-6)² * 10^(SPL/10)
      - Moyennage spatial de ces p² :
            p²_moy = (1/N) * Σ(p²)
      - Conversion en SPL moyen :
            SPL_moy = 10 * log10(p²_moy / (20e-6)²)
    
    Paramètre:
      spl_arrays : liste d'arrays numpy contenant les courbes SPL (en dB) issues des FFT (doivent avoir la même dimension).
    
    Retourne :
      Array numpy du SPL moyen en énergie (en dB).
    """
    ref = 20e-6
    ref_sq = ref**2
    # Conversion en p² pour chaque courbe SPL (dB)
    p_squared_arrays = [ ref_sq * 10**(spl / 10.0) for spl in spl_arrays ]
    # Moyenne spatiale en énergie (point par point)
    p_squared_mean = np.mean(p_squared_arrays, axis=0)
    # Conversion en SPL (dB)
    spl_mean_energy = 10 * np.log10(p_squared_mean / ref_sq)
    return spl_mean_energy


def smooth_curve(y, window_size=5):
    """
    Lisse la courbe y en appliquant une moyenne mobile sur une fenêtre de taille window_size.
    """
    if window_size < 1:
        return y
    window = np.ones(int(window_size)) / float(window_size)
    return np.convolve(y, window, mode='same')


################################################################################
# Mise à jour des sensibilités des microphones
################################################################################
def update_mic_sensitivity(chan, value):
    try:
        value_float = float(value)
        if value_float > 1:  # Si la valeur est en mV/Pa, la convertir en V/Pa
            value_float = value_float / 1000.0
        
        if chan < len(PARAMS["mic_sensitivities"]):
            PARAMS["mic_sensitivities"][chan] = value_float
            print(f"Sensibilité du micro {chan} mise à jour: {value_float} V/Pa")
        else:
            while len(PARAMS["mic_sensitivities"]) <= chan:
                PARAMS["mic_sensitivities"].append(1.0)
            PARAMS["mic_sensitivities"][chan] = value_float
            
        return True
    except ValueError:
        print(f"Erreur: La valeur '{value}' n'est pas un nombre valide")
        return False

################################################################################
# Acquisition complète et mise à jour de l'affichage
################################################################################
def perform_measurement():
    print("Démarrage de la configuration...")
    
    try:
        # Réinitialiser proprement le recorder
        if not reset_recorder(host):
            print("Réinitialisation du recorder échouée, nouvelle tentative...")
            sleep(2)
            reset_recorder(host)
        
        if not open_recorder(host):
            print("Impossible d'ouvrir le recorder, arrêt.")
            return {}, [{} for _ in range(PARAMS["nb_channels"])], 0
            
        module_info = get_module_info(host)
        if not module_info:
            return {}, [{} for _ in range(PARAMS["nb_channels"])], 0
        
        if not create_recording(host):
            return {}, [{} for _ in range(PARAMS["nb_channels"])], 0
            
        setup = get_default_input_setup(host)
        utility.update_value("destinations", ["socket"], setup)
        configure_input_channels(host, setup)
        
        prepare_generator(host)
        generator_setup = get_default_generator_setup(host)
        configure_generator(host, generator_setup)
        
        start_generator(host)
        sleep(0.5)
        
        port = get_streaming_port(host)
        start_measurement(host)
        
        sample_rate = calculate_sample_rate(setup, module_info)
        data_dict, interpretations = stream_data(PARAMS["ip"], port, sample_rate, PARAMS["duration_seconds"], PARAMS["nb_channels"])
        
        stop_measurement(host)
        stop_generator(host)
        
        return data_dict, interpretations, sample_rate
        
    except Exception as e:
        print(f"Erreur pendant la mesure: {e}")
        return {}, [{} for _ in range(PARAMS["nb_channels"])], 0
        
    finally:
        # S'assurer que tout est bien arrêté et fermé à la fin, même en cas d'erreur
        try:
            stop_measurement(host)
            stop_generator(host)
            # Ne pas fermer le recorder ici car cela empêcherait l'extraction des données
        except:
            pass

################################################################################
# Mise à jour des graphiques interactifs
################################################################################
def update_figure(data_dict, interpretations, sample_rate):
    global axs, fig

    if hasattr(axs, "ndim") and axs.ndim == 2:
        axs = axs.flatten()

    fig.subplots_adjust(hspace=0.5, top=0.92, bottom=0.15, left=0.1, right=0.95)

    for chan, ax in enumerate(axs):
        ax.clear()
        if chan not in data_dict or data_dict[chan].size == 0:
            ax.text(0.5, 0.5, f"Aucune donnée pour le canal {chan}",
                    horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        else:
            freq, s_dbspl = compute_fft(data_dict[chan], sample_rate, interpretations[chan], chan)
            s_dbspl_smoothed = smooth_curve(s_dbspl, window_size=10)
            if len(freq) > 0:
                s_dbspl_smoothed = smooth_curve(s_dbspl, window_size=PARAMS["smooth_qty"])

                min_freq_idx = np.argmax(freq >= 250)
                if min_freq_idx < len(freq):
                    max_idx = min_freq_idx + np.argmax(s_dbspl[min_freq_idx:])
                    max_freq = freq[max_idx]
                    max_spl = s_dbspl[max_idx]
                    peak_text = f"Pic max: {max_spl:.1f} dB @ {max_freq:.1f} Hz"
                else:
                    peak_text = "Pas de données >250 Hz"

                ax.semilogx(freq, s_dbspl_smoothed, color=PARAMS["graph_line_color"])

                octave_freqs = [31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000]
                ax.set_xticks(octave_freqs)
                ax.set_xticklabels([str(f) for f in octave_freqs], rotation=45, fontsize=10)
                ax.tick_params(axis='x', which='minor', bottom=False)  # Disable x minor ticks

                ax.grid(True, which='major', linestyle='-', alpha=0.7)
                ax.grid(True, which='minor', linestyle=':', alpha=0.4)
                ax.minorticks_on()  # Enable minor ticks globally

                if min_freq_idx < len(freq) and max_spl > 0:
                    ax.plot(max_freq, max_spl, 'ro', markersize=6)
                    y_pos = max_spl + 5 if max_spl < PARAMS["ylim"][1] - 10 else max_spl - 10
                    ax.annotate(f"{max_spl:.1f} dB\n@ {max_freq:.1f} Hz",
                                xy=(max_freq, max_spl),
                                xytext=(max_freq, y_pos),
                                textcoords='data',
                                ha='center',
                                fontsize=10,
                                bbox=dict(facecolor='lightblue', alpha=0.7, edgecolor='none', pad=3),
                                arrowprops=dict(arrowstyle='->', color='red', lw=1.5))

                ax.set_xlabel("Frequency [Hz]", fontsize=12)
                ax.set_ylabel("Amplitude [dBSPL]", fontsize=12)
                ax.set_xlim(PARAMS["xlim"])
                ax.set_ylim(PARAMS["ylim"])

                if chan < len(PARAMS["mic_sensitivities"]):
                    sensitivity_mV = PARAMS["mic_sensitivities"][chan] * 1000
                    ax_title = f"Canal {chan} - {sensitivity_mV:.1f}mV/Pa - {peak_text}"
                else:
                    ax_title = f"Canal {chan} - {peak_text}"
                ax.set_title(ax_title, fontsize=12, fontweight='bold')
                ax.set_facecolor(PARAMS["graph_background_color"])
                ax.tick_params(axis='both', which='major', labelsize=10)
            else:
                ax.text(0.5, 0.5, f"Impossible de calculer FFT pour le canal {chan}",
                        horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)

    fig.suptitle(PARAMS["graph_title"], fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0.15, 1, 0.95])
    fig.canvas.draw_idle()

def update_transmission_loss_plot(data_dict, interpretations, sample_rate):
    """
    Calcule et trace la Transmission Loss (TL) définie par :
         TL = (SPL moyen en énergie des canaux 1, 2 et 3) - (SPL du canal 0, micro 4189)

    La fonction propose des améliorations graphiques sur l’affichage de la courbe TL.
    """
    # Calcul de la FFT pour le canal de référence (micro 4189, canal 0)
    freq_ref, s_dbspl_ref = compute_fft(data_dict[0], sample_rate, interpretations[0], 0)
    
    # Récupération des courbes SPL (dB) pour les micros 1, 2 et 3
    spl_arrays = []
    for ch in [1, 2, 3]:
        freq, s_dbspl = compute_fft(data_dict[ch], sample_rate, interpretations[ch], ch)
        spl_arrays.append(s_dbspl)
    
    # Calcul du SPL moyen en énergie pour les trois micros (1, 2 et 3)
    spl_mean_energy = calculate_energy_average_SPL(spl_arrays)
    
    # Calcul de la Transmission Loss : différence entre le SPL moyen en énergie et le SPL du micro 4189
    TL = spl_mean_energy - s_dbspl_ref
    TL_smoothed = smooth_curve(TL, window_size=PARAMS["smooth_qty"])

    # Création et traçage de la figure TL avec améliorations graphiques
    fig_TL, ax_TL = plt.subplots(figsize=(12, 6))
    
    # Tracé de la courbe TL en échelle logarithmique pour l'axe des x
    ax_TL.semilogx(freq_ref, TL_smoothed, color='green', lw=2, label="Transmission Loss (TL)")
    ax_TL.set_facecolor(PARAMS["graph_background2_color"])
    
    # Titre et labels avec des polices améliorées
    ax_TL.set_title("Transmission Loss (TL [dB])", fontsize=14, fontweight='bold')
    ax_TL.set_xlabel("Fréquence [Hz]", fontsize=12)
    ax_TL.set_ylabel("TL [dB]", fontsize=12)
    
    # Ajout d'une grille complète (majeure et mineure)
    ax_TL.grid(which='both', linestyle='--', alpha=0.6)
    ax_TL.minorticks_on()

    # Définition des étiquettes en octaves sur l'axe des x pour une lecture simplifiée
    octave_freqs = [31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000]
    ax_TL.set_xticks(octave_freqs)
    ax_TL.set_xticklabels([str(f) for f in octave_freqs], rotation=45, fontsize=10)
    
    # Fixer la plage de fréquences selon les paramètres (xlim)
    ax_TL.set_xlim(PARAMS["xlim"])
    
    # Optionnel : Ajustement automatique de l'axe y selon la plage de TL (ou fixer des limites)
    ax_TL.set_ylim([min(TL)-5, max(TL)+5])
    
    # Recherche et annotation du pic maximal de TL
    if TL_smoothed.size:
        idx_max = np.argmax(TL_smoothed)
        peak_freq = freq_ref[idx_max]
        peak_TL = TL_smoothed[idx_max]
        ax_TL.plot(peak_freq, peak_TL, 'ro', markersize=8)
        y_offset = 5 if peak_TL < PARAMS["ylim"][1] - 10 else -10
        ax_TL.annotate(f"Pic: {peak_TL:.1f} dB\n@ {peak_freq:.1f} Hz", 
                       xy=(peak_freq, peak_TL), 
                       xytext=(peak_freq, peak_TL + y_offset),
                       textcoords='data',
                       arrowprops=dict(facecolor='red', shrink=0.05),
                       fontsize=10,
                       ha='center')
    
    ax_TL.legend(fontsize=12)
    plt.tight_layout()
    plt.show()
    
    

def on_sensitivity_change(text, chan):
    if update_mic_sensitivity(chan, text):
        if 'axs' in globals() and len(axs) > chan:
            sensitivity_mV = PARAMS["mic_sensitivities"][chan] * 1000
            axs[chan].set_title(f"{PARAMS['graph_title']} - Canal {chan} - Mic: {sensitivity_mV:.2f} mV/Pa")
            fig.canvas.draw_idle()

################################################################################
# Application principale et affichage interactif
################################################################################
if __name__ == '__main__':
    try:
        reset_recorder(host)
        sleep(1)
        
        data_dict, interpretations, sample_rate = perform_measurement()
        nb = PARAMS["nb_channels"]
        
        # Création des axes selon le nombre de micros
        if nb == 4:
            fig, axs = plt.subplots(2, 2, figsize=(12, 8))
            axs = axs.flatten()  # Passage en liste plate pour simplifier l'itération
        else:
            fig, axs = plt.subplots(nb, 1, figsize=(12, 4 * nb))
            if nb == 1:
                axs = [axs]
        
        # Mise à jour des graphiques classiques (spectres FFT lissés)
        update_figure(data_dict, interpretations, sample_rate)
        
        # Affichage de la Transmission Loss dans une nouvelle figure
        update_transmission_loss_plot(data_dict, interpretations, sample_rate)
        
        # Configuration du bouton pour relancer la mesure
        button_ax = fig.add_axes([0.8, 0.02, 0.15, 0.05])
        restart_button = Button(button_ax, "Relancer la mesure")
        restart_button.on_clicked(on_button_click)
        
        # Création des TextBox pour modifier la sensibilité de chaque micro
        sensitivity_boxes = []
        for i in range(nb):
            box_ax = fig.add_axes([0.1, 0.02 + i * 0.05, 0.2, 0.04])
            sensitivity_mV = PARAMS["mic_sensitivities"][i] * 1000 if i < len(PARAMS["mic_sensitivities"]) else 0
            text_box = TextBox(box_ax, f"Mic {i} (mV/Pa): ", initial=f"{sensitivity_mV:.2f}")
            
            def create_callback(chan):
                return lambda text: on_sensitivity_change(text, chan)
            
            text_box.on_submit(create_callback(i))
            sensitivity_boxes.append(text_box)
        
        plt.tight_layout(rect=[0, 0.15, 1, 1])
        plt.show()
        
    except Exception as e:
        print(f"Erreur dans le script principal: {e}")
    finally:
        print("Fermeture du script, nettoyage des ressources...")
        try:
            stop_measurement(host)
            stop_generator(host)
            requests.put(host + "/rest/rec/close")
        except Exception as ex:
            print(f"Erreur lors du nettoyage: {ex}")
        print("Fin du script.")