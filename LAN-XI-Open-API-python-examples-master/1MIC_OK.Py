from HelpFunctions.lanxi import LanXI
from HelpFunctions.Stream import streamHandler
from HelpFunctions.Buffer import DataBuffer
import HelpFunctions.utility as utility
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, TextBox
import numpy as np
import sys
import collections
import time
from scipy import signal
import pandas as pd
import os
from tkinter import Tk, filedialog, simpledialog
import threading

# ========================= PARAMÈTRES CONFIGURABLES =========================
# Paramètres du système d'acquisition
LAN_XI_IP = "169.254.254.218"     # Adresse IP du module Lan-XI

# Paramètres d'affichage généraux
CHUNK_SIZE = 2**12                # Taille de la fenêtre d'analyse
ANIMATION_INTERVAL = 5           # Intervalle de rafraîchissement en ms
FIGURE_SIZE = (12, 10)            # Taille de la figure (largeur, hauteur) en pouces

# Paramètres d'affichage temporel
TIME_WINDOW = 10.0                # Fenêtre d'affichage temporel en secondes
TIME_YRANGE = (-50, 50)       # Plage d'amplitude le graphique temporel
TIME_BACKGROUND = 'lightgreen'    # Couleur de fond du graphique temporel
TIME_LINECOLOR = 'blue'           # Couleur de la courbe temporelle
TIME_LINEWIDTH = 1.2              # Épaisseur de la ligne temporelle

# Paramètres d'affichage FFT
FFT_YRANGE = (30, 110)            # Plage d'amplitude en dB pour le graphique FFT
FFT_BACKGROUND = 'lightyellow'    # Couleur de fond du graphique FFT
FFT_LINECOLOR = 'red'             # Couleur de la courbe FFT
FFT_LINEWIDTH = 1.2               # Épaisseur de la ligne FFT

# Paramètres de moyennage
AVERAGE_TIME_WINDOW = 10          # Fenêtre de moyennage en secondes
FFT_AVERAGING_WEIGHT = 0.4        # Poids du spectre actuel dans la moyenne glissante (0-1)

# Référence acoustique (en Pa)
REFERENCE_PRESSURE = 20e-6        # 20 µPa (référence standard acoustique)

# Paramètres du filtre passe-bas
CUTOFF_FREQ = 40000               # Fréquence de coupure du filtre passe-bas (20 kHz)
FILTER_ORDER = 4                  # Ordre du filtre

# Paramètres d'exportation des données
DEFAULT_PROJECT_NAME = "Projet"   # Nom de projet par défaut
DEFAULT_SAVE_DIR = "output_data"  # Dossier de sauvegarde par défaut

# Paramètres du microphone
DEFAULT_MIC_SENSITIVITY = 42.54    # Sensibilité du microphone en mV/Pa
VOLTAGE_RANGE = 10.0              # Plage de tension du LAN-XI en volts crête-à-crête
# ==========================================================================

# IP of Lan-XI
Lanxi = LanXI(LAN_XI_IP)
Lanxi.setup_stream()

class FigHandler:
    def __init__(self):
        self.ChunkToShow = CHUNK_SIZE
        self.fftSize = self.ChunkToShow
        
        # Initialisation des buffers pour le moyennage temporel
        # Calcul du nombre d'échantillons à enregistrer pour la fenêtre de moyennage
        self.averaging_buffer_size = int(AVERAGE_TIME_WINDOW * Lanxi.sample_rate / self.ChunkToShow)
        self.time_buffer = collections.deque(maxlen=self.averaging_buffer_size)
        self.fft_buffer = collections.deque(maxlen=self.averaging_buffer_size)
        
        # Pour l'affichage des données temporelles sur une fenêtre glissante
        self.time_display_buffer_size = int(TIME_WINDOW * Lanxi.sample_rate)
        self.continuous_buffer = collections.deque(maxlen=self.time_display_buffer_size)
        
        # Timestamp de démarrage pour calculer le temps écoulé
        self.start_time = time.time()
        self.elapsed_time = 0
        
        # Variable pour gérer l'état de pause
        self.paused = False
        self.current_data = None
        self.current_spectrum = None
        
        # Paramètres de sauvegarde
        self.project_name = DEFAULT_PROJECT_NAME
        self.save_directory = DEFAULT_SAVE_DIR
        
        # Sensibilité du microphone
        self.mic_sensitivity = DEFAULT_MIC_SENSITIVITY  # en mV/Pa
        
        # Fenêtre pour la FFT
        self.win = np.hamming(self.fftSize)
        
        # Préparation des figures avec espace pour les boutons et la zone de texte
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=FIGURE_SIZE)
        
        # Configuration initiale pour le graphique temporel (sera mis à jour dynamiquement)
        self.ax1.set_facecolor(TIME_BACKGROUND)
        self.line1, = self.ax1.plot([], [], color=TIME_LINECOLOR, linewidth=TIME_LINEWIDTH)
        self.ax1.set_ylim(bottom=TIME_YRANGE[0], top=TIME_YRANGE[1])
        self.ax1.grid(True, linestyle='--', alpha=0.7)
        self.ax1.set_xlabel("Temps [s]", fontsize=10)
        self.ax1.set_ylabel("Amplitude [Pa]", fontsize=10)  # Changé de dB à Pa
        self.ax1.set_title(f"Signal Temporel (fenêtre de {TIME_WINDOW} s)", fontsize=12)
        
        # Axe fréquentiel (pour le graphique du bas)
        self.freq_axis = np.arange((self.fftSize // 2) + 1) / (float(self.fftSize) / Lanxi.sample_rate)
        
        # Configuration du graphique FFT
        self.ax2.set_facecolor(FFT_BACKGROUND)
        self.line2, = self.ax2.plot(self.freq_axis, np.zeros(len(self.freq_axis)), 
                                    color=FFT_LINECOLOR, linewidth=FFT_LINEWIDTH)
        self.ax2.set_xlim(left=150, right=20000)
        self.ax2.set_ylim(bottom=FFT_YRANGE[0], top=FFT_YRANGE[1])
        self.ax2.grid(True, linestyle='--', alpha=0.7)
        self.ax2.set_xlabel("Fréquence [Hz]", fontsize=10)
        self.ax2.set_ylabel("Amplitude [dB SPL]", fontsize=10)
        self.ax2.set_title(f"Spectre Fréquentiel (moyenné sur {AVERAGE_TIME_WINDOW} s)", fontsize=12)
        
        # Ajout d'un espace pour les boutons et la zone de texte
        plt.subplots_adjust(bottom=0.25)
        
        # Création du bouton de pause
        self.ax_pause = plt.axes([0.1, 0.15, 0.15, 0.05])
        self.btn_pause = Button(self.ax_pause, 'Pause / Lecture')
        self.btn_pause.on_clicked(self.toggle_pause)
        
        # Création du bouton de sélection de dossier
        self.ax_folder = plt.axes([0.3, 0.15, 0.15, 0.05])
        self.btn_folder = Button(self.ax_folder, 'Choisir dossier')
        self.btn_folder.on_clicked(self.choose_directory)
        
        # Création du bouton de réinitialisation
        self.ax_reset = plt.axes([0.7, 0.15, 0.15, 0.05])
        self.btn_reset = Button(self.ax_reset, 'Réinitialiser')
        self.btn_reset.on_clicked(self.reset_acquisition)
        
        # Création du bouton pour définir la sensibilité du microphone
        self.ax_sensitivity = plt.axes([0.5, 0.15, 0.15, 0.05])
        self.btn_sensitivity = Button(self.ax_sensitivity, 'Sensibilité Micro')
        self.btn_sensitivity.on_clicked(self.set_mic_sensitivity)
        
        # Création de la zone de texte pour le nom du projet
        self.ax_project = plt.axes([0.1, 0.08, 0.35, 0.05])
        self.txt_project = TextBox(self.ax_project, 'Nom du projet: ', initial=DEFAULT_PROJECT_NAME)
        self.txt_project.on_submit(self.update_project_name)
        
        # Création du bouton d'enregistrement
        self.ax_save = plt.axes([0.55, 0.08, 0.35, 0.05])
        self.btn_save = Button(self.ax_save, 'Enregistrer données')
        self.btn_save.on_clicked(self.save_data)
        
        # Texte affichant le dossier de sauvegarde et la sensibilité du micro
        self.ax_info_dir = plt.figtext(0.1, 0.02, f"Dossier: {self.save_directory}", fontsize=9)
        self.ax_info_mic = plt.figtext(0.6, 0.02, f"Sensibilité: {self.mic_sensitivity} mV/Pa", fontsize=9)
        
        # Désactiver le bouton d'enregistrement au démarrage
        self.btn_save.color = 'lightgray'
        self.btn_save.hovercolor = 'lightgray'
        self.btn_save.active = False
        
        # Initialisation du filtre passe-bas
        self.nyquist = 0.5 * Lanxi.sample_rate
        self.normal_cutoff = CUTOFF_FREQ / self.nyquist
        self.b, self.a = signal.butter(FILTER_ORDER, self.normal_cutoff, btype='low')
        
        # Ajustement de la mise en page
        self.fig.tight_layout(rect=[0, 0.25, 1, 0.95])
        
        # Gestion de la fermeture de la figure
        self.fig.canvas.mpl_connect('close_event', on_close)

    def convert_voltage_to_pa(self, data):
        """
        Convertit la tension mesurée en Pascal en utilisant la sensibilité du microphone.
        
        Args:
            data: Données de tension (normalisées entre -1 et 1)
            
        Returns:
            Données en Pascal
        """
        # Conversion des données normalisées en volts
        voltage_data = data * (VOLTAGE_RANGE / 2)  # Convertir les données normalisées en volts
        
        # Conversion des volts en Pascal en utilisant la sensibilité du microphone (mV/Pa)
        pa_data = voltage_data / (self.mic_sensitivity / 1000)  # Conversion mV en V en divisant par 1000
        
        return pa_data
    
    def set_mic_sensitivity(self, event):
        """Ouvre une boîte de dialogue pour définir la sensibilité du microphone"""
        # On crée un thread séparé pour ouvrir la boîte de dialogue afin d'éviter
        # le blocage de l'interface graphique
        threading.Thread(target=self._open_sensitivity_dialog).start()
    
    def _open_sensitivity_dialog(self):
        """Ouvre la boîte de dialogue dans un thread séparé"""
        # On cache la fenêtre principale Tk qui apparaît par défaut
        root = Tk()
        root.withdraw()
        
        # Ouvre la boîte de dialogue pour entrer la sensibilité
        new_sensitivity = simpledialog.askfloat(
            "Sensibilité du microphone", 
            "Entrer la sensibilité du microphone (mV/Pa):",
            initialvalue=self.mic_sensitivity,
            minvalue=0.1,
            maxvalue=1000
        )
        
        # Si une valeur a été entrée
        if new_sensitivity:
            self.mic_sensitivity = new_sensitivity
            # Mise à jour du texte d'information
            # Note: Cette opération doit être effectuée dans le thread principal
            plt.figure(self.fig.number)  # S'assurer qu'on travaille sur la bonne figure
            self.ax_info_mic.set_text(f"Sensibilité: {self.mic_sensitivity} mV/Pa")
            self.fig.canvas.draw_idle()
            print(f"Sensibilité du microphone définie à: {self.mic_sensitivity} mV/Pa")
        
        # Détruire la fenêtre Tk
        root.destroy()
    
    def update_project_name(self, text):
        """Met à jour le nom du projet"""
        self.project_name = text
        print(f"Nom du projet défini: {self.project_name}")
    
    def choose_directory(self, event):
        """Ouvre une boîte de dialogue pour choisir le répertoire de sauvegarde"""
        # On crée un thread séparé pour ouvrir la boîte de dialogue afin d'éviter
        # le blocage de l'interface graphique
        threading.Thread(target=self._open_directory_dialog).start()
    
    def _open_directory_dialog(self):
        """Ouvre la boîte de dialogue dans un thread séparé"""
        # On cache la fenêtre principale Tk qui apparaît par défaut
        root = Tk()
        root.withdraw()
        
        # Ouvre la boîte de dialogue pour sélectionner un dossier
        directory = filedialog.askdirectory(
            title="Sélectionner le dossier de sauvegarde",
            initialdir=self.save_directory if os.path.exists(self.save_directory) else "/"
        )
        
        # Si un dossier a été sélectionné
        if directory:
            self.save_directory = directory
            # Mise à jour du texte d'information
            # Note: Cette opération doit être effectuée dans le thread principal
            plt.figure(self.fig.number)  # S'assurer qu'on travaille sur la bonne figure
            self.ax_info_dir.set_text(f"Dossier: {self.save_directory}")
            self.fig.canvas.draw_idle()
            print(f"Dossier de sauvegarde: {self.save_directory}")
        
        # Détruire la fenêtre Tk
        root.destroy()
        
    def reset_acquisition(self, event):
        """Réinitialise les buffers de données et redémarre les mesures"""
        # Réinitialisation des buffers
        self.time_buffer.clear()
        self.fft_buffer.clear()
        self.continuous_buffer.clear()
        
        # Réinitialisation du temps de démarrage
        self.start_time = time.time()
        self.elapsed_time = 0
        
        # Réinitialiser les données courantes
        self.current_data = None
        self.current_spectrum = None
        
        # Réinitialiser les graphiques
        self.line1.set_data([], [])
        self.line2.set_ydata(np.zeros(len(self.freq_axis)))
        
        # Mise à jour du titre pour refléter la réinitialisation
        self.ax1.set_title(f"Signal Temporel (fenêtre de {TIME_WINDOW} s)", fontsize=12)
        
        print("Acquisition réinitialisée")
        
        # Forcer le rafraîchissement de la figure
        self.fig.canvas.draw_idle()
    
    
    def toggle_pause(self, event):
        """Bascule l'état de pause et active/désactive le bouton d'enregistrement"""
        self.paused = not self.paused
        
        if self.paused:
            self.btn_pause.label.set_text('Reprendre')
            self.btn_save.color = '0.85'
            self.btn_save.hovercolor = '0.95'
            self.btn_save.active = True
        else:
            self.btn_pause.label.set_text('Pause')
            self.btn_save.color = 'lightgray'
            self.btn_save.hovercolor = 'lightgray'
            self.btn_save.active = False
        
        self.fig.canvas.draw_idle()
    
    def save_data(self, event):
        """Enregistre les données actuelles en CSV et les graphiques en PNG"""
        if not self.paused or not self.btn_save.active:
            return
        
        # S'assurer que le dossier de sauvegarde existe
        if not os.path.exists(self.save_directory):
            try:
                os.makedirs(self.save_directory)
            except Exception as e:
                print(f"Erreur lors de la création du dossier: {e}")
                return
        
        # Horodatage pour nommer les fichiers
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        
        # Construire les noms de fichiers avec le nom du projet
        base_filename = f"{self.project_name}_{timestamp}"
        
        # Enregistrer les données temporelles en CSV
        if self.current_data is not None:
            time_data = pd.DataFrame({
                "temps_s": np.linspace(0, len(self.current_data) / Lanxi.sample_rate, len(self.current_data)),
                "amplitude_pa": self.current_data,  # Déjà converti en Pa
                "amplitude_db_spl": 20 * np.log10(np.abs(self.current_data) / REFERENCE_PRESSURE)  # Conversion en dB SPL
            })
            time_file = os.path.join(self.save_directory, f"TEMP_{base_filename}.csv")
            time_data.to_csv(time_file, index=False)
            
        # Enregistrer les données FFT en CSV
        if self.current_spectrum is not None and len(self.freq_axis) == len(self.current_spectrum):
            fft_data = pd.DataFrame({
                "frequence_hz": self.freq_axis,
                "amplitude_db_spl": self.current_spectrum
            })
            fft_file = os.path.join(self.save_directory, f"FFT_{base_filename}.csv")
            fft_data.to_csv(fft_file, index=False)
        
        # Enregistrer les métadonnées de l'acquisition
        metadata = {
            "Projet": self.project_name,
            "Date": time.strftime("%Y-%m-%d"),
            "Heure": time.strftime("%H:%M:%S"),
            "Sensibilité microphone (mV/Pa)": self.mic_sensitivity,
            "Fréquence d'échantillonnage (Hz)": Lanxi.sample_rate,
            "Fréquence de coupure filtre (Hz)": CUTOFF_FREQ,
            "Ordre du filtre": FILTER_ORDER,
            "Temps d'acquisition (s)": self.elapsed_time
        }
        metadata_file = os.path.join(self.save_directory, f"META_{base_filename}.csv")
        pd.DataFrame([metadata]).to_csv(metadata_file, index=False)
        
        # Enregistrer les graphiques en PNG
        temp_fig_file = os.path.join(self.save_directory, f"TEMP_{base_filename}.png")
        fft_fig_file = os.path.join(self.save_directory, f"FFT_{base_filename}.png")
        
        # Créer une figure temporaire pour sauvegarder chaque graphique individuellement
        # Figure temporaire pour le graphique temporel
        temp_fig, temp_ax = plt.subplots(figsize=(10, 6))
        temp_ax.set_facecolor(TIME_BACKGROUND)
        temp_ax.plot(self.line1.get_xdata(), self.line1.get_ydata(), color=TIME_LINECOLOR, linewidth=TIME_LINEWIDTH)
        temp_ax.set_ylim(self.ax1.get_ylim())
        temp_ax.set_xlim(self.ax1.get_xlim())
        temp_ax.grid(True, linestyle='--', alpha=0.7)
        temp_ax.set_xlabel("Temps [s]", fontsize=10)
        temp_ax.set_ylabel("Amplitude [Pa]", fontsize=10)
        temp_ax.set_title(f"Signal Temporel Filtré - {self.project_name} - Sensibilité: {self.mic_sensitivity} mV/Pa", fontsize=11)
        temp_fig.tight_layout()
        temp_fig.savefig(temp_fig_file)
        plt.close(temp_fig)
        
        # Figure temporaire pour le graphique FFT
        fft_fig, fft_ax = plt.subplots(figsize=(10, 6))
        fft_ax.set_facecolor(FFT_BACKGROUND)
        fft_ax.plot(self.freq_axis, self.line2.get_ydata(), color=FFT_LINECOLOR, linewidth=FFT_LINEWIDTH)
        fft_ax.set_ylim(self.ax2.get_ylim())
        fft_ax.set_xlim(self.ax2.get_xlim())
        fft_ax.grid(True, linestyle='--', alpha=0.7)
        fft_ax.set_xlabel("Fréquence [Hz]", fontsize=10)
        fft_ax.set_ylabel("Amplitude [dB SPL]", fontsize=10)
        fft_ax.set_title(f"Spectre Fréquentiel - {self.project_name} - Sensibilité: {self.mic_sensitivity} mV/Pa", fontsize=11)
        fft_fig.tight_layout()
        fft_fig.savefig(fft_fig_file)
        plt.close(fft_fig)
        
        # Message de confirmation
        print(f"Données enregistrées dans {self.save_directory}:")
        print(f"- Signal temporel: {time_file}")
        print(f"- Spectre FFT: {fft_file}")
        print(f"- Métadonnées: {metadata_file}")
        print(f"- Graphique temporel: {temp_fig_file}")
        print(f"- Graphique FFT: {fft_fig_file}")
    
    def _update(self, i):
        # Si en pause, ne pas mettre à jour les graphiques
        if self.paused:
            return self.line1, self.line2,
        
        # Mise à jour du temps écoulé
        self.elapsed_time = time.time() - self.start_time
        
        # Récupération des données actuelles
        data = DataBuffer.getPart(self.ChunkToShow)
        
        # Application du filtre passe-bas
        filtered_data = signal.filtfilt(self.b, self.a, data)
        
        # Conversion des données en Pascal en utilisant la sensibilité du microphone
        data_pa = self.convert_voltage_to_pa(filtered_data)
        
        # Sauvegarde des données courantes pour l'export (en Pascal)
        self.current_data = data_pa
        
        # Ajout des données temporelles au buffer pour moyennage
        self.time_buffer.append(data_pa)
        
        # Ajout des données au buffer continu pour l'affichage temporel
        for sample in data_pa:
            self.continuous_buffer.append(sample)
        
        # Préparer les données pour l'affichage temporel
        continuous_data = np.array(self.continuous_buffer)
        
        # Calculer l'axe temporel pour la fenêtre glissante
        window_end_time = self.elapsed_time
        window_start_time = max(0, window_end_time - TIME_WINDOW)
        
        # L'axe temporel représente la fenêtre actuelle de TIME_WINDOW secondes
        time_axis = np.linspace(
            window_start_time, 
            window_end_time, 
            len(continuous_data)
        )
        
        # Mise à jour de l'axe temporel et des données
        self.ax1.set_xlim(window_start_time, window_end_time)
        self.line1.set_data(time_axis, continuous_data)
        
        # Calcul du spectre FFT (sur données filtrées et converties en Pascal)
        freq, current_spectrum = utility.dbfft(data_pa, Lanxi.sample_rate, self.win, ref=REFERENCE_PRESSURE)
        
        # Sauvegarde du spectre courant pour l'export
        self.current_spectrum = current_spectrum
        
        # Ajout du spectre au buffer pour moyennage
        self.fft_buffer.append(current_spectrum)
        
        # Calcul de la moyenne spectrale
        if len(self.fft_buffer) > 0:
            fft_average = np.mean(np.array(self.fft_buffer), axis=0)
            self.line2.set_ydata(fft_average)
        
        # Mise à jour du titre du graphique temporel pour indiquer le temps écoulé
        self.ax1.set_title(f"Signal Temporel Filtré (Temps écoulé: {self.elapsed_time:.1f} s)", fontsize=12)
        
        return self.line1, self.line2,
    
    def startAnimation(self):
        self.ani = FuncAnimation(
            self.fig, 
            self._update, 
            interval=ANIMATION_INTERVAL, 
            blit=True
        )

# Create the stream and Rx data
streamer = streamHandler(Lanxi)

def on_close(event):
    print('Fermeture de la figure!')
    streamer.stopStream()
    sys.exit(0)

if __name__ == "__main__":
    try:
        # Affichage des paramètres actuels
        print(f"Démarrage de l'acquisition avec les paramètres suivants:")
        print(f"- Module LAN-XI: {LAN_XI_IP}")
        print(f"- Taille de fenêtre: {CHUNK_SIZE}")
        print(f"- Fenêtre d'affichage temporel: {TIME_WINDOW} secondes")
        print(f"- Fenêtre de moyennage: {AVERAGE_TIME_WINDOW} secondes")
        print(f"- Fréquence d'échantillonnage: {Lanxi.sample_rate} Hz")
        print(f"- Filtre passe-bas: {CUTOFF_FREQ} Hz (ordre {FILTER_ORDER})")
        print(f"- Sensibilité microphone: {DEFAULT_MIC_SENSITIVITY} mV/Pa")
        print(f"- Dossier de sauvegarde par défaut: {DEFAULT_SAVE_DIR}")
        
        # Démarrage du thread d'acquisition
        threading.Thread(target=streamer.startStream).start()
        
        # Création et démarrage de l'interface graphique
        fig = FigHandler()
        fig.startAnimation()
        plt.show()
        
    except Exception as e:
        print(f"Une erreur s'est produite : {e}")
        streamer.stopStream()