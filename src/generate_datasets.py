import os 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# import sklearn as skl
# import sklearn.utils, sklearn.preprocessing, sklearn.decomposition, sklearn.svm

import IPython.display as ipd

import librosa
import librosa.display

import os


def get_audio_path(track_id):
    track_number = '{:06d}'.format(track_id)
    file_name = track_number + '.mp3'
    AUDIO_DIR = './data/raw/fma_small'
    return os.path.join(AUDIO_DIR, track_number[:3], file_name)

def get_log_mel_spectrogram(filepath, display=False):

    try:
        x, sr = librosa.load(filepath, sr=None, mono=True)
    except Exception as e:
        print(f"[SKIP] Erreur lors du chargement de {filepath} : {e}")
        return None

    stft = np.abs(librosa.stft(x, n_fft=2048, hop_length=512)) # stft = Short-Time Fourier Transform

    # Mel-spectrogramme 
    mel = librosa.feature.melspectrogram(sr=sr, S=stft**2, n_mels=128) # sr = 

    # Conversion en dB (ancien logamplitude → remplacé)
    log_mel_spectrogram = librosa.power_to_db(mel, ref=np.max)

    if display:
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(log_mel_spectrogram, sr=sr, hop_length=512,
                                 x_axis='time', y_axis='mel')
        plt.colorbar(format='%+2.0f dB')
        plt.title("Log-mel spectrogram")
        plt.show()

    return log_mel_spectrogram



def get_music_label(track_id: int, df_genres: pd.DataFrame) -> str:
    """
    Retourne le genre_top pour un track_id donné dans le DataFrame passé.
    """
    # return df_genres["genre_top"].iloc[track_id]
    return df_genres.loc[track_id, "genre_top"]


def save_log_mel_spectrogram(track_id: int,
                             df_genres: pd.DataFrame,
                             split: str,
                             base_output_dir: str = "./data/spectrograms"):
    audio_path = get_audio_path(track_id)
    label = get_music_label(track_id, df_genres)

    out_dir = os.path.join(base_output_dir, split, label)
    os.makedirs(out_dir, exist_ok=True)

    filename = f"{int(track_id):06d}.png"
    out_path = os.path.join(out_dir, filename)

    if os.path.exists(out_path):
        return out_path

    spec = get_log_mel_spectrogram(audio_path, display=False)

    # 👉 si la lecture audio a échoué, on arrête là
    if spec is None:
        print(f"[SKIP] track_id={track_id} (fichier audio illisible)")
        return None

    spec_min = spec.min()
    spec_max = spec.max()
    spec_norm = (spec - spec_min) / (spec_max - spec_min + 1e-8)

    plt.imsave(out_path, spec_norm)
    return out_path


if __name__ == '__main__':
    train_genres = pd.read_csv('./data/raw/fma_metadata/small_train_genres.csv', index_col='track_id')
    val_genres   = pd.read_csv('./data/raw/fma_metadata/small_val_genres.csv',   index_col='track_id')
    test_genres  = pd.read_csv('./data/raw/fma_metadata/small_test_genres.csv',  index_col='track_id')

    splits = [
        ("train", train_genres),
        ("val",   val_genres),
        ("test",  test_genres),
    ]

    for split_name, df in splits:
        print(f"\n=== Génération des spectrogrammes pour {split_name} ({len(df)} pistes) ===")

        for i, track_id in enumerate(df.index):
            save_log_mel_spectrogram(track_id, df, split=split_name)

            # petit log toutes les 100 pistes pour voir l’avancement
            if (i + 1) % 100 == 0:
                print(f"... {i+1}/{len(df)} faits")



# if __name__ == '__main__':
#     train_genres = pd.read_csv('./data/raw/fma_metadata/small_train_genres.csv', index_col='track_id')
#     val_genres = pd.read_csv('./data/raw/fma_metadata/small_val_genres.csv', index_col='track_id')
#     test_genres = pd.read_csv('./data/raw/fma_metadata/small_test_genres.csv', index_col='track_id')

#     # On teste sur les 5 premières pistes du train
#     sample = train_genres.head()

#     for track_id, row in sample.iterrows():
#         out_path = save_log_mel_spectrogram(track_id, train_genres, split="train")
#         print(f"track_id = {track_id} -> sauvegardé dans : {out_path}")
