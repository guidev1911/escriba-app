"""
Escriba - transcrição de áudio em texto com separação de falantes.
Roda 100% local. Interface gráfica em tkinter (já vem com o Python).

Transcrição:        faster-whisper (modelos Whisper)
Separação de voz:   sherpa-onnx (pyannote segmentation + embeddings de voz)
"""

import logging
import os
import platform
import queue
import subprocess
import sys
import tarfile
import threading
import traceback
import urllib.request
from importlib import metadata
from logging.handlers import RotatingFileHandler
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

# ----------------------------------------------------------------------
# Configuração
# ----------------------------------------------------------------------
PASTA_MODELOS = Path.home() / ".escriba" / "modelos"

# Modelos de separação de falantes (baixados na primeira vez que for usar).
# Se algum link mudar, confira em https://github.com/k2-fsa/sherpa-onnx/releases
URL_SEGMENTACAO = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
)
URL_EMBEDDING = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "speaker-recongition-models/"
    "3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"
)
ARQ_SEGMENTACAO = PASTA_MODELOS / "sherpa-onnx-pyannote-segmentation-3-0" / "model.onnx"
ARQ_EMBEDDING = PASTA_MODELOS / "3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"

MODELOS_WHISPER = ["tiny", "base", "small", "medium", "large-v3"]
IDIOMAS = {
    "Automático": None, "Português": "pt", "Inglês": "en", "Espanhol": "es",
    "Francês": "fr", "Alemão": "de", "Italiano": "it",
}

# Paleta do tema escuro
COR_FUNDO = "#1e1e24"
COR_CAMPO = "#2a2a32"
COR_BOTAO = "#3a3a46"
COR_HOVER = "#4a4a5a"
COR_BORDA = "#44444f"
COR_TEXTO = "#e6e6ea"
COR_DESABILITADO = "#80808c"
COR_ACENTO = "#6c8cff"

# ----------------------------------------------------------------------
# Logs (arquivo em ~/.escriba/logs/escriba.log)
# ----------------------------------------------------------------------
PASTA_LOGS = Path.home() / ".escriba" / "logs"
ARQ_LOG = PASTA_LOGS / "escriba.log"


class _SaidaParaLog:
    """Substitui stdout/stderr quando não há console (modo --windowed)."""

    def __init__(self, nivel):
        self.nivel = nivel

    def write(self, texto):
        texto = texto.strip()
        if texto:
            logging.log(self.nivel, texto)

    def flush(self):
        pass

    def isatty(self):
        return False


def configurar_logs():
    PASTA_LOGS.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(
        ARQ_LOG, maxBytes=1_000_000, backupCount=3, encoding="utf-8"
    )
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(threadName)s: %(message)s")
    )
    raiz = logging.getLogger()
    raiz.setLevel(logging.DEBUG)
    raiz.addHandler(handler)

    # Se o programa foi aberto por um terminal, mostra os logs nele também
    if sys.stderr is not None:
        console = logging.StreamHandler(sys.stderr)
        console.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        raiz.addHandler(console)

    # Sem console, stdout/stderr são None e algumas bibliotecas quebram ao escrever
    if sys.stderr is None:
        sys.stderr = _SaidaParaLog(logging.WARNING)
    if sys.stdout is None:
        sys.stdout = _SaidaParaLog(logging.INFO)

    # Registra qualquer exceção não tratada, inclusive em threads
    sys.excepthook = lambda t, v, tb: logging.critical(
        "Exceção não tratada", exc_info=(t, v, tb)
    )
    threading.excepthook = lambda a: logging.critical(
        "Exceção não tratada na thread", exc_info=(a.exc_type, a.exc_value, a.exc_traceback)
    )

    logging.info("=== Escriba iniciado ===")
    logging.info("Python %s | %s", sys.version.split()[0], platform.platform())
    logging.info("Executável PyInstaller: %s", getattr(sys, "frozen", False))
    for pacote in ("faster-whisper", "ctranslate2", "av", "sherpa-onnx", "numpy", "onnxruntime"):
        try:
            versao = metadata.version(pacote)
        except metadata.PackageNotFoundError:
            versao = "NÃO ENCONTRADO"
        logging.info("Pacote %s: %s", pacote, versao)


# ----------------------------------------------------------------------
# Lógica
# ----------------------------------------------------------------------
def tempo_curto(s: float) -> str:
    s = int(s)
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def tempo_srt(s: float) -> str:
    ms = int(round(s * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    sec, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{sec:02d},{ms:03d}"


def baixar(url: str, destino: Path, log):
    destino.parent.mkdir(parents=True, exist_ok=True)
    log(f"Baixando {destino.name} (só na primeira vez)...")
    urllib.request.urlretrieve(url, destino)


def garantir_modelos_falantes(log):
    if not ARQ_SEGMENTACAO.exists():
        pacote = PASTA_MODELOS / "segmentacao.tar.bz2"
        baixar(URL_SEGMENTACAO, pacote, log)
        with tarfile.open(pacote, "r:bz2") as t:
            t.extractall(PASTA_MODELOS)
        pacote.unlink(missing_ok=True)
    if not ARQ_EMBEDDING.exists():
        baixar(URL_EMBEDDING, ARQ_EMBEDDING, log)


def separar_falantes(audio, n_falantes: int, log):
    """Retorna lista de (inicio, fim, id_do_falante)."""
    import sherpa_onnx

    garantir_modelos_falantes(log)
    config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                model=str(ARQ_SEGMENTACAO)
            ),
        ),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(ARQ_EMBEDDING)),
        clustering=sherpa_onnx.FastClusteringConfig(
            num_clusters=n_falantes if n_falantes > 0 else -1,
            threshold=0.5,  # usado quando o nº de falantes é automático
        ),
        min_duration_on=0.3,
        min_duration_off=0.5,
    )
    sd = sherpa_onnx.OfflineSpeakerDiarization(config)
    log("Identificando falantes...")
    resultado = sd.process(audio).sort_by_start_time()
    return [(r.start, r.end, r.speaker) for r in resultado]


def falante_do_trecho(inicio, fim, turnos, anterior):
    """Escolhe o falante com maior sobreposição de tempo com o trecho."""
    melhor, melhor_sobreposicao = anterior, 0.0
    for t_ini, t_fim, spk in turnos:
        sobreposicao = min(fim, t_fim) - max(inicio, t_ini)
        if sobreposicao > melhor_sobreposicao:
            melhor, melhor_sobreposicao = spk, sobreposicao
    return melhor


def processar(caminho, modelo, idioma, n_falantes, usar_gpu, separar, saida):
    """Roda em thread separada; comunica com a interface via fila."""
    def log(msg):
        logging.info(msg)
        saida.put(("log", msg))

    logging.info(
        "Iniciando: arquivo=%s modelo=%s idioma=%s falantes=%s gpu=%s separar=%s",
        caminho, modelo, idioma, n_falantes, usar_gpu, separar,
    )
    try:
        from faster_whisper import WhisperModel, decode_audio

        log("Lendo o áudio...")
        audio = decode_audio(str(caminho), sampling_rate=16000)

        dispositivo = "cuda" if usar_gpu else "cpu"
        log(f"Carregando modelo '{modelo}' (baixa na primeira vez)...")
        whisper = WhisperModel(
            modelo, device=dispositivo,
            compute_type="float16" if usar_gpu else "int8",
        )

        log("Transcrevendo...")
        segmentos, info = whisper.transcribe(
            audio, language=idioma, vad_filter=True, beam_size=5
        )
        trechos = []
        for seg in segmentos:  # gerador: a transcrição acontece aqui
            trechos.append((seg.start, seg.end, seg.text.strip()))
            if info.duration:
                saida.put(("prog", min(seg.end / info.duration, 1.0) * 85))

        turnos = []
        if separar:
            try:
                turnos = separar_falantes(audio, n_falantes, log)
            except Exception as e:  # segue sem falantes em vez de perder o texto
                logging.exception("Falha na separação de falantes")
                log(f"Aviso: não foi possível separar falantes ({e}).")

        # Junta trechos consecutivos do mesmo falante em blocos
        blocos, anterior = [], None
        for ini, fim, texto in trechos:
            spk = falante_do_trecho(ini, fim, turnos, anterior) if turnos else None
            anterior = spk if spk is not None else anterior
            if blocos and blocos[-1]["spk"] == spk:
                blocos[-1]["fim"] = fim
                blocos[-1]["texto"] += " " + texto
            else:
                blocos.append({"ini": ini, "fim": fim, "spk": spk, "texto": texto})

        saida.put(("prog", 100))
        saida.put(("pronto", blocos))
    except Exception as e:
        logging.exception("Erro durante a transcrição")
        saida.put(("erro", f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}"))


def rotulo(spk):
    return "" if spk is None else f"Falante {spk + 1}: "


def formatar_txt(blocos):
    return "\n\n".join(
        f"[{tempo_curto(b['ini'])}] {rotulo(b['spk'])}{b['texto']}" for b in blocos
    )


def formatar_srt(blocos):
    partes = []
    for i, b in enumerate(blocos, 1):
        partes.append(
            f"{i}\n{tempo_srt(b['ini'])} --> {tempo_srt(b['fim'])}\n"
            f"{rotulo(b['spk'])}{b['texto']}\n"
        )
    return "\n".join(partes)


# ----------------------------------------------------------------------
# Interface
# ----------------------------------------------------------------------
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Escriba - Transcrição com separação de falantes")
        self.geometry("820x640")
        self.minsize(640, 480)
        self.fila = queue.Queue()
        self.blocos = []
        self.arquivo = tk.StringVar()
        self.modelo = tk.StringVar(value="small")
        self.idioma = tk.StringVar(value="Português")
        self.n_falantes = tk.IntVar(value=0)
        self.separar = tk.BooleanVar(value=True)
        self.gpu = tk.BooleanVar(value=False)
        self._tema()
        self._montar()
        self.after(100, self._ler_fila)

    def _tema(self):
        self.configure(bg=COR_FUNDO)
        estilo = ttk.Style(self)
        estilo.theme_use("clam")  # tema base que aceita personalização de cores
        estilo.configure(
            ".", background=COR_FUNDO, foreground=COR_TEXTO,
            fieldbackground=COR_CAMPO, bordercolor=COR_BORDA,
            lightcolor=COR_FUNDO, darkcolor=COR_FUNDO,
            troughcolor=COR_CAMPO, focuscolor=COR_FUNDO, insertcolor=COR_TEXTO,
        )
        estilo.configure("TButton", background=COR_BOTAO, padding=6)
        estilo.map(
            "TButton",
            background=[("active", COR_HOVER), ("disabled", COR_CAMPO)],
            foreground=[("disabled", COR_DESABILITADO)],
        )
        for campo in ("TEntry", "TCombobox", "TSpinbox"):
            estilo.configure(campo, fieldbackground=COR_CAMPO, foreground=COR_TEXTO,
                             arrowcolor=COR_TEXTO, selectbackground=COR_ACENTO,
                             selectforeground="#ffffff")
        estilo.map(
            "TCombobox",
            fieldbackground=[("readonly", COR_CAMPO)],
            foreground=[("readonly", COR_TEXTO)],
            selectbackground=[("readonly", COR_CAMPO)],
            selectforeground=[("readonly", COR_TEXTO)],
        )
        estilo.map(
            "TCheckbutton",
            background=[("active", COR_FUNDO)],
            indicatorcolor=[("selected", COR_ACENTO), ("!selected", COR_CAMPO)],
        )
        estilo.configure("Horizontal.TProgressbar", background=COR_ACENTO,
                         troughcolor=COR_CAMPO, bordercolor=COR_BORDA)
        estilo.configure("Vertical.TScrollbar", background=COR_BOTAO,
                         troughcolor=COR_CAMPO, arrowcolor=COR_TEXTO,
                         bordercolor=COR_FUNDO)
        estilo.map("Vertical.TScrollbar", background=[("active", COR_HOVER)])
        # Lista suspensa do Combobox (é um widget tk comum, não ttk)
        self.option_add("*TCombobox*Listbox.background", COR_CAMPO)
        self.option_add("*TCombobox*Listbox.foreground", COR_TEXTO)
        self.option_add("*TCombobox*Listbox.selectBackground", COR_ACENTO)
        self.option_add("*TCombobox*Listbox.selectForeground", "#ffffff")

    def _montar(self):
        topo = ttk.Frame(self, padding=10)
        topo.pack(fill="x")
        ttk.Label(topo, text="Arquivo:").grid(row=0, column=0, sticky="w")
        ttk.Entry(topo, textvariable=self.arquivo).grid(row=0, column=1, sticky="ew", padx=6)
        ttk.Button(topo, text="Escolher...", command=self._escolher).grid(row=0, column=2)
        topo.columnconfigure(1, weight=1)

        opcoes = ttk.Frame(self, padding=(10, 0))
        opcoes.pack(fill="x")
        ttk.Label(opcoes, text="Modelo:").grid(row=0, column=0, sticky="w")
        ttk.Combobox(opcoes, textvariable=self.modelo, values=MODELOS_WHISPER,
                     width=10, state="readonly").grid(row=0, column=1, padx=(4, 14))
        ttk.Label(opcoes, text="Idioma:").grid(row=0, column=2, sticky="w")
        ttk.Combobox(opcoes, textvariable=self.idioma, values=list(IDIOMAS),
                     width=12, state="readonly").grid(row=0, column=3, padx=(4, 14))
        ttk.Label(opcoes, text="Nº de falantes (0 = automático):").grid(row=0, column=4)
        ttk.Spinbox(opcoes, from_=0, to=20, textvariable=self.n_falantes,
                    width=4).grid(row=0, column=5, padx=4)

        opcoes2 = ttk.Frame(self, padding=(10, 6))
        opcoes2.pack(fill="x")
        ttk.Checkbutton(opcoes2, text="Separar falantes",
                        variable=self.separar).pack(side="left", padx=(0, 14))
        ttk.Checkbutton(opcoes2, text="Usar GPU Nvidia (CUDA)",
                        variable=self.gpu).pack(side="left")

        acoes = ttk.Frame(self, padding=(10, 4))
        acoes.pack(fill="x")
        self.btn_iniciar = ttk.Button(acoes, text="Transcrever", command=self._iniciar)
        self.btn_iniciar.pack(side="left")
        self.btn_txt = ttk.Button(acoes, text="Salvar .txt", state="disabled",
                                  command=lambda: self._salvar("txt"))
        self.btn_txt.pack(side="left", padx=6)
        self.btn_srt = ttk.Button(acoes, text="Salvar .srt", state="disabled",
                                  command=lambda: self._salvar("srt"))
        self.btn_srt.pack(side="left")
        ttk.Button(acoes, text="Abrir pasta de logs",
                   command=self._abrir_logs).pack(side="right")

        self.barra = ttk.Progressbar(self, maximum=100)
        self.barra.pack(fill="x", padx=10, pady=(6, 0))
        self.status = ttk.Label(self, text="Escolha um arquivo de áudio ou vídeo.")
        self.status.pack(anchor="w", padx=10, pady=2)

        quadro = ttk.Frame(self, padding=10)
        quadro.pack(fill="both", expand=True)
        self.texto = tk.Text(
            quadro, wrap="word", font=("Segoe UI", 10),
            bg=COR_CAMPO, fg=COR_TEXTO, insertbackground=COR_TEXTO,
            selectbackground=COR_ACENTO, selectforeground="#ffffff",
            relief="flat", borderwidth=0, padx=10, pady=8,
            highlightthickness=1, highlightbackground=COR_BORDA,
            highlightcolor=COR_ACENTO,
        )
        rolagem = ttk.Scrollbar(quadro, command=self.texto.yview)
        self.texto.configure(yscrollcommand=rolagem.set)
        rolagem.pack(side="right", fill="y")
        self.texto.pack(side="left", fill="both", expand=True)

    def _abrir_logs(self):
        PASTA_LOGS.mkdir(parents=True, exist_ok=True)
        try:
            os.startfile(PASTA_LOGS)  # Windows
        except AttributeError:
            subprocess.Popen(
                ["open" if sys.platform == "darwin" else "xdg-open", str(PASTA_LOGS)]
            )

    def _escolher(self):
        caminho = filedialog.askopenfilename(
            title="Escolha o áudio ou vídeo",
            filetypes=[("Áudio e vídeo", "*.mp3 *.wav *.m4a *.flac *.ogg *.mp4 *.mkv *.webm *.mov"),
                       ("Todos os arquivos", "*.*")],
        )
        if caminho:
            self.arquivo.set(caminho)

    def _iniciar(self):
        caminho = Path(self.arquivo.get())
        if not caminho.is_file():
            messagebox.showwarning("Escriba", "Escolha um arquivo válido primeiro.")
            return
        self.btn_iniciar.config(state="disabled")
        self.btn_txt.config(state="disabled")
        self.btn_srt.config(state="disabled")
        self.texto.delete("1.0", "end")
        self.barra["value"] = 0
        threading.Thread(
            target=processar,
            args=(caminho, self.modelo.get(), IDIOMAS[self.idioma.get()],
                  int(self.n_falantes.get()), self.gpu.get(), self.separar.get(),
                  self.fila),
            daemon=True,
        ).start()

    def _ler_fila(self):
        try:
            while True:
                tipo, dado = self.fila.get_nowait()
                if tipo == "log":
                    self.status.config(text=dado)
                elif tipo == "prog":
                    self.barra["value"] = dado
                elif tipo == "pronto":
                    self.blocos = dado
                    self.texto.insert("1.0", formatar_txt(dado))
                    self.status.config(text="Pronto!")
                    self.btn_iniciar.config(state="normal")
                    self.btn_txt.config(state="normal")
                    self.btn_srt.config(state="normal")
                elif tipo == "erro":
                    self.status.config(text="Erro.")
                    self.btn_iniciar.config(state="normal")
                    self.texto.delete("1.0", "end")
                    self.texto.insert("1.0", "ERRO (copie tudo abaixo):\n\n" + dado)
                    messagebox.showerror(
                        "Escriba",
                        dado.splitlines()[0]
                        + "\n\nO erro completo está na caixa de texto da janela.",
                    )
        except queue.Empty:
            pass
        self.after(100, self._ler_fila)

    def _salvar(self, formato):
        base = Path(self.arquivo.get()).stem
        destino = filedialog.asksaveasfilename(
            defaultextension=f".{formato}", initialfile=f"{base}.{formato}",
            filetypes=[(formato.upper(), f"*.{formato}")],
        )
        if destino:
            conteudo = formatar_srt(self.blocos) if formato == "srt" else formatar_txt(self.blocos)
            Path(destino).write_text(conteudo, encoding="utf-8")
            self.status.config(text=f"Salvo em {destino}")


if __name__ == "__main__":
    configurar_logs()
    App().mainloop()
