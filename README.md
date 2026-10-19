# Escriba

Transcrição de áudio em texto com separação de falantes, com interface gráfica e **100% local**: nenhum áudio é enviado para a nuvem.

## Recursos

- Interface gráfica com tema escuro
- Transcrição com [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (modelos Whisper)
- Separação de falantes com [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)
- Aceita áudio e vídeo (mp3, wav, m4a, flac, ogg, mp4, mkv, webm, mov)
- Detecção automática de idioma ou idioma escolhido manualmente
- Exporta em `.txt` e `.srt` (legenda com tempos)
- Suporte opcional a GPU Nvidia (CUDA)

Exemplo de resultado:

```
[00:00:03] Falante 1: Bom dia, vamos começar a reunião.

[00:00:07] Falante 2: Bom dia! Tenho alguns pontos para apresentar.
```

## Requisitos

- Python 3.10 ou mais recente
- Internet na primeira execução, para baixar os modelos de IA (depois funciona offline)

## Instalação

### Windows (PowerShell)

```powershell
# Instalar Python e Git (pule o que já tiver)
winget install Python.Python.3.12
winget install Git.Git
```

Feche e abra o PowerShell de novo. Depois:

```powershell
git clone https://github.com/SEU_USUARIO/SEU_REPOSITORIO.git
cd SEU_REPOSITORIO

python -m venv .venv
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
python app.py
```

Se aparecer o erro "a execução de scripts foi desabilitada", rode uma vez:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

### macOS e Linux

```bash
git clone https://github.com/SEU_USUARIO/SEU_REPOSITORIO.git
cd SEU_REPOSITORIO

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
python app.py
```

No Ubuntu/Debian, instale antes o suporte à interface gráfica:

```bash
sudo apt install python3-tk
```

## Uso

Nas próximas vezes, basta ativar o ambiente e abrir o programa:

```powershell
cd SEU_REPOSITORIO
.venv\Scripts\Activate.ps1
python app.py
```

1. Clique em **Escolher...** e selecione o áudio ou vídeo.
2. Escolha o **modelo**, o **idioma** e o número de falantes (`0` = automático).
3. Clique em **Transcrever**.
4. Quando terminar, salve o resultado em **.txt** ou **.srt**.

### Qual modelo escolher?

| Modelo | Velocidade | Precisão |
|---|---|---|
| `tiny`, `base` | Muito rápido | Menor |
| `small` (padrão) | Equilibrado | Boa |
| `medium`, `large-v3` | Mais lento | Maior |

Para português, recomenda-se `small` ou superior.

## Primeira execução

Na primeira transcrição, o programa baixa automaticamente:

- o modelo Whisper escolhido;
- os modelos de separação de falantes (salvos em `~/.escriba/modelos`).

Depois disso, tudo funciona sem internet.

## Limitações

- A precisão da separação de falantes pode variar conforme o idioma e a qualidade do áudio. O modelo de identificação de voz usado foi treinado principalmente com chinês, então vale testar com seus áudios.
- Áudios com muito ruído, música ou vozes sobrepostas reduzem a qualidade da separação.
- Modelos grandes (`medium`, `large-v3`) são lentos sem GPU.

## Tecnologias

- [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
- [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)
- tkinter

## Licença

Defina a licença do projeto aqui (por exemplo, MIT).
