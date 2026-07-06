#README Bachelorarbeit Adrian Preisler

Projekt starten:

1. PowerShell öffnen.
2. In den Projektordner BA-Adrian-Preisler navigieren.
3. Eine `.env`-Datei mit `SECRET_KEY` anlegen, falls sie noch nicht existiert.
4. Den Befehl "docker compose up" eingeben.

Wichtig: Die Anführungszeichen " in dieser README dienen nur dazu, Befehle oder Pfade zu kennzeichnen. Sie müssen nicht mit eingegeben werden.

Nach dem Start ist das Projekt im Browser unter folgender Adresse erreichbar:
http://localhost:5173/

SECRET_KEY lokal erzeugen:

```bash
python3 -c "import secrets; print('SECRET_KEY=' + secrets.token_hex(32))" > .env
```


Docker auf macOS und Linux:

Standardstart, auch für macOS/Apple Silicon:
"docker compose up --build"

Die Standard-Konfiguration nutzt weiterhin linux/amd64, damit die Python/ML-Abhängigkeiten auf macOS stabil laufen. Auf Linux-x86_64 ist das ebenfalls die native Plattform. Nur wenn bewusst eine andere Architektur benötigt wird, kann "DOCKER_PLATFORM" überschrieben werden.

Linux mit NVIDIA-GPU:

Voraussetzungen auf dem Host:
- NVIDIA-Treiber installiert
- NVIDIA Container Toolkit installiert
- "nvidia-smi" funktioniert auf dem Host

Start mit GPU-Zugriff:
"docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build"

Nach dem Start kann geprüft werden, ob die GPU im Backend sichtbar ist:
"curl http://localhost:8503/status"

In der Antwort sollte "gpu.cuda_available" auf true stehen und mindestens ein Gerätename unter "gpu.device_names" erscheinen. Die LLM- und PyTorch-basierten Anomaly-Detection-Pfade verwenden dann automatisch CUDA, wenn PyTorch die GPU sieht.


Projekt nach Änderungen neu bauen:

Wenn Änderungen am Projekt vorgenommen wurden, müssen die betroffenen Container neu gebaut werden.

Nur das Frontend neu bauen:
"docker compose build frontend"

Nur das Backend neu bauen:
"docker compose build backend"

Das gesamte Projekt neu bauen:
"docker compose build"

Nach Änderungen an Python-Abhängigkeiten, GPU/CUDA-Konfiguration, Backend-Code oder Frontend-Code sollte der betroffene Container neu gebaut werden. Für reine README-Änderungen ist kein Rebuild nötig.


Sicherheit und Konfiguration:

- `SECRET_KEY` muss über die `.env`-Datei gesetzt werden. Flask-Session-Dateien werden nicht mehr ins Git übernommen.
- Flask Debug ist standardmäßig deaktiviert. Für lokale Fehlersuche kann `FLASK_DEBUG=true` gesetzt werden.
- Uploads werden begrenzt, damit sehr große CSV/XLSX-Dateien nicht unbegrenzt Speicher belegen. Die Grenzen können mit `DATAQUALITY_MAX_UPLOAD_ROWS` und `DATAQUALITY_MAX_UPLOAD_COLUMNS` angepasst werden.
- CSV/XLSX/ZIP-Downloads neutralisieren Spreadsheet-Formeln. Werte, die z. B. mit `=`, `+`, `-` oder `@` beginnen, werden beim Export als Text behandelt.
- Die LLM-Funktionen für Feature Type Detection können auf Linux mit NVIDIA-GPU über `docker-compose.gpu.yml` laufen. Ohne sichtbare CUDA-GPU wird das Modell auf CPU geladen, was sehr langsam sein kann.


Projekt öffentlich erreichbar machen:

Wenn das Projekt von außerhalb erreichbar sein soll, müssen die Docker-Images in das Docker-Repository hochgeladen werden.

Dafür folgende Befehle verwenden:
"docker push apreisler/ba-base:latest"
"docker push apreisler/ba-frontend:latest"
"docker push apreisler/ba-backend:latest"

Dadurch werden die Images in das Docker-Repository gepusht und können öffentlich bereitgestellt werden.

______________________________________________________________

Welche Aufgaben sind noch offen?

- Alles rund um Feature Type Inference, Anomaly Detection usw. wurde von mir nicht weiter gedebuggt.
Mögliche Fehler sollten daher noch überprüft werden.

-Der SEED für das Modelltraining ist im Backend aktuell zu Testzwecken auf einen fixen Seed gehardcoded.
Um einen Random SEED zu erhalten, muss im Script main_routes.py USE_RANDOM_SEED = False auf True gesetzt werden.

-Einige Fehler beim Upload der CSVs konnten nicht behoben werden.
Zum Beispiel kann beim Upload von redwine.csv keine Target-Var ausgewählt werden.

-Ein Loading-Screen für den Upload der CSVs ist aktuell ebenfalls noch nicht im Projekt eingebunden.

______________________________________________________________
