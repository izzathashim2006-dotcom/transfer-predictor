FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py rumours.py train_real_model.py predict.py predict_real.py build_real_dataset.py generate_dataset.py train_model.py ./
COPY templates/ templates/
COPY assets/ assets/
COPY model/ model/
COPY data/real_players.csv data/real_players.csv
COPY data/players.csv data/players.csv

ENV PORT=7860
EXPOSE 7860

CMD ["python", "app.py"]
