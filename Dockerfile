FROM python:3.14.2-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY iris_model.joblib predict.py ./

CMD ["python", "predict.py", "5.1", "3.5", "1.4", "0.2"]