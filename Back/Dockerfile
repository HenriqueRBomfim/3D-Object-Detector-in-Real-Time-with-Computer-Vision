
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN apt-get update && apt-get install -y --no-install-recommends \
      libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install uvicorn
RUN pip install "fastapi[standard]"
RUN pip install scikit-learn
RUN pip install numpy
COPY Back/ ./
COPY Scripts/ Scripts/
COPY Images/ Images
COPY normalized_landmarks.txt ./
COPY pose_landmarker.task ./
EXPOSE 8000
COPY model2.joblib ./

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]