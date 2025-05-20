# Dockerfile
FROM python:3.12

# Define o diretório de trabalho dentro do container
WORKDIR /app

# Copia o arquivo requirements.txt
COPY requirements.txt .

# Instala as dependências
RUN pip install --no-cache-dir -r requirements.txt

# Copiar o modelo e o parquet
COPY src/modeltraining/model_rf.joblib /app/modeltraining/model_rf.joblib
COPY src/data/applicants_processed.parquet /app/data/applicants_processed.parquet

# Copia o código fonte
COPY src/ /app/src

# Exponha a porta da aplicação
EXPOSE 5000

# Comando para iniciar a aplicação
CMD ["python", "src/main.py"]
