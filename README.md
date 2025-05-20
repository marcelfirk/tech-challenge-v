Tech Challenge V - Recruitment API

Sobre o Projeto

Este projeto é uma API desenvolvida para melhorar o processo de recrutamento, utilizando machine learning para avaliar a compatibilidade de candidatos com vagas de trabalho específicas. O objetivo é analisar candidatos com base em suas qualificações e prever o sucesso em uma vaga determinada.

Estrutura do Projeto

/tech-challenge-v/
├── data/
│   └── applicants_processed.parquet      # Base de candidatos já processada
├── modeltraining/
│   └── model_rf.joblib                   # Modelo treinado em formato .joblib
├── src/
│   ├── main.py                          # Arquivo principal da aplicação
│   └── routes/
│       └── prediction.py                # Rota /predict para previsões
├── Dockerfile                           # Dockerfile para build e execução da aplicação
├── requirements.txt                    # Dependências da aplicação
└── README.md                           # Documentação do projeto

🛠️ Tecnologias Utilizadas

Python 3.11

Flask

Pandas

Scikit-Learn

Joblib

Docker

Railway (Deploy)

🏗️ Como Executar o Projeto

1. Clonar o Repositório:

git clone https://github.com/marcelfirk/tech-challenge-v.git
cd tech-challenge-v

2. Criar e Ativar o Ambiente Virtual:

python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.\.venv\Scripts\activate  # Windows

3. Instalar as Dependências:

pip install -r requirements.txt

4. Executar a Aplicação:

python src/main.py

A aplicação estará disponível em http://localhost:8080.

🐳 Deploy com Docker

1. Build da Imagem:

docker build -t tech-challenge-v .

2. Executar o Container:

docker run -p 8080:8080 tech-challenge-v

📦 Deploy no Railway

Acesse Railway e crie um novo projeto.

Conecte o repositório do GitHub.

Defina o comando de execução como:

python src/main.py

Acompanhe os logs para garantir que a aplicação está rodando corretamente.

📌 Rotas da API

GET / - Verifica se a API está ativa.

POST /predict - Recebe os dados do candidato e retorna a previsão de compatibilidade com a vaga.

Exemplo de requisição:

{
    "cv_pt": "Engenheiro de software com 5 anos de experiência...",
    "app_prof_conhecimentos_tecnicos": "Python, Flask, Docker",
    "app_form_nivel_academico": "Superior Completo",
    "app_form_nivel_ingles": "Avançado"
}

✅ Considerações Finais

Esta API foi desenvolvida como parte do Tech Challenge V, utilizando machine learning para análise de perfis de candidatos.

O deploy foi realizado no Railway, permitindo testes e acesso remoto à API.
