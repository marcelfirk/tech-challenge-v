Tech Challenge V - API Recrutamento Decision

SOBRE O PROJETO

Este projeto é uma API desenvolvida para melhorar o processo de recrutamento da Decision, utilizando machine learning para avaliar a compatibilidade de candidatos com vagas de trabalho específicas. O objetivo é analisar candidatos com base em suas qualificações e prever o sucesso que ele teria no processo seletivo de uma vaga fornecida.

Documentação da API: https://documenter.getpostman.com/view/13216885/2sB2qZEMzs

ESTRUTURA

Na pasta data está localizado o arquivo applicants_processed.parquet, que contém a base de candidatos já processada (os dados foram fornecidos em formato .json pela Decision). Os dados usados para treinar o modelo estão também nessa pasta no arquivo processed_data.pkl. A pasta modeltraining armazena o modelo de Machine Learning (Random Forest) treinado, salvo no arquivo model_rf.joblib. O código-fonte da aplicação está na pasta src, que contém o arquivo principal main.py e a subpasta routes, onde está implementada a rota /predict no arquivo prediction.py. A rota predict recebe no corpo da requisição uma vaga de interesse em formato .json, concatena essa vaga com os candidatos processados que estão no arquivo parquet e prediz quais candidatos tem maior probabilidade de obterem sucesso no processo seletivo para a vaga de interesse.

BIBLIOTECAS E SERVIÇOS UTILIZADOS

Python 3.12

Flask

Pandas

Scikit-Learn

Joblib

Docker

Railway (Deploy)

EXECUÇÃO

O deploy foi realizado no Railway, permitindo testes e acesso remoto à API. Para acessar a API, as requisições devem ser feitas à URL https://tech-challenge-v-production.up.railway.app/predict sendo passado no corpo da requisição a vaga de interesse no formato .json com as informações necessárias. Um exemplo de execução está presente na documentação da API: https://documenter.getpostman.com/view/13216885/2sB2qZEMzs
