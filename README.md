# autor: Emanuel Boeira Martins

# Descrição:

Este projeto foi realizado como desáfio da trilha de IA do programa SCTEC.
Ele busca desenvolver um modelo de IA para predição do concelamento de diária em um hotel por meio da variável 'is_canceled'. Os dados publicos foram disponibilizados pale platafoma Kaggle em formato CSV.
    
O projeto foi executado em python.
    
O arquivo data-analysis.py foi utilizado para uma analise inicial dos dados. O modelo de predição foi desenvolvido no arquivo predict-model.py. O modelo utilizado foi o LogisticRegression e as métricas utilizadas foram Accuracy e AUC-ROC.

# Dependências:

- pandas
- seaborn
- matplotlib
- numpy
- scikit-learn

# Forma de execução:
    
Os arquivos devem der executados pelo terminal, no diretório principal do projeto:
    
    `python scripts/data-analysis.py` (para execução da análise de dados)

    `python scripts/predict-model.py` (para execução do modelo de predição)

# Análise dos dados:

- Análise de nulos:
    - As colunas 'agent' e 'company' tem muitos valores nulos.
    - As colunas 'country' e 'children' tem alguns valores nulos.

- Análise de mapa de calor:
    - A variável 'is_canceled' tem um relação comum com as outras variáveis.
    - A variável 'is_canceled' tem uma relação acima do normal com 'lead_time' e abixo do comum com 'total_speceial_requests'.

- Modelo de predição:
    - A coluna 'company' foi removida por conter muitos valores nulos.
    - As colunas 'reservation_status_date' e 'reservation_status' foram removido pois estavam atrapalhando o modelo por causar vazamento de dados (data leakage).
    - As colunas 'children' e 'country' foram preenchidas com a moda.
    - A coluna 'agent' foi preenchida com 1 como valor numérico representativo de no_agent.

# Conclusões:

- O modelo utilizado para predição foi de Logistic Regression.
- A precisão (Accuracy) foi de aproximadamente 0,8196.
- O método de análise de chute usado foi o ROC-AUC e resultou em aproximadamente 0,8962.
- Os valores de Accuracy e ROC-AUC indicam uma boa predição e pouca chance de predição baseada em chutes.
