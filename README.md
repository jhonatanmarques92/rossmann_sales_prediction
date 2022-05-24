# Projeto de predição de vendas das lojas Rossmann
<p align="center"><img src="https://github.com/jhonatanmarques92/rossmann_sales_prediction/blob/main/img/rossmann.png" width="650" height="280"></p>

### Observação: O contexto do problema de negócio não é real

## Questão de Negócio
O CFO da Rossmann, uma das maiores redes de drogaria da Europa, está planejando a reforma de algumas lojas, conforme as vendas de cada uma.  
Em uma reunião, ele pediu aos gerentes que enviassem previsões de venda diárias, nas próximas 6 semanas, de cada unidades, afim de distribuir o orçamento para as reformas.  
Após a reunião, os gerentes entraram em contato, requisitando uma previsão de venda das lojas.

## Entendimento de negócio
- **Motivação:** Reunião com os gerentes das lojas, afim de decidir como será dividido o orçamento para reformas nas lojas.  
- **Causa raiz do problema:** Auxiliar na decisão do valor para investir na reforma de cada loja.  
- **Dono do problema:** CFO da Rossmann.  
- **O formato da solução**
  - **Granularidade:** Previsão diária de cada loja, em um período de 6 semanas
  - **Tipo do problema:** Regressão com série temporal
  - **Potenciais métodos:** Regressão Linear, Regressão Lasso, Random Forest, XGBoost
  - **Forma de entrega:** Bot no telegram

## Informação dos dados
Os dados foram coletados de um competição do Kaggle: https://www.kaggle.com/competitions/rossmann-store-sales/data

| Colunas | Descrição |
| ------- | --------- |
|Id | um Id que representa uma duplicata (Store, Date) dentro do conjunto de teste|
|Store | um ID único para cada loja|
|Sales | o volume de vendas em qualquer dia|
|Customers | o número de clientes em um determinado dia|
|Open | um indicador para saber se a loja estava aberta: 0 = fechada, 1 = aberta|
|StateHoliday | indica um feriado estadual. Normalmente todas as lojas, com poucas exceções, fecham nos feriados estaduais. Observe que todas as escolas fecham nos feriados e finais de semana. a = feriado, b = feriado da Páscoa, c = Natal, 0 = Nenhum|
|SchoolHoliday | indica se (loja, data) foi afetado pelo fechamento de escolas públicas|
|StoreType  | diferencia entre 4 modelos de loja diferentes: a, b, c, d|
|Assortment | descreve um nível de sortimento: a = básico, b = extra, c = estendido|
|CompetitionDistance | distância em metros até a loja concorrente mais próxima|
|CompetitionOpenSince [Mês / Ano] | fornece o ano e mês aproximados em que o concorrente mais próximo foi aberto|
|Promo | indica se uma loja está fazendo uma promoção naquele dia|
|Promo2 | Promo2 é uma promoção contínua e consecutiva para algumas lojas: 0 = a loja não está participando, 1 = a loja está participando|
|Promo2Since [Ano / Semana] | descreve o ano e a semana em que a loja começou a participar da Promo2|
|PromoInterval | descreve os intervalos consecutivos em que a Promo2 é iniciada, nomeando os meses em que a promoção é reiniciada. Por exemplo, "fev, maio, ago, nov" significa que cada rodada começa em fevereiro, maio, agosto, novembro de qualquer ano para aquela loja|

## Estratégia de solução
Para solucionar o problema, foi utilizado o CRISP-DS, uma metodologia cíclica para o andamento de cada etapa do desenvolvimento do projeto.  
<p align="center"><img src="https://github.com/jhonatanmarques92/rossmann_sales_prediction/blob/main/img/crisp-ds.png" width="650" height="400"></p>  

- **Questão de negócio:** Recebimento do problema de negócio.  
- **Entendimento do negócio:** Fazer o levantamento quanto a motivação, problema raíz, dono do problema e o formato da solução.  
- **Coleta dos dados:** Coleta de dados através da competição o Kaggle.  
- **Limpeza dos dados:** Verificar inconsistência nos dados e derivá-los para o levantamento das hipóteses.  
- **Exploração dos dados:** Análise dos dados, procurando insights e entender melhor o impacto de cada variável no modelo.  
- **Modelagem dos dados:** Selecionar e preparar as variáveis para o treinamento dos modelos.  
- **Algorítmos de Machine Learning:** Treinamento dos modelos escolhidos.  
- **Avaliação do algorítmo:** Avaliação do impacto no negócio do melhor modelo selecionado.  
- **Modelo em produção:** Entrega do modelo em produção, para o acesso e uso de outras pessoas.  

## Insights dos dados
Foram levantadas algumas hipóteses, das quais, duas foram mais impactantes.  
- Lojas com competidores mais próximos deveriam vender menos.  
  - **Falsa**, pois lojas vendem mais com competidores próximos, na média.  
<p align="center"><img src="https://github.com/jhonatanmarques92/rossmann_sales_prediction/blob/main/img/h2.png" width="850" height="400"></p>  

- Lojas deveriam vender menos aos finais de semana.  
  - **Verdadeira**, lojas vendem levemente menos nos dias de semana.  
  <p align="center"><img src="https://github.com/jhonatanmarques92/rossmann_sales_prediction/blob/main/img/h10.png" width="850" height="500"></p> 
  
## Modelos de Machine Learning aplicados
Os modelos aplicados foram a baseline (média), Regressão Linear, Regressão Lasso, XGBoost.  
Abaixo segue o resultado do MAE, MAPE e RMSE de cada modelo.  

Modelo Baseline
|model_name | MAE | MAPE | RMSE|
|----------|------|-------|------|
|Average Model	| 1357.61|	20.95|	1825.54|

Modelos sem Cross Validation
|model_name | MAE | MAPE | RMSE|
|----------|------|-------|------|
|XGBoost Regressor	| 834.11|	12.50|	1202.28|
|Linear Regression Model|	1876.86|	30.12|	2657.20|
|Lasso|	1872.71|	29.29|	2683.25|  

Modelos com Cross Validation
|model_name|	MAE CV|	MAPE CV|	RMSE CV|
|----------|------|-------|------|
|XGBoost|	mean: 1030.79 std: +/- 199.41|	mean: 14.65 std: +/- 2.26|	mean: 1457.39 std: +/- 266.99|
|Linear Regression|	mean: 2082.92  std: +/- 272.87|	mean: 31.04  std: +/- 1.96|	mean: 2929.08  std: +/- 443.46|
|Lasso|	mean: 2078.82  std: +/- 310.76|	mean: 29.77  std: +/- 1.27|	mean: 2966.4  std: +/- 479.19|

Conforme a tabela acima, foi utilizado o XGBoost como modelo final.  

## Perfomance do modelo escolhido
Após escolhido os hiperparâmetros, o modelo XGBoost atingiu a seguinte performance
model_name|	MAE|	MAPE|	RMSE
|----------|------|-------|------|
XGBost Regressor|	652.35|	9.74|	935.34

## Resultado do negócio
No cenário do negócio, com os dados de teste, o modelo obteve o seguinte resultado
|cenario|	valores|
|----------|------|
|predictions|	R$ 288,081,024.00
|melhor_cenario|	R$ 288,812,366.33
|pior_cenario|	R$ 287,349,646.59

Gráfico comparando os valores de venda real dos dados de teste e as predições
<p align="center"><img src="https://github.com/jhonatanmarques92/rossmann_sales_prediction/blob/main/img/vendas-predicoes.png" width="650" height="600"></p>  

## 
