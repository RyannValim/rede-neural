# Projeto de Redes Neurais

Este projeto foi desenvolvido para a matéria de Sistemas Inteligentes, ministrada durante o 4° período da faculdade de Ciência da Computação na Universidade Positivo.

### Objetivo Geral

Aplicar os conceitos de aprendizado supervisionado e redes neurais multicamadas (MLP) em um contexto real de análise e predição de dados, utilizando datasets públicos e ferramentas de ciência de dados modernas.

---

#### 1° Etapa - Escolha e Preparação do Dataset

Deverá ser escolhido um tema atual e relevante utilizando um dataset público (*Kaggle*, *UCI Machine Learning Repository*, *Data.gov*, *IBGE*, etc.), assim fornecendo material para a análise da rede neural que será construída.

##### Tarefa desta etapa:

- Descrever o dataset: fonte, período, quantidade de amostras e atributos;
- Justificar quais atributos serão utilizados e o motivo da escolha;
- Realizar a padronização ou normalização dos dados;
- Definir a variável-alvo (*target*) para o treinamento da rede.

##### Tema escolhido:

> Analisar os datasets e as ideias que a professora deu, entre:
>
> - **Predição de consumo de energia em casas inteligentes (Smart Homes);**
> - **Análise de emoções e sentimentos em redes sociais (NLP);**
> - **Detecção de fake news ou desinformação digital;**
> - **Previsão de demanda de transporte urbano (mobilidade inteligente);**
> - **Análise de saúde mental e padrões de sono via dispositivos vestíveis (IoT);**
> - **Previsão de rendimento acadêmico de estudantes (educação preditiva);**
> - **Análise de sustentabilidade urbana; etc.**
>
> Guardar a fonte do dataset e o arquivo, pois será necessário entregar junto com o código e o relatório.

---

#### 2° Etapa - Implementação da Rede Neural

A rede neural precisa ser implementada utilizando o conceito de **multicamadas** (MLP - Multiple Layer Perceptron), com pelo menos três camadas ocultas. Deve ser capaz de classaificar ou prever algo com base nos atributos definidos.

##### Requisitos técnicos:

- Utilizar **Python** com T*ensorFlow*, *Keras* ou *PyTorch* (ou outra biblioteca equivalente);
- Testar pelo menos duas funções de ativação (*ReLU*, *Sigmoid*, *Tanh*, *LeakyReLU*, etc.);
- Ajustar e justificar a taxa de aprendizado, número de épocas e o tamanho do *batch*;
- Exibir o vetor de erros a cada iteração;
- Apresentar gráficos de desempenho (acurácia, perda, etc.).

---

#### 3° Etapa - Produzir um Relatório Técnico

Elaborar um relatório técnico do projeto completo cumprindo os seguintes requisitos:

* **Introdução e Contexto:** uma breve descrição do tema e da importância da análise escolhida;
* **Descrição do Dataset:** fonte, variáveis, período, quantidade de amostras;
* **Metodologia:** arquitetura da rede, funções de ativação, parâmetros ajustados;
* **Resultados Obtidos:** métricas, gráficos e comparação entre funções de ativação;
* **Discussão:** análise crítica dos resultados e limitações do modelo;
* **Conclusão:** lições aprendidas e potenciais melhorias;
* **Normas:** deve estar nas normas da ABNT, em um documento Word ou PDF.

---

### Para Entregar:

* Dataset utilizado;
* Código do modelo MLP (arquivo .py ou .ipynb);
* Relatório técnico completo;
* Data: até 20/11/2025.
