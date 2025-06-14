import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px
from sklearn.metrics import mean_squared_error, r2_score
import streamlit as st

# Importando o dataset e fazendo alguns ajustes
df = sns.load_dataset('tips')
df.columns = ['valor_conta','gorjeta','sexo','fumante','dia','horário','tamanho']
df = df[['valor_conta','sexo','fumante','dia','horário','tamanho','gorjeta']]
df['sexo'].replace({'Male':'Masculino',"Female":'Feminino'},inplace = True)
df['fumante'].replace({'Yes':'Sim','No':'Não'},inplace = True)
df['dia'].replace({'Thur':'Quinta','Fri':'Sexta','Sat':'Sábado','Sun':'Domingo'},
inplace = True)
df['horário'].replace({'Dinner':'Jantar','Lunch':'Almoço'},inplace = True)

# Índice de páginas na barra lateral do aplicativo
with st.sidebar:
    st.title("Menu")
    page = st.radio("Escolha uma opção",["Introdução:books:","Análise de dados:desktop_computer:","Previsão de gorjetas:robot_face:",])

# Primeira página
if page == "Introdução:books:":
    colunaA, colunaB = st.columns([4,1])
    with colunaA:
        st.title("Sobre este aplicativo")

    with colunaB:
        st.image('image_tips.jpeg')

    st.header("O que este aplicativo faz?")
    with st.expander("Clique aqui para ler o texto"):
        st.write("""
        Este aplicativo faz uma pequena análise de dados e apresenta alguns algoritmos
        de regressão para utilizados para se fazer previsão de gorjetas de um determinado
        restaurante. Os dados foram coletados do dataset 'tips', presente em diferentes 
        bibliotecas do Python, como Seaborn, Scikit-Learn, etc. 

        O dataset tips possui diferentes informações referentes aos clientes e seu consumo,
        tais como horário de atendimento, valor da conta, se os clientes fumam ou não, tamanho
        da mesa , etc, e o valor da gorjeta paga. A ideia deste aplicativo é permitir que você 
        possa  predizer o valor da gorjeta ajustando os diferentes parâmetros de entrada. Para 
        isso  são utilizados diferentes algoritmos de regressão, uma técnica de aprendizado 
        supervisionado de Machine Learninng, para tentar prever o valor da gorjeta em cada caso.

        Aqui você pode escolher entre 4 tipos de algoritmos de regressão: regressão linear simples,
        árvore de decisão (Decision Tree), floresta aleatória (Random Forest) e máquina de vetor de 
        suporte (SVR - Support Vector Regressor). Nos casos pertinentes você também poderá ajustar 
        os hiperparâmetros do modelo escolhido.

        A seguir você verá uma breve descrição dos dataset, depois disso é só navegar pelo menu 
        lateral e escolher entre permanecer ou voltar a esta página, visualizar uma breve análise de 
        dados sobre o dataset ou realizar previsões com o algoritmo desejado para diferentes cenários.
        Divirta-se!
        """)

    st.header("Informações sobre o dataset")
    st.write("""Os nomes entre parênteses a seguir correspondem aos nomes originais presentes no 
    dataset tips, para aqueles que o conhecem.""")
    with st.expander("Clique aqui para ler o texto"):
        st.markdown("""
        1. valor_conta (total_bill): Valor total da conta, incluindo comidas e bebidas.\n
        2. sexo (sex): O gênero do cliente (masculino ou feminino).
        3. fumante (smoker): Se o cliente é ou não fumante.\n
        4. dia (day): Dia da semana em que a transação ocorreu.\n
        5. horário (time): Horário em que a transação ocorreu (almoço ou jantar).\n
        6. Tamanho (size): Tamanho da mesa ou festa (número de pessoas) a que o cliente pertence.\n
        7. gorjeta (tip): Refere-se ao valor da gorjeta pago pelo cliente, o que se quer prever.
        """)

    st.dataframe(df)

# Segunda página
df1 = df.copy()
df1['porcentagem_gorjeta'] = round(100*(df1['gorjeta']/df1['valor_conta']),1)

if page == "Análise de dados:desktop_computer:":
    colunaA, colunaB = st.columns([4,1])
    with colunaA:
        st.title("Análise de dados")

    with colunaB:
        st.image('image_tips.jpeg')

    st.header("A seguir você verá algumas curiosidades sobre o dataset.")

    
    
    col001, col002 = st.columns(2)
    with col001:
        with st.container(border = True):
            st.subheader("Gênero x gorjeta")
            df01 = df1.groupby('sexo')['gorjeta'].mean().reset_index()
            st.table(df01)
            fig, ax = plt.subplots()
            ax.pie(df01['gorjeta'], labels=df01['sexo'], autopct='%1.1f%%', startangle=90)
            ax.axis('equal')  # Equal aspect ratio para manter forma de círculo
            st.pyplot(fig)

    with col002:
        with st.container(border = True):
            st.subheader("Gênero x gorjeta relativa")
            df02 = df1.groupby('sexo')['porcentagem_gorjeta'].mean().reset_index()
            st.table(df02)
            fig, ax = plt.subplots()
            ax.pie(df02['porcentagem_gorjeta'], labels=df02['sexo'], autopct='%1.1f%%', startangle=90)
            ax.axis('equal')  # Equal aspect ratio para manter forma de círculo
            st.pyplot(fig)

    with st.container(border = True):
        st.write("""Vemos que os homens pagam mais gorjetas em média do que as mulheres, 
        porém as mulheres pagam uma maior porcentagem de gorjeta em relação a conta. 
        Isso significa que as gorjetas das mulheres são mais generosas para um mesmo 
        valor de conta mas os homens se sobressaem nas gorjetas por consumirem mais.""")

    
    col003, col004 = st.columns(2)
    with col003:
        with st.container(border = True):
            st.subheader("Gênero e fumo x gorjetas")
            df03 = df1.groupby(['sexo','fumante'])['gorjeta'].mean().reset_index()
            st.table(df03)
            st.subheader("Gênero e fumo x gorjetas")
            fig = px.bar(df03, x='sexo', y='gorjeta', color='fumante', barmode='group',
                        labels={'gorjeta': 'Gorjeta Média', 'sexo': 'Sexo', 'fumante': 'Fumante'})
            st.plotly_chart(fig)

    with col004:
        with st.container(border = True):
            st.subheader("Gênero e fumo x gorjeta relativa")
            df04 = df1.groupby(['sexo','fumante'])['porcentagem_gorjeta'].mean().reset_index()
            st.table(df04)
            fig = px.bar(df04, x='sexo', y='porcentagem_gorjeta', color='fumante', barmode='group',
                        labels={'gorjeta': 'Gorjeta Média', 'sexo': 'Sexo', 'fumante': 'Fumante'})
            st.plotly_chart(fig)

    col005, col006 = st.columns(2)
    with col005:
        with st.container(border = True):
            st.subheader("Gênero x fumo")
            df05 = df1.groupby('sexo')['fumante'].count().reset_index()
            st.table(df05)
            st.bar_chart(df05,x='sexo',y='fumante')

    with col006:
        with st.container(border = True):
            st.write("""Vemos que as mulheres fumantes costumas pagar gorjetas maiores tanto
            proporcionalmente quanto em valor bruto, porém os homens fumantes costumam pagar 
            menos que os não fumantes. Como visto anteriormente homens consomem mais e pagam 
            mais valor bruto de gorjeta porém as mulheres costumam pagar proporcionalmente mais.
            A maior parte de frequentadores fumantes é do sexo masculino.""")

    st.markdown("---")

    col007, col008 = st.columns(2)
    with col007:
        with st.container(border = True):
            st.subheader("Horário x gorjeta")
            df06 = df1.groupby('horário')['gorjeta'].mean().reset_index()
            st.table(df06)
            st.bar_chart(df06,x='horário',y='gorjeta')

    with col008:
        with st.container(border = True):
            st.subheader("Horário x gorjeta relativa")
            df07 = df1.groupby('horário')['porcentagem_gorjeta'].mean().reset_index()
            st.table(df07)
            fig, ax = plt.subplots()
            ax.pie(df07['porcentagem_gorjeta'], labels=df07['horário'], autopct='%1.1f%%', startangle=90)
            ax.axis('equal')  # Equal aspect ratio para manter forma de círculo
            st.pyplot(fig)

    col009, col010 = st.columns(2)
    with col009:
        with st.container(border = True):
            st.subheader("Horário x gorjeta acumulada")
            df08 = df1.groupby('horário')['gorjeta'].sum().reset_index()
            st.table(df08)
            fig, ax = plt.subplots()
            ax.pie(df08['gorjeta'], labels=df08['horário'], autopct='%1.1f%%', startangle=90)
            ax.axis('equal')  # Equal aspect ratio para manter forma de círculo
            st.pyplot(fig)

    with col010:
        with st.container(border = True):
            st.subheader("Horário x fluxo de pessoas")
            df09 = df1.groupby('horário')['horário'].count().reset_index(name='qtde_pessoas')
            st.table(df09)
            st.bar_chart(df09,x='horário',y='qtde_pessoas')

    with st.container(border = True):
        st.write("""Vemos que a média das gorjetas é maior no almoço, porém o total de 
        gorjeta acumulada é muito maior no jantar. Isto se deve ao fato de que o
        restaurante é muito mais frequentado na hora do jantar do que no almoço.""")

    st.markdown("---")

    col011, col012 = st.columns(2)
    with col011:
        with st.container(border = True):
            st.subheader("Tamanho x gorjeta")
            df10 = df1.groupby('tamanho')['gorjeta'].mean().reset_index()
            st.table(df10)
            st.bar_chart(df10,x='tamanho',y='gorjeta')

    with col012:
        with st.container(border = True):
            st.subheader("Tamanho x gorjeta relativa")
            df11 = df1.groupby('tamanho')['porcentagem_gorjeta'].mean().reset_index()
            st.table(df11)
            st.bar_chart(df11,x='tamanho',y='porcentagem_gorjeta')

    col013, col014 = st.columns(2)
    with col013:
        with st.container(border = True):
            st.subheader("Distribuição de tamanhos")
            df12 = df.groupby("tamanho")["tamanho"].count().reset_index(name='frequência')
            st.table(df12)
            st.bar_chart(df12,x='tamanho',y='frequência')

        with col014:
            with st.container(border = True):
                st.write("""Vemos que as mesas com uma única pessoa costumam pagar maiores 
                gorjetas relativamente ao que consomem, porém pagam menor gorjeta bruta. 
                Isto significa que pessoas sozinhas costumam consumir menos e pagam gorjetas
                mais generosas. A maior parte das mesas são ocupadas por grupos de duas pessoas,
                seguidas por mesas de 3 e 4 pessoas. As de 1 e 6 pessoas são as mais raras.""")


# Terceira página

if page == "Previsão de gorjetas:robot_face:":
    colunaA, colunaB = st.columns([4,1])
    with colunaA:
        st.title("Previsão de gorjetas")

    with colunaB:
        st.image('image_tips.jpeg')

    st.markdown("---")

    st.header("Dados de gorjeta")
    n_linhas = st.slider('Controle o número de linhas do dataset que você quer ver',1,len(df),10)
    st.dataframe(df.head(n_linhas))

    st.markdown("---")

    st.header("Separação dos dados de treino e teste do modelo a ser utilizado")
    percent_treino = st.number_input("Porcentagem de dados de treino (entre 0 e 100):",
    min_value=0, max_value=100, value=70)
    st.write(f"Segundo sua escolha, {percent_treino}% dos dados serão usados \
    para treinar o modelo e {100 - (percent_treino)}% para testá-lo.")

    rs = st.number_input("Escolha um estado inicial para selecionar os dados de treino e teste",
    min_value=0, max_value=100, value=42)
    st.write(f"Você escolheu o estado inicial {rs} para selecionar os dados de treino e teste.")

    X = df[['valor_conta','sexo','fumante','dia','horário','tamanho']]
    Y = df[['gorjeta']]
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=1-(percent_treino/100),
    random_state=rs)

    n1 = st.slider("Regule a quantidade de dados de treino a ser vista",
    min_value=0, max_value=len(X_train), value=min(5,len(X_train)))
    col1, col2 = st.columns([3,1])
    with col1:
        st.table(X_train.head(n1))

    with col2:
        st.table(Y_train.head(n1))

    st.markdown("---")

    n2 = st.slider("Regule a quantidade de dados de teste a ser vista",
    min_value=0, max_value=len(X_test), value=min(5,len(X_test)))
    col1, col2 = st.columns([3,1])
    with col1:
        st.table(X_test.head(n2))

    with col2:
        st.table(Y_test.head(n2))


    st.header('Definição do modelo a ser utilizado')

    COL01, COL02 = st.columns([1,2])
    with COL01:
        modelo = st.radio('Escolha o mais modelo a ser utilizado',
        ['Regressão linear simples','Arvore de decisão',
        'Floresta de decisão','Support Vector Regressor (SVR)'])

    with COL02:
        if modelo == 'Regressão linear simples':
            st.image('image_regressao.jpeg')
            model = LinearRegression()

        if modelo == 'Arvore de decisão':
            st.image('image_decision_tree.png')
            model = DecisionTreeRegressor()

        if modelo == 'Floresta de decisão':
            st.image('image_random_forest.png')
            model = RandomForestRegressor()

        if modelo == 'Support Vector Regressor (SVR)':
            st.image('image_svr.jpeg')
            model = SVR()


    st.write(f"Por enquanto o modelo **{modelo.upper()}** é o selecionado para fazer previsões de gorjetas.")

    st.subheader("Personalização de hiperparâmetros")
    botao = st.toggle("Clique aqui para personalizar hiperparâmetros")
    if botao:
        with st.container(border = True):
            if modelo == 'Regressão linear simples':
                st.write("Este modelo não possui hiperparâmetros a serem utilizados.")
            
            if modelo == 'Arvore de decisão':
                kolun1, kolun2 = st.columns(2)
                with kolun1:
                    criterio = st.selectbox("Critério:",['squared_error', 'friedman_mse','absolute_error','poisson'])
                    spt = st.selectbox("Splitter:",['best','random'])
                    m_dep = st.selectbox("Profundidade máxima da árvore:",[None]+list(range(1,10000)))
                    nmin_s_split = st.number_input("Mínimo de amostras para dividir um nó:",step = 1,min_value=2)
                with kolun2:
                    nmin_s_leaf = st.number_input("Mínimo de amostras em uma folha:",step = 1,min_value=1)
                    mfeatures = st.selectbox("Número de features por split:",['auto','sqrt', 'log2','int','float',None])
                    mln = st.selectbox("Número máximo de folhas:",[None]+list(range(0,10000)))
                    semente = st.number_input("Random state:",step = 1)

                model = DecisionTreeRegressor(
                    criterion=criterio,  # Função de perda: 'squared_error' (antigo 'mse'), 'friedman_mse', 'absolute_error'
                    splitter=spt,           # 'best' ou 'random'
                    max_depth=m_dep,              # Profundidade máxima da árvore
                    min_samples_split=nmin_s_split,       # Mínimo de amostras para dividir um nó
                    min_samples_leaf=nmin_s_leaf,        # Mínimo de amostras em uma folha
                    max_features=mfeatures,         # Número de features a considerar em cada split (None, 'sqrt', 'log2' ou int/float)
                    max_leaf_nodes=mln,       # Número máximo de folhas
                    random_state=semente            # Para reprodutibilidade
    )

            if modelo == 'Floresta de decisão':
                kol1, kol2 = st.columns(2)

                with kol1:
                    num_arvores = st.number_input("Número de árvores:",step = 1,min_value = 1)
                    criterio = st.selectbox('Critério:',['squared_error', 'absolute_error','poisson'])
                    prof_max = st.selectbox("Profundidade máxima das árvores:",[None]+list(range(1,10000)))
                    min_s_split = st.number_input("Número mínimo de amostras para dividir um nó interno:",step = 1,min_value=2)
                    min_s_leaf = st.number_input("Mínimo de amostras por folha",step = 1,min_value=1)
                    
                with kol2:
                    m_feature = st.selectbox("Número de features por split:",[None, 'sqrt', 'log2']+list(range(0,10000)))
                    max_folhas = st.selectbox('Número máximo de folhas:',[None]+list(range(0,10000)))
                    boot = st.selectbox("Bootstrap (amostragem om reposição?):",[True,False])
                    rs = st.number_input('Random state:',step =1,min_value=1)
                    obb = st.selectbox("Oobscore:",[True,False])

                mmsamples = st.slider("Max_samples (entre 0 e 1, None se bootstrap = True):",min_value = 0.0, max_value = 1.0,value = 0.5,step = 0.01)
                

                if boot == True:
                    msamples = None
                    #  st.write("max_samples = None")
                else:
                    msamples = mmsamples


                model = RandomForestRegressor(
                    n_estimators=num_arvores,            # Número de árvores
                    criterion=criterio,   # Função de perda: 'squared_error', 'absolute_error', etc.
                    max_depth=prof_max,                # Profundidade máxima da árvore
                    min_samples_split=min_s_split,         # Mínimo de amostras para dividir um nó
                    min_samples_leaf=min_s_leaf,          # Mínimo de amostras em uma folha
                    min_weight_fraction_leaf=0.0,# Fração mínima do peso total das amostras em uma folha
                    max_features=m_feature,         # Máximo de features por split ('auto', 'sqrt', 'log2', float ou int)
                    max_leaf_nodes=max_folhas,         # Número máximo de folhas (None = ilimitado)
                    bootstrap=boot,              # Usar bootstrap?
                    oob_score=False,             # Usar amostras fora da bolsa para validação?
                    n_jobs=-1,                   # Usar todos os núcleos
                    random_state=rs,             # Semente aleatória para reprodutibilidade
                    verbose=0,                   # Nível de verbosidade
                    warm_start=False,            # Continuar o treinamento de um modelo anterior?
                    ccp_alpha=0.0,               # Poda de custo-complexidade
                    max_samples=msamples         # Fração de amostras para cada árvore (se bootstrap=True)
                )

            if modelo == 'Support Vector Regressor (SVR)':

                coluna1,coluna2 = st.columns(2)

                with coluna1:
                    ker = st.selectbox("Kernel:",['linear', 'poly', 'rbf', 'sigmoid',])
                    reg = st.slider("C:",min_value = 0.1,max_value = 1000.,step = 0.01)
                    eps = st.slider("Epsilon:",min_value = 0.1,max_value = 1.,step = 0.01)
                    deg1 = st.number_input("Grau do polinômio (se o kernel for polinomial)",min_value=1,step=1)
                    gam = st.selectbox("Coeficiente do kernel:",['scale','auto'])
                
                with coluna2:
                    bbias1 = st.number_input("Termo independente para kernel poly ou sigmmoid")
                    shr = st.selectbox("Shrinking:",[True,False])
                    caches = st.number_input("Cache_size:",min_value=1,step=1)
                    maxiter = st.slider("Número máximo de iterações (max_iter):",min_value=-1,max_value=1000,step=1)
                    toler = st.slider("Tolerância (tol):",min_value=0.001,max_value=0.1,step=0.001)

                deg = deg1 if ker == 'poly' else 3
                bbias = bbias1 if ker in ['poly','sigmoid'] else 0.0

                # Instanciando o modelo com hiperparâmetros preenchidos
                model = SVR(
                    kernel=ker,         # Função kernel: 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'
                    C=reg,                # Parâmetro de regularização (trade-off entre erro e margem)
                    epsilon=eps,          # Margem de tolerância onde nenhuma penalização é aplicada
                    degree=deg,             # Grau do polinômio se kernel='poly'
                    gamma=gam,        # Coeficiente do kernel: 'scale', 'auto' ou float
                    coef0=bbias,            # Término independente para kernels 'poly' e 'sigmoid'
                    shrinking=shr,       # Se deve usar a heurística do shrinking
                    tol=toler,             # Tolerância para critério de parada
                    cache_size=caches,       # Tamanho do cache em MB
                    max_iter=maxiter           # Número máximo de iterações (-1 = ilimitado)
                )


    else:
        if modelo == 'Regressão linear simples':
            model = LinearRegression()

        if modelo == 'Arvore de decisão':
            model = DecisionTreeRegressor()

        if modelo == 'Floresta de decisão':
            model = RandomForestRegressor()

        if modelo == 'Support Vector Regressor (SVR)':
            model = SVR()

    st.markdown("---")
    st.header("Previsão")
    st.write("Selecione a seguir os dados de entrada para ver visualizar a gorjeta prevista pelo modelo escolhido.")

    colA, colB = st.columns(2)
    with colA:
        dia = st.selectbox("Dia:",['Quinta','Sexta','Sábado','Domingo'])
    with colB:
        tamanho = st.selectbox("Tamanho:",range(1,11))

    A0, A1, A2, A3 = st.columns(4)
    with A0:
        sexo = st.radio("Sexo:",["Masculino","Feminino"])
    with A1:
        fumante = st.radio("Fumante:",["Sim","Não"])
    with A2:
        horario = st.radio("Horário:",["Almoço","Jantar"])
    with A3:
        valor_conta = st.number_input("Valor da conta:")

    entrada = pd.DataFrame([{'valor_conta':valor_conta, 'sexo':sexo, 'fumante':fumante, 'dia':dia, 'horário':horario, 'tamanho':tamanho}])

    le_sexo = LabelEncoder()
    le_sexo.fit(df['sexo'])

    le_fumante = LabelEncoder()
    le_fumante.fit(df['fumante'])

    le_horario = LabelEncoder()
    le_horario.fit(df['horário'])

    le_dia = LabelEncoder()
    le_dia.fit(df['dia'])

    entrada['sexo'] = le_sexo.transform(entrada['sexo'])
    entrada['fumante'] = le_fumante.transform(entrada['fumante'])
    entrada['horário'] = le_horario.transform(entrada['horário'])
    entrada['dia'] = le_dia.transform(entrada['dia'])

    X_train['sexo'] = le_sexo.transform(X_train['sexo'])
    X_train['fumante'] = le_fumante.transform(X_train['fumante'])
    X_train['horário'] = le_horario.transform(X_train['horário'])
    X_train['dia'] = le_dia.transform(X_train['dia'])

    X_test['sexo'] = le_sexo.transform(X_test['sexo'])
    X_test['fumante'] = le_fumante.transform(X_test['fumante'])
    X_test['horário'] = le_horario.transform(X_test['horário'])
    X_test['dia'] = le_dia.transform(X_test['dia'])

    X_train_enc = X_train.copy()
    
    model.fit(X_train,Y_train)

    previsao = pd.DataFrame(model.predict(entrada),columns=['Gorjeta prevista:'])
    previsao['Gorjeta prevista:'] = previsao['Gorjeta prevista:'].apply(lambda x: round(x,2))

    st.write("O valor de gorjeta predito pelo modelo escolhido considerando estes dados de entrada é .")
    st.table(previsao)

    st.markdown("---")

    st.header("Métricas comparativas para os dados de teste")


    Y_test_pred = model.predict(X_test)

    mse = mean_squared_error(Y_test,Y_test_pred)
    r2 = r2_score(Y_test,Y_test_pred)

    C001, C002  = st.columns(2)
    with C001:
        st.metric(label="Mean Squared Error", value=f"{mse:.2f}")
    with C002:
        st.metric(label="R² Score", value=f"{r2:.2%}")

    if r2 >= 0.7:
        st.success("O modelo possui coeficiente de determinação maior ou igual a 0.7, sugerindo que o mesmo apresenta boa capacidade preditiva.")
    else:
        st.warning("O modelo está com baixo desempenho. Tente ajustar os hiperparâmetros.")
