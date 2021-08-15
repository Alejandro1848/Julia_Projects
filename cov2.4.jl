### A Pluto.jl notebook ###
# v0.15.1

using Markdown
using InteractiveUtils

# ╔═╡ 9047582d-b0b2-4ff5-b84b-d9194b9053b0
using Pkg

# ╔═╡ eec9b297-df0b-4d1d-b54b-9f5c6f506df6
using PlutoUI

# ╔═╡ 2452c556-614c-471b-9611-8f4c17809dbf
using DataFrames

# ╔═╡ b790ff9a-d590-4cb0-975c-91473170c1cc
using CSV

# ╔═╡ 1d14e33b-aee6-43e5-b0df-401555d6cf67
using Plots

# ╔═╡ 52092e65-6275-4dab-af55-c546d80d4650
using GLM

# ╔═╡ f5886586-b2d0-4ce7-86cf-ec4c439b4ed1
using StatsBase

# ╔═╡ fc0f0e2b-7a7c-47e4-bd62-de69ad3e7e61
using Lathe

# ╔═╡ 37e1b9da-6ec1-40fd-aa5f-96ab9c36c502
using MLBase

# ╔═╡ 35bd5abe-5546-4556-8ecd-66edf6a89ade
using ClassImbalance

# ╔═╡ 9d2ff573-ece2-41d8-b3d8-fe3164c1c607
using ROCAnalysis

# ╔═╡ 92465c4f-9a97-4baf-902e-1620a6210034
#Prueba de entrenamiento dividida
using Lathe.preprocess:TrainTestSplit

# ╔═╡ 1a5bf86b-7ded-434d-a742-275e15b514be
using PyCall

# ╔═╡ 4c726d0c-f95e-11eb-3cd2-71abdb89daac
md"""# Proyecto de Ciencia de datos por Alejandro Juárez Toribio:

## COVID-19 en México.
- Data set extraído de: https://www.kaggle.com/omarlarasa/cov19-open-data-mexico

Contexto

México está en el Top 5 de países con más muertes por COV19. muchos pacientes mueren todos los días en los hospitales. Pero esta es la cuestión:

¿Cómo saber qué personas tienen mayor riesgo de morir por COV19? y luego, ¿qué podemos hacer con esa información?

Este conjunto de datos fue recopilado por las autoridades sanitarias mexicanas y contiene todos los registros sobre pacientes con COV19 en el momento en que se descargó este conjunto de datos (abril de 2021).

Contenido

Lo que hay dentro son datos sobre los pacientes, y hay muchas cosas que podemos saber sobre los pacientes, como dónde fueron hospitalizados, en qué estado de México, su edad y la fecha de la muerte.

También hay muchas columnas interesantes que nos dicen si el paciente tiene un problema de salud además de COVID19 y hay una lista larga sobre estos problemas como obesidad, hipertensión, asma, diabetes,….

Y hay otras características importantes: si el paciente fuma, si el paciente fue diagnosticado con neumonía o si el paciente fue intubado.
"""

# ╔═╡ 3ff84ad4-121b-4a05-a208-ae8dd5f1524c
md"""La variable dependiente que queremos predecir está en la columna ("CLASIFICACION_FINAL"). Las primeras 14 columnas son datos que identifican a los pacientes y datos poco relevantes para usarse como conjunto de entrenamiento. El resto son posibles variables X (predictores)."""

# ╔═╡ 656ec009-26f6-4e13-befc-22d7cf0c1833
#using Query

# ╔═╡ 42e2c092-d671-4509-940c-c02d17e0ccb9
#using ScikitLearn

# ╔═╡ 560908a7-ebc0-43f1-a0d3-4400a789bf44


# ╔═╡ 32fd347a-b369-4a59-b6b4-9bb58743ccaa
md"""###### Vemos la forma que tiene nuestro data set a analizar:"""

# ╔═╡ 827ddf81-517e-41ea-ab89-460021ec861d
#df_net=DataFrame(CSV.File("/Users/alex/Downloads/archive/XboxOne_GameSales.csv",delim=","))

# ╔═╡ 5db89743-cc56-46cb-a354-f255a5136795


df_cov=DataFrame(CSV.File("/Users/alex/Downloads/data.csv",delim=","))

# ╔═╡ d90c765d-3ae3-423e-90cb-1e36b0ff9aa9
df_cov

# ╔═╡ b65312de-1a28-479b-8007-cdc4bddc0868
md"""## Data Wrangling"""

# ╔═╡ 45f75439-d1b3-47a9-880c-0d6c18896554
md""""Usaremos como variables predictoras solo aquellas que tengan que ver con enfermedades que según estudios han mostrado ser factor decisivo para el desarrollo y agudización del Covid-19, además de la edad como un posible factor de riesgo. Todo esto con la finalidad de corroborar dichas aseveraciones que en este caso serán nuestra hipótesis."""

# ╔═╡ 24ced381-da15-453f-ac1c-276c4f13b9e9
delete!(df_cov,[:FECHA_ACTUALIZACION,:ID_REGISTRO,:ORIGEN,:SECTOR,:ENTIDAD_UM,:SEXO,:ENTIDAD_NAC,:ENTIDAD_RES,:MUNICIPIO_RES,:TIPO_PACIENTE,:FECHA_INGRESO,:FECHA_SINTOMAS,:FECHA_DEF,:INTUBADO,:NACIONALIDAD,:EMBARAZO,:HABLA_LENGUA_INDIG,:INDIGENA,:EPOC,:INMUSUPR,:OTRA_COM,:TOMA_MUESTRA_LAB,:RESULTADO_LAB,:TOMA_MUESTRA_ANTIGENO,:RESULTADO_ANTIGENO,:MIGRANTE,:PAIS_NACIONALIDAD,:PAIS_ORIGEN,:UCI])

# ╔═╡ 772f169b-0fca-4cdc-9aa4-9ddc1d63438c
md"""Para hacer una primera aproximación y reducir la carga computacional tomaremos solo 1000 pacientes del total de la población del DataFrame original."""

# ╔═╡ 089d21f7-b7e9-461e-bac6-eb121e5b49de
df_cov2=df_cov[1:20000,[:NEUMONIA,:EDAD,:DIABETES,:ASMA,:HIPERTENSION,:CARDIOVASCULAR,:OBESIDAD,:RENAL_CRONICA,:TABAQUISMO,:OTRO_CASO,:CLASIFICACION_FINAL]]

# ╔═╡ 0e1554b0-ec10-4693-9bba-5e2dc451c3c3
md"""Vemos un panorama de la estadística general de nuestros datos:"""

# ╔═╡ 239e5b31-79c7-4fa0-bb6f-817a6b73f401
describe(df_cov2)

# ╔═╡ c41eb4ab-158a-459d-9e19-b6fbaeb22c74
md"""Aquí sustituiremos valores de 99 y 98 que significan según la documentación del propio data set: 99->SE IGNORA, 98->NO ESPECIFIcdo por el valor 2->NO pues dada la media obtenida antes al usar el describe(df_cov2), este sería el valor idóneo para no modificar de menara significtiva el DataFrame original. Esto se hizo para cada variable predictora comose mostrará a continuación:"""

# ╔═╡ c5930081-53bb-4290-ac3e-5fee35046699
df_cov2[:NEUMONIA][df_cov2[:NEUMONIA].==99].=2

# ╔═╡ 69d3acc2-3594-4d16-9ce2-bf97196084aa
df_cov2[:NEUMONIA][df_cov2[:NEUMONIA].==98].=2

# ╔═╡ 5d2bc142-d4b5-4c5c-ab65-91349259269c
df_cov2[:NEUMONIA][df_cov2[:NEUMONIA].==2].=0

# ╔═╡ e2081f12-d658-46cb-91ab-f95ed4ec4c54
df_cov2[:DIABETES][df_cov2[:DIABETES].==99].=2

# ╔═╡ 2c015fb1-e500-4d8e-bb0e-88692e7b4c8f
df_cov2[:DIABETES][df_cov2[:DIABETES].==98].=2

# ╔═╡ 9450fad2-5155-42af-8504-67b7342a6c14
df_cov2[:DIABETES][df_cov2[:DIABETES].==2].=0

# ╔═╡ 1be1ae95-0531-4f28-b65b-1e8a3ee9ceb9
df_cov2[:ASMA][df_cov2[:ASMA].==99].=2

# ╔═╡ 645e54d1-cf4c-46f5-8e9f-0556f5428521
df_cov2[:ASMA][df_cov2[:ASMA].==98].=2

# ╔═╡ 5cd3fc04-2eb8-47eb-b754-89cf1ec6349a
df_cov2[:ASMA][df_cov2[:ASMA].==2].=0

# ╔═╡ 9dd62605-0d0c-4df9-9c1d-f041df9f64d3
df_cov2[:HIPERTENSION][df_cov2[:HIPERTENSION].==99].=2

# ╔═╡ bb717d40-3b9c-4ef9-b4c2-9a7cb8607710
df_cov2[:HIPERTENSION][df_cov2[:HIPERTENSION].==98].=2

# ╔═╡ fc0798d2-0d29-43af-b62b-3048b87cb7bb
df_cov2[:HIPERTENSION][df_cov2[:HIPERTENSION].==2].=0

# ╔═╡ be784d01-91ed-4d64-9a21-6ee856805e6e
df_cov2[:CARDIOVASCULAR][df_cov2[:CARDIOVASCULAR].==99].=2

# ╔═╡ fec5b493-76eb-411b-bc7f-a934c36b5dba
df_cov2[:CARDIOVASCULAR][df_cov2[:CARDIOVASCULAR].==98].=2

# ╔═╡ f9b2c19d-3c52-4c56-8dad-711947b8aa50
df_cov2[:CARDIOVASCULAR][df_cov2[:CARDIOVASCULAR].==2].=0

# ╔═╡ c42b5a68-e6d8-4c7f-a912-0fe4a2436b91
df_cov2[:OBESIDAD][df_cov2[:OBESIDAD].==99].=2

# ╔═╡ c6f77364-7d24-49f1-8c18-c3a3126520df
df_cov2[:OBESIDAD][df_cov2[:OBESIDAD].==98].=2

# ╔═╡ d6cf12ae-39db-4f5b-bee5-0efc8ba21427
df_cov2[:OBESIDAD][df_cov2[:OBESIDAD].==2].=0

# ╔═╡ 56fd56a4-a515-435b-af52-e7d9adaa7ef3
df_cov2[:RENAL_CRONICA][df_cov2[:RENAL_CRONICA].==99].=2

# ╔═╡ fd4452f3-23ee-47ab-ba50-a61c0aaa27f0
df_cov2[:RENAL_CRONICA][df_cov2[:RENAL_CRONICA].==98].=2

# ╔═╡ 655c8e30-2f51-493c-8fef-46bb23e33de7
df_cov2[:RENAL_CRONICA][df_cov2[:RENAL_CRONICA].==2].=0

# ╔═╡ 666e815c-d99c-43ea-9e52-fed3529ba160
df_cov2[:TABAQUISMO][df_cov2[:TABAQUISMO].==99].=2

# ╔═╡ 77489e96-b9d2-4954-858c-5ed89d690e71
df_cov2[:TABAQUISMO][df_cov2[:TABAQUISMO].==98].=2

# ╔═╡ 825f7e44-86b9-4572-a34b-4ddcfa2b17bb
df_cov2[:TABAQUISMO][df_cov2[:TABAQUISMO].==2].=0

# ╔═╡ 2904c4f0-b81f-4fb0-a6f5-c73eefb23642
df_cov2[:OTRO_CASO][df_cov2[:OTRO_CASO].==99].=2

# ╔═╡ 9c4cf9d5-7750-4f2f-81b0-91a4ecfd7c24
df_cov2[:OTRO_CASO][df_cov2[:OTRO_CASO].==98].=2

# ╔═╡ 593de815-2826-4f6d-89a9-da07f7446d81
df_cov2[:OTRO_CASO][df_cov2[:OTRO_CASO].==2].=0

# ╔═╡ 179d2b1e-5e31-4d38-a48c-9fbf3afead7f
df_cov2[:CLASIFICACION_FINAL][df_cov2[:CLASIFICACION_FINAL].==1].=1

# ╔═╡ c666078a-a2b6-4726-a514-82f57c7efb59
df_cov2[:CLASIFICACION_FINAL][df_cov2[:CLASIFICACION_FINAL].==2].=1

# ╔═╡ 9ef2ed03-d24b-42cb-9fb2-9481a899b9b0
df_cov2[:CLASIFICACION_FINAL][df_cov2[:CLASIFICACION_FINAL].==3].=1

# ╔═╡ 0169297b-d46d-4516-80f1-6cd4789f8731
df_cov2[:CLASIFICACION_FINAL][df_cov2[:CLASIFICACION_FINAL].==6].=1

# ╔═╡ 80f6dae8-cc4b-4ce8-a6b1-642d56098461
df_cov2[:CLASIFICACION_FINAL][df_cov2[:CLASIFICACION_FINAL].==4].=0

# ╔═╡ f7e524e9-c40e-4b63-9b35-4eda76fc7fed
df_cov2[:CLASIFICACION_FINAL][df_cov2[:CLASIFICACION_FINAL].==5].=0

# ╔═╡ ab3f98ba-ea41-4a3d-81ea-a8f648732859
df_cov2[:CLASIFICACION_FINAL][df_cov2[:CLASIFICACION_FINAL].==7].=0

# ╔═╡ 75501773-384a-4d6f-9adb-caa3eae52938
df_cov2

# ╔═╡ 841f76d8-0f24-4f53-a991-a57cf91ae2de
md"""
# ¡IMPORTANTE!

"""

# ╔═╡ e491dd89-c85e-4c31-8863-de7389f350ee
md"""
Como se ve en la siguiente línea de código, los datos están bastante desequilibrados. Necesita manejar el desequilibrio de clases antes de modelar, pero por ahora los dejaremos así.

"""

# ╔═╡ 387eca1d-98ae-44f9-af14-c5049f4a4cdf
countmap(df_cov2.CLASIFICACION_FINAL)

# ╔═╡ 6aa4d7f7-2306-4bc1-83bf-6188431c8310
train,test=TrainTestSplit(df_cov2,.75)

# ╔═╡ aaaf9cbe-3653-4899-bc9a-f6d257863c88
# Podemos ver la cantidad de filas que tiene cada conjunto tanto el de entrenamiento como el de prueba

# ╔═╡ c923ef0c-d319-4435-b495-f6cb83f266bf
nrow(train)

# ╔═╡ cfe17756-d81a-4c36-89f3-49e2e5483fb6
nrow(test)

# ╔═╡ 89b366f0-fb98-43f6-9387-5608b764af0f
#Modelo de entrenamiento de regresión logística
#Para ajustar un Generalized Linear Model(GLM), usamod la función glm(formula,data,family,link) donde:
#formula:usa los símbolos de las columas del DataFrame
#data: DataFrame a usar en este caso el conjunto de entrenamiento 
#family:elección entre diferentes distribuciones
#link elección sobre la lista de distribuciones disponibles, por ejemplo LogitLink() es un link válido para la familia Binomial()
begin
fm = @formula(CLASIFICACION_FINAL~NEUMONIA+EDAD+DIABETES+ASMA+HIPERTENSION+CARDIOVASCULAR+OBESIDAD+RENAL_CRONICA+TABAQUISMO+OTRO_CASO)
logit=glm(fm,train,Binomial(),ProbitLink())
end

# ╔═╡ 34b6f291-7bcb-4917-9ae0-8d0aaf72adba
md"""Según la tabla anterior al revisar los p-valores observamos que la Neumonia así como el contacto con otras personas enfermas de covid, la edad de los pacientes y el asma pueden ser aceptados como relevantes para el modelo logístico. Recordar que para rechazar la hipótesis nula el p-valor<0.05 """

# ╔═╡ c9e69eed-8d71-4c7b-9a16-045ca1cde242
md"""Ahora vamos a probar el rendimiento del modelo en el conjunto de datos de prueba"""

# ╔═╡ 8babe114-5b34-4f0d-9cca-c2847f7f5a28
#predicción de la variable objetivo sobre el conjunto de prueba
prediction=predict(logit,test)

# ╔═╡ 04648dc3-35b9-4d52-acdd-4bafa3d1a1e2
md"""La predicción del modelo GLM debe clasificarse como 0 o 1 esto es negativo o positivo a Covid-19 respectivamente. Lo más consistente sería entones fijar el umbral para clasificar usando la media de probabilidad i.e 0.5. Así, el siguiente paso sería convertir a clases el resultado anterior de manera que la probabilidad menor que 0.5 se trataría como 0 y si es mayor se trataría como 1."""

# ╔═╡ 9484e369-6028-4052-a80b-c271716f5620
#Convirtiendo el valor de probabilidad a una clase
begin
	prediction_class=[if x <0.5 0 else 1 end for x in prediction]#se recorre a toda la #lista
	prediction_df = DataFrame(y_actual = test.CLASIFICACION_FINAL,y_predicted=prediction_class,prob_predicted=prediction);
	
end

# ╔═╡ b797ceb6-191d-4dd8-b399-faaf215f85be
prediction_df.correctly_classified=prediction_df.y_actual.==prediction_df.y_predicted

# ╔═╡ 6ade89ec-71b5-43a7-bea3-b28974cd2786
md"""La exactitud del modelo es el número total de clases predichas correctamente por el modelo"""

# ╔═╡ db073131-d47e-45a7-bb3a-cecb62cca515
# Puntaje de exactitud
accuracy=mean(prediction_df.correctly_classified)

# ╔═╡ 7b25138c-8777-4252-913c-d5d00f59d671
md"""## Matriz de confusión
La matriz de confusión es una tabla que se usa a menudo para evaluar el desempeño de un modelo de clasificación. Para obtener más información sobre las métricas de rendimiento de un modelo de clasificación.

Puede utilizar la función de matriz de confusión para calcular la matriz de confusión. Pero no funciona con la clase 0. Se puede cambiar la clase 0 a 2. Usemos la función roc de la biblioteca MLBase. Proporciona valores positivos, negativos, verdaderos positivos, verdaderos negativos, falsos positivos y falsos negativos."""

# ╔═╡ 1e93da30-afea-48dd-ae4a-807078f69917

# confusion_matrix = confusmat(2,prediction_df.y_actual, prediction_df.y_predicted)
confusion_matrix = MLBase.roc(prediction_df.y_actual, prediction_df.y_predicted)


# ╔═╡ 0e767bb8-4ae1-407e-ba1d-8731f0d6656e
md"""
 
     - p :: T positivo en la verdad fundamental
     - n :: T negativo en la verdad fundamental
     - tp :: T predicción positiva correcta
     - tn :: T predicción negativa correcta
     - fp :: T (incorrecta) predicción positiva cuando la verdad del terreno es negativa
     - fn :: T # (incorrecta) predicción negativa cuando la verdad del terreno es positiva
fin

"""

# ╔═╡ 26dd53ed-b5e4-4a89-aa99-bf3e68b97271
md"""

# Curva ROC

La curva de características operativas del receptor es la métrica de evaluación utilizada para evaluar el modelo de clasificación en función de su poder predictivo para predecir la precisión de la clase uno y la precisión de la clase cero.

De hecho, el área bajo la curva ROC se puede utilizar como métrica de evaluación para comparar la eficacia de los modelos.

Recordar que se usa la función "pyimport" del paquete Pycall para importar cualquier paquete de Python a julia.
"""

# ╔═╡ 5b2cecef-f3de-4ca5-bd50-a6a12281f870
sklearn=pyimport("sklearn.metrics")

# ╔═╡ bdd56309-02db-40a4-b739-57c4d6932ed0
md"""Tracemos la curva ROC usando "roc_curve" de "sklearn.metrics" de python. Devuelve 3 salidas: tasa de falsos positivos, tasa de verdaderos positivos y umbrales"""

# ╔═╡ 83a491e0-ade8-4e5f-bd9e-5e42dfd0739e
# Calcua la razón de falsos positivos,razón de verdaderos positivos y umbrales
fpr, tpr, thresholds = sklearn.roc_curve(prediction_df.y_actual, prediction_df.prob_predicted)

# ╔═╡ 97a346fb-4455-43a5-8589-0427f2478f3f
md"""

La curva ROC no es más que la curva o el gráfico entre la tasa de falsos positivos y la tasa de verdaderos positivos. La tasa de falsos positivos es el eje y y la tasa de verdaderos positivos es el eje x.

Usar la función Plot:

"""

# ╔═╡ 76f43bb5-e10e-4efa-acd2-14e5e23495af
# Gráfica de la curva ROC:
begin
plot(fpr, tpr)
title!("Curva ROC ")
end

# ╔═╡ f1b1c98a-e210-4729-8c45-4d4ae650b2f5
md"""

Idealmente, la curva debería estar cerca de la línea del eje y y la línea superior del eje x, pero está lejos de ella. Eso significa que no es un buen modelo.

Esto se debe al desequilibrio de la clase alta. es decir, mientras se entrenaban los datos, la mayoría de los puntos de datos tenían clase 1, i.e casos positivos a COVID-19.

Veamos cómo solucionar este problema y manejar el desequilibrio de clases.

"""

# ╔═╡ d6d8b4dd-fc6c-447a-8813-fa92095a8150
md"""

# Manejo del desequilibrio de clases.

A estas alturas, ya conoce los problemas causados por el desequilibrio de clases. Ahora usaré una técnica de golpe para manejar el desequilibrio de clases.

En primer lugar, contaremos el número de clases presentes en los datos originales.

"""

# ╔═╡ 0dd623ac-dd0d-404b-86a6-003d6df3ee95
countmap(df_cov2.CLASIFICACION_FINAL)

# ╔═╡ 2a9d37c4-10b0-40bf-98e8-8f71bea3e1f8
md"""Ahora usemos la función smote para manejar el desequilibrio de clases."""

# ╔═╡ e26ebaf6-9434-4b29-8c18-adf7b22f2467
begin
X2, y2 =smote(df_cov2[!,[:NEUMONIA,:EDAD,:DIABETES,:ASMA,:HIPERTENSION,:CARDIOVASCULAR,:OBESIDAD,:RENAL_CRONICA,:TABAQUISMO,:OTRO_CASO]], df_cov2.CLASIFICACION_FINAL, k = 5, pct_under = 150, pct_over = 200)
df_balanced = X2
df_balanced.CLASIFICACION_FINAL = y2;

df_bal = df_balanced;
end

# ╔═╡ 21bf2587-eadf-4ee0-9543-d47396ea947a
# Count the classes
countmap(df_bal.CLASIFICACION_FINAL)

# ╔═╡ f909808a-0ede-499f-8b13-1b0d73566e8c
md"""

Ahora con las clases ya equilibradas volvemos a hacer los mismos cálculos anteriores

"""

# ╔═╡ de62e6c5-fd57-4365-b88a-71f179ab9494
begin 
# Train test split
train1, test1 = TrainTestSplit(df_bal,.75);

# Model Building
fm1 = @formula(CLASIFICACION_FINAL~NEUMONIA+EDAD+DIABETES+ASMA+HIPERTENSION+CARDIOVASCULAR+OBESIDAD+RENAL_CRONICA+TABAQUISMO+OTRO_CASO)
logit1 = glm(fm1, train1, Binomial(), ProbitLink())

# Predict the target variable on test data 
prediction1 = predict(logit1,test1)
end

# ╔═╡ 99888502-1bcb-4a0e-b438-4156cb22aa6f
begin
# Convert probability score to class
	prediction_class1 = [if x < 0.5 0 else 1 end for x in prediction1]

prediction_df1 = DataFrame(y_actual1 = test1.CLASIFICACION_FINAL, y_predicted1 = prediction_class1, prob_predicted1 = prediction1);
	
prediction_df1.correctly_classified1 = prediction_df1.y_actual1 .== prediction_df1.y_predicted1
	
accuracy1 = mean(prediction_df1.correctly_classified1)
print("La exactitud del modelo es : ",accuracy1)
end

# ╔═╡ 970c7b34-f877-4461-b51d-95f38b96f73b
# Calcua la razón de falsos positivos,razón de verdaderos positivos y umbrales
fpr1, tpr1, thresholds1 = sklearn.roc_curve(prediction_df1.y_actual1, prediction_df1.prob_predicted1)

# ╔═╡ a6f3c7f7-11b3-4fdf-9492-88f2f1899fc8
# Gráfica de la curva ROC:
begin
plot(fpr1, tpr1)
title!("Curva 2 ROC ")
end

# ╔═╡ a47411d2-0639-49bc-8666-ada0061f471e
accuracy1

# ╔═╡ e286b116-c72d-49a4-a9d1-f93fd31bbfe6
accuracy

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
CSV = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
ClassImbalance = "04a18a73-7590-580c-b363-eeca0919eb2a"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
GLM = "38e38edf-8417-5370-95a0-9cbb8c7f171a"
Lathe = "38d8eb38-e7b1-11e9-0012-376b6c802672"
MLBase = "f0e99cf1-93fa-52ec-9ecc-5026115318e0"
Pkg = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
PyCall = "438e738f-606a-5dbb-bf0a-cddfbfd45ab0"
ROCAnalysis = "f535d66d-59bb-5153-8d2b-ef0a426c6aff"
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"

[compat]
CSV = "~0.8.5"
ClassImbalance = "~0.8.7"
DataFrames = "~0.20.2"
GLM = "~1.4.2"
Lathe = "~0.0.9"
MLBase = "~0.8.0"
Plots = "~1.20.0"
PlutoUI = "~0.7.1"
PyCall = "~1.92.3"
ROCAnalysis = "~0.3.3"
StatsBase = "~0.32.2"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

[[Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "84918055d15b3114ede17ac6a7182f68870c16f7"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.3.1"

[[ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[Arpack]]
deps = ["Arpack_jll", "Libdl", "LinearAlgebra"]
git-tree-sha1 = "2ff92b71ba1747c5fdd541f8fc87736d82f40ec9"
uuid = "7d9fca2a-8960-54d3-9f78-7d1dccf2cb97"
version = "0.4.0"

[[Arpack_jll]]
deps = ["Libdl", "OpenBLAS_jll", "Pkg"]
git-tree-sha1 = "e214a9b9bd1b4e1b4f15b22c0994862b66af7ff7"
uuid = "68821587-b530-5797-8361-c406ea357684"
version = "3.5.0+3"

[[Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c3598e525718abcc440f69cc6d5f60dda0a1b61e"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.6+5"

[[CSV]]
deps = ["Dates", "Mmap", "Parsers", "PooledArrays", "SentinelArrays", "Tables", "Unicode"]
git-tree-sha1 = "b83aa3f513be680454437a0eee21001607e5d983"
uuid = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
version = "0.8.5"

[[Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "e2f47f6d8337369411569fd45ae5753ca10394c6"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.0+6"

[[CategoricalArrays]]
deps = ["Compat", "DataAPI", "Future", "JSON", "Missings", "Printf", "Reexport", "Statistics", "Unicode"]
git-tree-sha1 = "23d7324164c89638c18f6d7f90d972fa9c4fa9fb"
uuid = "324d7699-5711-5eae-9e2f-1d82baa6b597"
version = "0.7.7"

[[ClassImbalance]]
deps = ["Compat", "DataFrames", "Distributions", "LinearAlgebra", "Random", "Statistics", "StatsBase"]
git-tree-sha1 = "9503749483f4c3bfba567af52c26c14d00428b68"
uuid = "04a18a73-7590-580c-b363-eeca0919eb2a"
version = "0.8.7"

[[ColorSchemes]]
deps = ["ColorTypes", "Colors", "FixedPointNumbers", "Random"]
git-tree-sha1 = "9995eb3977fbf67b86d0a0a0508e83017ded03f2"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.14.0"

[[ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "024fe24d83e4a5bf5fc80501a314ce0d1aa35597"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.0"

[[Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "417b0ed7b8b838aa6ca0a87aadf1bb9eb111ce40"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.8"

[[Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "344f143fa0ec67e47917848795ab19c6a455f32c"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.32.0"

[[CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

[[Conda]]
deps = ["JSON", "VersionParsing"]
git-tree-sha1 = "299304989a5e6473d985212c28928899c74e9421"
uuid = "8f4d0f93-b110-5947-807f-2305c1781a2d"
version = "1.5.2"

[[Contour]]
deps = ["StaticArrays"]
git-tree-sha1 = "9f02045d934dc030edad45944ea80dbd1f0ebea7"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.5.7"

[[DataAPI]]
git-tree-sha1 = "ee400abb2298bd13bfc3df1c412ed228061a2385"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.7.0"

[[DataFrames]]
deps = ["CategoricalArrays", "Compat", "DataAPI", "Future", "InvertedIndices", "IteratorInterfaceExtensions", "Missings", "PooledArrays", "Printf", "REPL", "Reexport", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "7d5bf815cc0b30253e3486e8ce2b93bf9d0faff6"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "0.20.2"

[[DataStructures]]
deps = ["InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "88d48e133e6d3dd68183309877eac74393daa7eb"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.17.20"

[[DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[Distributions]]
deps = ["FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns"]
git-tree-sha1 = "55e1de79bd2c397e048ca47d251f8fa70e530550"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.22.6"

[[DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "a32185f5428d3986f47c2ab78b1f216d5e6cc96f"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.8.5"

[[Downloads]]
deps = ["ArgTools", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[EarCut_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "92d8f9f208637e8d2d28c664051a00569c01493d"
uuid = "5ae413db-bbd1-5e63-b57d-d24a61df00f5"
version = "2.1.5+1"

[[Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b3bfd02e98aedfa5cf885665493c5598c350cd2f"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.2.10+0"

[[FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "b57e3acbe22f8484b4b5ff66a7499717fe1a9cc8"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.1"

[[FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "LibVPX_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "Pkg", "Zlib_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "3cc57ad0a213808473eafef4845a74766242e05f"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.3.1+4"

[[FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays"]
git-tree-sha1 = "4863cbb7910079369e258dee4add9d06ead5063a"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "0.8.14"

[[FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "35895cf184ceaab11fd778b4590144034a167a2f"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.1+14"

[[Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "cbd58c9deb1d304f5a245a0b7eb841a2560cfec6"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.10.1+5"

[[FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "aa31987c2ba8704e23c6c8ba8a4f769d5d7e4f91"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.10+0"

[[Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pkg", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll"]
git-tree-sha1 = "dba1e8614e98949abfa60480b13653813d8f0157"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.5+0"

[[GLM]]
deps = ["Distributions", "LinearAlgebra", "Printf", "Random", "Reexport", "SparseArrays", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns", "StatsModels"]
git-tree-sha1 = "dc577ad8b146183c064b30e747e3afc6d6dfd62b"
uuid = "38e38edf-8417-5370-95a0-9cbb8c7f171a"
version = "1.4.2"

[[GR]]
deps = ["Base64", "DelimitedFiles", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Printf", "Random", "Serialization", "Sockets", "Test", "UUIDs"]
git-tree-sha1 = "182da592436e287758ded5be6e32c406de3a2e47"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.58.1"

[[GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Pkg", "Qt5Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "d59e8320c2747553788e4fc42231489cc602fa50"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.58.1+0"

[[GeometryBasics]]
deps = ["EarCut_jll", "IterTools", "LinearAlgebra", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "58bcdf5ebc057b085e58d95c138725628dd7453c"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.4.1"

[[Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "7bf67e9a481712b3dbe9cb3dac852dc4b1162e02"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.68.3+0"

[[Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[HTTP]]
deps = ["Base64", "Dates", "IniFile", "Logging", "MbedTLS", "NetworkOptions", "Sockets", "URIs"]
git-tree-sha1 = "44e3b40da000eab4ccb1aecdc4801c040026aeb5"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "0.9.13"

[[IniFile]]
deps = ["Test"]
git-tree-sha1 = "098e4d2c533924c921f9f9847274f2ad89e018b8"
uuid = "83e8ac13-25f8-5344-8a64-a9f2b223428f"
version = "0.5.0"

[[InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[InvertedIndices]]
deps = ["Test"]
git-tree-sha1 = "15732c475062348b0165684ffe28e85ea8396afc"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.0.0"

[[IterTools]]
git-tree-sha1 = "05110a2ab1fc5f932622ffea2a003221f4782c18"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.3.0"

[[IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "642a199af8b68253517b80bd3bfd17eb4e84df6e"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.3.0"

[[JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "8076680b162ada2a031f707ac7b4953e30667a37"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.2"

[[JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "d735490ac75c5cb9f1b00d8b5509c11984dc6943"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "2.1.0+0"

[[LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6250b16881adf048549549fba48b1161acdac8c"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.1+0"

[[LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e5b909bcf985c5e2605737d2ce278ed791b89be6"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.1+0"

[[LaTeXStrings]]
git-tree-sha1 = "c7f1c695e06c01b95a67f0cd1d34994f3e7db104"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.2.1"

[[Latexify]]
deps = ["Formatting", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "Printf", "Requires"]
git-tree-sha1 = "a4b12a1bd2ebade87891ab7e36fdbce582301a92"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.15.6"

[[Lathe]]
deps = ["DataFrames", "Random"]
git-tree-sha1 = "5f64e72da1435568cd8362d6d0f364d210df3e9e"
uuid = "38d8eb38-e7b1-11e9-0012-376b6c802672"
version = "0.0.9"

[[LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"

[[LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"

[[LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"

[[LibVPX_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "12ee7e23fa4d18361e7c2cde8f8337d4c3101bc7"
uuid = "dd192d2f-8180-539f-9fb4-cc70b1dcf69a"
version = "1.10.0+0"

[[Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "761a393aeccd6aa92ec3515e428c26bf99575b3b"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+0"

[[Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll", "Pkg"]
git-tree-sha1 = "64613c82a59c120435c067c2b809fc61cf5166ae"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.7+0"

[[Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "7739f837d6447403596a75d19ed01fd08d6f56bf"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.3.0+3"

[[Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c333716e46366857753e273ce6a69ee0945a6db9"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.42.0+0"

[[Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "42b62845d70a619f063a7da093d995ec8e15e778"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.16.1+1"

[[Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9c30530bf0effd46e15e0fdcf2b8636e78cbbd73"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.35.0+0"

[[Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Pkg", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "340e257aada13f95f98ee352d316c3bed37c8ab9"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.3.0+0"

[[Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7f3efec06033682db852f8b3bc3c1d2b0a0ab066"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.36.0+0"

[[LinearAlgebra]]
deps = ["Libdl"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[LogExpFunctions]]
deps = ["DocStringExtensions", "LinearAlgebra"]
git-tree-sha1 = "7bd5f6565d80b6bf753738d2bc40a5dfea072070"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.2.5"

[[Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[MLBase]]
deps = ["IterTools", "Random", "Reexport", "StatsBase", "Test"]
git-tree-sha1 = "f63a8d37429568b8c4384d76c4a96fc2897d6ddf"
uuid = "f0e99cf1-93fa-52ec-9ecc-5026115318e0"
version = "0.8.0"

[[MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "0fb723cd8c45858c22169b2e42269e53271a6df7"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.7"

[[Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "Random", "Sockets"]
git-tree-sha1 = "1c38e51c3d08ef2278062ebceade0e46cefc96fe"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.0.3"

[[MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"

[[Measures]]
git-tree-sha1 = "e498ddeee6f9fdb4551ce855a46f54dbd900245f"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.1"

[[Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "f8c673ccc215eb50fcadb285f522420e29e69e1c"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "0.4.5"

[[Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[NaNMath]]
git-tree-sha1 = "bfe47e760d60b82b66b61d2d44128b62e3a369fb"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "0.3.5"

[[NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7937eda4681660b4d6aeeecc2f7e1c81c8ee4e2f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+0"

[[OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"

[[OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "15003dcb7d8db3c6c857fda14891a539a8f2705a"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.10+0"

[[OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

[[OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[PCRE_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b2a7af664e098055a7529ad1a900ded962bca488"
uuid = "2f80f16e-611a-54ab-bc61-aa92de5b98fc"
version = "8.44.0+0"

[[PDMats]]
deps = ["Arpack", "LinearAlgebra", "SparseArrays", "SuiteSparse", "Test"]
git-tree-sha1 = "2fc6f50ddd959e462f0a2dbc802ddf2a539c6e35"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.9.12"

[[Parsers]]
deps = ["Dates"]
git-tree-sha1 = "bfd7d8c7fd87f04543810d9cbd3995972236ba1b"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "1.1.2"

[[Pixman_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b4f5d02549a10e20780a24fce72bea96b6329e29"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.40.1+0"

[[Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[PlotThemes]]
deps = ["PlotUtils", "Requires", "Statistics"]
git-tree-sha1 = "a3a964ce9dc7898193536002a6dd892b1b5a6f1d"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "2.0.1"

[[PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "Printf", "Random", "Reexport", "Statistics"]
git-tree-sha1 = "501c20a63a34ac1d015d5304da0e645f42d91c9f"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.0.11"

[[Plots]]
deps = ["Base64", "Contour", "Dates", "FFMPEG", "FixedPointNumbers", "GR", "GeometryBasics", "JSON", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "PlotThemes", "PlotUtils", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "UUIDs"]
git-tree-sha1 = "e39bea10478c6aff5495ab522517fae5134b40e3"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.20.0"

[[PlutoUI]]
deps = ["Base64", "Dates", "InteractiveUtils", "Logging", "Markdown", "Random", "Suppressor"]
git-tree-sha1 = "45ce174d36d3931cd4e37a47f93e07d1455f038d"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.1"

[[PooledArrays]]
deps = ["DataAPI"]
git-tree-sha1 = "b1333d4eced1826e15adbdf01a4ecaccca9d353c"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "0.5.3"

[[Preferences]]
deps = ["TOML"]
git-tree-sha1 = "00cfd92944ca9c760982747e9a1d0d5d86ab1e5a"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.2.2"

[[Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[PyCall]]
deps = ["Conda", "Dates", "Libdl", "LinearAlgebra", "MacroTools", "Serialization", "VersionParsing"]
git-tree-sha1 = "169bb8ea6b1b143c5cf57df6d34d022a7b60c6db"
uuid = "438e738f-606a-5dbb-bf0a-cddfbfd45ab0"
version = "1.92.3"

[[Qt5Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "xkbcommon_jll"]
git-tree-sha1 = "ad368663a5e20dbb8d6dc2fddeefe4dae0781ae8"
uuid = "ea2cea3b-5b76-57ae-a6ef-0a8af62496e1"
version = "5.15.3+0"

[[QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "12fbe86da16df6679be7521dfb39fbc861e1dc7b"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.4.1"

[[REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[ROCAnalysis]]
deps = ["DataFrames", "LinearAlgebra", "Printf", "Random", "RecipesBase", "SpecialFunctions"]
git-tree-sha1 = "e04ce44600445a6dac9c9a9bf48ea8aa5c80e24a"
uuid = "f535d66d-59bb-5153-8d2b-ef0a426c6aff"
version = "0.3.3"

[[Random]]
deps = ["Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[RecipesBase]]
git-tree-sha1 = "b3fb709f3c97bfc6e948be68beeecb55a0b340ae"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.1.1"

[[RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "RecipesBase"]
git-tree-sha1 = "2a7a2469ed5d94a98dea0e85c46fa653d76be0cd"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.3.4"

[[Reexport]]
deps = ["Pkg"]
git-tree-sha1 = "7b1d07f411bc8ddb7977ec7f377b97b158514fe0"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "0.2.0"

[[Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "4036a3bd08ac7e968e27c203d45f5fff15020621"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.1.3"

[[Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "bf3188feca147ce108c76ad82c2792c57abe7b1f"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.0"

[[Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "68db32dff12bb6127bac73c209881191bf0efbb7"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.3.0+0"

[[SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[Scratch]]
deps = ["Dates"]
git-tree-sha1 = "0b4b7f1393cff97c33891da2a0bf69c6ed241fda"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.1.0"

[[SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "a3a337914a035b2d59c9cbe7f1a38aaba1265b02"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.3.6"

[[Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[ShiftedArrays]]
git-tree-sha1 = "22395afdcf37d6709a5a0766cc4a5ca52cb85ea0"
uuid = "1277b4bf-5013-50f5-be3d-901d8477a67a"
version = "1.0.0"

[[Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[SortingAlgorithms]]
deps = ["DataStructures", "Random", "Test"]
git-tree-sha1 = "03f5898c9959f8115e30bc7226ada7d0df554ddd"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "0.3.1"

[[SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[SpecialFunctions]]
deps = ["OpenSpecFun_jll"]
git-tree-sha1 = "d8d8b8a9f4119829410ecd706da4cc8594a1e020"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "0.10.3"

[[StaticArrays]]
deps = ["LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "3240808c6d463ac46f1c1cd7638375cd22abbccb"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.2.12"

[[Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics"]
git-tree-sha1 = "19bfcb46245f69ff4013b3df3b977a289852c3a1"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.32.2"

[[StatsFuns]]
deps = ["LogExpFunctions", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "30cd8c360c54081f806b1ee14d2eecbef3c04c49"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "0.9.8"

[[StatsModels]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "Printf", "ShiftedArrays", "SparseArrays", "StatsBase", "StatsFuns", "Tables"]
git-tree-sha1 = "3db41a7e4ae7106a6bcff8aa41833a4567c04655"
uuid = "3eaba693-59b7-5ba5-a881-562e759f1c8d"
version = "0.6.21"

[[StructArrays]]
deps = ["Adapt", "DataAPI", "StaticArrays", "Tables"]
git-tree-sha1 = "000e168f5cc9aded17b6999a560b7c11dda69095"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.0"

[[SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[Suppressor]]
git-tree-sha1 = "a819d77f31f83e5792a76081eee1ea6342ab8787"
uuid = "fd094767-a336-5f1f-9728-57cf17d0bbfb"
version = "0.2.0"

[[TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "TableTraits", "Test"]
git-tree-sha1 = "d0c690d37c73aeb5ca063056283fde5585a41710"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.5.0"

[[Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[URIs]]
git-tree-sha1 = "97bbe755a53fe859669cd907f2d96aee8d2c1355"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.3.0"

[[UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[VersionParsing]]
git-tree-sha1 = "80229be1f670524750d905f8fc8148e5a8c4537f"
uuid = "81def892-9a0e-5fdd-b105-ffc91e053289"
version = "1.2.0"

[[Wayland_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "3e61f0b86f90dacb0bc0e73a0c5a83f6a8636e23"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.19.0+0"

[[Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll"]
git-tree-sha1 = "2839f1c1296940218e35df0bbb220f2a79686670"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.18.0+4"

[[XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "1acf5bdf07aa0907e0a37d3718bb88d4b687b74a"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.9.12+0"

[[XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "5be649d550f3f4b95308bf0183b82e2582876527"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.6.9+4"

[[Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4e490d5c960c314f33885790ed410ff3a94ce67e"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.9+4"

[[Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "12e0eb3bc634fa2080c1c37fccf56f7c22989afd"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.0+4"

[[Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fe47bd2247248125c428978740e18a681372dd4"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.3+4"

[[Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "b7c0aa8c376b31e4852b360222848637f481f8c3"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.4+4"

[[Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "0e0dc7431e7a0587559f9294aeec269471c991a4"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "5.0.3+4"

[[Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "89b52bc2160aadc84d707093930ef0bffa641246"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.7.10+4"

[[Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll"]
git-tree-sha1 = "26be8b1c342929259317d8b9f7b53bf2bb73b123"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.4+4"

[[Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "34cea83cb726fb58f325887bf0612c6b3fb17631"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.2+4"

[[Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "19560f30fd49f4d4efbe7002a1037f8c43d43b96"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.10+4"

[[Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6783737e45d3c59a4a4c4091f5f88cdcf0908cbb"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.0+3"

[[Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "daf17f441228e7a3833846cd048892861cff16d6"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.13.0+3"

[[Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "926af861744212db0eb001d9e40b5d16292080b2"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.0+4"

[[Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "0fab0a40349ba1cba2c1da699243396ff8e94b97"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.0+1"

[[Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll"]
git-tree-sha1 = "e7fd7b2881fa2eaa72717420894d3938177862d1"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.0+1"

[[Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "d1151e2c45a544f32441a567d1690e701ec89b00"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.0+1"

[[Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "dfd7a8f38d4613b6a575253b3174dd991ca6183e"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.9+1"

[[Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "e78d10aab01a4a154142c5006ed44fd9e8e31b67"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.1+1"

[[Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "4bcbf660f6c2e714f87e960a171b119d06ee163b"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.2+4"

[[Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "5c8424f8a67c3f2209646d4425f3d415fee5931d"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.27.0+4"

[[Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "79c31e7844f6ecf779705fbc12146eb190b7d845"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.4.0+3"

[[Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "cc4bf3fdde8b7e3e9fa0351bdeedba1cf3b7f6e6"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.0+0"

[[libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "acc685bcf777b2202a904cdcb49ad34c2fa1880c"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.14.0+4"

[[libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7a5780a0d9c6864184b3a2eeeb833a0c871f00ab"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "0.1.6+4"

[[libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "94d180a6d2b5e55e447e2d27a29ed04fe79eb30c"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.38+0"

[[libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "c45f4e40e7aafe9d086379e5578947ec8b95a8fb"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+0"

[[nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"

[[p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"

[[x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "d713c1ce4deac133e3334ee12f4adff07f81778f"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2020.7.14+2"

[[x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "487da2f8f2f0c8ee0e83f39d13037d6bbf0a45ab"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.0.0+3"

[[xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "ece2350174195bb31de1a63bea3a41ae1aa593b6"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "0.9.1+5"
"""

# ╔═╡ Cell order:
# ╟─4c726d0c-f95e-11eb-3cd2-71abdb89daac
# ╟─3ff84ad4-121b-4a05-a208-ae8dd5f1524c
# ╠═eec9b297-df0b-4d1d-b54b-9f5c6f506df6
# ╠═9047582d-b0b2-4ff5-b84b-d9194b9053b0
# ╠═2452c556-614c-471b-9611-8f4c17809dbf
# ╠═b790ff9a-d590-4cb0-975c-91473170c1cc
# ╠═656ec009-26f6-4e13-befc-22d7cf0c1833
# ╠═42e2c092-d671-4509-940c-c02d17e0ccb9
# ╠═1d14e33b-aee6-43e5-b0df-401555d6cf67
# ╠═52092e65-6275-4dab-af55-c546d80d4650
# ╠═f5886586-b2d0-4ce7-86cf-ec4c439b4ed1
# ╠═fc0f0e2b-7a7c-47e4-bd62-de69ad3e7e61
# ╠═37e1b9da-6ec1-40fd-aa5f-96ab9c36c502
# ╠═35bd5abe-5546-4556-8ecd-66edf6a89ade
# ╠═9d2ff573-ece2-41d8-b3d8-fe3164c1c607
# ╠═560908a7-ebc0-43f1-a0d3-4400a789bf44
# ╟─32fd347a-b369-4a59-b6b4-9bb58743ccaa
# ╠═827ddf81-517e-41ea-ab89-460021ec861d
# ╠═5db89743-cc56-46cb-a354-f255a5136795
# ╠═d90c765d-3ae3-423e-90cb-1e36b0ff9aa9
# ╟─b65312de-1a28-479b-8007-cdc4bddc0868
# ╟─45f75439-d1b3-47a9-880c-0d6c18896554
# ╠═24ced381-da15-453f-ac1c-276c4f13b9e9
# ╟─772f169b-0fca-4cdc-9aa4-9ddc1d63438c
# ╠═089d21f7-b7e9-461e-bac6-eb121e5b49de
# ╟─0e1554b0-ec10-4693-9bba-5e2dc451c3c3
# ╠═239e5b31-79c7-4fa0-bb6f-817a6b73f401
# ╟─c41eb4ab-158a-459d-9e19-b6fbaeb22c74
# ╠═c5930081-53bb-4290-ac3e-5fee35046699
# ╠═69d3acc2-3594-4d16-9ce2-bf97196084aa
# ╠═5d2bc142-d4b5-4c5c-ab65-91349259269c
# ╠═e2081f12-d658-46cb-91ab-f95ed4ec4c54
# ╠═2c015fb1-e500-4d8e-bb0e-88692e7b4c8f
# ╠═9450fad2-5155-42af-8504-67b7342a6c14
# ╠═1be1ae95-0531-4f28-b65b-1e8a3ee9ceb9
# ╠═645e54d1-cf4c-46f5-8e9f-0556f5428521
# ╠═5cd3fc04-2eb8-47eb-b754-89cf1ec6349a
# ╠═9dd62605-0d0c-4df9-9c1d-f041df9f64d3
# ╠═bb717d40-3b9c-4ef9-b4c2-9a7cb8607710
# ╠═fc0798d2-0d29-43af-b62b-3048b87cb7bb
# ╠═be784d01-91ed-4d64-9a21-6ee856805e6e
# ╠═fec5b493-76eb-411b-bc7f-a934c36b5dba
# ╠═f9b2c19d-3c52-4c56-8dad-711947b8aa50
# ╠═c42b5a68-e6d8-4c7f-a912-0fe4a2436b91
# ╠═c6f77364-7d24-49f1-8c18-c3a3126520df
# ╠═d6cf12ae-39db-4f5b-bee5-0efc8ba21427
# ╠═56fd56a4-a515-435b-af52-e7d9adaa7ef3
# ╠═fd4452f3-23ee-47ab-ba50-a61c0aaa27f0
# ╠═655c8e30-2f51-493c-8fef-46bb23e33de7
# ╠═666e815c-d99c-43ea-9e52-fed3529ba160
# ╠═77489e96-b9d2-4954-858c-5ed89d690e71
# ╠═825f7e44-86b9-4572-a34b-4ddcfa2b17bb
# ╠═2904c4f0-b81f-4fb0-a6f5-c73eefb23642
# ╠═9c4cf9d5-7750-4f2f-81b0-91a4ecfd7c24
# ╠═593de815-2826-4f6d-89a9-da07f7446d81
# ╠═179d2b1e-5e31-4d38-a48c-9fbf3afead7f
# ╠═c666078a-a2b6-4726-a514-82f57c7efb59
# ╠═9ef2ed03-d24b-42cb-9fb2-9481a899b9b0
# ╠═0169297b-d46d-4516-80f1-6cd4789f8731
# ╠═80f6dae8-cc4b-4ce8-a6b1-642d56098461
# ╠═f7e524e9-c40e-4b63-9b35-4eda76fc7fed
# ╠═ab3f98ba-ea41-4a3d-81ea-a8f648732859
# ╠═75501773-384a-4d6f-9adb-caa3eae52938
# ╟─841f76d8-0f24-4f53-a991-a57cf91ae2de
# ╟─e491dd89-c85e-4c31-8863-de7389f350ee
# ╠═387eca1d-98ae-44f9-af14-c5049f4a4cdf
# ╠═92465c4f-9a97-4baf-902e-1620a6210034
# ╠═6aa4d7f7-2306-4bc1-83bf-6188431c8310
# ╠═aaaf9cbe-3653-4899-bc9a-f6d257863c88
# ╠═c923ef0c-d319-4435-b495-f6cb83f266bf
# ╠═cfe17756-d81a-4c36-89f3-49e2e5483fb6
# ╠═89b366f0-fb98-43f6-9387-5608b764af0f
# ╟─34b6f291-7bcb-4917-9ae0-8d0aaf72adba
# ╟─c9e69eed-8d71-4c7b-9a16-045ca1cde242
# ╠═8babe114-5b34-4f0d-9cca-c2847f7f5a28
# ╟─04648dc3-35b9-4d52-acdd-4bafa3d1a1e2
# ╠═9484e369-6028-4052-a80b-c271716f5620
# ╠═b797ceb6-191d-4dd8-b399-faaf215f85be
# ╟─6ade89ec-71b5-43a7-bea3-b28974cd2786
# ╠═db073131-d47e-45a7-bb3a-cecb62cca515
# ╟─7b25138c-8777-4252-913c-d5d00f59d671
# ╠═1e93da30-afea-48dd-ae4a-807078f69917
# ╟─0e767bb8-4ae1-407e-ba1d-8731f0d6656e
# ╟─26dd53ed-b5e4-4a89-aa99-bf3e68b97271
# ╠═1a5bf86b-7ded-434d-a742-275e15b514be
# ╠═5b2cecef-f3de-4ca5-bd50-a6a12281f870
# ╠═bdd56309-02db-40a4-b739-57c4d6932ed0
# ╠═83a491e0-ade8-4e5f-bd9e-5e42dfd0739e
# ╟─97a346fb-4455-43a5-8589-0427f2478f3f
# ╠═76f43bb5-e10e-4efa-acd2-14e5e23495af
# ╟─f1b1c98a-e210-4729-8c45-4d4ae650b2f5
# ╟─d6d8b4dd-fc6c-447a-8813-fa92095a8150
# ╠═0dd623ac-dd0d-404b-86a6-003d6df3ee95
# ╠═2a9d37c4-10b0-40bf-98e8-8f71bea3e1f8
# ╠═e26ebaf6-9434-4b29-8c18-adf7b22f2467
# ╠═21bf2587-eadf-4ee0-9543-d47396ea947a
# ╟─f909808a-0ede-499f-8b13-1b0d73566e8c
# ╠═de62e6c5-fd57-4365-b88a-71f179ab9494
# ╠═99888502-1bcb-4a0e-b438-4156cb22aa6f
# ╠═970c7b34-f877-4461-b51d-95f38b96f73b
# ╠═a6f3c7f7-11b3-4fdf-9492-88f2f1899fc8
# ╠═a47411d2-0639-49bc-8666-ada0061f471e
# ╠═e286b116-c72d-49a4-a9d1-f93fd31bbfe6
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
