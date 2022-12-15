<p align=center><img src=https://assets.soyhenry.com/logos/LOGO-HENRY-04.png><p>

# <h1 align=center> **PROYECTO INDIVIDUAL Nº2** </h1>

# <h1 align=center>**`Machine Learning`**</h1>
*Alvaro Enrique Beleño Contreras*

*Ingeniero Industrial*

*Data Sciencie(In progress)*
<p align="center">
<img src="http://innovait.cat/wp-content/uploads/2022/03/Machine-learning.png"height=250>

</p>



<hr>  
Este trabajo consiste en relaizar un modelo de Machine Learning que pueda determinar si la estancia de un paciente en el hospital sera prolongada o no, para ello se suministra información historica de los pacientes por medio de un Dataset.

<hr>

## **Descripción del problema**

<p align="center">
<img src="https://www.kasikornresearch.com/SiteCollectionDocuments/analysis/business/Health/Hospital20_banner.jpg"height=250>


Un reconocido centro de salud me ha contratado para determinar la estancia hospitalaria de un paciente, para ello ha brindado información historica de los pacientes, como tambien parametros claves que ayudan a la determinación de los tipos de modelo a usar, nos dice que un que un paciente posee estancia hospitalaria prolongada si ha estado hospitalizado más de 8 días lo cual genera efectos negativos en el sistema de salud aumentando los costos, generando deficiencia en la accesibilidad de prestación de servicios de salud, saturando las unidades de hospitalización y urgencias, generando mayores efectos adversos como lo son las enfermedades intrahospitalarias.

Es por ello que se hace necesario realizar estudios a esta problematica, conociendo los diferentes perfiles de usuarios con la finalidad de predecir el tipo de estancia y de esta manera poder garantizar los recursos necesarios para la atención del paciente, realizar ajustes respecto a la oferta y demanda de los servicios de salud y los implementos asociados.​

<hr> 

## **Pasos para la elaboración del proyecto**
**1. Librerias utilizadas**
```python
import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_validate 
from IPython.display import clear_output
```
**2. Recolección de datos**

El hospital brindo dos Dataset que contiene datos historicos de los perfiles de los pacientes, los cuales tienen como nombre [hospitalizaciones_test.csv](https://raw.githubusercontent.com/soyHenry/Datathon/main/hospitalizaciones_test.csv) y [hospitalizaciones_train.csv](https://raw.githubusercontent.com/soyHenry/Datathon/main/hospitalizaciones_train.csv), las featues que contiene dichos dataset tienen la siguiente descripción:

* Available Extra Rooms in Hospital: Habitaciones adicionales disponibles en el hospital. Una habitación no es igual a un paciente, pueden ser individuales o compartidas.

* Department: Área de atención a la que ingresa el paciente.
Ward_Facility_Code: Código de la habitación del paciente.
doctor_name: Nombre de el/la doctor/a a cargo del paciente.

* staff_available: Cantidad de personal disponible al momento del ingreso del paciente.
patientid: Identificador del paciente.

* Age: Edad del paciente.

* gender: Género del paciente.

* Type of Admission: Tipo de ingreso registrado según la situación de ingreso del paciente.

* Severity of Illness: Gravedad de la enfermedad/condición/estado del paciente al momento del ingreso.

* health_conditions: Condiciones de salud del paciente.

* Visitors with Patient: Cantidad de visitantes registrados para el paciente.

* Insurance: Indica si la persona posee o no seguro de salud.

* Admission_Deposit: Pago realizado a nombre del paciente, con el fin de cubrir los costos iniciales de internación.

* Stay (in days): Días registrados de estancia hospitalaria.​


**3. EDA**

Se realiza un analisis exploratorio con la finalidad de conocer el dataset y verificar si tiene datos nulo o tipos de datos incorrectos, esto se hace de manera rapida con  ***.info()*** y ***.head()***, para ver mas a detalle se pueden remitir a [codigo_completo](https://github.com/Alvaro9721/PI_2_Machine_Learning/blob/main/PI_2_Machine_learning.ipynb).

```python
Df_hospitalizacion_train.info()
Df_hospitalizacion_train.head()
```

**3. Creación de variables Dummy**

Debido a que se necesita saber cuando un paciente tuvo estancia prologonda o no, se crea una nueva columna llamada ***tipo_estancia*** la cual posee variables binarias(0,1) donde 0 significa que dicho paciente no tuvo estancias prologanda y 1 si tuvo estancia prolongada, esta nueva *feature* se obtuvo con el siguiente codigo:
```python
import numpy as np

Df_train["tipo_estancia"]=np.where(Df_train["Días registrados"]>8,1,0) 
```

Posteriormente se convierten los datos de las demas *features* en valores numéricos por medio del procedimiento *.LabelEncoder()* de la libreria scikit-learn.
```python
from sklearn import preprocessing
le=preprocessing.LabelEncoder()

EJ: Df_train["Área de atención"]=le.fit_transform(Df_train["Área de atención"])
```

**4. Selección de variables predictoras**

Para seleccionar las variables que se van a utilizar en el modelo, se tuvo que recurrir a la correlacion que tiene todas las features con la ***Target*** que es ***tipo_estancia*** dicha correlación se represento por medio de un headmap.
```python
corr=Df_train.corr()

plt.figure(figsize=(13,8))
sns.heatmap(corr, cbar = True,  square = False, annot=True, fmt= '.2f',cmap= 'coolwarm')
```
Se pude ver en [codigo_completo](https://github.com/Alvaro9721/PI_2_Machine_Learning/blob/main/PI_2_Machine_learning.ipynb).



**5. Creación del modelo**

Para escoger el tipo de modelo se tuvo en cuenta que el hospital ya nos estaba dando las etiqueta(estancia prolongada o no) lo cual permite saber que estamos hablando de un modelo de aprendiza supervisado, otro caractrística importante es que se tienen variables binarias y se tiene que clasificar, lo cual descarta al modelo de regresión.

Se decide trabajar con el modelo de ***Arbol de decisión*** por su versatilidad, ademas de esto, todos los datos se convirtieron a **INT** y este modelo es muy útil para analizar datos cuantitativos y tomar una decisión basada en números, para escoger el hiperparametro se realizó una validación cruzada con que evaluo el modelo 5 veces cada profundida las cuales estaban en un rango de (1-20), gracias a esto se pudo obtener la mejor profundidad la cual es 4.

Despues de instanciar, evaluar, predecir el modelo se obtuvieron los siguientes resuldos que fueron exportados a un archivo .CSV y posteriormente enviados a la unidad hospitalaria.

```python
#predecimos con el modelo entrenado antiormente
y_pred_new= mod_1.predict(X_new)

prediccion=pd.DataFrame(y_pred_new.reshape(-1,1),columns=["pred"])
prediccion.head(5)

pred
0	1
1	1
2	1
3	1
4	1

```

```python
#Se exporta a .CSV

prediccion.to_csv("Alvaro9721.csv",index=False)
```

<hr>


<p align=center>
<img src = 'https://blog.soyhenry.com/content/images/2021/05/PRESENTACION-3.jpg' height=250><p>

