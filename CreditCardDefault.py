# Databricks notebook source
file_path = "dbfs:/FileStore/tables/creditCard/"

#Luetaan CSV
df = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load(file_path)

#Näytetään ensimmäiset rivit
display(df)
#Tarkistetaan datan rakenne
df.printSchema()

# COMMAND ----------

from pyspark.sql.functions import col, when, count, isnull

#Tarkistetaan sarakkeiden nimet
print(df.columns)
#Viimeinen sarake on ongelmallinen. Tarkistetaan sen sisältö
df.select(col("`default.payment.next.month`")).show(10)
#Sarake sisältää vain 0 ja 1, joten se on erittäin tärkeä. Nimetään sarake uudelleen.
df = df.withColumnRenamed("default.payment.next.month", "default_payment_next_month")
#Varmistetaan vielä ettei muita lukuja esiinny kuin 0 ja 1
df.select("default_payment_next_month").summary("count", "min", "max").show()

#Tarkistetaan onko puuttuvia arvoja ja poistetaan ne
df.select([count(when(isnull(c), c)).alias(c) for c in df.columns]).show()
df_clean =df.dropna()


# COMMAND ----------

import matplotlib.pyplot as plt
import seaborn as sns

#Muutetaan Spark DataFrame Pandas-muotoon visualisointia varten
pdf = df.toPandas()

#Histogrammi maksuhäiriöistä
sns.countplot(x="default_payment_next_month", data=pdf)
plt.title("Maksuhäiriöiden jakauma")
plt.xlabel("Maksuhäiriö seuraavana kuukautena (0 = ei, 1 = kyllä)")
plt.ylabel("Lukumäärä")
plt.show()

# COMMAND ----------

#Luottorajan ja maksuhäiriöiden välinen yhteys
sns.boxplot(x="default_payment_next_month", y="LIMIT_BAL", data=pdf)
plt.title("Luottorajan vaikutus maksuhäiriöihin")
plt.xlabel("Maksuhäiriö (0 = ei, 1 = kyllä)")
plt.ylabel("Luottoraja")
plt.show()
#-->Matalampi luottoraja on yhteydessä suurempaan maksuhäiriöriskiin


# COMMAND ----------

#Korrelaatio eri muuttujien välillä
plt.figure(figsize=(12, 6))
sns.heatmap(pdf.corr(), annot=True, fmt=" .2f", cmap="coolwarm")
plt.title("Korrelaatiomatriisi")
plt.show()
#-->Maksuhäiriön paras ennustaja on aiempi maksukäyttäytyminen

# COMMAND ----------

#Ennustusmallin rakentaminen
from pyspark.ml.feature import VectorAssembler

#Valitaan malliin tulevat muuttujat
feature_columns = ['LIMIT_BAL', 'AGE', 'PAY_0', 'PAY_2', 'PAY_3', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3']
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
df_transformed = assembler.transform(df)

#Jaetaan data opetus- ja testidataan
train_data, test_data = df_transformed.randomSplit([0.8, 0.2], seed=42)


# COMMAND ----------

#Koulutetaan logistinen regressiomalli
from pyspark.ml.classification import LogisticRegression

lr = LogisticRegression(featuresCol="features", labelCol="default_payment_next_month")
model = lr.fit(train_data)
predictions = model.transform(test_data)
predictions.select("default_payment_next_month").show(10)

# COMMAND ----------

#Mallin tarkkuuden arviointi (eli oikein ennustettujen tapausten osuus kaikista testidatan tapauksista)
from pyspark.ml.evaluation import BinaryClassificationEvaluator
evaluator =BinaryClassificationEvaluator(labelCol="default_payment_next_month")
accuracy = evaluator.evaluate(predictions)
print(f"Logistisen regression tarkkuus: {accuracy:.2f}")
#-->Malli ennustaa maksuhäiriöt oikein 72% ajasta, eli suurin osa asiakkaista ei saa maksuhäiriötä. Malli voi saavuttaa korkean tarkkuuden vain arvaamalla "ei maksuhäiriötä" useimmiten, joten kannattaa tarkistella muitakin mittareita.

# COMMAND ----------

#Kokeillaan päätöspuuta (Decision Tree), joka jakaa datan haarautumalla eri päätöksiin
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

#Luodaan päätöpuumalli
dt = DecisionTreeClassifier(labelCol="default_payment_next_month", featuresCol="features")
#Sovitetaan malli
dt_model = dt.fit(train_data)
#Tehdään ennuste testidatalla
dt_predictions = dt_model.transform(test_data)
#Arvioidaan tarkkuus
dt_evaluator = MulticlassClassificationEvaluator(labelCol="default_payment_next_month", metricName="accuracy")
dt_accuracy = dt_evaluator.evaluate(dt_predictions)

print(f"Päätöspuun tarkkuus: {dt_accuracy:.2f}")
#-->Päätöspuun tarkkuus on parempi kuin logistisen regression, eli se voi olla parempi malli.
#Tarkistetaan mitkä muuttujat vaikuttavat eniten
dt_model.featureImportances

# COMMAND ----------

#Satunnaismetsän (Random Forest) kokeileminen. Yhdistää useita päätöspuita ja tekee ennusteet niiden enemmistöpäätöksellä.
from pyspark.ml.classification import RandomForestClassifier

#Luodaan satunnaismetsämalli
rf = RandomForestClassifier(labelCol="default_payment_next_month", featuresCol="features", numTrees=100)
#Sovitetaan malli
rf_model = rf.fit(train_data)
#Tehdään ennuste testidatalla
rf_predictions = rf_model.transform(test_data)
#Arvioidaan tarkkuus
rf_evaluator = MulticlassClassificationEvaluator(labelCol="default_payment_next_month", metricName="accuracy")
rf_accuracy = rf_evaluator.evaluate(rf_predictions)

print(f"Satunnaismetsän tarkkuus: {dt_accuracy:.2f}")
#-->Tulos on sama kuin päätöspuun
