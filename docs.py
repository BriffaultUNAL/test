



class Perro:
    especie = 'mamífero'

    def __init__(self, nombre, raza):
        print(f"Creando perro {nombre}, {raza}")

        self.nombre = nombre
        self.raza = raza

    def ladra(self):
        print("Guau")

    def camina(self, pasos):
        print(f"Caminando {pasos} pasos")
        
mi_perro = Perro("Toby", "Bulldog")
mi_perro.ladra()
mi_perro.camina(10)

class Coffee:
        def __init__(self, name, price):
                self.name = name
                self.price = float(price)
        def check_budget(self, budget):
                if not isinstance(budget, (int, float)):
                        print('Enter float or int')
                        exit()
                if budget < 0: 
                    print('Sorry you don\'t have money') 
                    exit() 
        def get_change(self, budget):
                return budget - self.price
        
        def sell(self, budget):
                self.check_budget(budget)
                if budget >= self.price:
                        print(f'You can buy the {self.name} coffee')
                        if budget == self.price:
                                print('It\'s complete')
                        else:
                                print(f'Here is your change {self.get_change(budget)}$')

                        exit('Thanks for your transaction')        
                        
#df.loc[lambda X:X.Num_Estudiantes == 60]        

def measure_time(function):
    def wrapper(*args, **kwargs):
        import time

        start = time.time()
        result = function(*args, **kwargs)
        total = time.time() - start
        print(total, 'seconds' )
        return result

    return wrapper

#*args, **kwargs argunmentos dinamicos

@measure_time
def suma(a, b):
    import time
    time.sleep(1)
    return a + b

print(suma(10, 20))     

"""Paso	Detalle / Por qué es relevante
1 – Imports	Pipeline y ColumnTransformer aseguran que el mismo pre-procesamiento se aplique en entrenamiento y test, evitando data leakage.
2 – Simulación	Creamos un patrón realista: semanas cíclicas + campaña. Generar datos sintéticos agiliza pruebas sin exponer datos sensibles.
3 – Features	- dow y month capturan estacionalidad.
- is_q_end modela picos de fin de mes.
Las fechas casi siempre requieren ingeniería de calendario.
4 – Split temporal	En series de tiempo nunca barajes las filas; entrenas en pasado y pruebas en futuro. Por simplicidad usamos hold-out (30 días).
5 – Pre-procesamiento	- OneHotEncoder transforma categóricas en dummies; drop="first" evita multicolinealidad.
- RandomForest: buen baseline porque maneja no linealidad e interacciones sin escalar.
6 – Entrenamiento	.fit ajusta tanto el encoder como el modelo dentro de la misma tubería → una sola llamada.
7 – Evaluación	MAE (error absoluto medio) es fácil de explicar: “fallamos en ± X llamadas diarias”. Puedes comparar con tu margen de planificación.
8 – Importancia	Suma a 1; indica peso relativo de cada categoría (p. ej. si dow_0 ≃ 0.3, el lunes es crucial en la predicción). Comunica insights a Workforce Management.   
"""
# 1. ─── Imports básicos ────────────────────────────────────────────────
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# 2. ─── Simulamos 6 meses de datos diarios ────────────────────────────
rng = pd.date_range(start="2025-01-01", end="2025-06-30", freq="D")
n = len(rng)
df = pd.DataFrame({"date": rng})

# patr ón semanal: lunes pico, fines de semana baja
base_calls = 200 + 50*np.sin(2*np.pi*df["date"].dt.dayofweek/7)

# campaña de marketing en marzo eleva las llamadas
campaign = ((df["date"].dt.month == 3).astype(int) * 60)

# ruido aleatorio (Poisson aprox. llamadas)
df["calls"] = (base_calls + campaign + np.random.normal(0, 10, n)).round().astype(int)

# 3. ─── Feature engineering mínimo ────────────────────────────────────
df["dow"]      = df["date"].dt.dayofweek          # 0=lun … 6=dom
df["month"]    = df["date"].dt.month
df["is_q_end"] = (df["date"].dt.day > 25).astype(int)  # fin de mes

X = df[["dow", "month", "is_q_end"]]
y = df["calls"]

# 4. ─── Split train / test temporal (últimos 30 días = test) ──────────
X_train, X_test = X.iloc[:-30], X.iloc[-30:]
y_train, y_test = y.iloc[:-30], y.iloc[-30:]

# 5. ─── Pre-proceso: One-Hot a categóricas + modelo ───────────────────
cat_features = ["dow", "month", "is_q_end"]
preproc = ColumnTransformer(
    [("ohe", OneHotEncoder(drop="first"), cat_features)],
    remainder="passthrough"  # no hay numéricas extra, pero ilustrativo
)

model = RandomForestRegressor(
    n_estimators=300, random_state=42)  # árbol ensamble robusto

pipe = Pipeline(
    steps=[("prep", preproc),
           ("rf",   model)]
)

# 6. ─── Entrenamiento ────────────────────────────────────────────────
pipe.fit(X_train, y_train)

# 7. ─── Predicción y evaluación ──────────────────────────────────────
y_pred = pipe.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"MAE en test: {mae:.1f} llamadas/día")

# 8. ─── (Opcional) importancia de variables ──────────────────────────
importances = pipe.named_steps["rf"].feature_importances_
feature_names = pipe.named_steps["prep"].get_feature_names_out()
print(pd.Series(importances, index=feature_names).sort_values(ascending=False))        


"""Paso	Qué hace / Por qué es importante
1 – Imports	Incluimos ColumnTransformer y Pipeline para empaquetar todo el flujo (evita data leakage).
2 – Dataset sintético	- Reproduce causas reales de variación (tipo de llamada, experiencia, etc.).
- Añadimos ruido para emular variabilidad natural.
3 – X & y	Mantener las features separadas del target es crucial para evitar errores al escalar/encodear.
4 – Split de validación	Usamos stratify en call_type para conservar proporciones entre train y test.
En la vida real, con timestamp usarías un split temporal.
5 – Pre-procesamiento	- One-Hot Encoding convierte categóricas en columnas 0/1.
- StandardScaler centra y escala numéricas (GBR no lo exige, pero facilita comparaciones e importa en otros modelos).
6 – Modelo	GradientBoostingRegressor captura interacciones y no linealidad (e.g., impacto distinto del ivr_steps según call_type).
7 – Entrenar	pipe.fit ajusta todo de una vez: primero procesa X, luego entrena el modelo → misma lógica en predict().
8 – MAE	Error absoluto en segundos es intuitivo para supervisores: “fallamos ±24 s por llamada promedio”. Cambia a RMSE si necesitas penalizar más los errores grandes.
9 – Importancia	Muestra las features top: tip → agrupa columnas dummy del mismo campo para una vista de negocio (“call_type explica ≈40 % de la predicción”)."""


# 1. ─── Imports ───────────────────────────────────────────────────────
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

# 2. ─── Creamos un dataset sintético de 5 000 llamadas ───────────────
n = 5_000
rng = np.random.default_rng(42)

df = pd.DataFrame({
    "call_type": rng.choice(["billing", "tech", "upgrade"], n, p=[.5, .3, .2]),
    "customer_tier": rng.choice(["VIP", "Regular"], n, p=[.15, .85]),
    "agent_tenure_months": rng.integers(1, 60, n),     # experiencia agente
    "ivr_steps": rng.integers(1, 8, n),                # menús recorridos
    "first_call": rng.choice([0, 1], n, p=[.7, .3])    # primer contacto del día
})

# Duración base (seg) según tipo + ruido y efectos
base = (df["call_type"].map({"billing": 300, "tech": 420, "upgrade": 500})
        + df["ivr_steps"] * 15                       # cada menú añade ~15 s
        - df["agent_tenure_months"] * 2              # agentes expertos más rápidos
        + df["customer_tier"].map({"VIP": -30, "Regular": 0})
        + df["first_call"]  * 40                     # clientes nuevos tardan más
        + rng.normal(0, 25, n))                      # ruido gaussiano

df["duration_sec"] = base.round().astype(int)

# 3. ─── Seleccionamos features (X) y target (y) ───────────────────────
X = df.drop(columns="duration_sec")
y = df["duration_sec"]

# 4. ─── Hold-out temporal (últimos 10 %) ───────────────────────────────
# (si tuvieras timestamp real usarías split por fecha)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.10, random_state=42, stratify=df["call_type"]
)

# 5. ─── Pre-proceso: OHE para categóricas, escala numéricas ────────────
cat_cols = ["call_type", "customer_tier", "first_call"]
num_cols = ["agent_tenure_months", "ivr_steps"]

preproc = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(drop="first"), cat_cols),
        ("num", StandardScaler(),            num_cols)
    ]
)

# 6. ─── Modelo: Gradient Boosting (maneja no linealidad & interacción) ─
gbr = GradientBoostingRegressor(random_state=42)

pipe = Pipeline(steps=[("prep", preproc),
                      ("gbr",  gbr)])

# 7. ─── Entrenar ──────────────────────────────────────────────────────
pipe.fit(X_train, y_train)

# 8. ─── Evaluar ───────────────────────────────────────────────────────
y_pred = pipe.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"MAE: {mae:.1f} s — error medio absoluto (AHT)")

# 9. ─── Importancia global de variables ───────────────────────────────
importances = pipe.named_steps["gbr"].feature_importances_
feat_names  = pipe.named_steps["prep"].get_feature_names_out()
print(pd.Series(importances, index=feat_names)
        .sort_values(ascending=False)
        .head(8))


"""Paso	Qué hace / Concepto clave
1 – Imports	RandomForestClassifier gestiona relaciones no lineales y requiere poco tuning; class_weight="balanced" corrige el desbalance típico (pocos abandonos).
2 – Dataset sintético	Incluimos variables con causalidad plausible: ausencias recientes, turnos nocturnos, tiempo desde el último aumento, etc. prob_quit mezcla reglas simples para generar la etiqueta churn_risk.
3 – Separación X / y	Mantener target aparte evita filtrarlo accidentalmente en los features.
4 – train_test_split estratificado	Se asegura que la proporción de churners sea la misma en ambos conjuntos ⇒ estimación más estable.
5 – ColumnTransformer	- One-Hot Encoding: convierte categorías en columnas binarias (elimina problema de orden).
- StandardScaler: centra/escala numéricas → mejora estabilidad del bosque y es obligatorio si luego pruebas modelos lineales.
6 – Modelo	- 400 árboles reducen varianza.
- class_weight distribuye “peso” extra a la clase minoritaria (churn=1) y aumenta recall.
7 – Entrenamiento	.fit aplica el pipeline completo en un solo llamado ⇒ evita data-leakage (el scaler sólo “ve” train).
8 – Evaluación	- ROC-AUC: probabilidad de que el modelo rankee al azar dos agentes (uno churner, uno no) correctamente; > 0.75 ya es valioso.
- classification_report imprime precisión, recall, F1 de cada clase → revisa que el recall de churn ≥ 0.6 (mejor capturar la mayoría de riesgos).
9 – Importancia de features	Comunicas al HR que “las ausencias en los últimos 90 días y el desempeño bajo explican el 45 % del riesgo”, facilitando plan de acción."""


import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report

# 2. ─── Dataset sintético de 3 000 agentes ───────────────────────────
n = 3_000
rng = np.random.default_rng(7)

df = pd.DataFrame({
    "tenure_months"   : rng.integers(1, 48, n),                    # tiempo en la empresa
    "absences_90d"    : rng.integers(0, 6, n),                    # inasistencias recientes
    "shift_type"      : rng.choice(["day", "night"], n, p=[.7, .3]),
    "performance_q"   : rng.choice(["high", "mid", "low"], n, p=[.3, .5, .2]),
    "schedule_adher%" : rng.uniform(70, 105, n).round(1),          # % adherencia al horario
    "last_raise_mths" : rng.integers(0, 24, n),                   # meses desde último aumento
})

# Regla de negocio para la etiqueta (riesgo de absentismo / rotación)
prob_quit = (
      0.05
    + 0.04 * (df["absences_90d"] >= 3)
    + 0.06 * (df["performance_q"] == "low")
    + 0.03 * (df["shift_type"] == "night")
    + 0.03 * (df["last_raise_mths"] > 12)
    - 0.02 * (df["tenure_months"] > 24)
).clip(0, 1)

df["churn_risk"] = rng.binomial(1, prob_quit)   # 1 = alto riesgo

# 3. ─── X (features) y y (target) ────────────────────────────────────
X = df.drop(columns="churn_risk")
y = df["churn_risk"]

# 4. ─── Train / Test (estratificado para balancear la clase) ─────────
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# 5. ─── Pre-procesamiento: categóricas → OHE, numéricas → escala ─────
cat_cols = ["shift_type", "performance_q"]
num_cols = ["tenure_months", "absences_90d", "schedule_adher%", "last_raise_mths"]

preproc = ColumnTransformer(
    [("cat", OneHotEncoder(drop="first"), cat_cols),
     ("num", StandardScaler(),           num_cols)]
)

# 6. ─── Modelo: Random Forest (robusto, interpretabilidad básica) ────
rf = RandomForestClassifier(
        n_estimators=400,
        class_weight="balanced",   # compensa clases desbalanceadas
        random_state=42)

pipe = Pipeline([("prep", preproc),
                 ("clf",  rf)])

# 7. ─── Entrena ──────────────────────────────────────────────────────
pipe.fit(X_tr, y_tr)

# 8. ─── Evalúa ───────────────────────────────────────────────────────
proba_te = pipe.predict_proba(X_te)[:, 1]
pred_te  = pipe.predict(X_te)

print("ROC-AUC:", roc_auc_score(y_te, proba_te).round(3))
print(classification_report(y_te, pred_te, digits=3))

# 9. ─── Importancia de variables ─────────────────────────────────────
imp  = pipe.named_steps["clf"].feature_importances_
cols = pipe.named_steps["prep"].get_feature_names_out()
print(pd.Series(imp, index=cols)
        .sort_values(ascending=False)
        .head(10))




"""Paso	Preguntas clave	Artefacto entregable	Ejemplo concreto en un BPO
1. Alinear métricas	¿Qué KPI duele más (coste, NPS, SL A)?	Cuadro “técnico → negocio”	Un MAE de ±25 llamadas/h equivale a 2 agentes de sobre-o sub-staffing.
2. Segmentar decisiones	¿A quién o qué proceso afecto?	Regla de decisión (threshold, ranking)	Si prob_churn ≥ 0.6 → supervisor recibe alerta; 0.4–0.6 → lista de seguimiento.
3. Integrar en el flujo	¿Dónde vive el modelo? ¿Batch o real-time?	Endpoint REST, SQL view o archivo CSV	Pronóstico de volumen se exporta cada tarde al sistema WFM (Aspect, Verint).
4. Automatizar acción mínima	¿Puede ejecutarse sin mano humana?	Script, webhook, bot	API reprograma turnos; IVR enruta llamadas largas a agentes senior.
5. Medir el efecto	¿Qué varía después?	Dashboard A/B u “antes-después”	Tiempos de espera se reducen 15 %; coste de horas extra baja COP 180 M.
6. Cerrar el loop	¿El modelo sigue acertando?	Monitorización de drift + retrain	Si MAPE > 8 % durante 3 días, se dispara retrain semanal."""


"""2 ⟶ Explicar el “por qué” en lenguaje sencillo
2.1 Feature Importance (global)
¿Qué es?	Puntuación que resume cuánto contribuye cada variable al poder predictivo total del modelo.
Cómo se calcula	Árboles: caída de impureza (Gini/MSE)
Permutación: embarajar una columna y medir cuánto empeora el error.
Cómo contarlo al negocio	“Si quitáramos la variable absences_90d, nuestro error subiría un 12 %. Por tanto, las ausencias recientes son la palanca más fuerte.”
Visual típico	Barras horizontales ordenadas (top-10). Colores verde → impacto positivo, rojo → negativo (si aplicas SHAP global).

Analogía: “Tenemos un equipo de jugadores (features). La importancia global mide cuántos goles marca cada uno en toda la temporada.”

2.2 SHAP (local + global)
Concepto	Explicación para no técnicos
SHAP value	“Cuántos puntos suma o resta cada variable al resultado final de esta fila concreta comparado con el promedio.”
Base value	El “promedio de goles” de un partido hipotético con todos los equipos mezclados.
Propiedad clave	Las contribuciones SIEMPRE suman exactamente la predicción (transparencia contable).
Visuales	• Waterfall: barra apilada por una sola observación.
• Beeswarm: nube coloreada (todas las observaciones) para ver dirección e intensidad.

Metáfora del pastel
Imagina que la predicción es un pastel entero (porción de 550 s AHT). SHAP reparte el pastel en porciones y etiqueta cada trozo con el ingrediente responsable:
+150 s porque es call_type = tech
+80 s porque ivr_steps = 7
-60 s porque agent_tenure = 40 meses
… hasta completar los 550 s."""

"""5.1 Buenas prácticas de privacidad y ética en analítica
Pilar	Qué significa en la práctica	Herramientas / controles
Minimización	Recoge solo los campos estrictamente necesarios para el caso de uso.	Data-mapping previo, checklist de campos “must-have”.
Anonimización / seudonimización	Reemplaza identificadores directos (cédula, e-mail) por claves internas, y separa la tabla de equivalencias en un vault.	Hash + salt, tokenización, servicios KMS (AWS KMS, Azure Key Vault).
Propósito claro & consentimiento	Define la finalidad (e.g., “mejorar tiempos de espera”) y comunícala al cliente o a RR. HH.	Notices “just-in-time”, cláusulas de consentimiento, opt-out.
Limitación de retención	Establece ventanas: logs de transcripciones ≤ 90 días, métricas agregadas 2 años.	Políticas de borrado automático (lifecycle rules S3, GCS).
Seguridad y acceso mínimo	Políticas IAM basadas en rol; cifrado en reposo y en tránsito.	IAM, VPC, TLS, row-level security.
No maleficencia & equidad	Evalúa si el modelo perjudica a grupos protegidos (sexo, edad).	Métricas de equidad (disparate impact), auditorías externas.
Transparencia	Documenta lógica del modelo y los datos usados; proporciona explicaciones (SHAP) a usuarios afectados.	Model cards, data sheets, panel FAQ.

Marco rápido (5 preguntas para cada nuevo proyecto)

¿Qué datos personales requiere?

¿Podemos cumplir el objetivo con menos datos?

¿Durante cuánto tiempo son necesarios?

¿Quién tendrá acceso y bajo qué rol?

¿Qué daño potencial hay si el modelo se equivoca o se filtran los datos?

5.2 Manejo de datos incompletos o erróneos
Paso	Buen hábito	Detalle técnico
1. Perfilado inicial	Usa df.info(), df.describe(), tests de nulls/outliers, data profiling automático (Great Expectations, Soda).	Distingue entre MCAR, MAR, MNAR para decidir imputación.
2. Regla de negocio	Define qué es un valor “posible” (ej. AHT < 5 s o > 5 000 s = error).	Catálogo de “rango válido” por campo.
3. Depuración	Corrige formatos, unidades, fechas imposibles; crea columnas limpias, no sobre-escribe el original.	to_datetime(errors='coerce'), parseadores robustos, casting explícito.
4. Imputación	- Simple: media/mediana para numéricos, moda para categóricas (< 5 % faltante).
- Avanzada: KNNImputer, IterativeImputer (MICE), missForest.	Guarda flag is_imputed para trazabilidad.
5. Revisión de sesgo	Comprueba que la imputación no altere la distribución o favorezca a un grupo.	Comparación de histogramas antes/después, pruebas KS.
6. Pipeline reproducible	Automatiza limpieza e imputación en sklearn.Pipeline o Spark ETL para evitar código “en cuaderno suelto”.	Versiona artefactos en DVC o LakeFS.

5.3 Principios de gobernanza de datos y modelos predictivos
Dominio	Qué controlar	Práctica recomendada
Lineaje	Desde la ingesta hasta el dashboard.	Metadata automática (OpenLineage, Airflow + Marquez).
Catálogo & Glossary	Definición única por campo (“AHT”, “After-Call Work”).	Data Catalog (AWS Glue, Azure Purview), Business Glossary.
Versionado de datos	Cambios en esquema y particiones.	Delta Lake / Iceberg con control git-like.
Registro de modelos	Versiones, métricas, costumbre de envío a producción.	MLflow Model Registry, Evidently AI.
Políticas de acceso	Quién puede ver PII, quién puede re-entrenar.	RBAC, columnas enmascaradas, policy-as-code (OPA, Lake Formation).
Auditoría & logging	Query logs, inferencias servidas, drift.	CloudTrail, BigQuery Audit, Prometheus + Grafana.
Ciclo de vida	Caducidad de features y modelos; plan de retrain.	Cron Airflow, triggers por concept drift (> Δ error 8 %)."""

"""1. Conjuntos de Entrenamiento, Validación y Test
Conjunto	Qué se hace con él	Por qué es necesario
Entrenamiento (train)	El algoritmo ajusta sus parámetros internos (β de la regresión, pesos de una red, etc.).	El modelo “aprende” aquí; ver los demás conjuntos produciría fuga de información (data leakage).
Validación (val, dev)	Se *sintonizan los hiper-parámetros (n árboles, profundidad, λ de regularización).	Evita que el modelo se “haga trampa” optimizando contra el test.
Test (hold-out final)	Se calcula la métrica que vas a reportar al negocio o al artículo.	Simula rendimiento en datos futuros nunca vistos. Una sola mirada; no se re-entrena después.

Regla de oro: todo paso que calcule un número (media, varianza, umbral, pesos) sólo se ajusta con train y se reutiliza en val/test.

2. Métodos de Validación
2.1 Hold-out simple
python
Copiar
Editar"""
from sklearn.model_selection import train_test_split
X_train, X_tmp, y_train, y_tmp = train_test_split(X, y,
                                                  test_size=0.3,
                                                  random_state=42,
                                                  stratify=y)  # para clasifs. desequilibradas
X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp,
                                                test_size=0.5,
                                                random_state=42,
                                                stratify=y_tmp)
"""Ventaja: rápido; perfecto cuando tienes millones de filas.

Desventaja: un solo corte puede ser poco representativo → alta varianza de la métrica si el dataset es pequeño o muy heterogéneo.

2.2 K-Fold Cross-Validation (CV)
python
Copiar
Editar"""
from sklearn.model_selection import KFold, cross_validate
kf   = KFold(n_splits=5, shuffle=True, random_state=42)
scor = cross_validate(modelo, X, y,
                      cv=kf,
                      scoring=['neg_mean_absolute_error',
                               'r2'],
                      return_train_score=True)
print("MAE CV      :", -scor['test_neg_mean_absolute_error'].mean())
print("MAE std     :",  scor['test_neg_mean_absolute_error'].std())
print("Train vs Val:", -scor['train_neg_mean_absolute_error'].mean())
"""Variación	Cuándo usarla	Detalle
StratifiedKFold	Clasificación con clases raras.	Mantiene proporciones de cada clase en cada fold.
GroupKFold	Datos con unidades repetidas (cliente, agente).	Ningún “grupo” puede quedar repartido entre train y val. Evita leakage por identidad.
TimeSeriesSplit	Series temporales.	Respeta el orden cronológico: los folds son ventanas crecientes de pasado → futuro.

Ventaja: usa todos los datos para validar (repetidas veces) → menor varianza de la estimación final.
Costo: multiplica tiempo de entrenamiento por k.

3. Overfitting, Underfitting y la Curva Sesgo-Varianza
Situación	Síntomas	Gráfica típica	Soluciones
Underfitting	Alto error en train y val.	Curvas planas y altas.	Modelo más complejo, más features, transformar variables.
Overfitting	Error muy bajo en train, alto en val/test.	Gran separación entre curvas.	Regularización, más datos, limitar complejidad, usar CV/early-stopping.

3.1 Regularización → penalizar la complejidad
Modelo	Penalización	Hiper-parámetro
Ridge (L2)	∑β²	alpha
Lasso (L1)	∑	β
Elastic Net	mezcla L1+L2	alpha, l1_ratio
Árboles / RF	limitar profundidad, hojas mínimas	max_depth, min_samples_leaf
Boosting (XGBoost, LightGBM)	eta (learning rate), max_depth, subsample	—
Redes	Dropout, weight decay	tasa dropout, λ

Ejemplo en 6 líneas: Grid Search + CV
python
Copiar
Editar"""
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
param = {"alpha": np.logspace(-3, 2, 6)}
grid  = GridSearchCV(Ridge(), param, cv=5,
                     scoring="neg_root_mean_squared_error")
grid.fit(X_train, y_train)
print("α óptimo:", grid.best_params_)
print("RMSE CV :", -grid.best_score_)
4. Combinar todo en una Pipeline reproducible
python
Copiar
Editar
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso

pipe = Pipeline([
    ("scale", StandardScaler()),      # se ajusta sólo en train
    ("model", Lasso(alpha=0.1))
])
"""El Pipeline se entrena y valida completo dentro de cada fold → garantiza que el escalador no “huela” datos de validación.

En producción llamas pipe.predict(new_data) y obtienes la misma transformación.

5. Preguntas que podrían caerte en la prueba (y cómo responder)
Pregunta del entrevistador	Respuesta “de libro” (≤ 30 s)
¿Por qué no usar sólo train y test?	Ajustar hiper-parámetros mirando el test hace que lo “consumas”. Necesitas un set aparte o CV.
¿Cómo eliges k en K-Fold?	5 ó 10 en datasets grandes; Leave-One-Group-Out si hay pocos grupos; TimeSeriesSplit para tiempos.
¿Qué pasa si mi modelo es lento?	Usa ShuffleSplit con 3-4 folds aleatorios o hold-out estratificado; paraleliza CV (n_jobs=-1).
¿Cómo detectas data leakage?	Métrica inusualmente alta, features con alta correlación con el target obtenida post-evento (e.g. fecha_cierre al predecir duración). Evito esto en Pipeline.
¿Regularización siempre mejora?	Reduce varianza a costa de algo de sesgo. Si ya estás under-fitting, hará el error mayor; observo las curvas."""

"""1. Separar datos y entrenar el modelo más simple (baseline)
Qué demuestra	Código
Comprensión de la API «fit / predict» y de la métrica adecuada	python\nfrom sklearn.datasets import load_iris\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.metrics import accuracy_score\nfrom sklearn.tree import DecisionTreeClassifier\n\nX, y = load_iris(return_X_y=True)\nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n\nclf = DecisionTreeClassifier(random_state=42)\nclf.fit(X_train, y_train)\nprint(\"Acc:\", accuracy_score(y_test, clf.predict(X_test)))\n

2. Preprocesamiento + modelo en una Pipeline (evitar leakage)
Qué demuestra	Código
Uso de StandardScaler, buenas prácticas de ML productivo	python\nfrom sklearn.pipeline import Pipeline\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.linear_model import LogisticRegression\n\npipe = Pipeline([\n (\"scaler\", StandardScaler()),\n (\"model\", LogisticRegression(max_iter=1000))\n])\npipe.fit(X_train, y_train)\n

3. ColumnTransformer para datos mixtos (numéricos + categóricos)
Qué demuestra	Código
Manejo de features heterogéneas y transformaciones específicas	python\nimport pandas as pd\nfrom sklearn.compose import ColumnTransformer\nfrom sklearn.preprocessing import OneHotEncoder, StandardScaler\nfrom sklearn.linear_model import Ridge\n\n# Ejemplo artificial\ndf = pd.DataFrame({\n \"age\": [23, 45, 31, 35],\n \"income\": [40_000, 90_000, 60_000, 70_000],\n \"city\": [\"Bogotá\", \"Medellín\", \"Cali\", \"Bogotá\"],\n \"spend\": [300, 800, 400, 560]\n})\nX = df.drop(\"spend\", axis=1)\ny = df[\"spend\"]\n\nnumeric = [\"age\", \"income\"]\ncategorical = [\"city\"]\n\npre = ColumnTransformer([\n (\"num\", StandardScaler(), numeric),\n (\"cat\", OneHotEncoder(handle_unknown=\"ignore\"), categorical)\n])\n\nmodel = Pipeline([\n (\"pre\", pre),\n (\"reg\", Ridge(alpha=1.0))\n])\nmodel.fit(X, y)\n

4. Cross-validation rápida con cross_val_score
Qué demuestra	Código
Comprensión de varianza/overfitting y reproducibilidad (random_state)	python\nfrom sklearn.model_selection import cross_val_score\nimport numpy as np\n\nscores = cross_val_score(pipe, X, y, cv=5, scoring=\"accuracy\")\nprint(\"CV mean ± std:\", np.mean(scores), \"+/-\", np.std(scores))\n

5. GridSearchCV (búsqueda de hiperparámetros)
Qué demuestra	Código
Capacidad de exprimir rendimiento sin data leakage	python\nfrom sklearn.model_selection import GridSearchCV\n\nparam_grid = {\n \"model__C\": [0.1, 1, 10],\n \"model__penalty\": [\"l2\"],\n}\n\ngs = GridSearchCV(pipe, param_grid, cv=5, scoring=\"accuracy\", n_jobs=-1)\ngs.fit(X, y)\nprint(\"Best score:\", gs.best_score_)\nprint(\"Best params:\", gs.best_params_)\n

6. Métricas de clasificación avanzadas
Qué demuestra	Código
Conocer cuándo Accuracy no basta, y saber generar informes	python\nfrom sklearn.metrics import classification_report, confusion_matrix\n\nprint(confusion_matrix(y_test, pipe.predict(X_test)))\nprint(classification_report(y_test, pipe.predict(X_test)))\n

7. Guardar y cargar modelos (joblib)
Qué demuestra	Código
Reproducibilidad y deployment básico	python\nimport joblib\n\njoblib.dump(pipe, \"model.pkl\")\nmodel_loaded = joblib.load(\"model.pkl\")\nprint(model_loaded.predict(X_test[:3]))\n

8. Aprendizaje incremental con partial_fit
Qué demuestra	Código
Manejo de streams / big data y del formato partial_fit	python\nfrom sklearn.linear_model import SGDClassifier\nfrom sklearn.datasets import load_digits\nfrom sklearn.preprocessing import StandardScaler\n\nX, y = load_digits(return_X_y=True)\nscaler = StandardScaler()\nX = scaler.fit_transform(X)\n\nsgd = SGDClassifier(loss=\"log_loss\", random_state=42)\nclasses = np.unique(y)\n\nfor i in range(0, len(X), 256): # mini-batches\n sgd.partial_fit(X[i:i+256], y[i:i+256], classes=classes)\n"""

1. Línea básica con título y ejes
python
Copiar
Editar
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.figure()                         # nueva figura
plt.plot(x, y, linewidth=2)
plt.title("Seno simple")
plt.xlabel("x")
plt.ylabel("sin(x)")
plt.grid(True)
plt.show()
2. Varias líneas y leyenda
python
Copiar
Editar
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)
plt.figure()
plt.plot(x, np.sin(x), label="sin(x)")
plt.plot(x, np.cos(x), linestyle="--", label="cos(x)")
plt.legend()
plt.tight_layout()
plt.show()
3. Dispersión con mapeo de colores y barra de color
python
Copiar
Editar
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)
x = np.random.randn(200)
y = np.random.randn(200)
colors = np.sqrt(x**2 + y**2)        # magnitud

plt.figure()
sc = plt.scatter(x, y, c=colors, alpha=0.8)
plt.colorbar(sc, label="Magnitud")
plt.title("Scatter coloreado por magnitud")
plt.show()
4. Histograma + curva de densidad (kernel)
python
Copiar
Editar
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

data = np.random.normal(loc=0, scale=1, size=1000)

plt.figure()
plt.hist(data, bins=30, density=True, alpha=0.5)

kde = gaussian_kde(data)
x = np.linspace(data.min(), data.max(), 200)
plt.plot(x, kde(x), linewidth=2)
plt.title("Histograma con KDE")
plt.show()
5. Mosaico de subplots
python
Copiar
Editar
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(2, 2, figsize=(8, 6))

x = np.linspace(0, 2*np.pi, 100)
axes[0, 0].plot(x, np.sin(x));  axes[0, 0].set_title("sin")
axes[0, 1].plot(x, np.cos(x));  axes[0, 1].set_title("cos")
axes[1, 0].plot(x, np.tan(x));  axes[1, 0].set_title("tan")
axes[1, 1].plot(x, np.exp(x));  axes[1, 1].set_title("exp")

fig.tight_layout()
plt.show()
6. Barras con barras de error
python
Copiar
Editar
import matplotlib.pyplot as plt
import numpy as np

labels = ["A", "B", "C", "D"]
values = [20, 34, 30, 35]
errors = [2, 3, 4, 1]

plt.figure()
plt.bar(labels, values, yerr=errors, capsize=5)
plt.ylabel("Puntuación")
plt.title("Medias ± error estándar")
plt.show()
7. Boxplot por grupo categórico
python
Copiar
Editar
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(0)
group1 = np.random.normal(50, 10, size=200)
group2 = np.random.normal(55, 15, size=200)

plt.figure()
plt.boxplot([group1, group2], labels=["Grupo 1", "Grupo 2"])
plt.ylabel("Valor")
plt.title("Comparación de distribuciones")
plt.show()
8. Serie temporal con anotación
python
Copiar
Editar
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

dates = pd.date_range("2025-01-01", periods=180)
values = np.cumsum(np.random.randn(180))  # paseo aleatorio

plt.figure(figsize=(10, 4))
plt.plot(dates, values)
plt.title("Serie temporal simulada")
plt.axvline(pd.Timestamp("2025-04-01"), linestyle="--")
plt.annotate("Evento clave", xy=("2025-04-01", values[90]),
             xytext=("2025-02-15", values.max()),
             arrowprops=dict(arrowstyle="->"))
plt.tight_layout()
plt.show()

"""1. ¿Qué son los hiperparámetros?
Parámetros ↔ se aprenden durante el entrenamiento (β en regresión, pesos de redes).

Hiperparámetros ↔ decides antes de entrenar: profundidad de árbol, alpha de Lasso, número de vecinos en K-NN, tasa de aprendizaje en boosting, etc.

Influyen directamente en sesgo y varianza ⇒ rendimiento.

Para encontrar la combinación óptima se recorre un espacio de búsqueda y se mide la métrica (MAE, ROC-AUC…) mediante validación cruzada.

2. Grid Search vs Random Search
Característica	Grid Search	Random Search
Cómo funciona	Prueba todas las combinaciones de la rejilla.	Muestra combinaciones aleatorias de distribuciones dadas.
Ventaja	Garantiza evaluar cada punto definido.	Cubre espacios grandes con muchos menos experimentos; encuentra buenas regiones rápido.
Cuándo usar	Pocos hiperparámetros y/o valores discretos (K-NN: k = [3,5,7,9]).	Rangos amplios o continuos (Random Forest: n_estimators 50-500; max_depth 3-30).
Costo	Crece exponencialmente; puede volverse inabordable.	Proporcional al nº de iteraciones que fijes.

Regla práctica: Grid para ≤ 3 hiperparámetros con ~10 × 10 × 5 puntos; Random (o técnicas bayesianas) para todo lo demás.

3. Ejemplo 1 — Grid Search con Ridge (regresión)
python
Copiar
Editar
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# X, y ya preparados ──────────
cv       = KFold(n_splits=5, shuffle=True, random_state=42)
pipeline = make_pipeline(StandardScaler(), Ridge())

param_grid = {
    "ridge__alpha": np.logspace(-3, 2, 6)     # 0.001 → 100
}

gcv = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="neg_mean_absolute_error",
        cv=cv,
        n_jobs=-1)      # usa todos los núcleos

gcv.fit(X, y)

print("Mejor α :", gcv.best_params_["ridge__alpha"])
print("MAE CV  :", -gcv.best_score_)
Léelo así

ridge__alpha → prefijo ridge porque el modelo está dentro del Pipeline.

logspace explora 6 potencias de 10; GridSearchCV probará 6 modelos × 5 folds = 30 entrenamientos.

best_params_ y best_estimator_ quedan listos para predict() sobre test o producción.

4. Ejemplo 2 — Random Search con Random Forest (clasificación)
python
Copiar
Editar
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

rf = RandomForestClassifier(class_weight="balanced", random_state=42)

param_dist = {
    "n_estimators" : np.arange(100, 801, 50),     # 100…800
    "max_depth"    : np.arange(4, 31),            # 4…30
    "min_samples_leaf": [1, 2, 4, 6],
    "max_features" : ["sqrt", "log2", 0.5]        # n_features**0.5, log2, 50 %
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

rs = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_dist,
        n_iter=60,                 # 60 combinaciones aleatorias
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        random_state=42)

rs.fit(X, y)

print("Mejores hiperparámetros:", rs.best_params_)
print("ROC-AUC promedio CV   :", rs.best_score_)
Por qué funciona
Aun con un espacio de ≈ 28 k combinaciones posibles, sólo evaluamos 60 ⇒ gran ahorro.

Para la mayoría de modelos, las primeras 50-100 pruebas aleatorias capturan > 90 % del beneficio, porque muchas combinaciones malas son descartadas.

5. Buenas prácticas
Pre-procesamiento dentro de Pipeline
Escaladores, imputadores, codificadores… así cada fold se ajusta sólo con datos de train (sin leakage).

Escoge la métrica correcta

Regresión: MAE, RMSE, R².

Clasificación desequilibrada: ROC-AUC, F1, PR-AUC.

Fija random_state
Para tener reproducibilidad en Random Search y modelos estocásticos.

n_jobs = -1
Aprovecha todos los núcleos de la máquina; Grid Search paralelo mitiga el coste.

Validación temporal
En series de tiempo, usa TimeSeriesSplit dentro de Grid/Random Search.

Salvaguarda contra overfitting
Después de encontrar los hiperparámetros óptimos, re-entrena el modelo final en (train + val) y evalúalo en test.

6. Pasos resumidos para tu prueba
Definir el espacio

Discreto → lista explicitada (Grid).

Continuo → rango + distribución (loguniform, randint).

Elegir método y CV
GridSearchCV o RandomizedSearchCV + KFold / StratifiedKFold / TimeSeriesSplit.

Lanzar búsqueda
fit() ejecuta el ciclo:
for hp_comb in space: for fold in cv: train → val → métrica

Inspeccionar resultados
cv_results_ contiene cada combinación, puedes graficar hiperparámetro ↔ score con pandas/matplotlib.

Exportar el mejor modelo
Serializa con joblib.dump(rs.best_estimator_, 'rf_best.pkl').

Con estas nociones y ejemplos podrás justificar qué método usar, cómo configurarlo y cómo interpretar los resultados en cualquier entrevista o prueba técnica. ¡A practicar!"""

"""Abrev.	Significado / Descripción breve
A/B	A/B Testing → experimento con 2 variantes (control A, tratamiento B).
AIC	Akaike Information Criterion → métrica para comparar modelos (penaliza complejidad).
ANN	Artificial Neural Network → red neuronal artificial.
API	Application Programming Interface → punto de acceso programable.
AR / MA / ARMA / ARIMA / SARIMA	AutoRegressive, Moving Average, combinaciones para series temporales; SARIMA añade estacionalidad.
AUC / ROC-AUC	Area Under the (ROC) Curve → probabilidad de rankeo correcto entre clases.
AWS	Amazon Web Services (plataforma cloud).
BIC	Bayesian Information Criterion → similar a AIC con mayor penalización.
BERT	Bidirectional Encoder Representations from Transformers (modelo NLP).
BI	Business Intelligence.
CI/CD	Continuous Integration / Continuous Deployment (pipeline de entrega).
CNN	Convolutional Neural Network (visión por computadora).
CV	Cross-Validation → validación cruzada.
DAG	Directed Acyclic Graph → flujo de tareas (Airflow, etc.).
DBMS	Database Management System.
DL	Deep Learning.
EDA	Exploratory Data Analysis → análisis exploratorio.
ELT / ETL	Extract-Load-Transform / Extract-Transform-Load (ingesta de datos).
ELO	Expected Log-likelihood Optimisation (o rating de habilidad; contexto depende).
EM / EMR	Expectation-Maximization / Elastic MapReduce (AWS).
ETS	Error-Trend-Seasonal (modelo Holt-Winters).
F1	Harmonic mean de precisión y recall.
FN / FP / TP / TN	False Negative / False Positive / True Positive / True Negative.
GAN	Generative Adversarial Network.
GDPR	General Data Protection Regulation (UE).
GPU	Graphics Processing Unit.
GRU	Gated Recurrent Unit (variante RNN).
GS / RS	Grid Search / Random Search (búsqueda de hiper-parámetros).
IAM	Identity and Access Management (control de permisos).
ICA	Independent Component Analysis (reducción de dimensionalidad).
K-Fold	Validación cruzada en k particiones.
KNN	K-Nearest Neighbors.
KS-test	Kolmogorov–Smirnov test (comparación de distribuciones).
L1 / L2	Normas de regularización (Lasso / Ridge).
LDA	Linear Discriminant Analysis (clasificación) o Latent Dirichlet Allocation (tópicos).
LIME	Local Interpretable Model-agnostic Explanations.
LOOCV	Leave-One-Out Cross-Validation.
LSTM	Long Short-Term Memory (RNN avanzada).
MAE / MAPE / MASE	Mean Absolute Error / Percentage Error / Scaled Error.
MLE	Maximum Likelihood Estimation.
MLOps	Operaciones de Machine Learning (ciclo de vida en producción).
MNAR / MAR / MCAR	Tipos de datos faltantes: Missing Not At Random / At Random / Completely At Random.
MSE / RMSE	Mean Squared Error / Root MSE.
NLP	Natural Language Processing.
OLS / GLS	Ordinary / Generalized Least Squares (regresión).
PCA	Principal Component Analysis.
PII	Personally Identifiable Information.
PR-AUC	Área bajo la curva Precision-Recall.
R² / R-adj²	Coefficient of Determination y su versión ajustada.
RBF	Radial Basis Function kernel (SVM).
RFE	Recursive Feature Elimination (selección de variables).
RF	Random Forest.
RNN	Recurrent Neural Network.
ROC	Receiver Operating Characteristic curve.
SHAP	SHapley Additive exPlanations (interpretabilidad).
SMOTE	Synthetic Minority Over-sampling Technique (balanceo de clases).
SQL / NoSQL	Structured Query Language / bases de datos “no sólo SQL”.
SSE / SSR / SST	Sum of Squared Errors / Residuals / Total (ANOVA).
SVM / SVR	Support Vector Machine / Regression.
t-SNE / UMAP	Técnicas de proyección no lineal para visualización.
TF-IDF	Term Frequency–Inverse Document Frequency (ponderación de textos).
TPOT / Optuna / Hyperopt	Librerías de AutoML / optimización bayesiana.
TSF / TSS	Time Series Forecasting / Split.
VIF	Variance Inflation Factor (multicolinealidad).
XGB	XGBoost (extreme gradient boosting)."""

"""5 modelos de predicción “imprescindibles” en un call center
(Qué resuelven, por qué se prefieren y un micro-ejemplo para que veas la idea en práctica)

#	Modelo<br>(familia)	Caso de uso típico	¿Por qué encaja bien?	Ejemplo de 2-3 líneas de código
1	SARIMA / Prophet
(series temporales)	Pronóstico de volumen de llamadas por franja de 15 min o 1 h.	Capturan estacionalidad semanal y picos de campañas sin requerir gran volumen de features.	python<br>from prophet import Prophet<br>df_prophet = df[['ds','y']] # fecha, volumen<br>m = Prophet(weekly_seasonality=4).fit(df_prophet)<br>forecast = m.predict(future_df)
2	Gradient Boosting Regressor
(XGBoost / LightGBM)	Predicción de AHT (duración) y After-Call Work combinando: tipo de llamada, IVR steps, experiencia del agente, tier del cliente.	Maneja no linealidad, interacciones y outliers; se entrena rápido y da importancias robustas.	python<br>import lightgbm as lgb<br>lgb_reg = lgb.LGBMRegressor(max_depth=8, n_estimators=400)<br>lgb_reg.fit(X_train, y_train)
3	Random Forest Classifier	Absentismo / rotación de agentes (clasificación 0-1: se ausentará/renunciará).	Pocos supuestos estadísticos, interpreta “feature importance”, soporta mezcla numérico-categórica y desbalance.	python<br>from sklearn.ensemble import RandomForestClassifier<br>rf = RandomForestClassifier(n_estimators=500, class_weight='balanced').fit(X,y)
4	Logistic Regression
(baseline penalizado L1/L2)	Probabilidad de conversión en campañas outbound o de que un cliente acepte venta cruzada.	Fácil de explicar (“odds ratio”), rápido de entrenar, sirve de benchmark antes de usar modelos complejos.	python<br>from sklearn.linear_model import LogisticRegression<br>logit = LogisticRegression(penalty='l2', C=1).fit(X,y)
5	BERT-finetune / Sentiment Model
(NLP Transformer)	Predicción de CSAT a partir de transcripción: estima la satisfacción antes de la encuesta y alerta casos críticos."""

📦 Carga y exploración básica de datos

import pandas as pd

# Cargar CSV
df = pd.read_csv('archivo.csv')

# Ver las primeras y últimas filas
df.head()           # Primeras 5 filas
df.tail(3)          # Últimas 3 filas

# Tamaño
df.shape            # (filas, columnas)

# Nombres de columnas
df.columns.tolist()

# Tipos de datos por columna
df.dtypes

# Resumen estadístico
df.describe()

# Información general
df.info()

# Valores únicos
df['columna'].unique()

# Conteo de valores únicos
df['columna'].value_counts()
🔍 Filtrado y selección de datos

# Selección por nombre de columna
df['columna']
df[['col1', 'col2']]

# Filtrado condicional
df[df['edad'] > 30]
df[(df['sexo'] == 'F') & (df['ciudad'] == 'Bogotá')]

# Filtrado por valores faltantes
df[df['columna'].isnull()]
df[df['columna'].notnull()]
✍️ Creación y modificación de columnas

# Crear nueva columna
df['edad_doble'] = df['edad'] * 2

# Modificar valores con función
df['nombre_mayus'] = df['nombre'].str.upper()

# Reemplazar valores
df['sexo'] = df['sexo'].replace({'M': 'Masculino', 'F': 'Femenino'})

# Aplicar función fila por fila
df['categoria'] = df.apply(lambda x: 'Mayor' if x['edad'] > 60 else 'Adulto', axis=1)
🧼 Limpieza de datos

# Eliminar columnas o filas
df.drop(columns=['col1', 'col2'], inplace=True)
df.drop(index=[0, 1], inplace=True)

# Renombrar columnas
df.rename(columns={'antiguo': 'nuevo'}, inplace=True)

# Cambiar tipo de dato
df['fecha'] = pd.to_datetime(df['fecha'])

# Rellenar valores faltantes
df['salario'].fillna(0, inplace=True)

# Eliminar duplicados
df.drop_duplicates(inplace=True)

# Resetear índice
df.reset_index(drop=True, inplace=True)
📊 Agrupamiento y resumen

# Agrupar y resumir
df.groupby('ciudad')['ventas'].mean()
df.groupby(['ciudad', 'producto'])['ventas'].sum().reset_index()

# Contar elementos
df.groupby('producto').size()

# Agregaciones múltiples
df.groupby('ciudad')['ventas'].agg(['mean', 'sum', 'count'])
🔀 Ordenar y combinar datos

# Ordenar por columnas
df.sort_values(by='ventas', ascending=False)

# Unir dataframes (merge estilo SQL)
df_merged = pd.merge(df1, df2, on='id', how='left')

# Concatenar (apilar) dataframes
df_concat = pd.concat([df1, df2], axis=0)   # vertical
🧭 Indexación avanzada

# Establecer columna como índice
df.set_index('fecha', inplace=True)

# Selección por rango de índice
df.loc['2023-01-01':'2023-01-31']

# Resetear índice a número entero
df.reset_index(inplace=True)
📌 Extra: métodos útiles para pruebas

# Mapear valores
df['tipo'] = df['codigo'].map({1: 'A', 2: 'B'})

# Quitar espacios en texto
df['cliente'] = df['cliente'].str.strip()

# Ver nulos por columna
df.isnull().sum()

# Filtrar por múltiples valores
df[df['ciudad'].isin(['Bogotá', 'Medellín'])]
✅ Checklist para la prueba
Tarea común	Método clave
Leer datos	read_csv()
Explorar	head(), info(), describe()
Limpiar	drop(), fillna(), replace(), drop_duplicates()
Crear columnas	apply(), map(), str.upper()
Agrupar	groupby(), agg()
Filtrar	df[...], isin(), isnull()
Unir tablas	merge(), concat()
Ordenar	sort_values()

1. Variables, tipos y conversión

# Asignación
x = 10            # int
pi = 3.1416       # float
nombre = "Miguel" # str
es_mayor = True   # bool

# Conversión explícita
altura = "1.75"
altura = float(altura)   # str → float

# Ver tipo
type(altura)  # <class 'float'>
2. Operadores básicos
Tipo	Ejemplo	Resultado
Aritméticos	5 ** 2, 7 % 3	25, 1
Comparación	10 >= 8	True
Lógicos	a and b, not x	—
Asignación compuesta	x += 1	x = x + 1
Membership	'mig' in 'miguel'	True

3. Estructuras de datos fundamentales

# Lista (mutable)
nums = [1, 2, 3]
nums.append(4)

# Tupla (inmutable)
coords = (4, 5)

# Conjunto (únicos, sin orden)
vocales = {'a', 'e', 'i', 'o', 'u'}
vocales.add('y')

# Diccionario (clave-valor)
persona = {'nombre': 'Ana', 'edad': 30}
persona['edad'] += 1
4. Slicing y unpacking

lista = ['a','b','c','d','e']
lista[1:4]      # ['b','c','d']
lista[::-1]     # ['e','d','c','b','a']  (reversa)

a, b, *resto = lista
# a='a', b='b', resto=['c','d','e']
5. Control de flujo

# Condicional
if edad >= 18:
    print("Adulto")
elif edad >= 13:
    print("Adolescente")
else:
    print("Niño")

# Bucle for
for i in range(1, 4):
    print(i)   # 1 2 3

# Bucle while
contador = 0
while contador < 3:
    contador += 1
6. Funciones y lambda

def area_rect(base, altura=1):
    """Devuelve área de un rectángulo."""
    return base * altura

area_rect(4, 3)      # 12
area_rect(5)         # 5 (usa altura por defecto)

# Función anónima
al_cuadrado = lambda x: x**2
7. Comprensiones

# Lista de cuadrados
cuadrados = [n**2 for n in range(5)]          # [0,1,4,9,16]

# Dict de pares
pares = {n: n%2==0 for n in range(5)}         # {0:True,1:False,...}
8. Manejo de excepciones

try:
    resultado = 10 / 0
except ZeroDivisionError:
    resultado = None
finally:
    print("Siempre se ejecuta")
9. Archivos

with open('datos.txt', 'w') as f:
    f.write('Hola\n')

with open('datos.txt') as f:
    texto = f.read()
10. Módulos y import

import math
math.sqrt(16)      # 4.0

from statistics import mean
mean([1,2,3])      # 2
11. Clases y objetos (POO)

class Persona:
    def __init__(self, nombre):
        self.nombre = nombre

    def saluda(self):
        return f"Hola, soy {self.nombre}"

p = Persona("Laura")
p.saluda()          # "Hola, soy Laura"
12. Generadores y yield

def cuenta_regresiva(n):
    while n:
        yield n
        n -= 1

for num in cuenta_regresiva(3):   # 3 2 1
    print(num)
13. Decoradores (concepto clave)

def loguea(func):
    def wrapper(*args, **kwargs):
        print("Llamando:", func.__name__)
        return func(*args, **kwargs)
    return wrapper

@loguea
def suma(a, b):
    return a + b

suma(3, 4)   # imprime "Llamando: suma" → 7
14. enumerate, zip, map, filter

# enumerate
for idx, letra in enumerate(['a','b','c']):
    print(idx, letra)

# zip
for a, b in zip([1,2], ['uno','dos']):
    print(a, b)

# map / filter
dobles = list(map(lambda x: x*2, [1,2,3]))
pares = list(filter(lambda x: x%2==0, range(6)))
"""Checklist de conceptos básicos
Categoría	Métodos / palabras clave
Estructuras	list, dict, set, tuple, slicing
Control	if, for, while, range, comprensión
Funciones	def, lambda, *args, **kwargs
OOP	class, __init__, self, herencia
Librerías	import, from ... import ...
Utilidades	enumerate, zip, map, filter
Robustez	try / except / finally, with open()
Avanzado	yield (generadores), decoradores

🗄️ Guía-express de conceptos básicos de SQL
(Ejemplo + explicación en una sola línea para cada comando, ideal para un repaso antes de una prueba técnica)

Supón la tabla calls(id, agent_id, call_type, duration, started_at, city, revenue) y la tabla agents(agent_id, name, hire_date, team).

Categoría	Sintaxis mínima	Ejemplo & mini-explicación
Seleccionar columnas	SELECT col1, col2 FROM tabla;	SELECT id, duration FROM calls; → muestra solo id y duración.
Alias	SELECT col AS alias	SELECT duration AS aht FROM calls; renombra duration.
Filtrar filas	WHERE condición	SELECT * FROM calls WHERE duration > 300; → > 5 min.
AND / OR / NOT	cond1 AND cond2	WHERE city='Bogotá' AND call_type='tech'.
Ordenar	`ORDER BY col [ASC	DESC]`
LIMIT	LIMIT n [OFFSET m]	LIMIT 10; trae 10 filas.
DISTINCT	SELECT DISTINCT col	SELECT DISTINCT city FROM calls; lista ciudades únicas.
Funciones de agregación	COUNT, SUM, AVG, MAX, MIN	SELECT AVG(duration) FROM calls; = AHT global.
GROUP BY	GROUP BY col	SELECT city, COUNT(*) FROM calls GROUP BY city; llamadas por ciudad.
HAVING	filtro sobre agregados	HAVING COUNT(*) > 100; (solo ciudades con > 100 llamadas).
JOIN (INNER)	FROM A JOIN B ON cond	SELECT a.id, b.team FROM calls a JOIN agents b ON a.agent_id=b.agent_id;
LEFT / RIGHT JOIN	LEFT JOIN conserva filas de la izquierda	LEFT JOIN agents incluye llamadas sin agente emparejado.
Self-join	alias tabla	SELECT a.id, b.id FROM calls a, calls b WHERE a.agent_id=b.agent_id AND a.id<b.id;
UNION / UNION ALL	concatena resultados	SELECT city FROM intl_calls UNION SELECT city FROM calls;
Subconsulta	(SELECT …)	SELECT * FROM calls WHERE agent_id IN (SELECT agent_id FROM agents WHERE team='A');
CTE (WITH)	WITH tmp AS (…) SELECT …	sql WITH long_calls AS (SELECT * FROM calls WHERE duration>300) SELECT COUNT(*) FROM long_calls;
INSERT	INSERT INTO tabla (col1, col2) VALUES (v1, v2);	INSERT INTO agents(name, team) VALUES ('Ana','A');
UPDATE	UPDATE tabla SET col=v WHERE cond;	UPDATE calls SET city='Bogotá' WHERE id=5;
DELETE	DELETE FROM tabla WHERE cond;	DELETE FROM calls WHERE duration<5;
CREATE TABLE	definición esquemática	sql CREATE TABLE teams(id SERIAL PRIMARY KEY, name TEXT UNIQUE);
PRIMARY KEY	clave única + NOT NULL	id SERIAL PRIMARY KEY.
FOREIGN KEY	referencia otra tabla	agent_id INT REFERENCES agents(agent_id).
DEFAULT	valor por defecto	created_at TIMESTAMP DEFAULT NOW().
CHECK	restricción	CHECK (duration > 0).
INDEX	acelerar búsqueda	CREATE INDEX idx_calls_city ON calls(city);
VIEW	consulta guardada	CREATE VIEW aht_by_type AS SELECT call_type, AVG(duration) aht FROM calls GROUP BY call_type;
Window functions	OVER()	SELECT id, duration, AVG(duration) OVER(PARTITION BY call_type) AS avg_type FROM calls;
CASE	lógica condicional	SELECT id, CASE WHEN duration>600 THEN 'larga' ELSE 'corta' END AS categoria FROM calls;
COALESCE / NULL	reemplazar nulos	SELECT COALESCE(city,'(desconocido)') FROM calls;
DATE functions	DATE_TRUNC, EXTRACT	SELECT DATE_TRUNC('month', started_at) mes, COUNT(*) FROM calls GROUP BY mes;
CAST	cambiar tipo	SELECT CAST(duration AS FLOAT)/60 AS minutos FROM calls;
Transactions	BEGIN … COMMIT / ROLLBACK	asegura atomicidad de cambios.
Privileges	GRANT SELECT ON tabla TO user;	control de acceso."""

"""① Crear un Stored Procedure (SP) en MySQL
(obtiene el AHT diario de un rango de fechas para una campaña dada)

sql
Copiar
Editar
-- cambiamos el delimitador para que el ; interno no cierre el bloque
DELIMITER $$     

CREATE PROCEDURE sp_aht_diario (
    IN  p_campaign_id INT,          -- parámetro de entrada
    IN  p_fecha_ini   DATE,
    IN  p_fecha_fin   DATE
)
BEGIN
    /* 1)  agrupamos por fecha;    2)  calculamos la duración promedio   */
    SELECT
        DATE(call_date)      AS fecha_llamada,
        ROUND(AVG(duration)) AS aht_segundos
    FROM tb_llamadas
    WHERE campaign_id = p_campaign_id
      AND call_date   BETWEEN p_fecha_ini AND p_fecha_fin
    GROUP BY DATE(call_date)
    ORDER BY fecha_llamada;
END$$

DELIMITER ;      -- restauramos el delimitador normal
Cómo leerlo

DELIMITER $$ → evita que el ; dentro del cuerpo cierre el CREATE PROCEDURE.

IN → parámetros de entrada (podrías usar OUT o INOUT si quisieras devolver valores escalares).

BEGIN … END → bloque donde escribes SQL normal; admite variables, cursores, IF, WHILE.

GROUP BY DATE(call_date) → calcula el Average Handle Time por día.

Ejecución:

sql
Copiar
Editar
CALL sp_aht_diario(12, '2025-06-01', '2025-06-30');
② Ejemplo de ROW_NUMBER() (MySQL 8+/PostgreSQL/SQL Server)
sql
Copiar
Editar
/* Objetivo: quedarnos con la llamada más larga de cada agente en el día */

WITH llamadas_rank AS (
    SELECT
        id,
        agent_id,
        DATE(call_date)   AS dia,
        duration,
        ROW_NUMBER() OVER (
            PARTITION BY agent_id, DATE(call_date)      -- reinicia por agente y día
            ORDER BY duration DESC                       -- mayor duración = fila 1
        ) AS rn
    FROM tb_llamadas
)
SELECT *
FROM llamadas_rank
WHERE rn = 1;          -- sólo la fila top de cada partición
Puntos clave

ROW_NUMBER() asigna un contador incremental dentro de cada partición (PARTITION BY).

El ORDER BY dentro de la ventana dicta qué fila recibe rn = 1.

El CTE (WITH) hace el resultado legible; podrías usar una subconsulta si tu motor no soporta CTEs.

③ SP que usa ROW_NUMBER internamente (MySQL 8+)
sql
Copiar
Editar
DELIMITER $$

CREATE PROCEDURE sp_top_llamadas (
    IN  p_n INT,
    IN  p_fecha DATE
)
BEGIN
    WITH ranked AS (
        SELECT
            id,
            agent_id,
            duration,
            ROW_NUMBER() OVER (PARTITION BY agent_id ORDER BY duration DESC) AS rk
        FROM tb_llamadas
        WHERE DATE(call_date) = p_fecha
    )
    SELECT *
    FROM ranked
    WHERE rk <= p_n                -- top-N por agente
    ORDER BY agent_id, rk;
END$$

DELIMITER ;
p_n define cuántas llamadas “más largas” devolver por agente para un día concreto.

④ SP con parámetro OUT (devuelve una métrica scalar)
sql
Copiar
Editar
DELIMITER $$

CREATE PROCEDURE sp_ultimo_id_llamada (OUT p_last_id INT)
BEGIN
    SELECT MAX(id) INTO p_last_id
    FROM tb_llamadas;
END$$

DELIMITER ;

-- uso
CALL sp_ultimo_id_llamada(@ult);
SELECT @ult;    -- muestra el valor capturado en la variable de sesión"""

"""Métricas de error para regresión: MAE y MSE
Las dos métricas cuantifican la diferencia entre los valores predichos (ŷ) y los reales (y), pero lo hacen de manera distinta, lo que cambia su interpretación y las decisiones que tomarás con ellas.

1. MAE — Mean Absolute Error (Error Absoluto Medio)
Aspecto	Detalle
Fórmula	[
\text{MAE}= \frac{1}{n}\sum_{i=1}^{n}\left	y_i-\hat y_i \right
]	
Unidades	Las mismas que la variable objetivo (segundos, dólares, llamadas, …).
Intuición	“En promedio, nos equivocamos ± MAE unidades por observación.”
Ventajas	* Fácil de explicar a negocio (no eleva errores al cuadrado).
* Penaliza linealmente los errores: cada punto cuenta igual.
Sensibilidad a outliers	Baja: un error gigante aumenta el MAE en la misma proporción que uno “moderado”.
Cuándo preferirlo	• KPI entra en planes operativos en unidades absolutas (p. ej. ± 20 s de AHT).
• Dataset con outliers naturales que no quieres sobre–ponderar.

Ejemplo rápido
Si MAE = 15 llamadas en un forecast horario, significa que tu pronóstico se desvía en promedio una quinta parte de un agente (asumiendo 75 llamadas/hora por agente). Esa traducción directa es lo valioso del MAE.

2. MSE — Mean Squared Error (Error Cuadrático Medio)
Aspecto	Detalle
Fórmula	[
\text{MSE}= \frac{1}{n}\sum_{i=1}^{n}\bigl( y_i-\hat y_i \bigr)^{2}	
]	
Unidades	Cuadradas respecto al objetivo (si y son segundos, MSE son segundos²).
Intuición	Como los errores están al cuadrado, castiga más los fallos grandes que los pequeños.
Ventajas	* Matemáticamente conveniente: derivable, convexa para modelos lineales → base de la regresión OLS.
* Útil cuando quieres evitar a toda costa errores enormes.
Sensibilidad a outliers	Alta: un error × 10 se vuelve × 100 en la suma.
Cuándo preferirlo	• El costo de un gran error crece más que linealmente (p. ej. pronóstico de demanda que cause ruptura de inventario).
• Entrenamiento de muchos algoritmos (Regresión Lineal, XGBoost) que optimizan a priori MSE o su raíz (RMSE).

RMSE = √MSE se suele reportar porque recupera las unidades originales y mantiene la penalización cuadrática.

3. Diferencias clave MAE vs MSE
Dimensión	MAE	MSE / RMSE
Penalización	Lineal	Cuadrática
Outliers	Robusta	Muy sensible
Unidades	Originales	Cuadradas (o originales tras raíz)
Facilidad de comunicación	Alta (± unidades)	Media (segundos² no son intuitivos)
Base de optimización	Menos usada como función de coste (no derivable en 0)	Estándar en muchas librerías
Interpretación práctica	“Promedio de error absoluto”	“Error típico con énfasis en los grandes fallos”

4. ¿Cuál usar en la práctica?
Determina tu KPI

¿Importa el error medio absoluto? → MAE.

¿Es crítico evitar picos de error? → MSE / RMSE.

Observa la distribución de tu variable y de los residuos.
Con heavy tails, MAE puede dar una visión más estable; MSE puede dispararse.

Evalúa ambos al entrenar: a veces un modelo con RMSE más bajo puede tener MAE más alto y viceversa. Elige el que maximice el beneficio/coste real.

5. Cálculo en Python (scikit-learn)
python
Copiar
Editar
from sklearn.metrics import mean_absolute_error, mean_squared_error
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)  # squared=False devuelve raíz
Conclusión:
MAE te dice, en las mismas unidades que tu negocio, cuán lejos te quedas en promedio; MSE (o su raíz) enfatiza los errores grandes y es el motor matemático de muchos modelos. Conocer cuándo priorizar cada uno —y poder explicarlo en términos de costo operativo— es clave en una prueba de Data Science.








Tú dijiste:
podrias revisar las formulas que generaste ya que no se entiende de manera clara


ChatGPT Plus
¡Claro! Volvamos a escribir las fórmulas con una notación lo más “limpia” y paso-a-paso posible.

1. MAE – Mean Absolute Error (Error Absoluto Medio)
Supón que tienes n observaciones.
Para cada observación i tienes:

Valor real  → y_i

Valor predicho → ŷ_i

Calcula el error individual:

𝑒
𝑖
=
𝑦
𝑖
−
𝑦
^
𝑖
e 
i
​
 =y 
i
​
 − 
y
^
​
  
i
​
 
Toma el valor absoluto (quita el signo):

∣
𝑒
𝑖
∣
=
∣
𝑦
𝑖
−
𝑦
^
𝑖
∣
∣e 
i
​
 ∣=∣y 
i
​
 − 
y
^
​
  
i
​
 ∣
Suma todos los absolutos y promedia:

MAE
  
=
  
  
∣
𝑒
1
∣
  
+
  
∣
𝑒
2
∣
  
+
  
…
  
+
  
∣
𝑒
𝑛
∣
 
𝑛
MAE= 
n
∣e 
1
​
 ∣+∣e 
2
​
 ∣+…+∣e 
n
​
 ∣
​
 
2. MSE – Mean Squared Error (Error Cuadrático Medio)
Calcula el mismo error individual:

𝑒
𝑖
=
𝑦
𝑖
−
𝑦
^
𝑖
e 
i
​
 =y 
i
​
 − 
y
^
​
  
i
​
 
Ahora cuadra cada error (lo multiplicas por sí mismo):

𝑒
𝑖
 
2
  
=
  
(
𝑦
𝑖
−
𝑦
^
𝑖
)
2
e 
i
2
​
 =(y 
i
​
 − 
y
^
​
  
i
​
 ) 
2
 
Suma todos los cuadrados y promedia:

MSE
  
=
  
  
𝑒
1
2
  
+
  
𝑒
2
2
  
+
  
…
  
+
  
𝑒
𝑛
2
 
𝑛
MSE= 
n
e 
1
2
​
 +e 
2
2
​
 +…+e 
n
2
​
 
​
 
Si quieres el RMSE (Root MSE) --muy usado porque devuelve las unidades originales-- simplemente tomas la raíz cuadrada de la MSE:

RMSE
  
=
  
MSE
RMSE= 
MSE
​
 
Resumen rápido
Métrica	Operaciones clave	Qué hace
MAE	Absoluto → Promedio	Mide el error medio “tal cual”, cada fallo pesa igual.
MSE	Cuadrado → Promedio	Penaliza más los errores grandes porque 
𝑒
2
e 
2
  crece rápido.

Con esta notación paso-a-paso ya no debería haber ambigüedad. ¡Avísame si necesitas otro ejemplo numérico o un código corto para calcularlas!"""

"""Pensó durante 6 segundos
1. K-Nearest Neighbors (KNN) — Concepto y mecánica paso a paso
¿Qué es?
Un método no paramétrico: no construye una ecuación global, sino que usa el propio conjunto de entrenamiento como “memoria”.
En regresión, la predicción para un punto nuevo es el promedio (o promedio ponderado) de las K observaciones más cercanas.

Cómo funciona (flujo en 4 pasos)
Paso	Detalle	Puntos clave
1 Escalar	Normaliza variables (p. ej. StandardScaler).	Distancias Euclidianas se distorsionan si los rangos difieren.
2 Elegir K	El nº de vecinos (3, 5, 10 …).	• K pequeño = baja sesgo, alta varianza (overfitting).
• K grande = alta suavidad, posible underfitting.
3 Calcular distancias	Métrica típica: Euclidiana; altern.: Manhattan, Minkowski, Mahalanobis.	Se usan estructuras KD-Tree / Ball-Tree o brute-force para buscar los vecinos.
4 Agregar	Predicción = media (o media ponderada 1/distancia) de las K etiquetas.	En clasificación, es el voto mayoritario.

python
Copiar
Editar
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

knn = Pipeline([
    ("scale", StandardScaler()),
    ("model", KNeighborsRegressor(n_neighbors=5, weights="distance"))
])
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
Ventajas
Cero coste de entrenamiento (solo guarda los datos).

Se adapta a relaciones arbitrariamente no lineales.

Fácil de explicar: “tu valor se parece a los K más parecidos”.

Limitaciones
Coste de predicción O(n × d) (lento con muchos puntos).

Se degrada en alta dimensión (la “maldición de la dimensionalidad”).

Sensible a desbalance y ruido (outliers).

Cuándo usarlo en un call center
– Estimar AHT de un caso nuevo basado en tickets históricos con características similares (tipo de problema, canal, cliente).

2. Support Vector Regression (SVR) — Principios generales
Idea base
Extiende el concepto de Support Vector Machine (SVM) a regresión: busca una “tubo” de ancho ε alrededor de una función (lineal o no) tal que

la mayoría de los puntos queden dentro del tubo (error ≤ ε),

la función sea lo más plana posible (minimiza la norma de los coeficientes).

Formulación simplificada (caso lineal)
Minimizar

1
2
 
∥
𝑤
∥
2
  
+
  
𝐶
∑
𝑖
=
1
𝑛
(
𝜉
𝑖
+
𝜉
𝑖
∗
)
2
1
​
 ∥w∥ 
2
 +C 
i=1
∑
n
​
 (ξ 
i
​
 +ξ 
i
∗
​
 )
sujeto a

𝑦
𝑖
−
(
𝑤
⊤
𝑥
𝑖
+
𝑏
)
≤
 
𝜀
+
𝜉
𝑖
(
𝑤
⊤
𝑥
𝑖
+
𝑏
)
−
𝑦
𝑖
≤
 
𝜀
+
𝜉
𝑖
∗
𝜉
𝑖
,
𝜉
𝑖
∗
≥
0
y 
i
​
 −(w 
⊤
 x 
i
​
 +b)
(w 
⊤
 x 
i
​
 +b)−y 
i
​
 
ξ 
i
​
 ,ξ 
i
∗
​
 
​
  
≤ ε+ξ 
i
​
 
≤ ε+ξ 
i
∗
​
 
≥0
​
 
w: vector de pesos (pendientes).

ε: tolerancia (tamaño del tubo).

ξ, ξ*: slack variables (errores que caen fuera del tubo).

C: penaliza la suma de errores fuera del tubo → controla trade-off entre suavidad y error.

Núcleo (kernel) y no-linealidad
Mediante el kernel trick reemplazas 
𝑥
𝑖
⊤
𝑥
𝑗
x 
i
⊤
​
 x 
j
​
  por κ(x_i,x_j) (RBF, Polinómico, etc.), encontrando superficies no lineales sin trabajar en alta dimensión explícita.

python
Copiar
Editar
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

svr = Pipeline([
    ("scale", StandardScaler()),
    ("model", SVR(kernel="rbf", C=10, epsilon=0.1, gamma="auto"))
])
svr.fit(X_train, y_train)
y_pred = svr.predict(X_test)
Parámetros esenciales
Parámetro	Efecto	Heurística inicial
kernel	Forma de la función (lineal, rbf, poly).	Empieza con rbf para relaciones no lineales suaves.
C	Penaliza errores > ε.	C pequeño = más suaves, más sesgo; C grande = ajusta outliers.
epsilon	Ancho del tubo sin penalizar.	Ajusta a la tolerancia de error “aceptable” del negocio.
gamma	(RBF) Inverso de ancho del kernel.	"auto" o 1/n_features como base; afina con Grid / Random Search.

Ventajas
Soporta relaciones muy complejas sin explotar dimensionalidad.

Se centra en “soportes” → modelo sparse (solo usa unos cuantos puntos).

Control fino del bias–variance trade-off vía C y ε.

Desventajas
Escalado obligatorio; sensible a elección de hiper-parámetros.

Entrenamiento O(n²): costoso con > ~10 000 muestras.

Menos interpretable que un modelo lineal o árbol.

Ejemplo en BPO
Prever tiempo total de gestión de un chat (incluye AHT + after-chat) donde la relación con número de mensajes es curva: pocos mensajes → corto, muchos → largo, intermedios → impredecible. El kernel RBF captura esa forma.

Comparación rápida
Aspecto	KNN	SVR
Tipo	Instance-based (lazy learn)	Algoritmo de margen (modelo + kernels)
Entrenamiento	O(1)	O(n²)
Predicción	Lenta (dep. de n)	Rápida una vez entrenado
Escalado	Muy necesario	Necesario
Explicabilidad	Alta (vecinos reales)	Media (puedes usar SHAP)
Uso típico	Pequeños datasets, prototipado	Datos medianos, patrones no lineales suaves

Con estas bases puedes describir ambos algoritmos a cualquier entrevistador y elegir cuál aplicar según el problema y los datos disponibles."""

"""1. ¿Qué es un “bosque aleatorio”?
Un Random Forest (RF) es un ensamble que agrupa muchos árboles de decisión de regresión independientes y promedia sus predicciones.
Dos “inyecciones de aleatoriedad” crean árboles diversos (descorrelacionados):

Bootstrap (bagging): cada árbol se entrena con una muestra aleatoria con reemplazo de los registros de entrenamiento.

Sub-muestra de atributos: en cada nodo, el árbol sólo considera un subconjunto aleatorio de variables al buscar la mejor división.

El algoritmo final hace:

𝑦
^
RF
(
𝑥
)
  
=
  
1
𝑇
∑
𝑡
=
1
𝑇
  
𝑦
^
(
𝑡
)
(
𝑥
)
y
​
  
RF
​
 (x)= 
T
1
​
  
t=1
∑
T
​
  
y
​
  
(t)
 (x)
donde 
𝑇
T es el número de árboles y 
𝑦
^
(
𝑡
)
y
​
  
(t)
  la predicción del árbol t.

Intuición: cada árbol es un “juez” con un punto de vista sesgado pero distinto; la media de todos reduce las excentricidades individuales (varianza) sin aumentar demasiado el sesgo.

2. Ventajas clave frente a la regresión lineal
Dimensión	Random Forest	Regresión lineal
Forma de la relación	Capta no linealidad y interacciones complejas automáticamente.	Supone relación puramente lineal a menos de que agregues términos manualmente (x², log x…).
Escalado de variables	No requiere normalizar ni estandarizar.	Coeficientes sensibles a escalas; se suele estandarizar.
Outliers	Menos afectado (cada división sólo mira su rama).	Outliers pueden sesgar la línea global.
Feature engineering	Detecta divisiones óptimas sin que el analista cree features polinómicas.	Necesitas fabricar features para capturar curvaturas.
Robustez a multicolinealidad	Alta; variables redundantes compiten en splits y se diluyen.	Coeficientes inestables si las X están muy correlacionadas.
Overfitting	Controlado por n_estimators, max_depth, min_samples_leaf.	Puede sub-o over-fit, pero es menos flexible por naturaleza.
Interpretabilidad	“Caja gris”: se explica via importancia de variables o SHAP.	Coeficientes directos β → efecto marginal claro.

Conclusión: si sospechas relaciones curvas o interacciones imposibles de linealizar fácilmente (p. ej. tiempo de espera × tipo de llamada), RF suele ofrecer mejor performance con poca afinación.

3. Importancia de variables (feature importance)
Random Forest proporciona dos métricas globales; ambas responden “¿cuánto aporta cada variable al poder predictivo del bosque?”

Tipo	Cómo se calcula	Pros / Contras
Impureza (MDI – Mean Decrease in Impurity)	Para cada split de un árbol t calcula la reducción de MSE. Suma y promedia sobre todos los árboles.	Rápido (viene gratis al entrenar); sesgo hacia variables con más categorías o alta cardinalidad.
Permutación (MDA – Mean Decrease in Accuracy)	Para cada variable: baraja sus valores en el conjunto val, mide cuánto empeora el MAE/RMSE.	Model-agnostic, corrige sesgo de cardinalidad; más lento (requiere pasar muchas veces por el bosque).

Ejemplo práctico en scikit-learn
python
Copiar
Editar
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
import pandas as pd

rf = RandomForestRegressor(
        n_estimators=400,
        max_depth=None,
        random_state=42
).fit(X_train, y_train)

# 1) Importancia por impureza
imp_mdi = pd.Series(rf.feature_importances_, index=X_train.columns) \
             .sort_values(ascending=False)

# 2) Importancia por permutación
result = permutation_importance(rf, X_val, y_val,
                                n_repeats=10, random_state=42)
imp_mda = pd.Series(result.importances_mean, index=X_train.columns) \
             .sort_values(ascending=False)

print("Top-5 MDI\n", imp_mdi.head())
print("Top-5 MDA\n", imp_mda.head())
Cómo interpretarlo para negocio

“ivr_steps explica el 31 % de la reducción total de error; si aumentaran los pasos en el IVR, esperaríamos un salto en la duración de llamada.
agent_tenure contribuye un 18 %: agentes nuevos incrementan el AHT. Esto confirma que invertir en entrenamiento reducirá costos.”

Para explicaciones fila-a-fila (predicciones individuales), complementa con SHAP:

python
Copiar
Editar
import shap
expl = shap.TreeExplainer(rf)
shap_values = expl.shap_values(X_val.iloc[0])     # un solo caso
shap.waterfall(shap.Explanation(shap_values,
                                 expl.expected_value,
                                 X_val.iloc[0]))
4. Parámetros esenciales a afinar
Hiper-parámetro	Influencia	Heurística inicial
n_estimators	Nº de árboles: ↑ → ↓ varianza, ↑ costo.	300-500 en datasets < 100 k filas.
max_depth	Profundidad máxima de cada árbol.	None (deja que crezca) + min_samples_leaf ≥ 5.
min_samples_leaf	Mín. de registros por hoja → suaviza predicción.	1 % de n si hay mucho ruido.
max_features	Nº de variables candidatas por split.	auto (= √d) en regresión.

Para obtener los valores óptimos → Grid/Random Search con validación cruzada (n_jobs=-1 para paralelizar).

5. Cuándo no usar Random Forest
Necesitas predicciones fuera de rango (extrapolación) → RF sólo interpola dentro del dominio visto.

~> 1 M filas + > 200 features → RAM y tiempo crecen; considera Gradient Boosting o XGBoost con histogram-based splits.

Requieres coeficientes claros (elasticidad en %) → usa regresión (lineal, Poisson) o GAMs."""

"""Clasificación vs Regresión — las dos grandes familias de problemas supervisados
Dimensión	Clasificación	Regresión
Qué predice	Etiquetas discretas (clases).
Ej.: “churn / no-churn”, “tipo de llamada: billing, tech, ventas”.	Valores continuos (numéricos).
Ej.: duración de llamada en segundos, monto de facturación.
Salida típica del modelo	- Probabilidades por clase (P(clase=k))
- Etiqueta final (la de mayor prob.)	- Número real (Ŷ = 237 s de AHT).
Función de pérdida	Cross-entropy, hinge, log loss, etc.	MSE, MAE, Huber, Poisson deviance.
Métricas de evaluación	Accuracy, ROC-AUC, F1, Precision/Recall, Kappa, Log-loss.	MAE, RMSE, R², MAPE, sMAPE.
Modelos más usados	Logistic Regression, Random Forest Classifier, XGBoost Classifier, SVM, Redes neuronales de clasificación.	Linear/Polynomial Regression, Random Forest Regressor, Gradient Boosting Regressor (XGB/LGBM), SVR, Redes neuronales de regresión.
Curva de decisión	Frontera que separa regiones de clase en el espacio de variables.	Superficie (curva) que aproxima la relación X → Y.
Umbral crítico	Sí: se fija un threshold sobre la probabilidad para decir «clase=1».	No: se predice directamente el valor; el negocio define márgenes de error aceptables.
Coste típico de error	Asimétrico (FN vs FP pueden costar distinto).	Generalmente proporcional a la magnitud del error, o cuadrático (RMSE).
Ejemplo en un call center	Detectar si una llamada acabará insatisfecha (CSAT bajo) → bandera roja al supervisor.	Estimar cuántos segundos tomará resolver la próxima llamada para dimensionar staffing.

Pistas rápidas para la entrevista
¿Qué pasa si convierto un problema continuo en categórico?
– Pierdes información (duración 179 s ≈ 181 s) y precisas más datos para igual precisión.

¿Puedo usar un modelo de regresión para clasificación?
– Sí, si conviertes la regresión a un scoring y aplicas un umbral, pero no optimizarás métricas de clasificación de forma directa.

¿Cómo elijo métrica?
– Clasificación: depende del balance entre FP y FN (p. ej. ROC-AUC para ranking, F1 si la clase positiva es rara).
– Regresión: MAE si el negocio entiende unidades absolutas; RMSE si penalizas mucho los grandes errores.

¿Se pueden combinar?
– Modelos mixtos (multi-output): p.ej. predecir clase de ticket y tiempo de resolución en un solo pipeline.

Con esto podrás explicar claramente la diferencia de naturaleza, objetivos y métricas de cada tipo de problema en Data Science."""

"""¿Qué es el criterio de parada en un árbol de decisión?
Cuando entrenamos un árbol, el algoritmo va dividiendo el conjunto de datos en nodos cada vez más puros (menos mezcla de clases o menor varianza, según sea clasificación o regresión).
Si lo dejáramos crecer sin control, acabaríamos con:

Hojas de un solo registro (error casi 0 en entrenamiento, pero altísimo riesgo de overfitting).

Tiempo y memoria innecesarios.

El criterio de parada define cuándo dejar de dividir un nodo y declarar que es hoja definitiva.
En otras palabras: “¿Hasta qué punto vale la pena seguir complicando el árbol?”

Criterios de parada más comunes
Parámetro (en scikit-learn)	Qué controla	Ejemplo típico
max_depth	Profundidad máxima (niveles).	max_depth=6 limita el árbol a 6 decisiones en cadena.
min_samples_split	Mínimo de registros para intentar un nuevo split.	min_samples_split=20 → si el nodo tiene 19 o menos, se vuelve hoja.
min_samples_leaf	Mínimo de registros que debe contener cada hoja resultante.	min_samples_leaf=5 evita hojas diminutas.
min_impurity_decrease	Umbral mínimo de reducción de impureza (Gini, Entropía, MSE) para aceptar un split.	min_impurity_decrease=0.001 requiere ganancia “significativa”.
max_leaf_nodes	Límite global de hojas.	max_leaf_nodes=50 fuerza poda temprana.
min_weight_fraction_leaf	Igual que min_samples_leaf pero proporcional al total (útil con sample weights).	0.02 ⇒ cada hoja ≥ 2 % del peso total.

En árboles de regresión, “impureza” = varianza (MSE).
En clasificación, puede ser Gini o Entropía.

¿Cómo interactúan?
El algoritmo intenta todos los splits posibles para la variable candidata.

Calcula la reducción de impureza.

Sólo acepta el split si cumple todos los requisitos (ej.: produce ≥ min_samples_leaf, profundidad < max_depth, y la ganancia supera min_impurity_decrease).

Si ningún split válido existe, el nodo se declara hoja y ahí termina su rama.

¿Por qué es crucial?
Evitar overfitting: Limitar la profundidad o exigir un tamaño mínimo de hoja impide que el árbol “memorice” ruido.

Generalización y estabilidad: Un árbol muy ajustado varía mucho si cambias ligeramente los datos (alta varianza).

Eficiencia computacional: Menos nodos = modelo más ligero y predicciones más rápidas.

Interpretabilidad: Árboles más compactos son más fáciles de explicar a negocio.

Ejemplo rápido en Python
python
Copiar
Editar
from sklearn.tree import DecisionTreeRegressor

tree = DecisionTreeRegressor(
    max_depth=6,
    min_samples_leaf=10,
    min_impurity_decrease=0.0005,
    random_state=42
).fit(X_train, y_train)
Con esos hiperparámetros:

Nunca habrá más de 6 saltos desde la raíz a una hoja.

Cada hoja contendrá ≥ 10 registros.

Un split se acepta sólo si reduce la varianza en ≥ 0.0005.

Resumen en una frase
El criterio de parada es el conjunto de reglas que le dice al algoritmo de árbol “¡basta de dividir aquí!”, equilibrando precisión y simplicidad para que el modelo sea útil y no se sobreajuste."""

"""“Kit estadístico” esencial para un data scientist
(Conceptos explicados en lenguaje llano + recordatorio de cómo se calculan)

1. Descriptivos básicos - “fotografía” inicial del dato
Cálculo	Qué mide	Cómo se obtiene (idea)
Media (promedio)	El centro “equilibrado”.	Suma de valores ÷ n.
Mediana	Punto que corta la serie al 50 %.	Ordenas y tomas el del medio.
Moda	Valor más repetido.	Cuenta apariciones.
Rango	Extensión total.	Máx – mín.
Varianza (σ²) / Desviación estándar (σ)	Qué tan dispersos están.	Promedio de (valor − media)²; √ para σ.
IQR (rango intercuartílico)	Dispersión robusta a outliers.	Q3 – Q1 (75 % – 25 %).
Skewness	Asimetría (cola a la derecha/izq.).	Fórmula de 3er momento estandarizado.
Kurtosis	“Picos” y colas.	4º momento estandarizado.

2. Probabilidad & distribuciones - “comportamiento esperado”
Bernoulli / Binomial: éxitos-fracasos (clic/no clic).

Poisson: recuentos por intervalo (llamadas por minuto).

Normal: muchas medias y errores tienden a ella (CLT).
– Cálculo útil: z-score = (valor − media)/σ → cuántas σ te alejas.

Exponencial: tiempo entre eventos independientes.

Uniforme: todos los valores del rango igual probabilidad.

3. Muestreo y el “error natural”
Concepto	Para qué sirve
Tamaño de muestra (n)	Balancea coste vs precisión. Fórmulas dependen de variabilidad y confianza.
Error estándar (SE)	Desviación de la media de muestras. SE = σ/√n.
Teorema del límite central (CLT)	La distribución de la media → normal si n es “grande”, sin importar la forma original.
Bootstrap	Re-muestreo con reemplazo para estimar incertidumbre cuando las fórmulas no aplican.

4. Inferencia: convertir muestra → conclusiones
Herramienta	Para qué	Regla rápida
Intervalo de confianza (IC)	Rango plausible para el parámetro.	Media ± (se × z_{α/2}).
Pruebas t / z	¿Media ≠ valor H₀? / ¿Medias distintas?	Si p-value < α (0.05) → “diferencia significativa”.
Chi-cuadrado	Tablas de frecuencias, bondad de ajuste.	
ANOVA	>2 medias de grupos. F grande → al menos una media difiere.	
Tests no paramétricos (Mann-Whitney, Wilcoxon)	Igualdad de medianas sin asumir normalidad.	
Valor-p	Probabilidad de ver datos ≥ extremos asumiendo H₀. Pequeño → evidencia contra H₀.	
Tamaño de efecto (Cohen d, r)	“Cuán grande” es la diferencia, no sólo si existe.	

5. Relaciones entre variables
Covarianza: dirección de la relación (difícil de leer por unidades).

Correlación (Pearson r): fuerza lineal entre −1 y 1.
Spearman ρ si la relación es monótona pero no lineal.

Regresión lineal: ajusta ŷ = β₀ + β₁x.
– β₁: cambio esperado en y por unidad de x.
– R²: % de varianza explicada.

Índice VIF: detecta multicolinealidad en regresores.

6. Evaluación de modelos (verás estos nombres en ML)
Métrica	Tipo de problema	Intuición
MAE / MSE / RMSE	Regresión	Error medio (absoluto o cuadrático).
R² ajustado	Regresión	Varianza explicada penalizando nº de features.
Accuracy, Precision, Recall, F1	Clasificación	Aciertos generales vs aciertos clase positiva.
ROC-AUC / PR-AUC	Clasificación desequilibrada	Capacidad de ranking entre clases.
Log Loss / Cross-Entropy	Probabilístico	Penaliza predicciones seguras y equivocadas.

7. Otros “imperdibles”
Bayes: Posterior ∝ Prior × Verosimilitud → actualiza creencias.

AIC / BIC: comparan modelos penalizando complejidad.

Autocorrelación & ACF: dependencia temporal en series.

Durbin-Watson: detecta autocorrelación de residuos.

Cómo estudiarlos rápido
Descriptivos → practica pandas.describe() y fórmulas manuales.

Inferencia → simula coin-toss en Python y verifica IC y valor-p.

Regresión & Correlación → usa scipy.stats.linregress.

Evaluación modelos → sklearn.metrics tiene casi todas las métricas.

Con este glosario tendrás los cálculos estadísticos indispensables dominados para cualquier rol de Data Science. Si quieres ejemplos de código o ejercicios concretos de algún apartado, ¡avísame!"""
"""
Cómo se calcula R² (coeficiente de determinación)
…y qué significa en una regresión lineal simple
1. Concepto intuitivo 🚦
R² responde a esta pregunta:

¿Qué proporción de la variabilidad de y (la variable que quieres predecir) es explicada por la recta que has ajustado con x?

Valor-guía*

0 → la recta no explica nada; predecirías igual usando la media de y.

1 → la recta explica el 100 % de la variación (todos los puntos caen exactamente sobre la línea).

< 0 → (sólo aparece en predicción fuera de muestra) el modelo es peor que usar la media.

2. Fórmula paso a paso
Imagina que tienes n pares 
(
𝑥
𝑖
,
𝑦
𝑖
)
(x 
i
​
 ,y 
i
​
 ) y has ajustado la recta

𝑦
^
𝑖
=
𝛽
0
+
𝛽
1
𝑥
𝑖
y
^
​
  
i
​
 =β 
0
​
 +β 
1
​
 x 
i
​
 
Varianza total de y (“cuánto se mueven los valores reales”)

SST
=
∑
𝑖
=
1
𝑛
 
(
𝑦
𝑖
−
𝑦
ˉ
)
2
SST= 
i=1
∑
n
​
 (y 
i
​
 − 
y
ˉ
​
 ) 
2
 
Errores del modelo (“lo que la recta no acierta”)

SSR
=
∑
𝑖
=
1
𝑛
 
(
𝑦
𝑖
−
𝑦
^
𝑖
)
2
SSR= 
i=1
∑
n
​
 (y 
i
​
 − 
y
^
​
  
i
​
 ) 
2
 
R²

𝑅
2
=
1
  
−
  
SSR
SST
R 
2
 =1− 
SST
SSR
​
 
Interpretación:

Si la recta reduce mucho SSR respecto a SST, el cociente es pequeño y R² se acerca a 1.

Si la recta apenas mejora la media, SSR ≈ SST y R² ≈ 0.

3. Ejemplo numérico rápido
𝑥
x	
𝑦
y	
𝑦
^
y
^
​
  (recta)
1	2	2.2
2	3	3.1
3	2	4.0

𝑦
ˉ
=
(
2
+
3
+
2
)
/
3
=
2.33
y
ˉ
​
 =(2+3+2)/3=2.33

SST = 
(
2
−
2.33
)
2
+
(
3
−
2.33
)
2
+
(
2
−
2.33
)
2
=
0.11
+
0.44
+
0.11
=
0.66
(2−2.33) 
2
 +(3−2.33) 
2
 +(2−2.33) 
2
 =0.11+0.44+0.11=0.66

SSR = 
(
2
−
2.2
)
2
+
(
3
−
3.1
)
2
+
(
2
−
4.0
)
2
=
0.04
+
0.01
+
4
=
4.05
(2−2.2) 
2
 +(3−3.1) 
2
 +(2−4.0) 
2
 =0.04+0.01+4=4.05

𝑅
2
=
1
−
4.05
0.66
≈
−
5.13
R 
2
 =1− 
0.66
4.05
​
 ≈−5.13
R² es negativo: fuera de muestra la recta lo hizo peor que la simple media (caso inventado para que se vea el concepto).

4. Cálculo en Python (scikit-learn)
python
Copiar
Editar
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import numpy as np

X = np.array([[1],[2],[3]])
y = np.array([2,3,2])

model = LinearRegression().fit(X, y)
y_pred = model.predict(X)

print("R²:", r2_score(y, y_pred))   # coincide con model.score(X, y)
5. Matices importantes
R² alto no = buen modelo: puede estar sobre-ajustado (fíjate en R² sobre datos de validación).

Comparación de modelos con distinto nº de variables: usa R² ajustado para penalizar el exceso de features.

𝑅
adj
2
=
1
−
(
1
−
𝑅
2
)
 
𝑛
−
1
𝑛
−
𝑝
−
1
R 
adj
2
​
 =1−(1−R 
2
 ) 
n−p−1
n−1
​
 
donde 
𝑝
p = n.º de predictores.

Unidades: R² es adimensional; siempre entre −∞ y 1.

Relación no lineal: un R² bajo puede indicar que la relación no es lineal, no necesariamente que “no exista”.

Resumen en 1 frase
R² = 1 − (errores de la recta ÷ variación total); indica la fracción de la dispersión de y que tu modelo lineal logra capturar."""

"""¿Cómo gana dinero y opera un call center/BPO en Colombia?
Caso referencia: GroupCOS
1. Propuesta de valor
Externalización integral: asume procesos de atención, ventas y back-office que no son core para el cliente (telecom, retail, fintech, salud).

Nearshoring: presta servicio en español e inglés desde Colombia y México — huso horario compatible con EE. UU. y Canadá, y costos ~50 % menores que on-shore. 
latam.groupcosbpo.com.co
Connext

Enfoque “tech + talento”: combina plataformas de contact center omnicanal, RPA y analítica con agentes certificados. 
groupcosbpo.com
LinkedIn

Calidad certificada: PCI-DSS, ISO 27001, Friendly Biz, etc.; tranquiliza a clientes de sectores regulados. 
Comercial Group COS -

2. Líneas de servicio y flujos de ingresos
Línea	Qué vende	Cómo se tarifica*
Inbound CX (soporte, post-venta)	Horas/agente o “FTE” + bono por SLA (TMO, FCR).	$/hora o $/FTE/mes
Outbound ventas/cobranza	Contacto a leads, recordatorios de pago.	$/venta cerrada o comisión %
BPO back-office	Digitación, KYC, verificación, chat moderación.	$/transacción
RPA & analítica	Bots, speech-to-text, dashboards (arriba de Vicidial, Genesys, 3CX, etc.).	Proyecto cerrado + fee de mantenimiento
Nearshore staffing	Equipos dedicados (tech-support lvl 1–2).	Costo + margen (modelo seat leasing)

*Los contratos suelen mezclar tarifas fijas + variables por KPI.

3. Cost structure y márgenes típicos
Rubro	% del costo total aprox.	Gestión en GroupCOS
Talento (salario + prestaciones)	55–65 %	Reclutamiento continuo, academia interna – Learning & Development.
Tecnología (licencias CCaaS, CRM, RPA)	10–15 %	Alianzas con Avaya, Zendesk; bots propios.
Infraestructura (sites, energía, seguridad)	8–10 %	Sedes Bogotá, Bucaramanga; redundancia TIER III.
QA & compliance	4–6 %	Auditoría 100 % de llamadas con speech-analytics.
G&A	10–15 %	Dirección, ventas, certificaciones.

Márgenes EBITDA para players medianos en Colombia: 12–18 %; se eleva con automatización (bots atienden ≈30 % de interacciones sin agente).

4. Operación diaria (modelo Delivery)
Discovery → diagnóstico del proceso del cliente.

Blueprint & KPIs → definición de SLA (AHT, CSAT, NPS, ventas/hr).

Ramp-up → reclutamiento, entrenamiento, prueba piloto.

Run & Monitor → reporting en tiempo real; metodología interna de mejora (Lean + Design Thinking). 
groupcosbpo.com

Continuous Improvement → análisis de voz, dashboards Power BI, retrain ML.

5. Diferenciales frente a otros call centers
Factor	Cómo lo capitaliza GroupCOS
Tecnología propietaria	RPA y speech-analytics propios para reducir AHT.
Certificaciones	Abre puertas a bancos/fintech que exigen ISO 27001, PCI-DSS.
Multipaís	“Follow-the-sun” con sites Colombia–México; cobertura 24/7 sin turnos extremos.
Cultura bilingüe	Agentes con B2–C1 permiten subir tarifas en USD.

6. Indicadores de éxito que reporta al cliente
Costo por contacto vs baseline.

AHT y FCR (First-Call Resolution).

CSAT / NPS mensual.

Ventas incremental por hora en campañas outbound.

ROI de RPA (horas-hombre liberadas).

Conclusión
El negocio de GroupCOS (y call centers BPO similares en Colombia) se basa en monetizar talento local + diferencial de costos + tecnología propia para entregar atención y procesos back-office a empresas globales. El ingreso combina tarifas por agente/ transacción y bonos por desempeño; el éxito depende de escalar equipos rápido, mantener calidad certificada y, cada vez más, automatizar para proteger márgenes."""