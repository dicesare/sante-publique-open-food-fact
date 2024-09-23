#%% md
#  le jeu de données [Open Food](https://world.openfoodfacts.org/) (ou disponible à ce [lien](https://s3-eu-west-1.amazonaws.com/static.oc-static.com/prod/courses/files/parcours-data-scientist/P2/fr.openfoodfacts.org.products.csv.zip) en téléchargement).
# 
# > Merci d'exécuter ce notebook car il va générer un fichier c.s.v qui sera utilisé pour le notebook P3_Plotly qui est dédié a l'affichage "Voila"
#%% md
# 
#%%
import matplotlib.figure as fig
from matplotlib.patches import Polygon
import pandas as pa
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import linregress
from matplotlib.cbook import boxplot_stats
import statsmodels.api as sm
#%%

off = pa.read_csv('../fr.openfoodfacts.org.products/fr.openfoodfacts.org.products.csv', sep='\t')
#%%
off.columns = off.columns.str.replace('-','_')
#%% md
# ## Statistique descriptives et représentation graphique
# ### Nettoyez et analysez votre jeu de données
# #### Présentation de notre dataframe
#%% md
# On dispose d'un jeu de données contenant 320772 individus et 162 variables
#%%
off.shape
#%% md
# ### isnull & sum
# Profitons du fait de voir des `NaN` pour utiliser la combinaison de commande pratique de 🐼 `.isnull()` & `sum()`.
# > [Isnull](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.isnull.html?highlight=isnull#pandas.DataFrame.isnull) nous retourne un tableau de booléens de même taille que notre Dataframe Les valeurs `NaN`, telles que None ou [numpy.NaN](https://numpy.org/doc/stable/reference/constants.html?highlight=nan#numpy.NaN), sont mappées aux valeurs `True`. Tout le reste est mappé sur des valeurs ``False``.
# > [Sum](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.sum.html?highlight=sum#pandas.DataFrame.sum) Renvoie la somme des valeurs sur l'axe demandé.
# > > *par defaut : `DataFrame.sum(axis=None, skipna=None, level=None, numeric_only=None, min_count=0, **kwargs)`*
#%%
off.isnull().sum()
#%% md
# ### dtypes
# Affichons le types de nos colonnes `.dtypes`.
# > [DTypes](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.dtypes.html?highlight=dtypes#pandas.DataFrame.dtypes) Renvoie les dtypes dans le DataFrame.
# > `value_counts` permet de comptabiliser les qualitatives et quantitatives
#%%
off.dtypes.value_counts()
#%% md
# ### Nunique
# Regardons en détails nos qualitatives, savoir si nous pourrions définir des groupes pour l'étude à venir
#%%
columns_qual = off.columns[off.dtypes == object]
off[columns_qual].nunique().sort_values()
#%%
off.pnns_groups_1.replace('unknown', np.NaN,inplace=True)
off.pnns_groups_1=off.pnns_groups_1.str.replace('-',' ')
off.pnns_groups_1=off.pnns_groups_1.str.capitalize()
pnns_groups_1 = off.pnns_groups_1.dropna()
#%%
pnns_groups_1.describe()
#%%
off.pnns_groups_2.replace('unknown', np.NaN,inplace=True)
off.pnns_groups_2=off.pnns_groups_2.str.capitalize()
pnns_groups_2 = off.pnns_groups_2.dropna()
#%%
pnns_groups_2.describe()
#%% md
# On observe dans cette table que les produits alimentaires sont répartis en 9 groupes d’aliments (pnns_groups_1), qui eux-mêmes, sont séparés en sous-groupe (pnns_groups_2), 36 au total. On remarque aussi 4/5 (soit 251 883 individus) des aliments n’ont pas de groupe ou de sous-groupe.
# Les 13 groupes d'aliments sont les suivants :
#%%
pnns_groups_1.unique()
#%%
off.pnns_groups_2.unique()
#%% md
# Pour les variables quantitatives on retrouve bien les 60 constituants comprenant les énergies (au nombre de 3) et les 7 principales familles de nutriments :
# + Les protéines comportant les variables protéines et protéines brutes,
# + Les glucides (11) comportant les variables glucides, sucre, amidon, polyols et alcool,
# + Les lipides (38) comportant les différentes variables d’acides gras saturés (au nombre de 17) et les variables lipides, acides organiques et cholestérol,
# + Les minéraux (8) comportant les variables sel, cendres, calcium, magnésium, phosphore, potassium, sodium et chlorure,
# + Les oligoéléments (6) comportant les variables fer, zinc, iode, manganèse, sélénium et cuivre,
# L’eau
# + Les vitamines (14) comprenant les 12 variables vitamines et les variables rétinol et bêta-carotène.
# 
# Les nutriments ne sont pas tous de la même unité : les énergies sont en kcal/100g ou kJ/100g, les nutriments énergétiques (lipides, glucides et protéines) sont en g/100g excepté le cholestérol qui est en mg/100g, les minéraux sont en mg/100g excepté le sel et les cendres qui sont en g/100g, l’eau est en g/100g, les oligoéléments sont en mg/100g excepté l’iode et le sélénium qui sont en μg/100g et les vitamines sont en
# μg/100g excepté les vitamines E, C, B1, B2, B3, B5 et B6 qui sont en mg/100g.
# On observe 3 valeurs énergétiques différentes. En effet, l’énergie des aliments peut-être calculée de plusieurs méthodes :
# + Les valeurs d’énergie, Règlement UE N° 1169/2011 prend en compte la teneur en protéines brutes, c’est-à-dire la teneur en azote total multipliée par le facteur 6,25, quel que soit l’aliment.
# + Les valeurs d’énergie, N x facteur Jones avec fibres, en kJ/100g ou en kcal/100 g, sont calculées en prenant en compte les teneurs en protéines qui sont elles-mêmes estimées sur la base de la teneur en azote total et de facteurs spécifiques (dits facteurs de Jones), qui peuvent différer d’une famille d’aliments à une autre.
# 
# 
#%% md
# ##### Amputation des individus sur la localité France
#%% md
# 
#%% md
# ### Series.str
# > [Str](https://pandas.pydata.org/docs/reference/api/pandas.Series.str.html)
# Fonctions de chaîne vectorisées pour les séries et les index.
# Les NA restent NA à moins qu'ils ne soient traités autrement par une méthode particulière. Inspiré des méthodes de chaîne de Python, avec une certaine inspiration du package stringr de R
# >[contains](https://pandas.pydata.org/docs/reference/api/pandas.Series.str.contains.html)
# Teste si un motif ou une expression régulière est contenu dans une chaîne d'une série ou d'un index.
# 
# ci-dessous on utilisera les fonctions pour ne retenir que les individus dont la diffusion se situe en France
#%%
off_fr = off.where(off.countries.str.contains('fr', flags=re.IGNORECASE, regex=True)).dropna(subset=['countries']).copy()
off_fr.shape
#%%
# vérifions
off_fr.countries.unique()
#%% md
# #### Amputation sur les colonnes présentant trop valeurs manquantes
# ##### Fonction
#%% md
# Créer une nouvelle fonction qui détermine si la valeur en paramètre est manquante:
#%%
def mean_col_missing(x):
    return x.isnull().mean()
#%% md
# selection de colonne sur un seuil minimum de donnée manquante
#%%
def select_threshold(data, seuil):
    data = data[data.apply(mean_col_missing).where(data.apply(mean_col_missing) < seuil).dropna().keys()].copy()
    return data
#%% md
# On applique cette fonction pour chaque colonne:
#%%
print("Moyennes valeurs manquantes par colonne:")
off_fr.apply(mean_col_missing, axis=0) #axis=0 définit que la fonction sera bien appliquée sur chaque colonne
#%% md
# On applique nos fonctions pour sélectionner seulement dont les données manquantes sont inférieurs à un seuil
#%%
off_clean_col = select_threshold(off_fr,0.55)
off_clean_col
#%% md
# #### Recherche de colonne qualitative ordinale avec des valeurs uniques.
#%%
off_clean_col.isnull().sum()
#%% md
# Parmis les colonnes, la colonne `code` correspond aux codes barres des produits, exactement ce que nous recherchons.
# > Nous pourrons nous servir de cette variable pour la recherche de valeurs dupliquées.
# 
# La fonction ci-dessous va controller et convertir nos donnees pour corriger les erreurs lexicales de type numérique entière.
# Sachant qu'un [code barre](https://fr.wikipedia.org/wiki/Code-barres_EAN) répond à une norme.
#%% md
# #### Fonction
#%%
def numeric_lexical_error(x):
    if type(x) is int:
        y = x
    elif len(x)<19:
        y = pa.to_numeric(x, downcast='integer')
    else:
        y=np.nan
    return y
#%% md
# Recherche de code trop long remplacer par de NaN et récupération des index pour amputation.
# Après avoir traité les codes on les transforme en qualitative ordinal.
#%%
code_numeric = off_clean_col.code.apply(numeric_lexical_error)
i_code_numeric = code_numeric[code_numeric.isna()].index
off_code_oversize = off.iloc[i_code_numeric]
select_threshold(off_code_oversize, 0.50) # aucune quantitative ---> suppression
off_fr_uni = off_clean_col.drop(i_code_numeric).copy()
off_fr_uni.code.astype(str)
off_fr_uni.code = off_fr_uni.code.apply(lambda _:str(_))
#%%
off_fr_uni.code.describe()
#%% md
# ## Recherche des valeurs dupliquées du Dataframe
# 
# ### Critère
# 
# Dans notre Data frame chaque code se doit d'être unique.
# + `.duplicated()` fonction [duplicated](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.duplicated.html?highlight=duplicated#pandas.DataFrame.duplicated) qui renvoie une série booléenne indiquant les lignes en double.
# + `.drop_duplicates()` fonction [drop_duplicates](https://pandas.pydata.org/docs/reference/api/pandas.Series.drop_duplicates.html#pandas.Series.drop_duplicates) qui renvoie la série avec les valeurs en double supprimées.
# Méthode pour gérer la suppression des doublons :
#  + `first` : Supprime les doublons sauf pour la première occurrence.
#  + `last` : supprime les doublons à l'exception de la dernière occurrence.
#  + `False` : supprimez tous les doublons.
#%% md
# #### fonction
#%%
# la fonction supprime tous les individus passés par la variable i_rows qui transmet le(s) index
def select_drop(x, i_rows):
    y = x.drop(i_rows).copy()
    return y

def list_duplicated(x):
   return x[x.duplicated()]

# façon plus élégante de faire un describe ;-)
def describe_columns(x):
    return x.describe()
#%% md
# 
#%%
# keep permet d'afficher tous les doubles pas seulement les valeurs en doubles
off_fr_uni[off_fr_uni.duplicated(subset=['code'],keep=False)]
#%%
my_list = [519,9892,67371,80034,283605,315440,320747]
# print(type(my_list))
off_drop_code_dupl = off_fr_uni.apply(select_drop, i_rows=my_list)
off_fr_dupl = off_drop_code_dupl.apply(list_duplicated)
off_fr_dupl.isna().sum()
#%% md
# On recherche maintenant les duplicata des produits d'une même marque
#%%
produit = pa.DataFrame(off_fr_dupl.product_name)
marques = pa.DataFrame(off_fr_dupl.brands)
reference = produit.join(marques)
reference_dupl = reference.duplicated()
reference_dupl.sum()
#%%
i_product_dupli_nan = off_fr_dupl[reference_dupl].where(off_fr_dupl.product_name.isna()).dropna(subset=['code']).index
off_drop_prod_na = off_fr_dupl.apply(select_drop, i_rows=i_product_dupli_nan)
off_drop_prod_na.apply(describe_columns)
#%%
off_drop_prod_na = off_drop_code_dupl.apply(select_drop, i_rows=i_product_dupli_nan)
off_drop_prod_na.apply(describe_columns)
#%% md
# ## Analyse des variables qualitatives
# ### Diagrammes en Bâtons du nombre d’aliments
#%%
column = pnns_groups_1
off_drop_prod_na.pnns_groups_1.value_counts().sort_values(ascending=False)
#%%
off_drop_prod_na.pnns_groups_1.value_counts().describe()
#%%

sns.set()
histogramme = sns.countplot(data=off_drop_prod_na, x="pnns_groups_1")
histogramme.set_title(f"Histogramme correspondant au \nnombre d'aliments dans chaque groupe", fontsize=25)
for p in histogramme.patches:
    histogramme.annotate(f'\n{p.get_height()}', (p.get_x()+0.2, p.get_height()), ha='center', va='top', color='white', size=18)
histogramme.set_xlabel("Nom des groupes d'aliments", fontsize=20)
histogramme.set_ylabel("nombres d'individus", fontsize=20)
histogramme.set_xticklabels(histogramme.get_xticklabels(), rotation=40, ha="right", fontdict={'fontsize': 14})
plt.legend(labels=['Somme des individus du groupe'], fontsize=14)
plt.gcf().set_size_inches(15,8)
plt.tight_layout()
plt.show()
#%%
off_drop_prod_na.pnns_groups_2.value_counts().describe()
#%%
off_drop_prod_na.pnns_groups_2.value_counts().sort_values(ascending=False)
#%% md
# 
#%%
sns.set()
histogramme = sns.countplot(data=off_drop_prod_na, x="pnns_groups_2")
histogramme.set_title(f"Histogramme correspondant au \nnombre d'aliments dans chaque sous-groupe", fontsize=25)
histogramme.set_xlabel("Nom des sous-groupes d'aliments", fontsize=20)
histogramme.set_ylabel("nombres d'individus", fontsize=20)
histogramme.set_xticklabels(histogramme.get_xticklabels(), rotation=40, ha="right", fontdict={'fontsize': 14})
plt.legend(labels=['Somme des individus du groupe'], fontsize=14)
plt.gcf().set_size_inches(15,8)
plt.tight_layout()
plt.show()
#%% md
# ### Commentaires
# 
# Pour la variable catégorielle pnns_groups_1, on observe que le groupe contenant le plus grand nombre d’aliments sont les snacks sucrés avec 9882 aliments et le plus bas est le groupe snack salé avec 2124 aliments.
# De plus, pour la variable catégorielle pnns_groups_2, on constate que le groupe contenant le plus grand nombre d’aliments est plat cuisiné avec 4948 aliments et le plus bas est le groupe Produits salés et gras avec 15 aliments.
# On en déduit alors, qu’importe la variable catégorielle, les aliments ne sont pas répartis de façon homogène dans les différents sous-groupes.
# On observe que le sous-groupe contenant le plus d’aliments, plat cuisiné, n’appartient pas au groupe possédant le plus grand nombre d’aliments snack sucré.
#%%
off_drop_prod_na.pnns_groups_1.value_counts().keys()
#%%
data = off_drop_prod_na.pnns_groups_1.value_counts()
group_name = data.keys()
labels_pnns_1 = group_name
explode = [0.1, 0, 0, 0, 0, 0, 0, 0, 0.1]
colors = sns.color_palette('bright')
plt.subplots(figsize=(13, 13))
plt.pie(data,
        colors = colors,
        autopct = lambda x: str(round(x, 2)) + '%',
        explode=explode,
        shadow=True,
        startangle = 90,
        textprops = {'color': 'Green','fontsize':16},
        wedgeprops = {'linewidth': 3},
        frame = 'true' ,
        center = (0.1,0.1),
        rotatelabels = 'true')
plt.title(f"Camembert représentatif du \nnombre d'aliments dans chaque groupe", fontsize=25)
plt.legend(labels=labels_pnns_1, title='Groupe des aliments', fontsize = 'large', title_fontsize = "20", loc = 2, bbox_to_anchor = (1,1))
plt.show()
#%% md
# ### Commentaire
# 
# Pour la variable catégorielle pnns_groups_1, la catégorie snack sucré contient 18.86% des aliments, 5 catégories sont aux alentours des 12 % de l’ensemble des aliments ou encore la catégorie snack salé ne contient que 4.05% des aliments.
# On en conclut qu’il y a une répartition uniforme des produits dans chaque groupe. En effet, comme on a pu le voir précédemment, nous avons des groupes qui ont plus de sous-groupes que d’autres, mais avec peu de produits. D’autre part, nous avons des groupes qui ont moins de sous-groupes que d’autres, mais contenant beaucoup de produits.
# 
#%% md
# 
#%% md
# ## Imputation & recherche des valeurs quantitatives nutritionnelles aberrantes
#%% md
# 
#%% md
# #### Fonction
#%%
def value_outlet(x):
    sup_100 = x.apply(lambda y: y > 100)
    return x[sup_100]

def replace_empty_val(x,i_rows):
    y = x.fillna(i_rows).copy()
    return y

def replace_mean(df):
    return df.fillna(df.mean()).copy()

def replace_median(df):
    return df.fillna(df.median()).copy()
#%%
off_quantitative = off_drop_prod_na[[
    'code',
    'additives_n',
    'ingredients_from_palm_oil_n',
    'ingredients_that_may_be_from_palm_oil_n',
    'energy_100g',
    'fat_100g',
    'saturated_fat_100g',
    'carbohydrates_100g',
    'sugars_100g',
    'fiber_100g',
    'proteins_100g',
    'salt_100g',
    'sodium_100g',
    'nutrition_score_fr_100g'
]].copy()
off_nutri_100g = off_quantitative.iloc[:,5:13].copy()
#%% md
# ### Representation de notre Dataframe Quantitatif avant traitement
#%%
data_brut = off_quantitative.drop(columns=['code','energy_100g','ingredients_that_may_be_from_palm_oil_n','ingredients_from_palm_oil_n'])
label_x=["additifs","Matières grasses","Acides gras saturés ","Glucides","Sucres","Fibres alimentaires","Protéines","Sel","Sodium",'Points Nutri-score']
sns.set_theme(style="ticks", palette="pastel")

fig, ax = plt.subplots(figsize=(15, 6))
fig.canvas.manager.set_window_title('A Boxplot Quantitative uni-variate')
fig.subplots_adjust(left=0.075, right=0.95, top=1.9, bottom=.75)

# Initialize the figure with a logarithmic y axis
# ax.set_yscale("log")
ax.xaxis.grid(True, linestyle='-', which='major', color='lightgrey',
               alpha=0.5)
ax.set(xlabel="Informations nutritionnelles pour 100gr." , ylabel="nombres d'individus")
# ax.set_xscale(rt)
boxplot_num = sns.boxplot(data=data_brut, medianprops=dict(color="red", alpha=0.7),showfliers=False)
# stripplot_num = sns.stripplot(data=data,jitter=0.0001,palette="Set2", size=4, marker="D",
#                    edgecolor="gray", alpha=.25, linewidth=1)
boxplot_num.set_title("Représentation Uni-varié des quantitatives avant imputation par boxplot", fontsize=20)
boxplot_num.set_xticklabels(label_x, rotation=30, fontsize=10)
sns.despine(offset=10, trim=True)
#%%
off_nutri_outlet = off_nutri_100g.apply(value_outlet).copy()
i_outlet = off_nutri_outlet.index
off_nutri_dp_outlet = off_nutri_100g.apply(select_drop,i_rows=i_outlet).copy()
#%%
off_nutri_100g_drop_sub_cat = off_nutri_dp_outlet.drop(columns=['saturated_fat_100g','sugars_100g','sodium_100g'])
# calculons la somme des valeurs manquantes
sum_isna_row = off_nutri_100g_drop_sub_cat.isna().sum(axis=1)
# on va recuperer tout les individus dont ils manquent qu'une valeur dans notre Dataframe quantitatif nutritionnel
row_one_na = sum_isna_row.apply(lambda x:x==1)
# voir si la somme des valeurs de chaque colonne par ligne soit <= a 100
sum_nutri = off_nutri_100g_drop_sub_cat[row_one_na].sum(axis=1)
# verifions si des valeurs sont superieur à 100
sum_up_100 = sum_nutri.apply(lambda x:x>100)
# supprimons ces erreurs
i_sum_up_100 = sum_up_100.where(sum_up_100 == True).dropna().index
sum_nutri_clean = sum_nutri.drop(index=i_sum_up_100).copy()
off_nutri_clean = off_nutri_100g_drop_sub_cat.apply(select_drop,i_rows=i_sum_up_100).copy()
# calculons notre diferrence
dif_nutri = sum_nutri_clean.apply(lambda x: 100-x)
# on va remplacer les valeurs NaN dans notre Df par notre calcul precedent
# on selectionne les individus par les index du tableau et on remplace par la methode fillna
off_nutri_calc = off_nutri_clean.loc[dif_nutri.index].apply(replace_empty_val,i_rows=dif_nutri)
# verifie que notre traitement est celui attendu
# off_nutri_calc.sum(axis=1).where(off_nutri_calc.sum(axis=1) != 100.0).value_counts()
off_nutri_replace = off_nutri_100g_drop_sub_cat.copy()
off_nutri_replace.loc[off_nutri_calc.index, :] = off_nutri_calc[:]
#%%
off_nutri_replace.describe()
#%%
off_nutri_replace.where((off_nutri_replace.sum(axis=1) > 100)).dropna()
#%%
off_nutri_sum_sup_100 = off_nutri_replace.where(
    (off_nutri_replace.sum(axis=1) > 100.1) &
    (off_nutri_replace.fat_100g != off_nutri_replace.fat_100g.median()) &
    (off_nutri_replace.carbohydrates_100g != off_nutri_replace.carbohydrates_100g.median()) &
    (off_nutri_replace.fiber_100g != off_nutri_replace.fiber_100g.median()) &
    (off_nutri_replace.proteins_100g != off_nutri_replace.proteins_100g.median()) &
    (off_nutri_replace.salt_100g != off_nutri_replace.salt_100g.median())
).dropna()
off_nutri_sum_sup_100.count()
#%% md
# Sur la somme des colonnes pour compenser les arrondis on tolère une certaine marge.
# 
#%%
off_nutri_sum_sup_100.where((off_nutri_replace.sum(axis=1) > 106)).dropna()
#%%
columns_select = ['code','product_name','brands','fat_100g','carbohydrates_100g','fiber_100g','proteins_100g','salt_100g']
off_drop_prod_na[columns_select].loc[off_nutri_sum_sup_100.where(off_nutri_replace.sum(axis=1)>110).dropna().index]
#%%
off_quantitative_replace = off_quantitative.copy()
off_quantitative_replace.loc[off_nutri_replace.index, :] = off_nutri_replace[:]
# off_quantitative_replace.apply(select_drop, i_rows=off_nutri_replace.index)
off_quantitative_replace.fillna(off_quantitative.iloc[:], inplace=True)
#%%
off_quantitative_replace.drop(index=off_quantitative_replace.apply(select_drop, i_rows=off_nutri_replace.index).index, inplace=True)
off_quantitative_replace.drop(columns=['ingredients_from_palm_oil_n','ingredients_that_may_be_from_palm_oil_n']).describe()
#%% md
# ## Imputation des valeurs Manquantes
#%% md
# fonctions
#%%
def replace_imput_iter(x,column_ref,column_nan):
    df = pa.DataFrame(column_ref).join(column_nan)
    df.loc[df.index,:] = IterativeImputer().fit_transform(df)[:]
    column_nan.loc[column_nan.index]=df[column_nan.name]
    return x

def linear_regression(y,a,b):
    x = (y - b)/a
    amputation_outlet(x)

def linear_regression_imputation(data_mv,col_imput, col_ref):
    # pour réaliser notre imputation affine on doit selectionne les colonnes pour effectuer un groupages des individus
    # boucle sur les valeurs de sous groupe
    nutri_val = data_mv.nutrition_grade_fr.dropna().unique()
    for n in nutri_val:
        data_frame_subgroup = data_mv.loc[data_mv.nutrition_grade_fr==n] # dataframe divise en sous groupe pour une imputation plus affine
        data_frame_ref = data_frame_subgroup.dropna() # datafrane de reference sans valeur manquante
        x = data_frame_ref[col_imput] # doit recevoir la colonne a impute d'un dataframe sans valeur manquante comme reference pour le calcule dans la fonction linregress()
        y = data_frame_ref[col_ref] # doit recevoir la colonne a de support d'un dataframe sans valeur manquante comme reference pour le calcule dans la fonction linregress()
        slop, interc, r_val, p_value, std_err = linregress(x, y)
        # on remplace les valeurs manquantes par la fonction qui retourne la valeur calculée issu de la droite de régression linaire
        data_mv[col_imput].fillna(data_frame_subgroup[col_ref].apply(linear_regression,a=slop,b=interc),inplace=True)
    return data_mv

def amputation_outlet(x):
    if x > 100 :
        return 100
    elif x < 0 :
        return 0
    else :return x
#%% md
# 
#%% md
# ### Imputation par la médiane
#%%
off_quantitative_subset = off_quantitative_replace.dropna(subset=['saturated_fat_100g','sugars_100g','sodium_100g']).copy()
off_quantitative_subset.describe()
#%%
off_quantitative_replace.describe()
#%%
off_quantitative_replace_median = off_quantitative_replace.copy()
off_quantitative_imput_median = off_quantitative_subset.apply(replace_median)
off_quantitative_replace_median.loc[off_quantitative_imput_median.index, :] = off_quantitative_imput_median[:]
print(off_quantitative_replace_median.shape)
off_quantitative_replace_median.describe()
#%%
data_med = off_quantitative_replace_median.drop(columns=['code','energy_100g','ingredients_that_may_be_from_palm_oil_n','ingredients_from_palm_oil_n'])
label_x=["additifs","Matières grasses","Acides gras saturés ","Glucides","Sucres","Fibres alimentaires","Protéines","Sel","Sodium",'Points Nutri-score']
sns.set_theme(style="ticks", palette="pastel")

fig, ax = plt.subplots(figsize=(15, 6))
fig.canvas.manager.set_window_title('A Boxplot Quantitative uni-variate')
fig.subplots_adjust(left=0.075, right=0.95, top=1.9, bottom=.75)

# Initialize the figure with a logarithmic y axis
# ax.set_yscale("log")
ax.xaxis.grid(True, linestyle='-', which='major', color='lightgrey',
               alpha=0.5)
ax.set(xlabel="Informations nutritionnelles pour 100gr." , ylabel="nombres d'individus")
# ax.set_xscale(rt)
boxplot_num = sns.boxplot(data=data_med, medianprops=dict(color="red", alpha=0.7),showfliers=False)
# stripplot_num = sns.stripplot(data=data,jitter=0.0001,palette="Set2", size=4, marker="D",
#                    edgecolor="gray", alpha=.25, linewidth=1)
boxplot_num.set_title("Représentation Uni-varié des quantitatives imputation médiane par boxplot", fontsize=20)
boxplot_num.set_xticklabels(label_x, rotation=30, fontsize=10)
sns.despine(offset=10, trim=True)
#%% md
# ### Imputation par Iterative Imputer
#%%
my_columns = ['nutrition_grade_fr','pnns_groups_1','pnns_groups_2']
off_quantitative_join_score_pnns = off_quantitative_replace.join(off_drop_prod_na[my_columns]).copy()
off_quantitative_join_score_pnns
#%% md
# 
#%%
off_quantitative_replace_iter = off_quantitative_join_score_pnns.copy()
off_quantitative_iter = off_quantitative_subset.copy()
off_quantitative_iter.apply(replace_imput_iter,column_ref=off_quantitative_iter.carbohydrates_100g,column_nan=off_quantitative_iter.sugars_100g)
off_quantitative_iter.apply(replace_imput_iter,column_ref=off_quantitative_iter.fat_100g,column_nan=off_quantitative_iter.saturated_fat_100g)
off_quantitative_iter.apply(replace_imput_iter,column_ref=off_quantitative_iter.salt_100g,column_nan=off_quantitative_iter.sodium_100g)
off_quantitative_iter.apply(replace_imput_iter,column_ref=off_quantitative_iter.sugars_100g ,column_nan=off_quantitative_iter.carbohydrates_100g)
off_quantitative_iter.apply(replace_imput_iter,column_ref=off_quantitative_iter.saturated_fat_100g ,column_nan=off_quantitative_iter.fat_100g)
off_quantitative_iter.apply(replace_imput_iter,column_ref=off_quantitative_iter.sodium_100g ,column_nan=off_quantitative_iter.salt_100g)
#%%
this_columns = ['fat_100g','saturated_fat_100g','carbohydrates_100g','sugars_100g','fiber_100g','proteins_100g','salt_100g','sodium_100g']
replace_columns = off_quantitative_iter[this_columns]
drop_outlet = off_quantitative_iter.drop(columns=['code','energy_100g','ingredients_that_may_be_from_palm_oil_n','ingredients_from_palm_oil_n','additives_n','nutrition_score_fr_100g'])
treatments = drop_outlet.apply(lambda x:x.apply(amputation_outlet))
off_quantitative_iter[this_columns]=treatments
print(off_quantitative_iter.shape)
off_quantitative_iter.describe()
#%%
off_quantitative_replace_iter.fillna(off_quantitative_iter,inplace=True)
#%%
print(off_quantitative_replace_iter.shape)
off_quantitative_replace_iter.describe(include='all')
#%%

# off_quantitative_replace_iter = off_quantitative_replace_iter.join(off_drop_prod_na[my_columns]).copy()
liste ={'fonctions':['var','skew','kurt']}
stat = pa.DataFrame(liste)
stat.join(off_quantitative_replace_iter.drop(columns=['code','nutrition_grade_fr','pnns_groups_1','pnns_groups_2']).apply(lambda x: [x.var(), x.skew(),x.kurtosis()]))
#%%
data_iter = off_quantitative_replace_iter.drop(columns=['code','energy_100g','ingredients_that_may_be_from_palm_oil_n','ingredients_from_palm_oil_n'])
label_x=["additifs","Matières grasses","Acides gras saturés ","Glucides","Sucres","Fibres alimentaires","Protéines","Sel","Sodium",'Points Nutri-score']
sns.set_theme(style="ticks", palette="pastel")

fig, ax = plt.subplots(figsize=(15, 6))
fig.canvas.manager.set_window_title('A Boxplot Quantitative uni-variate')
fig.subplots_adjust(left=0.075, right=0.95, top=1.9, bottom=.75)

# Initialize the figure with a logarithmic y axis
# ax.set_yscale("log")
ax.xaxis.grid(True, linestyle='-', which='major', color='lightgrey',
               alpha=0.5)
ax.set(xlabel="Informations nutritionnelles pour 100gr." , ylabel="nombres d'individus")
# ax.set_xscale(rt)
boxplot_num = sns.boxplot(data=data_iter, medianprops=dict(color="red", alpha=0.7),showfliers=False)
# stripplot_num = sns.stripplot(data=data,jitter=0.0001,palette="Set2", size=4, marker="D",
#                    edgecolor="gray", alpha=.25, linewidth=1)
boxplot_num.set_title("Représentation Uni-varié des quantitatives imputation itérative par boxplot", fontsize=20)
boxplot_num.set_xticklabels(label_x, rotation=30, fontsize=10)
sns.despine(offset=10, trim=True)
#%% md
# ### Imputation par droite de regression lineaire
#%%
off_quantitative_regress= linear_regression_imputation(off_quantitative_join_score_pnns,'fat_100g','saturated_fat_100g')
off_quantitative_regress= linear_regression_imputation(off_quantitative_join_score_pnns,'carbohydrates_100g','sugars_100g')
off_quantitative_regress= linear_regression_imputation(off_quantitative_join_score_pnns,'salt_100g','sodium_100g')
print(off_quantitative_regress.shape)
off_quantitative_regress.describe()
# off_quantitative_regress
# off_quantitative_regress.isna().sum()
#%%
stat.join(off_quantitative_regress.drop(columns=['code','nutrition_grade_fr','pnns_groups_1','pnns_groups_2']).apply(lambda x: [x.var(), x.skew(),x.kurtosis()]))
#%%
data_regr = off_quantitative_regress.drop(columns=['code','energy_100g','ingredients_that_may_be_from_palm_oil_n','ingredients_from_palm_oil_n'])
label_x=["additifs","Matières grasses","Acides gras saturés ","Glucides","Sucres","Fibres alimentaires","Protéines","Sel","Sodium",'Points Nutri-score']
sns.set_theme(style="ticks", palette="pastel")

fig, ax = plt.subplots(figsize=(15, 6))
fig.canvas.manager.set_window_title('A Boxplot Quantitative uni-variate')
fig.subplots_adjust(left=0.075, right=0.95, top=1.9, bottom=.75)

# Initialize the figure with a logarithmic y axis
# ax.set_yscale("log")
ax.xaxis.grid(True, linestyle='-', which='major', color='lightgrey',
               alpha=0.5)
ax.set(xlabel="Informations nutritionnelles pour 100gr." , ylabel="nombres d'individus")
# ax.set_xscale(rt)
boxplot_num = sns.boxplot(data=data_regr, medianprops=dict(color="red", alpha=0.7),showfliers=False)
# stripplot_num = sns.stripplot(data=data,jitter=0.0001,palette="Set2", size=4, marker="D",
#                    edgecolor="gray", alpha=.25, linewidth=1)
boxplot_num.set_title("Représentation Uni-varié des quantitatives imputation régression linéaire par boxplot", fontsize=20)
boxplot_num.set_xticklabels(label_x, rotation=30, fontsize=10)
sns.despine(offset=10, trim=True)
#%% md
# ### Observations:
# Pour diverses raisons, de nombreux ensembles de données du monde réel contiennent des valeurs manquantes, souvent codées sous forme de blancs, de NaN ou d'autres espaces réservés. De tels ensembles de données sont cependant incompatibles avec les estimateurs scikit-learn qui supposent que toutes les valeurs d'un tableau sont numériques et que toutes ont et conservent une signification. Une stratégie de base pour utiliser des ensembles de données incomplets consiste à supprimer des lignes et/ou des colonnes entières contenant des valeurs manquantes. Cependant, cela se fait au prix de la perte de données qui peuvent être précieuses (même si elles sont incomplètes). Une meilleure stratégie consiste à imputer les valeurs manquantes, c'est-à-dire à les déduire de la partie connue des données.
# Dans ce notebook j’ai pu faire diverse imputation comme avec la médiane ou la moyenne, mais aussi une fonction qui utilise la droite de regression linaire pour imputer, représenter par les graphiques ci-dessus.
# Un type d'algorithme d'imputation est uni-varié, qui impute des valeurs dans l'ième dimension de caractéristique en utilisant uniquement les valeurs non manquantes dans cette dimension de caractéristique (par exemple impute. SimpleImputer). En revanche, les algorithmes d'imputation multivariée utilisent l'ensemble complet des dimensions d'entités disponibles pour estimer les valeurs manquantes (par exemple impute. IterativeImputer).
# Une approche plus sophistiquée consiste à utiliser la IterativeImputer classe, qui modélise chaque caractéristique avec des valeurs manquantes en fonction d'autres caractéristiques et utilise cette estimation pour l'imputation. Il le fait de manière itérative : à chaque étape, une colonne de caractéristiques est désignée comme sortie `y` et les autres colonnes de caractéristiques sont traitées comme des entrées `X`. Un régresseur est adapté sur `(X,y)` pour `y` connu. Ensuite, le régresseur est utilisé pour prédire les valeurs manquantes de `y`.Ceci est fait pour chaque caractéristique de manière itérative, puis est répété pour les cycles d'imputation max_iter. Les résultats du dernier tour d'imputation sont retournés.
#%% md
# ### Imputation de la variable Energie, Règlement UE N 1169/2011 (kcal/100g)
#%% md
# Nous allons nous concentrer sur la variable Energie, Règlement UE N 1169/2011 (kcal/100g) car elle est determinée par d’autres variables de la table. Cette valeur permet de savoir aussi l’apport nutritionnelle des aliments.
#%% md
# D’après les informations fournies par l'Anses sur la table, pour l’ensemble des aliments, la valeur énergétique a été calculée en utilisant les coefficients suivants :
# 
# +    pour les lipides : 37 kJ/g (9 kcal/g) ;
# +   pour l’alcool (ethanol) : 29 kJ/g (7 kcal/g) ;
# +    pour les protéines : 17 kJ/g (4 kcal/g) ;
# +    pour les glucides (a l’exception des polyols) : 17 kJ/g (4 kcal/g) ;
# +    pour les acides organiques : 13 kJ/g (3 kcal/g) ;
# +    pour les polyols : 10 kJ/g (2,4 kcal/g) ;
# +    pour les fibres alimentaires : 8 kJ/g (2 kcal/g).
#%% md
# fonction
#%%
def calcul_nrj(data,i_rows):
    fat=data.fat_100g[i_rows]
    carbo=data.carbohydrates_100g[i_rows]
    prot=data.proteins_100g[i_rows]
    fib = data.fiber_100g[i_rows]
    y = 17 * ((fat * 38 / 17) + carbo + prot + (8 * fib / 17))
    return y


def whisker(data_whisker, whis=1.5):
    q1 = data_whisker.quantile(q=0.25)
    q3 = data_whisker.quantile(q=0.75)
    iqr = q3 - q1
    wr = whis * iqr
    lower_whisker = q1 - wr
    upper_whisker = q3 + wr
    lower_whisker[lower_whisker<0]=0
    return pa.concat([lower_whisker,upper_whisker],axis=1,keys=['lower_whisker','upper_whisker'])
#%%
energy = off_quantitative_replace_iter.energy_100g
i_energy = energy.where(energy > 4000).dropna().index
off_quantitative_nrg = off_quantitative_replace_iter.apply(select_drop,i_rows=i_energy)
energy = off_quantitative_nrg.energy_100g
energy.describe()
#%%
energy_score = pa.DataFrame(off_quantitative_nrg.energy_100g).join(off_quantitative_nrg.nutrition_grade_fr).dropna()
energy_score.groupby(['nutrition_grade_fr']).describe(include='all')
#%% md
# 
#%%
energy_aliment_group = pa.DataFrame(off_quantitative_nrg.energy_100g).join(off_drop_prod_na.pnns_groups_1)
fat_aliment_group = pa.DataFrame(off_quantitative_nrg.fat_100g).join(off_drop_prod_na.pnns_groups_1)
energy_aliment_group.describe(include='all')
#%%
boxplot_c = sns.boxplot(data=off_quantitative_nrg, x='energy_100g', hue ="energy_100g",color='red')
boxplot_c.set_xlabel("Valeur énergétique (kj/100g)", fontsize=20)
boxplot_c.set_ylabel("nombres d'individus", fontsize=20)
plt.title(f"boite à moustache de la valeur énergétique par nombres d'individus", fontsize=25)
plt.gcf().set_size_inches(15,4)
plt.tight_layout()
plt.show()
#%%
fig, ax = plt.subplots(figsize=(16, 6))
fig.subplots_adjust(left=0.075, right=0.95, top=1.9, bottom=.75)
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}

plt.rc('font', **font)
energy_ags = energy_aliment_group.join(off_quantitative_replace_iter.nutrition_grade_fr)
sns.set()
sns.set_theme(style="ticks", palette="pastel")
sns.despine(offset=10,)
boxplot_nrj_ali = sns.boxplot(
    x='pnns_groups_1',
    y='energy_100g',
    data=energy_ags,
    showmeans=True,
    medianprops=dict(color="red", alpha=0.7),
    capprops=dict(color="purple", alpha=0.7))
boxplot_nrj_ali.set_title("Représentation Boxplot\n de la valeur énergétique par groupe d'aliments  ", fontsize=20)
# boxplot_nrj_ali.set_xlabel("Nom des groupes d'aliments", fontsize=20)
boxplot_nrj_ali.set_xticklabels(boxplot_nrj_ali.get_xticklabels(), rotation=40, ha="right", fontsize=14)
ax.xaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.5)
ax.set_xlabel('Répartition des individus selon leurs groupes.', fontsize=20)
ax.set_ylabel('Valeur énergetique (en kj/100g)', fontsize=20)

plt.show()
#%%
order_nrj = np.sort(energy_ags.pnns_groups_1.dropna().unique())
order_nrj
#%%
boxplot_stats(off_quantitative_nrg.energy_100g.dropna().values)
#%%
for i,t in enumerate(off_quantitative_nrg.groupby('pnns_groups_1')):
    test = pa.DataFrame(t[1])
    nrj = (test.energy_100g.dropna())
    stat = boxplot_stats(nrj.values)
    print(stat)
#%%
energy_ags = energy_aliment_group.join(off_quantitative_replace_iter.nutrition_grade_fr)
sns.set()
sns.set_theme(style="ticks", palette="pastel")
sns.despine(offset=10, trim=True)
boxplot_nrj_ali = sns.boxenplot(x='pnns_groups_1', y='energy_100g', data=energy_ags,)
boxplot_nrj_ali.set_title("Représentation Boxenplot\n de la valeur énergétique par groupe d'aliments  ", fontsize=20)
boxplot_nrj_ali.set_xlabel("Nom des groupes d'aliments", fontsize=20)
boxplot_nrj_ali.set_xticklabels(boxplot_nrj_ali.get_xticklabels(), rotation=40, ha="right", fontsize=14)
boxplot_nrj_ali.set_ylabel("Valeur énergetique (en kj/100g)", fontsize=20)
plt.gcf().set_size_inches(45,16)
plt.show()
#%%
caps = []
for t in off_quantitative_nrg.groupby('pnns_groups_1'):
    # print(pa.DataFrame(t))
    test = pa.DataFrame(t[1])
    print(test.pnns_groups_1.unique())
    nrj = (test.energy_100g.dropna())
    bp = plt.boxplot(nrj, showmeans=True)
    caps.append([item.get_ydata()[0] for item in bp['caps']])
columns_group = off_quantitative_nrg.pnns_groups_1.dropna().unique()
#%%
nrj_groupby_pnns = energy_aliment_group.groupby(['pnns_groups_1'])
#%%
nrj_groupby_pnns.describe(include='all')
#%%
whisker(nrj_groupby_pnns)
#%% md
# 
#%%
nrj_groupby_pnns.quantile()
#%%
label_x=['a','b','c','d','e']
sns.set_theme(style="ticks", palette="pastel")

fig, ax = plt.subplots(figsize=(15, 6))
fig.subplots_adjust(left=0.075, right=0.95, top=1.9, bottom=.75)
boxplot_num = sns.boxplot(x='nutrition_grade_fr', y='energy_100g', data=energy_score, medianprops=dict(color="red", alpha=0.7))
swarmplot_num = sns.swarmplot(x="nutrition_grade_fr", y="energy_100g", data=energy_score, color='#7d0013', size=1)
boxplot_num.set_title("Représentation Boxplot de la valeur énergétique par grade nutritionnel  ", fontsize=20)
boxplot_num.set_xticklabels(label_x, fontsize=10)
ax.xaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.5)
ax.set(xlabel="Répartition des individus selon leurs Nutriscore." , ylabel="Valeur énergetique (en kj/100g)")
sns.despine(offset=10, trim=True)
plt.show()
#%%
label_x=['a','b','c','d','e']
sns.set_theme(style="ticks", palette="pastel")

boxplot_num = sns.displot(data=energy_score, x='energy_100g', hue='nutrition_grade_fr',hue_order=label_x, kind='kde', height=8,multiple='fill')
sns.despine(offset=10, trim=True)

# ax.set(xlabel="Répartition des individus selon leurs Nutriscore." , ylabel="Valeur énergetique (en kj/100g)")
plt.show()
#%%
fat_ags = fat_aliment_group.join(off_quantitative_replace_iter.nutrition_grade_fr)
#%%
energy_ags = energy_aliment_group.join(off_quantitative_replace_iter.nutrition_grade_fr)
sns.set()
sns.set_theme(style="ticks", palette="pastel")
sns.despine(offset=10, trim=True)
stripplot_nrj_ali = sns.stripplot(x="pnns_groups_1",
                                  y="energy_100g",
                                  data=energy_ags,
                                  hue="nutrition_grade_fr",
                                  hue_order=['a','b','c','d','e'],
                                  palette="Set2",
                                  size=4,
                                  jitter=True,
                                  split=True
                                  )
stripplot_nrj_ali.set_title("Représentation Stripplot de la valeur énergétique\n par groupe d'aliments et des nutri-score  ", fontsize=20)
stripplot_nrj_ali.set_xlabel("Nom des groupes d'aliments", fontsize=20)
stripplot_nrj_ali.set_xticklabels(stripplot_nrj_ali.get_xticklabels(), rotation=40, ha="right", fontsize=14)
stripplot_nrj_ali.set_ylabel("Valeur énergetique (en kj/100g)", fontsize=20)
handles, labels = stripplot_nrj_ali.get_legend_handles_labels()
plt.legend(handles[0:5],
           labels[0:5],
           title='Nutri-score des aliments',
           fontsize = 'large',
           title_fontsize = "20",
           loc = 2,
           bbox_to_anchor = (1.05,1),
           borderaxespad=0.)
plt.gcf().set_size_inches(15,8)
plt.show()
#%%
energy_ags_drna = energy_ags.dropna()
for categ in energy_ags_drna.pnns_groups_1.unique():
    sous_categ = energy_ags_drna[energy_ags_drna.pnns_groups_1 == categ]
    boxplot_nrj_ali = sns.boxplot(y='pnns_groups_1', x='energy_100g', data=sous_categ, medianprops=dict(color="red", alpha=0.7),)
    stripplot_nrj_ali = sns.stripplot(y="pnns_groups_1",
                                  x="energy_100g",
                                  data=sous_categ,
                                  hue="nutrition_grade_fr",
                                  hue_order=['a','b','c','d','e'],
                                  palette="Set2",
                                  size=4,
                                  jitter=True,
                                  split=True
                                  )
    stripplot_nrj_ali.set_title(f"Représentation Stripplot de la valeur énergétique\n pour la categorie {categ} et des nutri-score  ", fontsize=20)
    stripplot_nrj_ali.set_ylabel("Nom des groupes d'aliments", fontsize=20)
    stripplot_nrj_ali.set_xlabel("Valeur énergetique (en kj/100g)", fontsize=20)
    handles, labels = stripplot_nrj_ali.get_legend_handles_labels()
    plt.legend(handles[0:5],
               labels[0:5],
               title='Nutri-score des aliments',
               fontsize = 'large',
               title_fontsize = "20",
               loc = 2,
               bbox_to_anchor = (1.05,1),
               borderaxespad=0.)
    plt.gcf().set_size_inches(15,8)
    plt.show()

#%% md
# ### Observations
# Lorsque nous observons la distribution de la variable numérique Energies, Règlement UE N° 1169/2011 (kJ/100 g) en fonction des différents groupes, on s’aperçoit qu’elles sont différentes. En effet, pour le groupe boissons et celui des fruits et légumes, il n’y a pas de grande dispersion entre les valeurs énergétiques et le nutri-score A est le plus représentatif. Elles se situent entre 0 kJ/100 g et 365 kJ/100 g. Or, pour le groupe matière grasse et sauce, on a une très grande distribution des valeurs énergétiques, comprises entre 0 et 6784 kJ/100 g le nutri-score D est le plus représentatif. On peut donc dire que dans ce groupe que 75% des aliments ont un très grand apport énergétique, entre 1033 kJ et 6784 kJ/100 g, par rapport aux autres aliments des autres groupes. Le groupe snack sucré est le second groupe ayant des aliments qui ont un apport énergétique entre 557.0 et 3125.0 kJl/100 g, soit une différence de 2568 kJ/100 g, mais ont la représentation de nutri-score E la plus importante, ainsi que dans les boissons. On remarque aussi que les groupes boissons, fruits, légumes, légumineux et oléagineux et viande, œufs, poissons ont de nombreuses valeurs extrêmes. De plus, les aliments du groupe boissons sont les derniers en apport énergétique, 50% de ces aliments ont une valeur énergétique comprise entre 0 et 184 kJ/100 g.
# 
#%%

lo_up_wr = whisker(nrj_groupby_pnns)
nrj_desc = nrj_groupby_pnns.describe()
median = energy_aliment_group.energy_100g.median()
energy_aliment_group[energy_aliment_group.energy_100g > median].count()/energy_aliment_group.dropna().count()
# energy_aliment_group.dropna().count()
#%%
stat_groupe = pa.concat([nrj_desc,lo_up_wr],axis=1)
stat_groupe.loc['Fat and sauces','upper_whisker']=caps[3][1]
stat_groupe
#%% md
# 
#%% md
# 
#%%
sns.set_theme(style="ticks", palette="pastel")
sns.displot(data=energy_aliment_group, x='energy_100g', hue='pnns_groups_1', kind='kde', height=8,multiple='fill',)
sns.despine(offset=10, trim=True)
plt.gcf().set_size_inches(15,8)
plt.show()
#%%
sns.set_theme(style="ticks", palette="pastel")
histogramme = sns.displot(data=energy_aliment_group, x='energy_100g',kde=True, height=8, color='green')
histogramme.set_xlabels("energie (kJ/100g", fontsize=20)
histogramme.set_ylabels("nombres d'individus", fontsize=20)
sns.despine(offset=10, trim=True)
plt.title(f"Repartition des valeur énergétique \npar nombres d'individus", fontsize=25)
plt.legend(labels=['Estimation de la densité du noyau','valeur énergétique'], fontsize=14)
plt.gcf().set_size_inches(15,8)
# plt.tight_layout()
plt.show()
#%% md
# ### Observations
# Lorsque nous regardons la densité des aliments en fonction de la valeur énergétique, on observe une décroissance, donc plus la valeur énergétique augmente, plus le nombre d’aliments diminue. Ainsi, sur l’histogramme, on observe un pic avec plus de 3400 aliments qui ont une valeur énergétique comprise entre 180 et 240 kJ/100 g alors qu’entre 3150 et 32 kcal/100 g, mais aussi entre 3550 et 3750, on peut voir qu'aucun aliment ne possède de valeur énergétique. À partir de 2250 kJ/100 g on voit une nette décroissance du nombre d’aliments en fonction de l’énergie.
#%%
energy_ags
#%%
# generate a boxplot to see the data distribution by genotypes and years. Using boxplot, we can easily detect the
# differences between different groups
boxplot_nrj_ags_m = sns.boxplot(x="pnns_groups_1",
                                y="energy_100g",
                                hue="nutrition_grade_fr",
                                hue_order=['a','b','c','d','e'],
                                data=energy_ags,
                                palette="Set1",
                                showmeans=True,
                                medianprops=dict(color="red", alpha=0.7),
                                capprops=dict(color="purple", alpha=0.7),
                                )
boxplot_nrj_ags_m.set_title("Anova de la valeur énergétique par groupe d'aliments \net des nutri-score  ", fontsize=20)
boxplot_nrj_ags_m.set_xlabel("Nom des groupes d'aliments", fontsize=20)
boxplot_nrj_ags_m.set_xticklabels(boxplot_nrj_ags_m.get_xticklabels(), rotation=40, ha="right", fontsize=14)
boxplot_nrj_ags_m.set_ylabel("Valeur énergetique (en kj/100g)", fontsize=20)
handles, labels = boxplot_nrj_ags_m.get_legend_handles_labels()
plt.legend(handles[0:5],
           labels[0:5],
           title='Nutri-score des aliments',
           fontsize = 'large',
           title_fontsize = "20",
           loc = 2,
           bbox_to_anchor = (1.05,1),
           borderaxespad=0.)
plt.gcf().set_size_inches(20,8)
#%% md
# Ce graphique montre l'intérêt de traiter les valeurs aberrantes, qui de manière disparate sur les different nutri-score auront une influence significative sur l'étude de variance.
#%%
# reshape the d dataframe suitable for statsmodels package
energy_ags_melt = pa.melt(energy_ags.reset_index(), id_vars=['index','pnns_groups_1','nutrition_grade_fr'], value_vars=['energy_100g']).dropna()
# replace column names
energy_ags_melt.columns = ['index','pnns_groups_1','nutrition_grade_fr','treatments', 'value']
energy_ags_melt
#%%
energy_ags
#%%
# reshape the d dataframe suitable for statsmodels package
fat_ags_melt = pa.melt(fat_ags.reset_index(), id_vars=['index','pnns_groups_1','nutrition_grade_fr'], value_vars=['fat_100g']).dropna()
# replace column names
fat_ags_melt.columns = ['index','pnns_groups_1','nutrition_grade_fr','treatments', 'value']
#%%
fat_ags.groupby('nutrition_grade_fr').pnns_groups_1.describe()
#%%
fat_ags_melt.pnns_groups_1.unique()
#%%
fat_ags_melt
#%%
beverages = energy_ags_melt.where(energy_ags_melt.pnns_groups_1=='Beverages').dropna().copy()
sugary_snacks = energy_ags_melt.where(energy_ags_melt.pnns_groups_1=='Sugary snacks').dropna().copy()
composite_foods = energy_ags_melt.where(energy_ags_melt.pnns_groups_1=='Composite foods').dropna().copy()
fruits_and_vegetables = energy_ags_melt.where(energy_ags_melt.pnns_groups_1=='Fruits and vegetables').dropna().copy()
milk_and_dairy_products = energy_ags_melt.where(energy_ags_melt.pnns_groups_1=='Milk and dairy products').dropna().copy()
fat_and_sauces = energy_ags_melt.where(energy_ags_melt.pnns_groups_1=='Fat and sauces').dropna().copy()
salty_snacks = energy_ags_melt.where(energy_ags_melt.pnns_groups_1=='Salty snacks').dropna().copy()
cereals_and_potatoes = energy_ags_melt.where(energy_ags_melt.pnns_groups_1=='Cereals and potatoes').dropna().copy()
fish_meat_eggs = energy_ags_melt.where(energy_ags_melt.pnns_groups_1=='Fish meat eggs').dropna().copy()
#%%
energy_ags_melt.drop(['index'],axis=1).groupby('pnns_groups_1').describe()
#%%
from statsmodels.formula.api import ols

group_name = energy_ags_melt.pnns_groups_1.unique()
anova_table = []
for name in group_name:
    data_group = energy_ags_melt.where(energy_ags_melt.pnns_groups_1== name).dropna()
    model = ols(f'value ~ C(nutrition_grade_fr)', data=data_group).fit()
    print(sm.stats.anova_lm(model, typ=2))
#%%
nrj_melt = energy_ags_melt.drop(['index'],axis=1).groupby('pnns_groups_1')
#%%
# ANOVA table using bioinfokit v1.0.3 or later (it uses wrapper script for anova_lm)
from bioinfokit.analys import stat
res = stat()
res.anova_stat(df=energy_ags, res_var='energy_100g', anova_model='energy_100g ~ C(pnns_groups_1)')
res.anova_summary
# output (ANOVA F and p value)
# note: if the data is balanced (equal sample size for each group), Type 1, 2, and 3 sums of squares
# (typ parameter) will produce similar results.
#%%
# get ANOVA table as R like output
import statsmodels.api as sm


model = ols('energy_100g ~ C(pnns_groups_1) + C(nutrition_grade_fr) + C(pnns_groups_1):C(nutrition_grade_fr)', data=energy_ags).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
anova_table
#%%
res = stat()
res.anova_stat(df=energy_ags, res_var='energy_100g', anova_model='energy_100g~C(pnns_groups_1)+C(nutrition_grade_fr)+C(pnns_groups_1):C(nutrition_grade_fr)')
res.anova_summary
#%%
from statsmodels.graphics.factorplots import interaction_plot
fig = interaction_plot(x=energy_ags['pnns_groups_1'], trace=energy_ags['nutrition_grade_fr'], response=energy_ags['energy_100g'],
    colors=['red','blue', 'green', 'pink', 'orange'])
plt.legend(handles[0:5],
           labels[0:5],
           title='Nutri-score des aliments',
           fontsize = 'large',
           title_fontsize = "20",
           loc = 2,
           bbox_to_anchor = (1.05,1),
           borderaxespad=0.)
plt.gcf().set_size_inches(20,8)
plt.show()
#%%
sm.qqplot(res.anova_std_residuals, line='45')
plt.xlabel("Theoretical Quantiles")
plt.ylabel("Standardized Residuals")
plt.show()

# histogram
plt.hist(res.anova_model_out.resid, bins='auto', histtype='bar', ec='k')
plt.xlabel("Residuals")
plt.ylabel('Frequency')
plt.show()

# Shapiro-Wilk test
import scipy.stats as stats
w, pvalue = stats.shapiro(res.anova_model_out.resid)
print(f"La statistique de test : {w}, La valeur de p pour le test d'hypothèse : {pvalue}")
#%%
res = stat()
res.levene(df=energy_ags, res_var='energy_100g', xfac_var=['pnns_groups_1', 'nutrition_grade_fr'])
res.levene_summary
#%%
sugary_snacks
#%%
# Ordinary Least Squares (OLS) model
model = ols(f'value ~ C(nutrition_grade_fr)', data=sugary_snacks).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
anova_table
# output (ANOVA F and p value)
#%% md
# Le test de F montre une grande dispersion donc la moyenne des groupes est tres écartée, comme la probabilité (PR) = 0 nos données sont incompatibles avec l'hypothèse nulle et le modèle est globalement significatif, donc nous pouvons conclure que toutes les moyennes des groupes ne sont pas égales.
#%%
# ANOVA table using bioinfokit v1.0.3 or later (it uses wrapper script for anova_lm)
from bioinfokit.analys import stat
res = stat()
res.anova_stat(df=sugary_snacks, res_var='value', anova_model='value ~ C(nutrition_grade_fr)')
res.anova_summary
# output (ANOVA F and p value)
# note: if the data is balanced (equal sample size for each group), Type 1, 2, and 3 sums of squares
# (typ parameter) will produce similar results.
#%%
sm.qqplot(res.anova_std_residuals, line='45')
plt.xlabel("Theoretical Quantiles")
plt.ylabel("Standardized Residuals")
plt.show()

# histogram
plt.hist(res.anova_model_out.resid, bins='auto', histtype='bar', ec='k')
plt.xlabel("Residuals")
plt.ylabel('Frequency')
plt.show()

# Shapiro-Wilk test
import scipy.stats as stats
w, pvalue = stats.shapiro(res.anova_model_out.resid)
print(f"La statistique de test : {w}, La valeur de p pour le test d'hypothèse : {pvalue}")
#%% md
# ### Observation
# Comme les résidus standardisés se situent autour de la ligne de 45 degrés, cela suggère que les résidus sont à peu près normalement distribués
# 
# Dans l'histogramme, la distribution semble approximativement normale et suggère que les résidus sont à peu près normalement distribués
#%% md
# ### Interpretation
# La valeur p obtenue à partir de l'analyse ANOVA est significative ( p < 0,05) et, par conséquent, nous concluons qu'il existe des différences significatives entre les traitements.
#%% md
# D'après l'analyse ANOVA, nous savons que les différences de traitement sont statistiquement significatives, mais l'ANOVA ne dit pas quels traitements sont significativement différents les uns des autres. Pour connaître les paires de traitements différents significatifs, nous effectuerons une analyse de comparaison par paires multiples (comparaison post hoc ) pour toutes les comparaisons non planifiées à l'aide du test honnêtement significativement différent (HSD) de Tukey .
#%%
# ANOVA table using bioinfokit v1.0.3 or later (it uses wrapper script for anova_lm)
from bioinfokit.analys import stat
res = stat()
res.tukey_hsd(df=sugary_snacks, res_var='value', xfac_var='nutrition_grade_fr', anova_model='value ~ C(nutrition_grade_fr)')
res.tukey_summary
# output (ANOVA F and p value)
# note: if the data is balanced (equal sample size for each group), Type 1, 2, and 3 sums of squares
# (typ parameter) will produce similar results.
#%% md
# Les résultats ci-dessus du HSD de Tukey suggèrent, quasiment toutes les autres comparaisons par paires pour les traitements rejettent l'hypothèse nulle ( p <0,05) et indiquent des différences statistiquement significatives.
# Sauf pour nutri-score B/A qui tres proche des 5%, ce qui laisse supposer des similitudes statistiques.
# 
#%%
res.levene(df=sugary_snacks, res_var='value', xfac_var='nutrition_grade_fr')
res.levene_summary
#%% md
# la valeur p résultante du test de Levene est inférieure à un niveau de signification (typiquement 0,05), il est peu probable que les différences obtenues dans les variances d'échantillon se soient produites sur la base d'un échantillonnage aléatoire d'une population à variances égales. Ainsi, l'hypothèse nulle d'égale variance est rejetée et il est conclu qu'il existe une différence entre les variances dans la population.
#%%
# generate a boxplot to see the data distribution by genotypes and years. Using boxplot, we can easily detect the
# differences between different groups
fig, ax = plt.subplots(figsize=(16, 6))
fig.subplots_adjust(left=0.075, right=0.95, top=1.9, bottom=.75)
font = {'family' : 'arial',
        'weight' : 'bold',
        'size'   : 22}

plt.rc('font', **font)
boxplot_nrj_ags_m = sns.boxplot(x="pnns_groups_1",
                                y="value",
                                hue="nutrition_grade_fr",
                                hue_order=['a','b','c','d','e'],
                                data=sugary_snacks,
                                palette="Set1",
                                showmeans=True,
                                medianprops=dict(color="red", alpha=0.7),
                                capprops=dict(color="purple", alpha=0.7),
                                showfliers=False)
boxplot_nrj_ags_m.set_title("Anova de la valeur énergétique pour les snack sucré \net des nutri-score  ", fontsize=20)
boxplot_nrj_ags_m.set_xlabel("Nom des groupes d'aliments", fontsize=20)
boxplot_nrj_ags_m.set_xticklabels(boxplot_nrj_ags_m.get_xticklabels(), rotation=40, ha="right", fontsize=14)
boxplot_nrj_ags_m.set_ylabel("Valeur énergetique (en kj/100g)", fontsize=20)
handles, labels = boxplot_nrj_ags_m.get_legend_handles_labels()
plt.legend(handles[0:5],
           labels[0:5],
           title='Nutri-score des aliments',
           fontsize = 'large',
           title_fontsize = "20",
           loc = 2,
           bbox_to_anchor = (1.05,1),
           borderaxespad=0.)
ax.xaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.5)
plt.show()
#%% md
# 
#%%
# ANOVA table using bioinfokit v1.0.3 or later (it uses wrapper script for anova_lm)
from bioinfokit.analys import stat
res = stat()
res.tukey_hsd(df=beverages, res_var='value', xfac_var='nutrition_grade_fr', anova_model='value ~ C(nutrition_grade_fr)')
res.tukey_summary
# output (ANOVA F and p value)
# note: if the data is balanced (equal sample size for each group), Type 1, 2, and 3 sums of squares
# (typ parameter) will produce similar results.
#%%
# generate a boxplot to see the data distribution by genotypes and years. Using boxplot, we can easily detect the
# differences between different groups
fig, ax = plt.subplots(figsize=(16, 6))
fig.subplots_adjust(left=0.075, right=0.95, top=1.9, bottom=.75)
font = {'family' : 'arial',
        'weight' : 'bold',
        'size'   : 22}

plt.rc('font', **font)
boxplot_nrj_ags_m = sns.boxplot(x="pnns_groups_1",
                                y="value",
                                hue="nutrition_grade_fr",
                                hue_order=['a','b','c','d','e'],
                                data=beverages,
                                palette="Set1",
                                showmeans=True,
                                medianprops=dict(color="red", alpha=0.7),
                                capprops=dict(color="purple", alpha=0.7),
                                showfliers=False)
boxplot_nrj_ags_m.set_title("Anova de la valeur énergétique pour les boissons \net des nutri-score  ", fontsize=20)
boxplot_nrj_ags_m.set_xlabel("Nom des groupes d'aliments", fontsize=20)
boxplot_nrj_ags_m.set_xticklabels(boxplot_nrj_ags_m.get_xticklabels(), rotation=40, ha="right", fontsize=14)
boxplot_nrj_ags_m.set_ylabel("Valeur énergetique (en kj/100g)", fontsize=20)
handles, labels = boxplot_nrj_ags_m.get_legend_handles_labels()
plt.legend(handles[0:5],
           labels[0:5],
           title='Nutri-score des aliments',
           fontsize = 'large',
           title_fontsize = "20",
           loc = 2,
           bbox_to_anchor = (1.05,1),
           borderaxespad=0.)
ax.xaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.5)
plt.show()
#%% md
# 
#%%
res = stat()
res.levene(df=fat_and_sauces, res_var='value', xfac_var='nutrition_grade_fr')
res.levene_summary
#%%
res = stat()
res.anova_stat(df=fat_and_sauces, res_var='value', anova_model='value ~ C(nutrition_grade_fr)')
res.anova_summary
#%%
# res.anova_std_residuals are standardized residuals obtained from ANOVA (check above)
sm.qqplot(res.anova_std_residuals, line='45')
plt.xlabel("Theoretical Quantiles")
plt.ylabel("Standardized Residuals")
plt.show()

# histogram
plt.hist(res.anova_model_out.resid, bins='auto', histtype='bar', ec='k')
plt.xlabel("Residuals")
plt.ylabel('Frequency')
plt.show()
#%% md
# ### Observations
# Comme les résidus standardisés se situent autour de la ligne de 45 degrés, cela suggère que les résidus sont à peu près normalement distribués
# 
# Dans l'histogramme, la distribution ressemble approximativement à une courbe pluri-normale et suggère que les résidus sont sur 3 zones de distributions avec un étalement sur la droite
#%%
res = stat()
res.levene(df=beverages, res_var='value', xfac_var='nutrition_grade_fr')
res.levene_summary
#%%
res = stat()
res.anova_stat(df=beverages, res_var='value', anova_model='value ~ C(nutrition_grade_fr)')
res.anova_summary
#%%
# res.anova_std_residuals are standardized residuals obtained from ANOVA (check above)
sm.qqplot(res.anova_std_residuals, line='45')
plt.xlabel("Theoretical Quantiles")
plt.ylabel("Standardized Residuals")
plt.show()

# histogram
plt.hist(res.anova_model_out.resid, bins='auto', histtype='bar', ec='k')
plt.xlabel("Residuals")
plt.ylabel('Frequency')
plt.show()
#%% md
# ### Observations
# Comme les résidus standardisés se situent de manière plus disparate autour de la ligne de 45 degrés, cela suggère que les résidus ne sont que tres peu normalement distribués.
# 
# Dans l'histogramme, la distribution ressemble approximativement à une courbe pluri-normale et suggère que les résidus sont sur 4 zones de distributions avec un étalement sur la droite
#%%
res = stat()
res.levene(df=beverages, res_var='value', xfac_var='nutrition_grade_fr')
res.levene_summary
#%% md
# En statistique, le Test de Levene est une statistique déductive utilisée pour évaluer l'égalité de variance pour une variable calculée pour deux groupes ou plus1.
# 
# Certaines procédures statistiques courantes supposent que les variances des populations à partir desquelles différents échantillons sont prélevés sont égales. Le test de Levene évalue cette hypothèse. Il teste l'hypothèse nulle que les variances de population sont égales (appelées « homogénéité de la variance » ou homoscédasticité). Si la valeur p résultante du test de Levene est inférieure à un niveau de signification (typiquement 0,05), il est peu probable que les différences obtenues dans les variances d'échantillon se soient produites sur la base d'un échantillonnage aléatoire d'une population à variances égales. Ainsi, l'hypothèse nulle d'égale variance est rejetée et il est conclu qu'il existe une différence entre les variances dans la population.
#%% md
# ### Etude de la corrélation entre les variables nutritionnelles
#%% md
# 
#%%
off_quantitative_corr = off_quantitative_iter.drop(columns=['code','additives_n','ingredients_that_may_be_from_palm_oil_n','ingredients_from_palm_oil_n']).corr()
off_quantitative_corr
#%%
mask = np.zeros_like(off_quantitative_corr)
mask[np.triu_indices_from(mask)] = True
with sns.axes_style("white"):
    fig, ax = plt.subplots(figsize=(10,10))
    fig.subplots_adjust(top=0.9)
    fig.suptitle(f"Analyse de corrélation par Heatmap des variables quantitatives", fontsize=25)
    ax = sns.heatmap(off_quantitative_corr, mask=mask, square=True, linewidths=.5, annot=True)
    ax.set_xticklabels(off_quantitative_corr, rotation=50, fontsize=14)
    ax.set_yticklabels(off_quantitative_corr, fontsize=14)
    plt.tight_layout()
    plt.show()
#%%
lmplot = sns.lmplot(data = off_quantitative_replace_iter,
                x='fat_100g',
                y='saturated_fat_100g',
                col='pnns_groups_1',
                col_wrap=3,
                hue='pnns_groups_1',
                x_jitter=0.0001,
                palette="Set2",
                height=6,
                line_kws={'color': 'red'},
                )
lmplot.fig.suptitle(f"Analyse de bivarié avec droite de régression linéaire \ndes variables quantitatives par groupe d'aliment", fontsize=25)
lmplot.fig.subplots_adjust(top=1.5)
sns.set_theme(style="ticks", palette="pastel")
sns.despine(offset=10, trim=True)
plt.tight_layout()
plt.show()
#%%
lmplot = sns.lmplot(data = off_quantitative_replace_iter,
                x='fat_100g',
                y='saturated_fat_100g',
                col='nutrition_grade_fr',
                col_order=['a','b','c','d','e'],
                hue_order=['a','b','c','d','e'],
                col_wrap=3,
                hue='nutrition_grade_fr',
                x_jitter=0.0001,
                palette="Set2",
                height=6,
                line_kws={'color': 'red'},
                # legend=False,
                )
# lmplot.ax.legend(bbox_to_anchor=(1.01, 0.5), ncol=5)
lmplot.fig.suptitle(f"Analyse de bivarié avec droite de régression linéaire des variables quantitatives", fontsize=25)
lmplot.fig.subplots_adjust(top=1.5)
sns.set_theme(style="ticks", palette="pastel")
sns.despine(offset=10, trim=True)
plt.tight_layout()
plt.show()
#%%
lmplot = sns.lmplot(data = off_quantitative_replace_iter,
                x='carbohydrates_100g',
                y='sugars_100g',
                col='nutrition_grade_fr',
                col_order=['a','b','c','d','e'],
                hue_order=['a','b','c','d','e'],
                col_wrap=3,
                hue='nutrition_grade_fr',
                x_jitter=0.0001,
                palette="Set2",
                height=6,
                line_kws={'color': 'red'},
                # legend=False,
                )
# lmplot.ax.legend(bbox_to_anchor=(1.01, 0.5), ncol=5)
lmplot.fig.suptitle(f"Analyse de bivarié avec droite de régression linéaire des variables quantitatives", fontsize=25)
lmplot.fig.subplots_adjust(top=1.5)
sns.set_theme(style="ticks", palette="pastel")
sns.despine(offset=10, trim=True)
plt.tight_layout()
plt.show()
#%% md
# ## Générer le CSV pour l'affichage avec voila (notebook P3_Plotly)
#%%
columns_select = off_quantitative_replace_iter.columns.values
off_drop_prod_na[columns_select]=off_quantitative_replace_iter
off_drop_prod_na.to_csv("open_food_fact_clean", sep=';',index=False)
#%%
off_drop_prod_na.shape
#%%
X = off_quantitative_replace_iter.drop(columns=['code','additives_n','ingredients_that_may_be_from_palm_oil_n','ingredients_from_palm_oil_n','nutrition_grade_fr','energy_100g'])
X.dropna()
#%% md
# ## Analyse de la Composante Principale
#%% md
# ### Fonction
#%%
from matplotlib.collections import LineCollection
from matplotlib.transforms import Affine2D

def display_circles(pcs, n_comp, pca, axis_ranks, labels=None, label_rotation=0, lims=None):
    for d1, d2 in axis_ranks: # On affiche les 3 premiers plans factoriels, donc les 6 premières composantes
        if d2 < n_comp:

            # initialisation de la figure
            fig, ax = plt.subplots(figsize=(7,6))

            # détermination des limites du graphique
            if lims is not None :
                xmin, xmax, ymin, ymax = lims
            elif pcs.shape[1] < 30 :
                xmin, xmax, ymin, ymax = -1, 1, -1, 1
            else :
                xmin, xmax, ymin, ymax = min(pcs[d1,:]), max(pcs[d1,:]), min(pcs[d2,:]), max(pcs[d2,:])

            # affichage des flèches
            # s'il y a plus de 30 flèches, on n'affiche pas le triangle à leur extrémité
            if pcs.shape[1] < 30 :
                plt.quiver(np.zeros(pcs.shape[1]), np.zeros(pcs.shape[1]),
                   pcs[d1,:], pcs[d2,:],
                   angles='xy', scale_units='xy', scale=1, color="grey")
                # (voir la doc : https://matplotlib.org/api/_as_gen/matplotlib.pyplot.quiver.html)
            else:
                lines = [[[0,0],[x,y]] for x,y in pcs[[d1,d2]].T]
                ax.add_collection(LineCollection(lines, axes=ax, alpha=.1, color='black'))

            # affichage des noms des variables
            if labels is not None:
                for i,(x, y) in enumerate(pcs[[d1,d2]].T):
                    if xmin <= x <= xmax and ymin <= y <= ymax:
                        plt.text(x, y, labels[i], fontsize='14', ha='center', va='center', rotation=label_rotation, color="blue", alpha=0.5)

            # affichage du cercle
            circle = plt.Circle((0,0), 1, facecolor='none', edgecolor='b')
            plt.gca().add_artist(circle)

            # définition des limites du graphique
            plt.xlim(xmin, xmax)
            plt.ylim(ymin, ymax)

            # affichage des lignes horizontales et verticales
            plt.plot([-1, 1], [0, 0], color='grey', ls='--')
            plt.plot([0, 0], [-1, 1], color='grey', ls='--')

            # nom des axes, avec le pourcentage d'inertie expliqué
            plt.xlabel('F{} ({}%)'.format(d1+1, round(100*pca.explained_variance_ratio_[d1],1)))
            plt.ylabel('F{} ({}%)'.format(d2+1, round(100*pca.explained_variance_ratio_[d2],1)))

            plt.title("Cercle des corrélations (F{} et F{})".format(d1+1, d2+1))
            plt.show(block=False)

def display_factorial_planes(X_projected, n_comp, pca, axis_ranks, labels=None, alpha=1, illustrative_var=None):
    for d1,d2 in axis_ranks:
        if d2 < n_comp:

            # initialisation de la figure
            fig = plt.figure(figsize=(7,6))

            # affichage des points
            if illustrative_var is None:
                plt.scatter(X_projected[:, d1], X_projected[:, d2], alpha=alpha)
            else:
                illustrative_var = np.array(illustrative_var)
                for value in np.unique(illustrative_var):
                    selected = np.where(illustrative_var == value)
                    plt.scatter(X_projected[selected, d1], X_projected[selected, d2], alpha=alpha, label=value)
                plt.legend()

            # affichage des labels des points
            if labels is not None:
                for i,(x,y) in enumerate(X_projected[:,[d1,d2]]):
                    colors_tab = pa.DataFrame([('a','darkgreen'),('b','green'),('c','yellow'),('d','orange'),('e','red')])
                    color=colors_tab[1].where(colors_tab[0]==labels[i]).dropna().values[0]
                    plt.text(x, y, labels[i],
                              fontsize='14', ha='center',va='center',color=color)

            # détermination des limites du graphique
            boundary = np.max(np.abs(X_projected[:, [d1,d2]])) * 1.1
            plt.xlim([-boundary,boundary])
            plt.ylim([-boundary,boundary])

            # affichage des lignes horizontales et verticales
            plt.plot([-100, 100], [0, 0], color='grey', ls='--')
            plt.plot([0, 0], [-100, 100], color='grey', ls='--')

            # nom des axes, avec le pourcentage d'inertie expliqué
            plt.xlabel('F{} ({}%)'.format(d1+1, round(100*pca.explained_variance_ratio_[d1],1)))
            plt.ylabel('F{} ({}%)'.format(d2+1, round(100*pca.explained_variance_ratio_[d2],1)))

            plt.title("Projection des individus (sur F{} et F{})".format(d1+1, d2+1))
            plt.show(block=False)

def display_scree_plot(pca):
    scree = pca.explained_variance_ratio_*100
    plt.bar(np.arange(len(scree))+1, scree)
    plt.plot(np.arange(len(scree))+1, scree.cumsum(),c="red",marker='o')
    plt.xlabel("rang de l'axe d'inertie")
    plt.ylabel("pourcentage d'inertie")
    plt.title("Eboulis des valeurs propres")
    plt.show(block=False)

def pareto(data) :
    from matplotlib.ticker import PercentFormatter
    y = list(data)
    x = range(len(data))
    ycum = np.cumsum(y)/sum(y)*100
    fig, ax = plt.subplots()
    ax.bar(x,y,color="yellow")
    ax2 = ax.twinx()
    ax2.plot(x,ycum, color="C1", marker="D", ms=7)
    ax2.axhline(y=80,color="r")
    ax2.yaxis.set_major_formatter(PercentFormatter())
    ax.tick_params(axis="y", colors="C0")
    ax2.tick_params(axis="y", colors="C1")
    plt.ylim(0,110)
    plt.show()
#%%


X_df = off_quantitative_replace_iter.drop(columns=['code','additives_n','ingredients_that_may_be_from_palm_oil_n','ingredients_from_palm_oil_n','nutrition_grade_fr','energy_100g','pnns_groups_1','pnns_groups_2','saturated_fat_100g','sugars_100g','sodium_100g']).dropna()
X_df_score = X_df.join(off_quantitative_replace_iter.nutrition_grade_fr)
nutri = X_df_score.nutrition_grade_fr.astype('category')
Y = nutri.cat.codes
print(X_df.isna().sum()) ; print(f"valeur de  Y: {Y.min()}")

#%%
from pandas.plotting import scatter_matrix

Axes = scatter_matrix(X_df,c=Y, alpha=0.2, figsize=(50, 50))
#y ticklabels
[plt.setp(item.yaxis.get_majorticklabels(), 'size', 15) for item in Axes.ravel()]
#x ticklabels
[plt.setp(item.xaxis.get_majorticklabels(), 'size', 15) for item in Axes.ravel()]
#y labels
[plt.setp(item.yaxis.get_label(), 'size', 20) for item in Axes.ravel()]
#x labels
[plt.setp(item.xaxis.get_label(), 'size', 20) for item in Axes.ravel()]
plt.show()
#%% md
# > Méthode de base sklearn
#%%
X = X_df.copy()
X
#%%
from sklearn.preprocessing import scale

X = scale(X)
X_df_pca = pa.DataFrame(X,columns=X_df.columns.values,index=X_df.index.values)
X_df_pca.describe()
#%%
X
#%% md
# > Autre méthode, pour les dataframe (pandas) - on obtient ainsi un X au format dataframe.
#%%
n_comp = 3
features = X_df_pca.columns
names = X_df_pca.index
mypca = PCA(n_components=n_comp)
# Modèle d'ACP
mypca.fit(X)
# Pourcentage de la variance expliquée par chacune des composantes sélectionnées.
print(f"Valeurs de variance : {mypca.singular_values_}") # Valeurs de variance
print(f"Pourcentages : {mypca.explained_variance_ratio_}") #  Pourcentages
# Axes principaux dans l'espace des caractéristiques, représentant les directions de la variance maximale dans les données. Les composantes sont triées par variance expliquée.
print(f"Axes principaux dans l'espace des caractéristiques, représentant les directions de la variance maximale dans les données. Les composantes sont triées par variance expliquée.\n{mypca.components_}") #
# Résultats de l'ACP
X_scaled = mypca.fit_transform(X)
print(f"Bruit estimé lié à la covariance : {mypca.noise_variance_}")
#%%
pca_full = PCA()
pca_full.fit(X)
#%%
X_scaled_df = pa.DataFrame(X_scaled,columns=X_df.columns[:n_comp].values,index=X_df.index.values)
X_scaled_df
#%% md
# ### Eboulis des valeurs propres
#%%
display_scree_plot(pca_full)
#%%
pareto(mypca.explained_variance_ratio_)
#%% md
# 
#%% md
# ### Cercle de Corrélation
#%%
pcs = pca_full.components_
display_circles(pcs, n_comp, mypca, [(0,1),(2,3),(4,5)], labels = np.array(features))
#%% md
# ### Projection des individus
#%%
X_scaled
#%%
X_projected = mypca.transform(X)
#%%
display_factorial_planes(X_projected, n_comp, mypca, [(0,1),(2,3),(4,5)], labels = np.array(nutri))
plt.show()
#%%
Axes = scatter_matrix(X_scaled_df,c=nutri.cat.codes, alpha=0.2, figsize=(20, 20))
#y ticklabels
[plt.setp(item.yaxis.get_majorticklabels(), 'size', 15) for item in Axes.ravel()]
#x ticklabels
[plt.setp(item.xaxis.get_majorticklabels(), 'size', 15) for item in Axes.ravel()]
#y labels
[plt.setp(item.yaxis.get_label(), 'size', 20) for item in Axes.ravel()]
#x labels
[plt.setp(item.xaxis.get_label(), 'size', 20) for item in Axes.ravel()]
plt.show()
#%%
X_df_pca = pa.DataFrame(X,columns=X_df.columns.values,index=X_df.index.values)
# type(X)
# X.shape == X_df.shape
X_df_pca

#%%
X_score = X_df_pca.join(off_quantitative_replace_iter.nutrition_grade_fr)
X_group_score = X_score.groupby(['nutrition_grade_fr'])
X_group_score.count()
#%%
X_score
#%%
np.array(nutri.unique())
#%%
pa.DataFrame(X_projected)
#%%
X_df_pca
pca_df_full = PCA()
pca_df_full.fit(X_df_pca)
X_df_projected = pca_df_full.transform(X_df_pca)
X_df_projected
#%%
X_df_projected_score = pa.DataFrame(X_df_projected,index=X_df_pca.index,columns=X_df_pca.columns).join(off_quantitative_replace_iter.nutrition_grade_fr)
#%%
from mpl_toolkits.mplot3d import Axes3D
def display_factorial_3D(X_projected, n_comp, pca, axis_ranks, labels=None, alpha=1, illustrative_var=None,angle_view=None):
    sns.set(style = "darkgrid")
    for d1,d2,d3 in axis_ranks:
        if d3 < n_comp:

            # initialisation de la figure
            fig = plt.figure(figsize=(16,16))
            ax = fig.add_subplot(111, projection = '3d')
            # affichage des points
            if illustrative_var is None:
                ax.scatter(X_projected[:, d1], X_projected[:, d2],X_projected[:, d3], alpha=alpha)
            else:
                illustrative_var = np.array(illustrative_var)
                for value in np.unique(illustrative_var):
                    selected = np.where(illustrative_var == value)
                    ax.scatter(X_projected[selected, d1], X_projected[selected, d2], X_projected[selected, d3], alpha=alpha, label=value)
                ax.legend()

            # affichage des labels des points
            if labels is not None:
                for i,(x,y,z) in enumerate(X_projected[:,[d1,d2,d3]]):
                    colors_tab = pa.DataFrame([('a','darkgreen'),('b','green'),('c','yellow'),('d','orange'),('e','red')])
                    color=colors_tab[1].where(colors_tab[0]==labels[i]).dropna().values[0]
                    ax.text(x, y, z, labels[i],
                              fontsize='14', ha='center',va='center',color=color)

            # détermination des limites du graphique
            x_boundary_up = np.max(np.abs(X_projected[:, [d1]])) * 1.1
            x_boundary_dw = np.min(np.abs(X_projected[:, [d1]])) * 1.1
            y_boundary_up = np.max(np.abs(X_projected[:, [d2]])) * 1.1
            y_boundary_dw = np.min(np.abs(X_projected[:, [d2]])) * 1.1
            z_boundary_up = np.max(np.abs(X_projected[:, [d3]])) * 1.1
            z_boundary_dw = np.min(np.abs(X_projected[:, [d3]])) * 1.1
            ax.set_xlim3d([-x_boundary_dw,x_boundary_up])
            ax.set_ylim3d([-y_boundary_dw,y_boundary_up])
            ax.set_zlim3d([-z_boundary_dw,z_boundary_up])


            # affichage des lignes horizontales et verticales
            ax.plot([-100, 100], [0, 0],[0, 0], color='grey', ls='--')
            ax.plot([0, 0], [-100, 100],[0, 0], color='grey', ls='--')
            ax.plot([0, 0],[0, 0], [-100, 100], color='grey', ls='--')

            # nom des axes, avec le pourcentage d'inertie expliqué
            ax.set_xlabel('F{} ({}%)'.format(d1+1, round(100*pca.explained_variance_ratio_[d1],1)))
            ax.set_ylabel('F{} ({}%)'.format(d2+1, round(100*pca.explained_variance_ratio_[d2],1)))
            ax.set_zlabel('F{} ({}%)'.format(d3+1, round(100*pca.explained_variance_ratio_[d3],1)))
            if angle_view is not None:
                ax.view_init(angle_view[0], angle_view[1])
            ax.set_title("Projection des individus sur F{}, F{} et F{})".format(d1+1, d2+1, d3+1))
            plt.show(block=False)

#%%
display_factorial_3D(X_projected,5, pca_df_full, [(0,1,2),(3,4,5),(6,7,8)], labels = np.array(nutri), angle_view=[20,30])
plt.show()
#%%
X_df_projected

#%%
sns.set(style = "darkgrid")

fig = plt.figure(figsize=(16,16))
ax = fig.add_subplot(111, projection = '3d',anchor='NW')

x = X_projected[:,0:1]
y = X_projected[:,1:2]
z = X_projected[:,2:3]

ax.set_xlabel("F1")
ax.set_ylabel("F2")
ax.set_zlabel("F3")
ax.view_init(10, 35)
ax.scatter(x, y, z)
ax.view_init(20,300)
plt.show()
#%%
import scipy.stats as st
#%%
fat_sauce = off_quantitative_replace_iter[off_quantitative_replace_iter.pnns_groups_1=='Fat and sauces'].dropna().copy()
print(fat_sauce.shape)
sns.scatterplot(data=fat_sauce, x='fat_100g',y='additives_n')
#%%
sns.lmplot(data=fat_sauce, x='fat_100g',y='additives_n',line_kws={'color': 'red'},)
#%%
st.pearsonr(fat_sauce.fat_100g,fat_sauce.additives_n)[0]
np.cov(fat_sauce.fat_100g,fat_sauce.additives_n,ddof=0)[1,0]
#%%
fat_add = off_quantitative_replace_iter[['fat_100g','additives_n']].dropna().copy()
fat_add.describe(include='all')
#%%
sns.scatterplot(data=fat_add, x='fat_100g',y='additives_n')
#%%
off_quantitative_replace_iter.pnns_groups_1.value_counts()
#%%
data_add = off_quantitative_replace_iter.copy()
sns.lmplot(data=data_add,
           x='fat_100g',
           y='additives_n',
           line_kws={'color': 'red'},
           col='nutrition_grade_fr',
           col_order=['a','b','c','d','e'],
           col_wrap=3,hue='nutrition_grade_fr',)
#%%
select_group = 'Composite foods'
data_add_cf = off_quantitative_replace_iter[off_quantitative_replace_iter.pnns_groups_1==select_group].copy()
sns.lmplot(data=data_add_cf,
           x='fat_100g',
           y='additives_n',
           line_kws={'color': 'red'},
           col='nutrition_grade_fr',
           col_order=['a','b','c','d','e'],
           col_wrap=3,hue='nutrition_grade_fr',)
#%%
select_group = 'Sugary snacks'
data_add_ss = off_quantitative_replace_iter[off_quantitative_replace_iter.pnns_groups_1==select_group].copy()
sns.lmplot(data=data_add_ss,
           x='sugars_100g',
           y='additives_n',
           line_kws={'color': 'red'},
           col='nutrition_grade_fr',
           col_order=['a','b','c','d','e'],
           col_wrap=3,hue='nutrition_grade_fr',)
#%%
data_add
#%%
X1_df = data_add_ss.drop(columns=['code','ingredients_that_may_be_from_palm_oil_n','ingredients_from_palm_oil_n','nutrition_grade_fr','energy_100g','pnns_groups_1','pnns_groups_2','fat_100g','carbohydrates_100g','sodium_100g'])
X1_df_score = X1_df.join(off_quantitative_replace_iter.nutrition_grade_fr)
nutri = X_df_score.nutrition_grade_fr.astype('category')
Y1 = nutri.cat.codes
print(X1_df.isna().sum()) ; print(f"valeur de  Y: {Y1.min()}")
#%%
X1 = X1_df.dropna().copy()
st.pearsonr(X1.sugars_100g,X1.additives_n)#pas de correlation linéaire
#%%
X2 = scale(X1)
X2_df_pca = pa.DataFrame(X1,columns=X1.columns.values,index=X1.index.values)
X2_df_pca.describe()
#%% md
# ### Observations
# Éboulie des valeurs nous retrouvons notre coude sur le rang 3
#%%
pca_full = PCA()
pca_full.fit(X2)
display_scree_plot(pca_full)
#%%
n_comp = 3
features = X2_df_pca.columns
names = X2_df_pca.index
mypca = PCA(n_components=n_comp)
# Modèle d'ACP
mypca.fit(X2)
# Pourcentage de la variance expliquée par chacune des composantes sélectionnées.
print(f"Valeurs de variance : {mypca.singular_values_}") # Valeurs de variance
print(f"Pourcentages : {mypca.explained_variance_ratio_}") #  Pourcentages
# Axes principaux dans l'espace des caractéristiques, représentant les directions de la variance maximale dans les données. Les composantes sont triées par variance expliquée.
print(f"Axes principaux dans l'espace des caractéristiques, représentant les directions de la variance maximale dans les données. Les composantes sont triées par variance expliquée.\n{mypca.components_}") #
# Résultats de l'ACP
X2_scaled = mypca.fit_transform(X2)
print(f"Bruit estimé lié à la covariance : {mypca.noise_variance_}")
#%%
pareto(mypca.explained_variance_ratio_)
#%%
pcs = mypca.components_
display_circles(pcs, n_comp, mypca, [(0,1),(2,3),(4,5)], labels = np.array(features))
#%%
X2_projected = mypca.transform(X2)
display_factorial_planes(X2_projected, n_comp, mypca, [(0,1),(2,3),(4,5)], labels = np.array(nutri))
plt.show()
#%%
X2_projected
#%%
X2_projected[:,2:3]