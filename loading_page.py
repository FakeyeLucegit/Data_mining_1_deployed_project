import streamlit as st
import numpy as np
import pandas as pd
from streamlit_option_menu import option_menu
from sklearn.impute import KNNImputer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as xp  # Correct import statement
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

# Suppress warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
st.set_option('deprecation.showfileUploaderEncoding', False)
st.set_option('deprecation.showPyplotGlobalUse', False)


# Titre de l'application
#st.title('Data analyst')
#df = pd.read_csv("DataAnalyst.csv")
#df.drop(['Unnamed: 0'], axis=1, inplace=True)

#data=pd.DataFrame(df)

with st.sidebar:

    st.logo("Logo_Efrei_PAris_2.png")
    selected = option_menu(
        menu_title="Menu",
        options= ["Initial data","Data preprocessing","Visualization","Clustering","Learning Evaluation","Objective"]
    
    )  
    uploaded_file=st.file_uploader("Cliquez ici pour choisir un fichier CSV", type="csv")   

    

  
# Bouton de téléchargement de fichier CSV



# Initialiser la variable de session pour le DataFrame
if 'df' not in st.session_state:
    st.session_state.df = None


print("-------------------------part 1")

if selected == "Initial data":
        # Titre de l'application
    st.title("Application d'analyse de données")



    if uploaded_file is not None:
        # Lecture du fichier CSV
        #df = pd.read_csv(uploaded_file, sep=" ")
        #df = pd.read_csv(uploaded_file, sep=',')
        df = pd.read_csv(uploaded_file,delim_whitespace=True)

        # Ajouter les noms des colonnes manuellement
        noms_colonnes = ['sequence_name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'class']
        df.columns = noms_colonnes

        st.session_state.df = df

        # Affichage des premières et dernières lignes du fichier
        st.subheader("Aperçu des données")
        st.write("Premières lignes:")
        st.write(df.head())
        st.write("Dernières lignes:")
        st.write(df.tail())
        # Résumé statistique
        st.subheader("Résumé statistique")
        st.write("Nombre de lignes et de colonnes:")
        st.write(df.shape)
        st.write("Noms des colonnes:")
        st.write(df.columns)
        st.write("Nombre de valeurs manquantes par colonne:")
        st.write(df.isnull().sum())
        st.write("Description des variables")
        st.write(df.describe())
    
print("----------------------------------")
if selected == "Data preprocessing":
    if st.session_state.df is not None:
        df = st.session_state.df #session enregistrée
        st.subheader("Prétraitement des données")
        st.write("Données actuelles:")
        st.write(df)

    st.subheader("Traitement des valeurs manquantes")
                    # Choix de la méthode de traitement       
    method = st.selectbox("Choisissez une méthode pour traiter les valeurs manquantes", 
                              ["Choisir","Supprimer les lignes/colonnes", "Remplacer par la moyenne", 
                               "Remplacer par la médiane", "Remplacer par le mode", 
                               "Imputation KNN"])

        # Option pour supprimer les valeurs manquantes
    if method=="Supprimer les lignes/colonnes":
            st.session_state.df = df
            # Afficher les valeurs manquantes avant suppression
            missing_values = df[df.isnull().any(axis=1)]
            st.write("Valeurs manquantes avant suppression:")
            st.write(missing_values)

            # Supprimer les lignes avec des valeurs manquantes
            df_cleaned=df.dropna(inplace=True)
            
            # Afficher le DataFrame nettoyé
            st.write("DataFrame après suppression des valeurs manquantes:")
            st.write(df)
    
    elif method == "Remplacer par la moyenne":
        # Afficher les valeurs manquantes avant suppression
        missing_values = df[df.isnull().any(axis=1)]
        st.write("Valeurs manquantes avant suppression:")
        st.write(missing_values)

        # Remplacer par la moyenne
        df.fillna(df.mean(), inplace=True)
        st.session_state.df = df
        st.write("Valeurs manquantes remplacées par la moyenne.")
        st.write(df)
    
    elif method == "Remplacer par la médiane":
        # Afficher les valeurs manquantes avant suppression
        missing_values = df[df.isnull().any(axis=1)]
        st.write("Valeurs manquantes avant suppression:")
        st.write(missing_values)

        # Remplacer par la mediane
        df_remplaced_mode= df.fillna(df.median(), inplace=True)
        st.session_state.df = df
        st.write("Valeurs après remplacement par la médiane.")
        st.write(df)

    elif method == "Remplacer par le mode":
        st.session_state.df = df
        # Afficher les valeurs manquantes avant suppression
        missing_values = df[df.isnull().any(axis=1)]
        st.write("Valeurs manquantes avant suppression:")
        st.write(missing_values)

        for column in df.columns:
            df[column].fillna(df[column].mode()[0], inplace=True)

        # Mettre à jour la session state avec le DataFrame modifié
        st.session_state.df = df

        # Afficher un message de confirmation et le DataFrame modifié
        st.write("Valeurs manquantes remplacées par le mode.")
        st.write(df)
    
    elif method == "Imputation KNN":
        st.session_state.df = df
        # Afficher les valeurs manquantes avant suppression
        missing_values = df[df.isnull().any(axis=1)]
        st.write("Valeurs manquantes avant suppression:")
        st.write(missing_values)

        # Remplacer par l imputation
        imputer = KNNImputer()
        df[df.columns] = imputer.fit_transform(df)
        st.session_state.df = df
        st.write("Valeurs manquantes imputées avec KNN.")
        st.write(df)
    
    
print("--------------------------------")
if selected == "Visualization":
    if st.session_state.df is not None:
        df = st.session_state.df
        st.subheader("Visualisation des données")
        st.write(df)

        # Checklist pour choisir les types de visualisations
        options = st.multiselect("Choisissez les types de visualisation", 
                                 ["Univariée", "Bivariée", "Multivariée"])

        # Visualisation univariée des données
        if "Univariée" in options:
            st.write("Visualisation univariée des données:")
            selected_column = st.selectbox("Sélectionnez une colonne pour visualiser l'histogramme", df.columns)
            if selected_column:
    # Créer l'histogramme interactif avec plotly
                fig = xp.histogram(df, x=selected_column, nbins=10, title=f"Histogramme de {selected_column}")
                fig.update_layout(showlegend=False)
                
                # Afficher le graphique dans Streamlit
                st.plotly_chart(fig)
            else:
                st.warning("Veuillez sélectionner une colonne pour visualiser l'histogramme.")

        # Visualisation bivariée des données
        elif "Bivariée" in options:
            st.write("Visualisation bivariée des données avec histogrammes conditionnés:")
            selected_columns = st.multiselect("Sélectionnez deux colonnes pour visualiser leur relation", df.columns)

# Sélecteur de type de graphique
            plot_type = st.selectbox("Sélectionnez le type de graphique", ["Histogramme", "Boxplot"])

            if len(selected_columns) == 2:
                if plot_type == "Histogramme":
                    # Créer l'histogramme bivarié interactif avec plotly
                    fig = xp.histogram(df, x=selected_columns[0], title=f"Histogramme de {selected_columns[0]} par {selected_columns[1]}")
                    fig.update_layout(bargap=0.1, showlegend=False)  # Enlever la légende
                elif plot_type == "Boxplot":
                    # Créer le boxplot bivarié interactif avec plotly
                    fig = xp.box(df, x=selected_columns[0], y=selected_columns[1], title=f"Boxplot de {selected_columns[0]} par {selected_columns[1]}")
                    fig.update_layout(showlegend=False)  # Enlever la légende

                # Afficher le graphique dans Streamlit
                st.plotly_chart(fig)
            else:
                st.warning("Veuillez sélectionner deux colonnes pour visualiser leur relation.")
            
        elif "Multivariée" in options:
            # Calculer la matrice de corrélation
            # Sélectionner uniquement les colonnes avec des types de données numériques
            df_numeric = df.select_dtypes(include=['number'])

            # Vérifier le résultat
            corr_matrix = df_numeric.corr()

            # Afficher la matrice de corrélation
            print(corr_matrix)
                            # Créer un heatmap interactif avec plotly
            fig = xp.imshow(corr_matrix.values,
                x=list(corr_matrix.columns),
                    y=list(corr_matrix.index),
                    color_continuous_scale='Viridis')

            # Mise à jour du layout
            fig.update_layout(title="Matrice de Corrélation avec Légende des Couleurs",
                            xaxis_title="Variables",
                            yaxis_title="Variables",
                            coloraxis_colorbar=dict(title='Corrélation', tickvals=[-1, 0, 1], ticktext=['Faible', 'Moyenne', 'Élevée'])) #afficher la legende

            # Afficher le graphique dans Streamlit
            st.plotly_chart(fig)





# Clustering
if selected == "Clustering":
    if st.session_state.df is not None:
        df = st.session_state.df
        st.subheader("Clustering")

        st.write("Sélectionnez les colonnes pour le clustering:")
        features = st.multiselect("Colonnes à utiliser", df.columns[:-1])  # Exclure la colonne 'class'

        if len(features) < 2:
            st.warning("Veuillez sélectionner au moins deux colonnes pour le clustering.")
        else:
            X = df[features]
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            algo = st.selectbox("Choisissez l'algorithme de clustering", ["K-Means", "DBSCAN"])

            if algo == "K-Means":
                n_clusters = st.slider("Nombre de clusters", min_value=2, max_value=10, value=3)
                kmeans = KMeans(n_clusters=n_clusters)
                df['cluster'] = kmeans.fit_predict(X_scaled)
            
                plt.figure(figsize=(10, 7))
                plt.scatter(df[features[0]], df[features[1]], c=df['cluster'], cmap='viridis')
                plt.xlabel(features[0])
                plt.ylabel(features[1])
                plt.title(f"K-Means Clustering (n_clusters={n_clusters})")
                plt.colorbar(label='Cluster')
                st.pyplot(plt)
            
                # Cluster statistics
                st.subheader("Statistiques des clusters")
                cluster_stats = df.groupby('cluster')[features].agg(['count', 'mean'])
                st.write(cluster_stats)

                
            elif algo == "DBSCAN":
                eps = st.slider("Paramètre eps", min_value=0.1, max_value=10.0, value=0.5)
                min_samples = st.slider("Nombre minimum d'échantillons", min_value=1, max_value=20, value=5)
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                df['cluster'] = dbscan.fit_predict(X_scaled)
            
                st.write("Résultats de DBSCAN:")
                st.write(df.head())
            
                # Visualization with matplotlib
                if 'cluster' in df.columns:
                    plt.figure(figsize=(10, 7))
                    plt.scatter(df[features[0]], df[features[1]], c=df['cluster'], cmap='viridis')
                    plt.title(f"DBSCAN Clustering (eps={eps}, min_samples={min_samples})")
                    plt.colorbar(label='Cluster')
                    st.pyplot(plt)
                else:
                    st.error("La colonne 'cluster' n'existe pas dans le DataFrame.")
            
                # Cluster statistics
                st.subheader("Statistiques des clusters")
                cluster_stats = df.groupby('cluster')[features].agg(['count', 'mean'])
                st.write(cluster_stats)






# Learning Evaluation
if selected == "Learning Evaluation":
    if st.session_state.df is not None:
        df = st.session_state.df
        st.subheader("Évaluation de l'apprentissage")

        # Afficher les premières lignes du DataFrame pour débogage
        #st.write("Aperçu du DataFrame :")
        #st.write(df.head())

        # Convertir les attributs binaires en 0 et 1
        if 'lip' in df.columns:
            df['lip'] = df['lip'].map({'yes': 1, 'no': 0})
        if 'chg' in df.columns:
            df['chg'] = df['chg'].map({'yes': 1, 'no': 0})

        # Utiliser LabelEncoder pour les colonnes catégorielles
        le = LabelEncoder()
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = le.fit_transform(df[col].astype(str))

        # Sélectionner la colonne cible
        target_column = st.selectbox("Sélectionnez la colonne cible pour la prédiction", df.columns)

        if target_column not in df.columns:
            st.error(f"La colonne cible '{target_column}' n'existe pas dans les données.")
        else:
            features = [col for col in df.columns if col != target_column]

            X = df[features]
            y = df[target_column]

            # Afficher les premières valeurs de la colonne cible pour le débogage
            st.write(f"Premières valeurs de la colonne cible '{target_column}':")
            st.write(y.head())

            # Vérifiez le type de la colonne cible
            st.write(f"Type des valeurs dans la colonne cible '{target_column}':")
            st.write(y.dtypes)

            # Vérifiez les valeurs manquantes dans la colonne cible
            st.write(f"Nombre de valeurs manquantes dans la colonne cible '{target_column}':")
            st.write(y.isna().sum())

            # Vérifiez que y n'est pas vide et ne contient pas uniquement des valeurs NaN
            if y.empty or y.isna().all():
                st.error(f"La colonne cible '{target_column}' est vide ou contient uniquement des valeurs manquantes.")
            else:
                # Traiter les valeurs manquantes dans X et y
                imputer_X = SimpleImputer(strategy='mean')  # Pour les données numériques
                X = imputer_X.fit_transform(X)

                imputer_y = SimpleImputer(strategy='most_frequent')  # Imputer les valeurs manquantes pour y
                y = imputer_y.fit_transform(y.values.reshape(-1, 1)).ravel()

                # Vérifiez les dimensions de X et y
                st.write(f"Dimensions de X: {X.shape}")
                st.write(f"Dimensions de y: {y.shape}")

                # Diviser les données en ensembles d'entraînement et de test
                if X.shape[0] == y.shape[0]:  # Vérifier que X et y ont le même nombre d'échantillons
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

                    # Déterminer si le problème est de régression ou de classification
                    unique_y = np.unique(y)
                    is_classification = len(unique_y) < 20  # Un seuil arbitraire pour classification

                    if is_classification:
                        model_type = st.selectbox("Choisissez le modèle de classification", ["Arbre de Décision"])
                        if model_type == "Arbre de Décision":
                            model = DecisionTreeClassifier()
                            model.fit(X_train, y_train)
                            y_pred = model.predict(X_test)

                            st.write("Évaluation du modèle d'Arbre de Décision:")
                            st.write("Rapport de classification:")
                            st.write(classification_report(y_test, y_pred))

                            # Visualisation
                            fig = plt.figure()
                            sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d')
                            plt.title("Matrice de Confusion")
                            st.pyplot(fig)

                    else:
                        model_type = st.selectbox("Choisissez le modèle de régression", ["Régression Linéaire"])
                        if model_type == "Régression Linéaire":
                            model = LinearRegression()
                            model.fit(X_train, y_train)
                            y_pred = model.predict(X_test)

                            st.write("Évaluation du modèle de Régression Linéaire:")
                            st.write("Erreur quadratique moyenne:", mean_squared_error(y_test, y_pred))

                            # Visualisation
                            fig = plt.figure()
                            plt.scatter(y_test, y_pred)
                            plt.xlabel("Valeurs Réelles")
                            plt.ylabel("Valeurs Prédites")
                            plt.title("Régression Linéaire: Valeurs Réelles vs Prédites")
                            st.pyplot(fig)
                else:
                    st.error("Les dimensions de X et y ne sont pas compatibles. Vérifiez vos données.")
    else:
        st.warning("Aucune donnée chargée dans la session. Veuillez charger un fichier de données.")

if selected == "Objective":
    # Définir le texte
    auteur = "Auteur"
    corps = "FAKEYE Luce et ASSALE Oriane, étudiantes en Master à Efrei Paris"

    # Afficher le contenu avec un titre en bleu et le corps en italique
    st.markdown("## **Objectif du Projet**")
    st.markdown("""
    Ce projet a pour but de :
    1. Analyser et prétraiter les données biologiques.
    2. Appliquer des techniques de clustering pour identifier des groupes dans les données.
    3. Évaluer les performances des modèles d'apprentissage automatique sur les données.
    4. Visualiser les résultats pour interpréter les données efficacement.
    """)

    st.markdown(f"## <span style='color:blue'>Auteur:</span>", unsafe_allow_html=True)
    st.markdown(f"*{corps}*", unsafe_allow_html=True)

