# ======================================================
#%% 
# LIBRARIES
# ======================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from scipy.stats import f_oneway
from scipy.stats import kruskal
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from itertools import combinations_with_replacement
import pickle


#%% =====================================================
# DATA LOADER
# =====================================================

class FlightDelayDataLoader:

    def __init__(self, filename, test_size=0.2, random_state=42):

        self.filename = filename
        self.test_size = test_size
        self.random_state = random_state

        self.features_train = None
        self.features_test = None
        self.target_train = None
        self.target_test = None

        self._load_data()


    def _load_data(self):

        df = pd.read_csv(self.filename, low_memory=False)

        print("Flight dataset loaded:", df.shape)

        # Remove cancelled flights
        if "CANCELLED" in df.columns:
            df = df[df["CANCELLED"] == 0]

        # Create target variable
        df["FLIGHT_DELAYED"] = (df["ARR_DELAY"] > 15).astype(int)

        # Remove leakage columns
        leakage_columns = [
            "ARR_DELAY",
            "DEP_DELAY",
            "DEP_TIME",
            "ARR_TIME",
            "WHEELS_OFF",
            "WHEELS_ON",
            "TAXI_OUT",
            "TAXI_IN",
            "AIR_TIME",
            "ELAPSED_TIME",
            "DELAY_DUE_CARRIER",
            "DELAY_DUE_WEATHER",
            "DELAY_DUE_NAS",
            "DELAY_DUE_SECURITY",
            "DELAY_DUE_LATE_AIRCRAFT",
            "CANCELLED",
            "CANCELLATION_CODE",
            "DIVERTED",
            "FL_NUMBER"

        ]

        df = df.drop(columns=[c for c in leakage_columns if c in df.columns])

        X = df.drop(columns=["FLIGHT_DELAYED"])
        y = df["FLIGHT_DELAYED"]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state
        )

        self.features_train = X_train
        self.features_test = X_test

        self.target_train = y_train
        self.target_test = y_test

        print("Train/Test split completed")
        print("Training samples:", X_train.shape)
        print("Testing samples:", X_test.shape)


class FlightDelayDataManipulator(FlightDelayDataLoader):

    def __init__(self, filename, test_size=0.2, random_state=42):

        super().__init__(filename, test_size, random_state)

       
        self.create_airline_names()
        self.create_airport_city_features()
        self.create_delay_aggregations()
        self.create_advanced_features()

    # =====================================================
    # AIRLINE NAMES (para gráficos)
    # =====================================================
    def create_airline_names(self):

        airline_map = {
            "AA": "American Airlines",
            "DL": "Delta Airlines",
            "UA": "United Airlines",
            "WN": "Southwest",
            "B6": "JetBlue",
            "AS": "Alaska Airlines",
            "NK": "Spirit Airlines",
            "F9": "Frontier Airlines"
        }

        for df in [self.features_train, self.features_test]:
            if "AIRLINE" in df.columns:
                df["airline_name"] = df["AIRLINE"].map(airline_map).fillna(df["AIRLINE"])

        print("Airline names created")


    # =====================================================
    # AIRPORT / CITY FEATURES (para labels legíveis)
    # =====================================================
    def create_airport_city_features(self):

        for df in [self.features_train, self.features_test]:

            if "ORIGIN_CITY" in df.columns:
                df["origin_city_clean"] = df["ORIGIN_CITY"].str.split(",").str[0]

            if "DEST_CITY" in df.columns:
                df["dest_city_clean"] = df["DEST_CITY"].str.split(",").str[0]

        print("City clean features created")


    # =====================================================
    # AGGREGATED DELAY FEATURES (muito importantes)
    # =====================================================
    def create_delay_aggregations(self):

        # ⚠️ usar apenas treino para evitar leakage
        train_df = self.features_train.copy()
        train_df["DELAYED"] = self.target_train.values

        # ----------------------------------------
        # Delay por companhia
        # ----------------------------------------
        if "AIRLINE" in train_df.columns:
            airline_delay_rate = train_df.groupby("AIRLINE")["DELAYED"].mean()

            for df in [self.features_train, self.features_test]:
                df["airline_delay_rate"] = df["AIRLINE"].map(airline_delay_rate)

        # ----------------------------------------
        # Delay por aeroporto origem
        # ----------------------------------------
        if "ORIGIN" in train_df.columns:
            airport_delay_rate = train_df.groupby("ORIGIN")["DELAYED"].mean()

            for df in [self.features_train, self.features_test]:
                df["origin_delay_rate"] = df["ORIGIN"].map(airport_delay_rate)

        # ----------------------------------------
        # Delay por mês
        # ----------------------------------------
        if "flight_month" in train_df.columns:
            month_delay_rate = train_df.groupby("flight_month")["DELAYED"].mean()

            for df in [self.features_train, self.features_test]:
                df["month_delay_rate"] = df["flight_month"].map(month_delay_rate)

        # ----------------------------------------
        # Volume de voos por aeroporto (origem + destino)
        # ----------------------------------------
        if "ORIGIN" in train_df.columns and "DEST" in train_df.columns:

            airport_counts = pd.concat([
                train_df["ORIGIN"],
                train_df["DEST"]
            ]).value_counts()

            for df in [self.features_train, self.features_test]:
                df["airport_total_traffic"] = df["ORIGIN"].map(airport_counts)

        print("Delay aggregation features created")
    # =====================================================
    # ADVANCED FEATURES (NOVAS FEATURES IMPORTANTES)
    # =====================================================
    def create_advanced_features(self):

        train_df = self.features_train.copy()
        train_df["DELAYED"] = self.target_train.values

        # Média global (baseline)
        global_delay_rate = train_df["DELAYED"].mean()

        # ----------------------------------------
        # 1. Peak hours (horas de maior tráfego)
        # ----------------------------------------
        for df in [self.features_train, self.features_test]:
            if "departure_hour" in df.columns:
                df["is_peak_hour"] = df["departure_hour"].isin([7,8,9,17,18,19]).astype(int)
                df["is_night_flight"] = df["departure_hour"].isin([0,1,2,3,4,5]).astype(int)

        # ----------------------------------------
        # 2. Route delay (origem + destino)
        # ----------------------------------------
        if "ORIGIN" in train_df.columns and "DEST" in train_df.columns:

            route_delay = train_df.groupby(["ORIGIN", "DEST"])["DELAYED"].mean()

            for df in [self.features_train, self.features_test]:
                routes = list(zip(df["ORIGIN"], df["DEST"]))
                df["route_delay_rate"] = pd.Series(routes).map(route_delay)

        # ----------------------------------------
        # 3. Airline vs global performance
        # ----------------------------------------
        if "AIRLINE" in train_df.columns:

            airline_delay = train_df.groupby("AIRLINE")["DELAYED"].mean()

            for df in [self.features_train, self.features_test]:
                df["airline_vs_global"] = df["AIRLINE"].map(airline_delay) - global_delay_rate

        # ----------------------------------------
        # 4. Airport congestion (proxy)
        # ----------------------------------------
        if "ORIGIN" in train_df.columns:

            airport_traffic = train_df["ORIGIN"].value_counts()

            for df in [self.features_train, self.features_test]:
                df["origin_congestion"] = df["ORIGIN"].map(airport_traffic)

        # ----------------------------------------
        # 5. Difference origin vs destination activity
        # ----------------------------------------
        if "ORIGIN" in train_df.columns and "DEST" in train_df.columns:

            airport_traffic = train_df["ORIGIN"].value_counts()

            for df in [self.features_train, self.features_test]:
                origin = df["ORIGIN"].map(airport_traffic)
                dest = df["DEST"].map(airport_traffic)
                df["traffic_diff"] = origin - dest

        # ----------------------------------------
        # 6. Distance normalized by time (proxy efficiency)
        # ----------------------------------------
        for df in [self.features_train, self.features_test]:
            if "DISTANCE" in df.columns and "CRS_ELAPSED_TIME" in df.columns:
                df["distance_time_ratio"] = df["DISTANCE"] / (df["CRS_ELAPSED_TIME"] + 1)

        # ----------------------------------------
        # 7. Short haul vs long haul
        # ----------------------------------------
        for df in [self.features_train, self.features_test]:
            if "DISTANCE" in df.columns:
                df["is_long_flight"] = (df["DISTANCE"] > 1500).astype(int)

        # ----------------------------------------
        # 8. Airline + Month interaction (muito forte)
        # ----------------------------------------
        if "AIRLINE" in train_df.columns and "flight_month" in train_df.columns:

            airline_month_delay = train_df.groupby(
                ["AIRLINE", "flight_month"]
            )["DELAYED"].mean()

            for df in [self.features_train, self.features_test]:
                keys = list(zip(df["AIRLINE"], df["flight_month"]))
                df["airline_month_delay"] = pd.Series(keys).map(airline_month_delay)

        # ----------------------------------------
        # 9. Day of week delay
        # ----------------------------------------
        if "flight_day_of_week" in train_df.columns:

            dow_delay = train_df.groupby("flight_day_of_week")["DELAYED"].mean()

            for df in [self.features_train, self.features_test]:
                df["dow_delay_rate"] = df["flight_day_of_week"].map(dow_delay)

        # ----------------------------------------
        # 10. Flight density (hora)
        # ----------------------------------------
        if "departure_hour" in train_df.columns:

            hour_density = train_df["departure_hour"].value_counts()

            for df in [self.features_train, self.features_test]:
                df["hour_traffic"] = df["departure_hour"].map(hour_density)

        print("Advanced features created")

#%% =====================================================
# DATA CLEANING
# =====================================================

class FlightDelayDataCleaning:

    def __init__(self, data_loader):

        self.data_loader = data_loader


    def remove_duplicates(self):

        df = self.data_loader.features_train

        before = df.shape[0]

        df = df.drop_duplicates()

        self.data_loader.target_train = self.data_loader.target_train.loc[df.index]
        self.data_loader.features_train = df

        after = df.shape[0]

        print("Removed duplicates:", before - after)


    def handle_missing_values(self):

        train_df = self.data_loader.features_train
        test_df = self.data_loader.features_test

        numeric_cols = train_df.select_dtypes(include=["int64", "float64"]).columns
        categorical_cols = train_df.select_dtypes(include=["object", "string", "category"]).columns

        train_df[numeric_cols] = train_df[numeric_cols].fillna(train_df[numeric_cols].median())
        test_df[numeric_cols] = test_df[numeric_cols].fillna(test_df[numeric_cols].median())

        train_df[categorical_cols] = train_df[categorical_cols].fillna("Unknown")
        test_df[categorical_cols] = test_df[categorical_cols].fillna("Unknown")

        print("Missing values handled")

#%% =====================================================
# PREPROCESSING
# =====================================================

class FlightDelayPreprocessing:

    

    def __init__(self, data_loader):

        self.data_loader = data_loader

        
        self.data_loader.features_train_original = self.data_loader.features_train.copy()
        self.data_loader.features_test_original = self.data_loader.features_test.copy()

        self._preprocess()


    def _preprocess(self):

        train_df = self.data_loader.features_train
        test_df = self.data_loader.features_test

        numeric_cols = train_df.select_dtypes(include=["int64", "float64"]).columns
        categorical_cols = train_df.select_dtypes(include=["object", "string", "category"]).columns

        # Encode categorical variables
        for col in categorical_cols:

            encoder = LabelEncoder()

            train_df[col] = encoder.fit_transform(train_df[col].astype(str))

            test_df[col] = test_df[col].map(
                lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1
            )

        # Scale numeric variables
        scaler = StandardScaler()

        train_df[numeric_cols] = scaler.fit_transform(train_df[numeric_cols])
        test_df[numeric_cols] = scaler.transform(test_df[numeric_cols])

        print("Preprocessing completed")






#%% =====================================================
# PIPELINE EXECUTION
# =====================================================

data_loader = FlightDelayDataManipulator("../data/flights_sample_3m.csv")

print("\nBefore cleaning")
print(data_loader.features_train.shape)

cleaner = FlightDelayDataCleaning(data_loader)
cleaner.remove_duplicates()
cleaner.handle_missing_values()

print("\nAfter cleaning")
print(data_loader.features_train.shape)

preprocessing = FlightDelayPreprocessing(data_loader)

print("\nAfter preprocessing")
print(data_loader.features_train.shape)


#%% =====================================================
# SAVE DATASET
# =====================================================

os.makedirs("../output", exist_ok=True)

with open("../output/flight_delay_dataset.pkl", "wb") as f:

    pickle.dump(data_loader, f)

print("Dataset saved")


#%% =====================================================
# LOAD DATASET
# =====================================================

with open("../output/flight_delay_dataset.pkl", "rb") as f:

    loaded_dataset = pickle.load(f)

print("Dataset loaded")
print(loaded_dataset.features_train.shape)


#%% =====================================================
# EXPLORATORY DATA ANALYSIS
# =====================================================

class FlightDelayEDA:

    def __init__(self, data_loader, output_folder="../output/plots"):

        self.data_loader = data_loader
        self.output_folder = output_folder

        os.makedirs(self.output_folder, exist_ok=True)


    def run(self):

        self.plot_correlation_heatmap()
        self.plot_feature_vs_target()
        self.plot_top_distributions()


    # ------------------------------------------------
    # CORRELATION HEATMAP
    # ------------------------------------------------
    def plot_correlation_heatmap(self):

        df = self.data_loader.features_train.copy()
        df["DELAYED"] = self.data_loader.target_train.values

        selected_features = [
            "airline_delay_rate",
            "origin_delay_rate",
            "month_delay_rate",
            "airport_total_traffic",
            "route_delay_rate",
            "distance_time_ratio",
            "origin_congestion",
            "traffic_diff",
            "DELAYED"
        ]

        df = df[[col for col in selected_features if col in df.columns]]

        plt.figure(figsize=(10, 8))
        sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")

        plt.title("Correlation Heatmap (Engineered Features)")
        plt.savefig(f"{self.output_folder}/correlation_heatmap.png")
        plt.close()

        print("Correlation heatmap saved.")


    # ------------------------------------------------
    # FEATURE VS TARGET
    # ------------------------------------------------
    def plot_feature_vs_target(self):

        df = self.data_loader.features_train.copy()
        df["DELAYED"] = self.data_loader.target_train.values

        features = [
            "airline_delay_rate",
            "origin_delay_rate",
            "month_delay_rate",
            "route_delay_rate"
        ]

        for feature in features:

            if feature in df.columns:

                plt.figure(figsize=(8, 5))
                sns.boxplot(x="DELAYED", y=feature, data=df)

                plt.title(f"{feature} vs Delay")
                plt.savefig(f"{self.output_folder}/{feature}_vs_delay.png")
                plt.close()

        print("Feature vs target plots saved.")


    # ------------------------------------------------
    # DISTRIBUTIONS IMPORTANTES
    # ------------------------------------------------
    def plot_top_distributions(self):

        df = self.data_loader.features_train

        features = [
            "airport_total_traffic",
            "distance_time_ratio",
            "origin_congestion",
            "traffic_diff"
        ]

        for feature in features:

            if feature in df.columns:

                plt.figure(figsize=(8, 5))
                sns.histplot(df[feature], bins=50, kde=True)

                plt.title(f"Distribution of {feature}")
                plt.savefig(f"{self.output_folder}/dist_{feature}.png")
                plt.close()

        print("Top distributions saved.")
#%% =====================================================
# RUN EDA + VISUALIZATION
# =====================================================

eda = FlightDelayEDA(data_loader)
eda.run()


#%% =====================================================
# FEATURE ANALYSIS
# =====================================================

class FlightDelayFeatureAnalysis:

    def __init__(self, data_loader, output_folder="outputs/plots"):

        self.data_loader = data_loader
        self.output_folder = output_folder

        os.makedirs(self.output_folder, exist_ok=True)


    # ------------------------------------------------
    # PCA ANALYSIS
    # ------------------------------------------------
    def perform_pca(self, explained_variance_threshold=0.80,
                    plot_pca=True, add_pca=True):

        train_df = self.data_loader.features_train.copy()

        # só numéricas
        numeric_df = train_df.select_dtypes(include=["int64", "float64"])

        if numeric_df.empty:
            raise ValueError("No numeric data available for PCA.")

        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_df)

        pca = PCA()
        pca.fit(scaled_data)

        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

        n_components = np.argmax(
            cumulative_variance >= explained_variance_threshold
        ) + 1

        print(f"Selected {n_components} PCA components")

        # aplicar PCA
        pca = PCA(n_components=n_components)
        transformed = pca.fit_transform(scaled_data)

        if add_pca:
            for i in range(n_components):
                train_df[f"PCA_{i+1}"] = transformed[:, i]

            self.data_loader.features_train = train_df

        if plot_pca:
            plt.figure(figsize=(8,5))
            plt.plot(cumulative_variance, marker='o')
            plt.axhline(y=explained_variance_threshold, color='r', linestyle='--')
            plt.title("Cumulative Explained Variance (PCA)")
            plt.xlabel("Number of Components")
            plt.ylabel("Variance Explained")
            plt.savefig(f"{self.output_folder}/pca_variance.png")
            plt.close()


    # ------------------------------------------------
    # FEATURE IMPORTANCE (MUTUAL INFORMATION)
    # ------------------------------------------------
    def relevant_feature_identification(self, num_features=10):

        X = self.data_loader.features_train
        y = self.data_loader.target_train

        mi_scores = mutual_info_classif(X, y)

        mi_df = pd.DataFrame({
            "feature": X.columns,
            "importance": mi_scores
        }).sort_values(by="importance", ascending=False)

        top_features = mi_df.head(num_features)

        print("\nTop relevant features:")
        print(top_features)

        plt.figure(figsize=(10,6))
        sns.barplot(x="importance", y="feature", data=top_features)

        plt.title("Top Feature Importance (Mutual Information)")
        plt.savefig(f"{self.output_folder}/feature_importance_MI.png")
        plt.close()

        return top_features["feature"].tolist()


#%% =====================================================
# FEATURE GENERATION
# =====================================================

class FlightDelayFeatureAnalysisAndGenerator(FlightDelayFeatureAnalysis):

    def __init__(self, data_loader):
        super().__init__(data_loader)


    # ------------------------------------------------
    # MAIN
    # ------------------------------------------------
    def generate_features(self):

        self.add_time_features()
        self.add_route_features()
        self.add_interaction_features()

        print("Feature generation completed.")


    # ------------------------------------------------
    # TIME FEATURES
    # ------------------------------------------------
    def add_time_features(self):

        df_train = self.data_loader.features_train
        df_test = self.data_loader.features_test

        for df in [df_train, df_test]:

            if "departure_hour" in df.columns:

                df["is_morning"] = df["departure_hour"].between(6, 12).astype(int)
                df["is_afternoon"] = df["departure_hour"].between(12, 18).astype(int)
                df["is_night"] = df["departure_hour"].between(18, 24).astype(int)

            if "DISTANCE" in df.columns and "CRS_ELAPSED_TIME" in df.columns:

                df["flight_speed_est"] = df["DISTANCE"] / (df["CRS_ELAPSED_TIME"] + 1)


    def add_route_features(self):

        df_train = self.data_loader.features_train
        df_test = self.data_loader.features_test

        if {"ORIGIN", "DEST"}.issubset(df_train.columns):

            df_train["route"] = (
                df_train["ORIGIN"].astype(str) + "_" +
                df_train["DEST"].astype(str)
            )

            df_test["route"] = (
                df_test["ORIGIN"].astype(str) + "_" +
                df_test["DEST"].astype(str)
            )

            route_freq = df_train["route"].value_counts()

            df_train["route_frequency"] = df_train["route"].map(route_freq)
            df_test["route_frequency"] = df_test["route"].map(route_freq).fillna(0)

    # ------------------------------------------------
    # INTERACTION FEATURES
    # ------------------------------------------------
    def add_interaction_features(self):

        df_train = self.data_loader.features_train
        df_test = self.data_loader.features_test

        features = ["DISTANCE", "departure_hour", "flight_speed_est"]

        features = [f for f in features if f in df_train.columns]

        for f1, f2 in combinations_with_replacement(features, 2):

            if f1 != f2:

                df_train[f"{f1}_x_{f2}"] = df_train[f1] * df_train[f2]
                df_test[f"{f1}_x_{f2}"] = df_test[f1] * df_test[f2]

        print("Interaction features created.")
#%% 
# FEATURE ANALYSIS PIPELINE
# ------------------------------------------------

feature_analysis = FlightDelayFeatureAnalysisAndGenerator(data_loader)

# PCA
feature_analysis.perform_pca()

# Generate new features
feature_analysis.generate_features()

# Feature importance
relevant_features = feature_analysis.relevant_feature_identification(
    len(data_loader.features_train.columns)
)


# Save dataset
with open("data_loader_with_new_features.pkl", "wb") as f:
    pickle.dump(data_loader, f)


# Load dataset
with open("data_loader_with_new_features.pkl", "rb") as f:
    data_loader_loaded = pickle.load(f)


print("Training data shape:", data_loader_loaded.features_train.shape)
print("Training target shape:", data_loader_loaded.target_train.shape)
print("Testing data shape:", data_loader_loaded.features_test.shape)
print("Testing target shape:", data_loader_loaded.target_test.shape)


#%% =====================================================
# HYPOTHESIS TESTING
# =====================================================

class FlightDelayHypothesisTesting:

    def __init__(self, data_loader):

        self.data_loader = data_loader


    # ------------------------------------------------
    # ANOVA TEST
    # ------------------------------------------------

    def _perform_anova_test(self, feature):

        X = self.data_loader.features_train
        y = self.data_loader.target_train

        data_groups = [
            X.loc[y == label, feature].dropna()
            for label in y.unique()
        ]

        if len(data_groups) < 2:
            return None, False

        f_statistic, p_value = f_oneway(*data_groups)

        significant = p_value < 0.05

        return p_value, significant


    # ------------------------------------------------
    # KRUSKAL TEST
    # ------------------------------------------------

    def _perform_kruskal_test(self, feature):

        X = self.data_loader.features_train
        y = self.data_loader.target_train

        data_groups = [
            X.loc[y == label, feature].dropna()
            for label in y.unique()
        ]

        if len(data_groups) < 2:
            return None, False

        h_statistic, p_value = kruskal(*data_groups)

        significant = p_value < 0.05

        return p_value, significant


    # ------------------------------------------------
    # ANOVA RESULTS
    # ------------------------------------------------

    def anova_results(self):

        print("\n==============================")
        print("ANOVA RESULTS")
        print("==============================")

        X = self.data_loader.features_train

        numeric_features = X.select_dtypes(
            include=["int64", "float64"]
        ).columns

        for feature in numeric_features:

            p_value, significant = self._perform_anova_test(feature)

            if p_value is None:
                continue

            print(
                f"Feature: {feature:25} | "
                f"p-value: {p_value:.6f} | "
                f"Significant: {significant}"
            )


    # ------------------------------------------------
    # KRUSKAL-WALLIS RESULTS
    # ------------------------------------------------

    def kruskal_wallis_results(self):

        print("\n==============================")
        print("KRUSKAL-WALLIS RESULTS")
        print("==============================")

        X = self.data_loader.features_train

        numeric_features = X.select_dtypes(
            include=["int64", "float64"]
        ).columns

        for feature in numeric_features:

            p_value, significant = self._perform_kruskal_test(feature)

            if p_value is None:
                continue

            print(
                f"Feature: {feature:25} | "
                f"p-value: {p_value:.6f} | "
                f"Significant: {significant}"
            )


    # ------------------------------------------------
    # T-TEST
    # ------------------------------------------------

    def t_test_results(self):

        print("\n==============================")
        print("T-TEST RESULTS")
        print("==============================")

        X = self.data_loader.features_train
        y = self.data_loader.target_train

        numeric_features = X.select_dtypes(
            include=["int64", "float64"]
        ).columns

        classes = y.unique()

        for feature in numeric_features:

            print(f"\nFeature: {feature}")

            for class1, class2 in itertools.combinations(classes, 2):

                data_class1 = X.loc[y == class1, feature].dropna()
                data_class2 = X.loc[y == class2, feature].dropna()

                if len(data_class1) < 2 or len(data_class2) < 2:
                    continue

                t_statistic, p_value = ttest_ind(
                    data_class1,
                    data_class2,
                    nan_policy="omit"
                )

                result = "Significant" if p_value < 0.05 else "Not Significant"

                print(
                    f"Class {class1} vs {class2} → {result} "
                    f"(p={p_value:.6f})"
                )


#%% =====================================================
# RUN HYPOTHESIS TESTING
# =====================================================

hypothesis_tester = FlightDelayHypothesisTesting(data_loader)

hypothesis_tester.anova_results()

hypothesis_tester.kruskal_wallis_results()

hypothesis_tester.t_test_results()