import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression

class HealthAnalyzer:
    """
    En klass för att hantera datan som efterfrågas i notbooken.
    """
    
    def __init__(self, file_path):
        """
        Initerar analysen genom att ladda in datasetet i en dataframe
        """
        self.df = pd.read_csv(file_path)
        
        print(f"Data laddad! Rader: {len(self.df)}")

    def show_basic_info(self):
        """
        Skriver ut de fem första raderna för att se att allt ser ok ut.
        """
        return self.df.head()
    
    def plot_blood_pressure_distribution(self):
        """
        Ritar ett histogram över blodtrycket.
        """
        plt.figure(figsize=(8, 5))
        sns.histplot(self.df['systolic_bp'], kde=True) 
        plt.title("Fördelning av Systoliskt Blodtryck")
        plt.xlabel("Blodtryck (mmHg)")
        plt.show()

    def weight_defined_by_sex(self):
        """
        Ritar en boxplot över vikt per kön.
        """
        female_weight = self.df.loc[self.df["sex"] == "F", "weight"]
        male_weight   = self.df.loc[self.df["sex"] == "M", "weight"]

        fig, ax = plt.subplots(figsize=(6,5))
        ax.boxplot([female_weight, male_weight], tick_labels=["F", "M"], showmeans=True)
        ax.set_title("Boxplot över vikt per kön")
        ax.set_xlabel("Kön")
        ax.set_ylabel("Vikt (kg)")
        ax.grid(True, axis="y")
        plt.show()

    def percentage_of_smokers(self):
        """
        Ritar ut en graf som visar hur stor andelen i % som röker eller inte röker i studien.
        """
        smoker_counts = self.df["smoker"].value_counts()
        smoker_percent = smoker_counts / len(self.df) * 100

        fig, ax = plt.subplots(figsize=(6,5))
        smoker_percent.plot(kind="bar", color=["green", "red"], edgecolor="black")
        ax.set_title("Andel rökare vs icke-rökare (%)")
        ax.set_xlabel("Röker")
        ax.set_ylabel("Andel (%)")
        ax.grid(True, axis="y")
        plt.show()
        print(smoker_counts)

    def simulated_sickness_mean(self, n_sim=1000):
        """
        En simulation görs med en förutbestämd seed(42) och n_sim = 1000 personer (kan justeras)
        Detta skall visa på en lite större stickprovsstorlek och testa om
        mönstren från de observerade data håller i en större population.
        """
        np.random.seed(42)

        # Totalt antal personer i datasetet
        n_total = len(self.df)

        # Antal personer med sjukdomen
        n_disease = self.df['disease'].sum()

        # Verklig sannolikhet
        p_disease = n_disease / n_total
        print(f"Andel med sjukdomen i datasetet: {p_disease:.3f}")
        
        # Simulera 1000 personer: 1 = sjuk, 0 = frisk
        simulated = np.random.binomial(1, p_disease, n_sim)

        # Beräkna andelen i simuleringen
        simulated_mean = simulated.mean()
        print(f"Andel med sjukdomen i simuleringen: {simulated_mean:.3f}")

    def analyze_bp_bootstrap(self, B=5000, confidence=0.95):
        """
        Beräknar konfidensintervall med Bootstrap och ritar en graf.
        Antal iterationer (B) och konfidensgrad (confidence) kan justeras
        """
        data = self.df["systolic_bp"].dropna().values
        n = len(data)
            
        boot_means = np.empty(B)
        for b in range(B):
            boot_sample = np.random.choice(data, size=n, replace=True)
            boot_means[b] = np.mean(boot_sample)
            
        alpha = (1 - confidence) / 2 
        lo, hi = np.percentile(boot_means, [100*alpha, 100*(1-alpha)])
        mean_val = np.mean(data)
            
        print(f"--- Bootstrap-analys av Blodtryck ({B} iterationer) ---")
        print(f"Stickprovsmedelvärde: {mean_val:.2f} mmHg")
        print(f"95% Konfidensintervall: ({lo:.2f}, {hi:.2f}) mmHg")
            
        fig, ax = plt.subplots(figsize=(7, 3))
            
        ax.hist(boot_means, bins=30, edgecolor="black", alpha=0.7)   
        ax.axvline(mean_val, color="tab:green", linestyle="--", label="Stickprovsmedel")
        ax.axvline(lo, color="tab:red", linestyle="--", label="2.5%")
        ax.axvline(hi, color="tab:red", linestyle="--", label="97.5%") 
        ax.set_title("Bootstrap-fördelning av medel + 95% intervall")
        ax.set_xlabel("Resamplat medelvärde (mmHg)")
        ax.set_ylabel("Antal")
        ax.legend()
        ax.grid(True, axis="y")
        plt.show()

    def hypothesis_smokers(self):
        """
        Testar för att se om hypotesen att rökare har högre blodtryck än de som inte röker.
        Tar också ett Welch-test för att se hur varianserna ser ut.
        """

        clean_df = self.df.dropna(subset=["smoker", "systolic_bp"]).copy()

        smokers_bp = clean_df.loc[clean_df["smoker"] == "Yes", "systolic_bp"].values
        nonsmokers_bp = clean_df.loc[clean_df["smoker"] == "No", "systolic_bp"].values

        t_stat_w, p_val_w = stats.ttest_ind(smokers_bp, nonsmokers_bp , equal_var=False)
        print(f"Welch t-test: t = {t_stat_w:.3f}, p-värde = {p_val_w:.4f}")

        fig, ax = plt.subplots()
        ax.boxplot([smokers_bp, nonsmokers_bp], showmeans=True)
        ax.set_xticklabels(["Rökare", "Icke rökare"])
        ax.set_ylabel("Systoliskt bp (mmHg)")
        ax.set_title("Jämförelse av blodtryck mellan rökare och icke-rökare")
        ax.grid(True, axis="y")
        plt.show()

    def linear_regression(self):
        """
        Utför en linjär regression för att se linjen om hur åldern påverkar blodtrycket
        """

        x = self.df[["age"]].values
        y = self.df["systolic_bp"].values

        linreg = LinearRegression()
        linreg.fit(x, y)

        sk_intercept = float(linreg.intercept_)
        sk_slope = float(linreg.coef_[0])
        sk_r2 = float(linreg.score(x, y))
        sk_pred_12 = float(linreg.predict(np.array([[50.0]]))[0])

        print(f"""
            -- Linjär regression: --
            intercept = {sk_intercept:.2f}
            slope = {sk_slope:.2f}
            R² = {sk_r2:.3f}
            prognos vid 50 år = {sk_pred_12:.1f}
        """)

        plt.figure(figsize=(6,4))
        plt.scatter(x, y, alpha=0.6, label="Data", zorder=1)

        grid_x = np.linspace(0, 100, 200)
        grid_y_sklearn = sk_intercept + sk_slope * grid_x

        plt.plot(grid_x, grid_y_sklearn, color="tab:blue", linestyle="--", linewidth=2, label="Linjär regression", zorder=3)
        plt.xlabel("Ålder (år)")
        plt.ylabel("Systoliskt Blodtryck (mmHg)")
        plt.title("Ålder vs Blodtryck")
        plt.legend()
        plt.show()

    def analyze_bmi_disease(self):
        """
        Testar en jämförelse av BMI: Friska vs Sjuka
        """

        clean_df = self.df.dropna(subset=["disease", "weight", "height"]).copy()

        clean_df["bmi"] = clean_df["weight"] / ((clean_df["height"] / 100) ** 2)

        bmi_healthy = clean_df.loc[clean_df["disease"] == 0 , "bmi"].values
        bmi_sick = clean_df.loc[clean_df["disease"] == 1 , "bmi"].values


        print(f"""
            --- BMI-analys ---
            Snitt-BMI Friska: = {bmi_healthy.mean():.1f}
            Snitt-BMI Sjuka: = {bmi_sick.mean():.1f}
        """)

        fig, ax = plt.subplots()
        ax.boxplot([bmi_healthy, bmi_sick], showmeans=True)
        ax.set_xticklabels(["Friska", "Sjuka"])
        ax.set_ylabel("BMI")
        ax.set_title("Jämförelse av BMI: Friska vs Sjuka")
        ax.grid(True, axis="y")
        plt.show()