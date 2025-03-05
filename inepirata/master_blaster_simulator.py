import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import minimize
import seaborn as sns
from scipy.stats import f
import scipy.stats as stats


class DegreeAssignmentSimulator:
    def __init__(self, n_students, n_degrees, mean_enem=500, std_enem=100,normal_sd=1.35, selection_parameter=0.75, max_co_meso=20,simulation_type="FCFS"):
        if simulation_type not in ["FCFS", "Utility Function"]:
            raise ValueError("Invalid simulation_type. Must be 'FCFS' or 'Utility Function'.")
        
        self.n_students = n_students
        self.n_degrees = n_degrees
        self.mean_enem = mean_enem
        self.std_enem = std_enem
        self.normal_sd = normal_sd
        self.selection_parameter = selection_parameter
        self.random_percentage = 1 - selection_parameter
        self.simulation_type = simulation_type
        self.max_co_meso = max_co_meso

    def generate_students(self):
        np.random.seed(42)
        self.enem_scores = np.random.normal(loc=self.mean_enem, scale=self.std_enem, size=self.n_students)
        self.ages = self.generate_student_ages()
        self.races = self.generate_student_races()
        self.public_hs = self.generate_student_public_hs()
        self.female = np.random.choice([0, 1], size=self.n_students, p=[0.5, 0.5])  # Assuming a 50-50 distribution

        self.student_data = pd.DataFrame({
            'student_id': range(self.n_students),
            'enem_score': self.enem_scores,
            'age': self.ages,
            'nonwhite': self.races,
            'public_hs': self.public_hs,
            'female': self.female
        })

    def generate_student_ages(self):
        dfn = 4  # Degrees of freedom for the numerator
        dfd = 12  # Degrees of freedom for the denominator
        scale = 3  # Scale factor to adjust the range
        loc = 18  # Minimum age offset

        ages_f = f.rvs(dfn, dfd, size=self.n_students) * scale + loc
        out_of_range_mask = (ages_f < 18) | (ages_f > 45)
        ages_f[out_of_range_mask] = np.random.uniform(18, 45, size=out_of_range_mask.sum())
        return ages_f
    
    def generate_student_races(self):
        """
        Ensures that 56% of students are non-white.
        """
        self.enem_rank = self.enem_scores.argsort().argsort()  # Rank based on ENEM scores
        # Calculate probabilities for being non-white
        nonwhite_probs = (1 / (self.enem_rank + 1)) ** 0.8
        nonwhite_probs /= nonwhite_probs.sum()
        # Assign races ensuring 56% are non-white
        total_nonwhite = int(self.n_students * 0.56)
        races = np.zeros(self.n_students, dtype=int)  # Default: white (0)
        nonwhite_indices = np.random.choice(
            self.n_students, size=total_nonwhite, replace=False, p=nonwhite_probs
        )
        races[nonwhite_indices] = 1  # Non-white (1)
        return races

    def generate_student_public_hs(self):
        """
        Ensures that 58% of students are from public high schools.
        """
        enem_rank = self.enem_scores.argsort().argsort()  # Rank based on ENEM scores
        # Calculate probabilities for attending public high schools
        public_hs_probs = (1 / (enem_rank + 1)) ** 0.8
        public_hs_probs /= public_hs_probs.sum()
        # Assign public HS status ensuring 58% are from public HS
        total_public_hs = int(self.n_students * 0.58)
        public_hs = np.zeros(self.n_students, dtype=int)  # Default: private HS (0)
        public_hs_indices = np.random.choice(
            self.n_students, size=total_public_hs, replace=False, p=public_hs_probs
        )
        public_hs[public_hs_indices] = 1  # Public HS (1)
        return public_hs

    def calculate_value_added(self):
        degree_percentiles = np.linspace(0.01, 0.99, self.n_degrees)
        self.degrees = pd.DataFrame({
            'degree': [f'Degree_{i + 1}' for i in range(self.n_degrees)],
            'degree_value_added': stats.norm.ppf(degree_percentiles, loc=0, scale=1)+0.5
        })

        va_rank = self.degrees['degree_value_added'].rank(ascending=True)
        online_probs = ((1 / va_rank) ** 0.8) / ((1 / va_rank) ** 0.8).sum()
        private_probs = ((1 / va_rank) ** 0.8) / ((1 / va_rank) ** 0.8).sum()

        # Assign "online" status
        n_online = int(self.n_degrees * 0.15)
        self.degrees['online'] = 0
        online_indices = np.random.choice(self.degrees.index, size=n_online, replace=False, p=online_probs)
        self.degrees.loc[online_indices, 'online'] = 1

        # Assign "private" status
        n_private = int(self.n_degrees * 0.75)
        self.degrees['private'] = 0
        private_indices = np.random.choice(self.degrees.index, size=n_private, replace=False, p=private_probs)
        self.degrees.loc[private_indices, 'private'] = 1

        self.degrees['degree'] = self.degrees['degree'].str.lstrip('Degree_')


    def assign_degrees(self):
        self.calculate_value_added()

        if self.simulation_type=="FCFS":
            ranked_students = np.argsort(self.enem_scores)
            degree_assignments = np.zeros(self.n_students, dtype=int)
            batch_size = self.n_students // self.n_degrees

            for i in range(self.n_degrees):
                start_idx = i * batch_size
                end_idx = (i + 1) * batch_size
                degree_assignments[ranked_students[start_idx:end_idx]] = i

            n_random = int(self.n_students * self.random_percentage)
            np.random.seed(42)
            random_indices = np.random.choice(self.n_students, n_random, replace=False)
            degree_assignments[random_indices] = np.random.choice(self.n_degrees, size=n_random)

            self.student_data['degree_assignment'] = degree_assignments
            self.student_data['assigned_degree_value_added'] = self.degrees.loc[degree_assignments, 'degree_value_added'].values

        if self.simulation_type=="Utility Function":
            def calculate_utility(enem_score, degree_va):
                deterministic_part = degree_va * (-0.5 + 0.001 * enem_score) * np.exp(-degree_va)
                random_part = np.random.gumbel(0, 0.1)
                return deterministic_part + random_part

            utilities = []
            assignments = []

            np.random.seed(42)
            
            for _, student in self.student_data.iterrows():
                student_utilities = [
                    calculate_utility(student['enem_score'], va)
                    for va in self.degrees['degree_value_added']
                ]
                probabilities = np.exp(student_utilities) / np.sum(np.exp(student_utilities))
                assigned_degree = np.random.choice(self.degrees.index, p=probabilities)
                utilities.append(max(student_utilities))  # Keep only the max utility value
                assignments.append(assigned_degree)

            self.student_data['degree_assignment'] = assignments
            self.student_data['assigned_degree_value_added'] = self.degrees.loc[assignments, 'degree_value_added'].values

        self.student_data['degree_assignment'] = self.student_data['degree_assignment'].add(1)


    def assign_degrees_fcfs(self):
        self.calculate_value_added()
        ranked_students = np.argsort(self.enem_scores)
        degree_assignments = np.zeros(self.n_students, dtype=int)

        base_batch_size = self.n_students // self.n_degrees  # Floor division
        remainder = self.n_students % self.n_degrees  # Students left after even distribution

        start_idx = 0
        for i in range(self.n_degrees):
            # Allocate extra student to the first 'remainder' degrees
            batch_size = base_batch_size + (1 if i < remainder else 0)
            end_idx = start_idx + batch_size
            degree_assignments[ranked_students[start_idx:end_idx]] = i
            start_idx = end_idx  # Move to the next batch

        # Random reassignment
        self.n_random = int(self.n_students * self.random_percentage)
        np.random.seed(42)
        random_indices = np.random.choice(self.n_students, self.n_random, replace=False)
        degree_assignments[random_indices] = np.random.choice(self.n_degrees, size=self.n_random)

        self.student_data['degree_assignment_fcfs'] = degree_assignments
        self.student_data['degree_value_added_fcfs'] = self.degrees.loc[degree_assignments, 'degree_value_added'].values
        self.student_data['degree_assignment_fcfs'] = self.student_data['degree_assignment_fcfs'].add(1)


    def assign_degrees_utility(self):
        self.calculate_value_added()

        def calculate_utility(enem_score, degree_va):
            deterministic_part = degree_va * (-0.5 + 0.001 * enem_score) * np.exp(-degree_va)
            random_part = np.random.gumbel(0, 0.1)
            return deterministic_part + random_part

        utilities = []
        assignments = []

        np.random.seed(42)

        for _, student in self.student_data.iterrows():
            student_utilities = [
                calculate_utility(student['enem_score'], va)
                for va in self.degrees['degree_value_added']
            ]
            probabilities = np.exp(student_utilities) / np.sum(np.exp(student_utilities))
            assigned_degree = np.random.choice(self.degrees.index, p=probabilities)
            utilities.append(max(student_utilities))  # Keep only the max utility value
            assignments.append(assigned_degree)

        self.student_data['degree_assignment_utility'] = assignments
        self.student_data['degree_value_added_utility'] = self.degrees.loc[
            assignments, 'degree_value_added'
        ].values
        self.student_data['degree_assignment_utility'] = self.student_data['degree_assignment_utility'].add(1)

    def assign_degrees_utility2(self):
        self.calculate_value_added()

        def calculate_utility(enem_score, degree_va):
            deterministic_part = -0.3 * degree_va + 0.001 * (enem_score * degree_va)
            random_part = np.random.gumbel(0, 0.1)
            return deterministic_part + random_part

        utilities = []
        assignments = []

        # Initialize a counter for assigned students per degree
        degree_counts = {degree: 0 for degree in self.degrees.index}
        max_students_per_degree = len(self.student_data) // len(self.degrees)

        for _, student in self.student_data.iterrows():
            # Calculate utility for each degree
            student_utilities = [
                calculate_utility(student['enem_score'], va)
                for va in self.degrees['degree_value_added']
            ]

            # Normalize utilities to ensure fairness across degrees
            normalized_utilities = np.array(student_utilities)
            normalized_utilities -= normalized_utilities.mean()  # Center utilities
            probabilities = np.exp(normalized_utilities) / np.sum(np.exp(normalized_utilities))

            # Assign degree based on normalized probabilities
            assigned_degree = np.random.choice(
                self.degrees.index, p=probabilities
            )

            # Ensure balanced assignment by checking counts
            while degree_counts[assigned_degree] >= max_students_per_degree:
                probabilities[assigned_degree] = 0  # Exclude oversubscribed degree
                probabilities /= probabilities.sum()  # Re-normalize probabilities
                assigned_degree = np.random.choice(self.degrees.index, p=probabilities)

            utilities.append(max(student_utilities))  # Keep max utility for each student
            assignments.append(assigned_degree)

            # Update degree count
            degree_counts[assigned_degree] += 1

        # Update student data with assignments and utilities
        self.student_data['degree_assignment_utility'] = assignments
        self.student_data['degree_value_added_utility'] = self.degrees.loc[
            assignments, 'degree_value_added'
        ].values
        self.student_data['degree_assignment_utility'] = self.student_data['degree_assignment_utility'].add(1)

    def assign_degree_characteristicsOLD(self):
        n_online = int(self.n_degrees * 0.2)
        self.online_probs = np.ones(self.n_degrees) / self.n_degrees  # Uniform probability for simplicity
        online_indices = np.random.choice(self.degrees.index, size=n_online, replace=False, p=self.online_probs)
        self.degrees['online'] = 0
        self.degrees.loc[online_indices, 'online'] = 1

        n_private = int(self.n_degrees * 0.7)
        private_probs = np.ones(self.n_degrees) / self.n_degrees  # Uniform probability for simplicity
        private_indices = np.random.choice(self.degrees.index, size=n_private, replace=False, p=private_probs)
        self.degrees['private'] = 0
        self.degrees.loc[private_indices, 'private'] = 1
        self.degrees['degree'] = self.degrees['degree'].lstrip('Degree_')


    def assign_co_meso(self,online_multiple_co_meso=True):
        np.random.seed(42)  # Para reprodutibilidade

        # Definir os possíveis co_meso (regiões) disponíveis
        self.degrees['co_meso'] = -1  # Inicializa com um placeholder
        available_meso = np.arange(self.max_co_meso)  # Lista de regiões possíveis

        # Para cursos presenciais: atribuir um único co_meso para cada curso
        in_person_degrees = self.degrees[self.degrees['online'] == 0].index
        assigned_meso = np.random.choice(available_meso, size=len(in_person_degrees), replace=True)
        self.degrees.loc[in_person_degrees, 'co_meso'] = assigned_meso

        # Para cursos online: cada curso pode estar em múltiplas regiões
        online_degrees = self.degrees[self.degrees['online'] == 1].index
        max_online_regions = min(self.max_co_meso, 5)  # Até 5 regiões por curso online
        online_meso_assignments = {}

        for degree in online_degrees:
            num_regions = np.random.randint(1, max_online_regions + 1)  # Escolhe de 1 a 5 regiões
            regions = np.random.choice(available_meso, size=num_regions, replace=False)
            online_meso_assignments[degree] = regions

        # Expandir o dataset para refletir que cursos online podem estar em várias regiões
        expanded_degrees = []
        if online_multiple_co_meso:
            for _, row in self.degrees.iterrows():
                if row['online'] == 1:
                    for region in online_meso_assignments[row.name]:
                        new_row = row.copy()
                        new_row['co_meso'] = region
                        expanded_degrees.append(new_row)
                else:
                    expanded_degrees.append(row)
        else:
            for _, row in self.degrees.iterrows():
                expanded_degrees.append(row)

        self.degrees = pd.DataFrame(expanded_degrees).reset_index(drop=True)

        # Criar efeitos fixos para cada co_meso
        self.co_meso_fixed_effects = pd.DataFrame({
            'co_meso': np.arange(self.max_co_meso),  # Criar efeitos para todas as regiões possíveis
            'co_meso_fe': np.random.normal(0, 0.8, size=self.max_co_meso)  # Efeitos fixos normais
        })

        # Juntar os efeitos fixos no dataset de degrees
        self.degrees = self.degrees.merge(self.co_meso_fixed_effects, on='co_meso', how='left')

        # Garantir que co_meso seja do tipo inteiro
        self.degrees['co_meso'] = self.degrees['co_meso'].astype(int)


    def generate_students_without_degrees(self):
        # Generate students who do not have degrees
        n_no_degree_students = int(self.n_students * 0.1)  # Let's assume 10% of students have no degrees
        
        np.random.seed(42)  # Ensure reproducibility
        enem_scores_no_degree = np.random.normal(loc=self.mean_enem-50, scale=self.std_enem, size=n_no_degree_students)
        ages_no_degree = self.generate_student_ages()[:n_no_degree_students]
        races_no_degree = self.generate_student_races()[:n_no_degree_students]
        public_hs_no_degree = self.generate_student_public_hs()[:n_no_degree_students]
        female_no_degree = np.random.choice([0, 1], size=n_no_degree_students, p=[0.5, 0.5])  # Assuming a 50-50 distribution

        students_no_degree = pd.DataFrame({
            'student_id': range(self.n_students, self.n_students + n_no_degree_students),
            'enem_score': enem_scores_no_degree,
            'age': ages_no_degree,
            'nonwhite': races_no_degree,
            'public_hs': public_hs_no_degree,
            'female': female_no_degree,
            'degree_assignment_fcfs': 0,
            'degree_value_added_fcfs': 0,
            'degree_assignment_utility': 0,
            'degree_value_added_utility': 0,
            'degree_assignment': 0,
            'assigned_degree_value_added':0
        })

        # Add these students to the main student_data DataFrame
        self.student_data = pd.concat([self.student_data, students_no_degree], ignore_index=True)

    def merge_degree_characteristics(self):
        self.student_data.degree_assignment = self.student_data.degree_assignment.astype(int)
        self.student_data.degree_assignment_fcfs = self.student_data.degree_assignment_fcfs.astype(int)
        self.student_data.degree_assignment_utility = self.student_data.degree_assignment_utility.astype(int)
        self.degrees.degree = self.degrees.degree.astype(int)

        np.random.seed(42)  # Ensuring reproducibility

        # Helper function: For each student, randomly select one matching row from self.degrees
        def random_merge(student_df, degree_col):
            merged_df = student_df.copy()

            # Merge with degrees, but assign a random row per student-degree match
            merged_df = merged_df.merge(
                self.degrees,
                how="left",
                left_on=degree_col,
                right_on="degree"
            )

            # Group by student ID and sample one row per student
            merged_df = merged_df.groupby("student_id").apply(lambda x: x.sample(n=1)).reset_index(drop=True)

            return merged_df

        # First Merge: FCFS
        self.final_df = random_merge(self.student_data, "degree_assignment_fcfs").rename(
            columns={"online": "online_fcfs", "private": "private_fcfs"}
        ).drop(columns=["degree", "degree_value_added"])

        # Second Merge: Utility-Based
        self.final_df = random_merge(self.final_df, "degree_assignment_utility").rename(
            columns={"online": "online_utility", "private": "private_utility"}
        ).drop(columns=["degree", "degree_value_added"])

        # Third Merge: General Assignment
        self.final_df = random_merge(self.final_df, "degree_assignment").drop(columns=["degree", "degree_value_added"])

        self.final_df[["online_fcfs",
                       "online_utility",
                       "private_fcfs",
                       "private_utility",
                       "online",
                       "private"]] = self.final_df[["online_fcfs",
                                                    "online_utility",
                                                    "private_fcfs",
                                                    "private_utility",
                                                    "online",
                                                    "private"]].fillna(0)



    def assign_co_meso_no_degree(self):
        np.random.seed(42)  # For reproducibility
        
        # Identify individuals without a degree
        no_degree_mask = self.final_df['degree_assignment'] == 0
        
        # Identify missing co_meso or co_meso_fe values
        missing_meso_mask = self.final_df['co_meso'].isna() | self.final_df['co_meso_fe'].isna()
        
        # Filter out those who both lack a degree and have missing region info
        individuals_to_assign = self.final_df[no_degree_mask & missing_meso_mask].index
        
        # Get available co_meso and co_meso_fe pairs from degrees dataset
        valid_meso_pairs = self.degrees[['co_meso', 'co_meso_fe']].drop_duplicates().values
        
        # Assign a random pair to each individual needing it
        random_assignments = np.random.choice(len(valid_meso_pairs), size=len(individuals_to_assign))
        
        self.final_df.loc[individuals_to_assign, ['co_meso', 'co_meso_fe']] = valid_meso_pairs[random_assignments]
        self.final_df['co_meso'] = self.final_df['co_meso'].astype(int)

    def calculate_income(self):
        """
        Calculate income for students based on their characteristics and degree value-added.
        """
        beta = [2.8, 0.80, 0.01, -0.06]  # Coefficients for log_income
        
        # Calculate log(age)
        self.final_df['log_age'] = np.log(self.final_df['age'])

        #np.random.seed(42)
        # Calculate log_income
        self.final_df['log_income'] = (
            beta[0] +
            beta[1] * self.final_df['log_age'] +
            beta[2] * self.final_df['enem_score'] +
            beta[3] * self.final_df['female'] +
            self.final_df['assigned_degree_value_added'] +
            self.final_df['co_meso_fe'] +
            np.random.normal(0, self.normal_sd, size=len(self.final_df))
        )

        # Calculate income
        self.final_df['income'] = np.exp(self.final_df['log_income'])

    def calculate_income_no_degree(self):
        """
        Calculate income for students without degrees, assuming lower average income.
        """
        n_no_degree = self.final_df[self.final_df['degree_assignment_fcfs'] == 0].shape[0]

        # Select rows corresponding to no-degree students
        no_degree_data = self.final_df[self.final_df['degree_assignment_fcfs'] == 0].copy()

        beta = [2.8, 0.80, 0.01, -0.06]  # Coefficients for log_income
        
        # Calculate log(age) for no-degree students
        no_degree_data['log_age'] = np.log(no_degree_data['age'])

        #np.random.seed(42)
        # Assign log_income for no-degree group
        no_degree_data['log_income'] = (
            beta[0] +
            beta[1] * no_degree_data['log_age'] +
            beta[2] * no_degree_data['enem_score'] +
            beta[3] * no_degree_data['female'] +
            no_degree_data['co_meso_fe'] +
            np.random.normal(0, self.normal_sd, size=n_no_degree)  # Worse income on average
        )

        # Calculate income for no-degree group
        no_degree_data['income'] = np.exp(no_degree_data['log_income'])

        # Merge back into the main DataFrame
        self.final_df.loc[no_degree_data.index, ['log_income', 'income']] = no_degree_data[['log_income', 'income']]

    def visualize_income_distribution(self):
        """
        Visualize the income distribution of students with and without degrees.
        """
        plt.figure(figsize=(10, 6))
        plt.hist(self.final_df['income'], bins=50, alpha=0.7, label='With Degree')
        plt.hist(
            self.final_df[self.final_df['degree_assignment_fcfs'] == 0]['income'],
            bins=50,
            alpha=0.7,
            label='Without Degree',
            color='orange'
        )
        plt.xlabel('Income')
        plt.ylabel('Frequency')
        plt.title('Income Distribution by Degree Status')
        plt.legend()
        plt.show()

    def visualize_results(self):
        plt.figure(figsize=(10, 6))
        plt.scatter(self.student_data.enem_score, self.student_data.degree_value_added_fcfs, alpha=0.8, s=0.2, cmap='winter')
        plt.xlabel('ENEM Score')
        plt.ylabel('Degree Value Added (FCFS)')
        plt.title('First-Come-First-Served Degree Assignment')
        #plt.colorbar(label='Degree')
        plt.show()

        plt.figure(figsize=(10, 6))
        plt.scatter(self.student_data.enem_score, self.student_data.degree_value_added_utility, alpha=0.8, s=0.2, cmap='winter')
        # Add trendline
        z = np.polyfit(self.student_data.enem_score, self.student_data.degree_value_added_utility, 1)
        p = np.poly1d(z)
        plt.plot(self.student_data.enem_score, p(self.student_data.enem_score), color='red', linewidth=0.3, linestyle='-.', label='Trendline')
        plt.xlabel('ENEM Score')
        plt.ylabel('Degree Value Added (Utility-Based)')
        plt.title('Utility-Based Degree Assignment')
        #plt.colorbar(label='Degree Value Added')
        plt.legend()
        plt.show()
    
    def assign_degrees_fcfs2(self):
        self.calculate_value_added()
        ranked_students = np.argsort(self.enem_scores)
        degree_assignments = np.zeros(self.n_students, dtype=int)
        batch_size = self.n_students // self.n_degrees

        for i in range(self.n_degrees):
            start_idx = i * batch_size
            end_idx = (i + 1) * batch_size
            degree_assignments[ranked_students[start_idx:end_idx]] = i

        self.n_random = int(self.n_students * self.random_percentage)
        np.random.seed(42)
        random_indices = np.random.choice(self.n_students, self.n_random, replace=False)
        degree_assignments[random_indices] = np.random.choice(self.n_degrees, size=self.n_random)

        self.student_data['degree_assignment_fcfs'] = degree_assignments
        self.student_data['degree_value_added_fcfs'] = self.degrees.loc[degree_assignments, 'degree_value_added'].values
        self.student_data['degree_assignment_fcfs'] = self.student_data['degree_assignment_fcfs'].add(1)


    def merge_degree_characteristicsOLD(self):
        self.student_data.degree_assignment = self.student_data.degree_assignment.astype(int)
        self.student_data.degree_assignment_fcfs = self.student_data.degree_assignment_fcfs.astype(int)
        self.student_data.degree_assignment_utility = self.student_data.degree_assignment_utility.astype(int)
        self.degrees.degree = self.degrees.degree.astype(int)

        self.final_df = pd.merge(self.student_data,
                                 self.degrees,
                                 how="left",
                                 left_on="degree_assignment_fcfs",
                                 right_on="degree").drop(
                                     columns=["degree","degree_value_added"]
                                     ).rename(columns={"online":"online_fcfs",
                                                       "private":"private_fcfs"})
        
        self.final_df = pd.merge(self.final_df,
                                 self.degrees,
                                 how="left",
                                 left_on="degree_assignment_utility",
                                 right_on="degree").drop(
                                     columns=["degree","degree_value_added"]
                                     ).rename(columns={"online":"online_utility",
                                                       "private":"private_utility"})

        self.final_df = pd.merge(self.final_df,
                                 self.degrees,
                                 how="left",
                                 left_on="degree_assignment",
                                 right_on="degree").drop(
                                     columns=["degree","degree_value_added"]
                                     )
        
        self.final_df[["online_fcfs",
                       "online_utility",
                       "private_fcfs",
                       "private_utility",
                       "online",
                       "private"]] = self.final_df[["online_fcfs",
                                                    "online_utility",
                                                    "private_fcfs",
                                                    "private_utility",
                                                    "online",
                                                    "private"]].fillna(0)