# **Inepacote**  

This package provides two main classes for generating synthetic or simulated data that integrates higher education and national exam information with long-term labor market outcomes.  

## **Installation**

Not currently in Pypi, can be installed by running pip install linking directly to this github repository:

```python
!pip install git+https://github.com/enricoruggieri/inepacote.git
```

## **Classes**  

### **1. SyntheticDataSampler**  
Throught the use of Gaussian copulas, generates synthetic data by sampling from chunks of real data distributions, preserving key statistical properties.  

#### **Example Usage:**  
```python
from inepirata.gaussian_copula_synthetizer import SyntheticDataSampler

sampler = SyntheticDataSampler()
sampled_data = sampler.sample(500)
```
### **2. DegreeAssignmentSimulator**  
Simulates the process of student degree selection, allocation, and its long-term impact on labor market outcomes. It includes different degree assignment mechanisms and allows for the incorporation of co-meso effects.  

#### **Example Usage:**  
```python
from inepirata.master_blaster_simulator import DegreeAssignmentSimulator

# Initialize the simulator
simulator = DegreeAssignmentSimulator(
    n_students=2000,       # Number of students
    n_degrees=350,         # Number of degrees available
    normal_sd=2,           # Standard deviation for normal distribution
    simulation_type="Utility Function",  # Type of degree assignment model
    max_co_meso=20         # Maximum number of co-meso assignments
)

# Simulate the student cohort and degree assignment process
simulator.generate_students()
simulator.assign_degrees()  
simulator.assign_degrees_fcfs()  # First-Come, First-Served degree allocation  
simulator.assign_degrees_utility()  # Utility-based degree allocation  

# Simulate co-meso effects (regional influences)
simulator.assign_co_meso(online_multiple_co_meso=True)  

# Generate students who did not complete a degree
simulator.generate_students_without_degrees()  

# Merge degree characteristics with assigned students
simulator.merge_degree_characteristics()  

simulator.assign_co_meso_no_degree()  
simulator.calculate_income()
simulator.calculate_income_no_degree()
