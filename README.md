PSO Petrol Pump Queueing Simulator

## Project Overview

This project is a **web-based queueing simulation system** developed to analyze the performance of a **PSO petrol pump** using **queueing theory and discrete-event simulation**.

The simulator models vehicle arrivals, service at fuel dispensers, queue formation, and system performance using **standard queueing models**.
It also includes **statistical validation (Chi-square test)** to justify model assumptions, as required in academic simulation projects.

---

## Objectives

* Model a real-world petrol pump using queueing theory
* Identify the correct queueing model using data
* Validate assumptions using **Chi-square goodness-of-fit test**
* Measure system performance through simulation
* Provide visual and numerical outputs for analysis

---

## Queueing Models Supported (9)

The simulator supports **all standard queueing models**:

### Markovian Arrival & Service

* **M/M/1**
* **M/M/2**
* **M/M/c**

### Markovian Arrival, General Service

* **M/G/1**
* **M/G/2**
* **M/G/c**

### General Arrival & General Service

* **G/G/1**
* **G/G/2**
* **G/G/c**

> In this project, based on data analysis, the **PSO petrol pump is identified as an M/M/2 system**.

---

## Data Description

The simulator works in two modes:

### 1️⃣ Data-Driven Mode (CSV)

Uses real observed data.

**Required CSV columns:**

* `Arrival_min` → arrival time of vehicles
* `ServiceDuration_min` → fueling time

This data is used to:

* Estimate λ (arrival rate)
* Estimate μ (service rate)
* Perform Chi-square tests
* Run simulation

---

### 2️Rate-Based Mode (λ, μ)

Uses user-defined parameters:

* Arrival rate λ (Poisson arrivals)
* Service rate μ (Exponential service)

Used for **what-if analysis and comparison**.

---

## Statistical Validation (Chi-Square Test)

The project applies **Chi-square goodness-of-fit tests** to validate assumptions:

### Inter-arrival Times

* **H₀:** Inter-arrival times follow Exponential distribution
* Result: *Fail to reject H₀*
  ✔️ Poisson arrivals accepted

### Service Times

* **H₀:** Service times follow Exponential distribution
* Result: *Fail to reject H₀*
  ✔️ Exponential service accepted

This confirms the **M/M/2 model selection** statistically.

---

## Simulation Engine

The simulator is built using:

* **Python**
* **Discrete Event Simulation (DES)**
* **FIFO queue discipline**

The engine:

* Generates arrivals and service times
* Assigns customers to servers
* Tracks waiting time, service time, and completion
* Computes performance metrics

---

## Outputs & Visualizations

### Key Performance Indicators (KPIs)

* Average Waiting Time
* Average Service Time
* Average Turnaround Time
* Average Queue Length (Lq)
* Average System Length (L)
* Throughput

### Graphs

* Waiting Time Histogram
* Service Time Histogram
* Inter-arrival Time Histogram
* Queue Length Over Time
* **Service Timeline (Gantt Chart)**

---

## Interpretation

* Histograms are **right-skewed**, consistent with exponential distributions
* Low waiting time indicates efficient service
* Server utilization shows balanced workload
* Gantt chart visualizes parallel service at multiple dispensers

---

## Project Structure

```
pso_queue_sim/
│
├── PSO_general_streamlit_app.py   # Main Streamlit web app
├── engine.py                      # Simulation engine (DES logic)
├── processes.py                   # Arrival & service distributions
├── metrics.py                     # KPI calculations & dataframes
├── disciplines.py                 # FIFO queue discipline
├── README.md                      # Project documentation
└── data/
    └── PSO_full_200_dataset.csv   # Sample dataset
```

---

##  How to Run the Project

### 1️⃣ Install required libraries

```bash
pip install streamlit numpy pandas matplotlib plotly scipy
```

### 2️⃣ Run the simulator

```bash
streamlit run PSO_general_streamlit_app.py
```

### 3️⃣ Open browser

```
http://localhost:8501
```

---

## Conclusion

This project successfully demonstrates:

* Practical application of queueing theory
* Statistical validation using Chi-square tests
* Discrete-event simulation for real-world systems
* Performance analysis of a petrol pump

The simulator provides a **decision-support tool** for analyzing and improving service systems.

---

## Academic Use

This project fulfills requirements for:

* Simulation & Modeling courses
* Operations Research
* Industrial Engineering
* Data-driven system analysis

