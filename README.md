# Fitness Tracker Based on ML

Imagine yourself at the gym, pushing through your strength training routine, but wouldn’t it be amazing if you could ditch the expensive personal trainer and instead have a smart companion right on your wrist? Picture a device that not only tracks every move you make during your weightlifting sessions but also counts your repetitions, ensuring you stay on track with your fitness goals. That’s precisely the problem we’re tackling: the lack of automated tracking systems for free weight exercises, particularly in the realm of strength training.

[Project Web site](https://aem-fitness-app.onrender.com/)

[Project Medium Article](https://medium.com/@monaanafe3/smart-strength-training-tracker-ef6fc68c5fda)


## Project Video Demonstration


https://github.com/Veto2922/Fitness-tracker-based-on-ML-2/assets/73383924/7008bb17-113c-43cd-9c31-26f7f75c0c4d


## Project Block Diagram
Below is a block diagram illustrating the architecture of the Fitness Tracker Based on ML:
![block diagram](https://github.com/Veto2922/Fitness-tracker-based-on-ML-2/assets/73383924/e8990d07-a6f0-4d1e-93ea-8d5902014141)



## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Contributing](#contributing)
- [Contact Information](#contact-information)

## Installation

To get started with the project, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Veto2922/Fitness-tracker-based-on-ML-2.git
   cd Fitness-tracker-based-on-ML-2
2. **Set up a virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
3. **Install the required packages**:
   ```bash
   pip install -r requirements.txt

## Usage
To use the project:
1. **Start the Flask application**:
   ```bash
   flask run
2. **Use the web interface**:
   - Open your web browser and go to `http://localhost:5000.`
   - Input the accelerometer and gyroscope files.
   - Get the predicted exercise and repetitions count.
  
   

## Features
- **Automated Exercise Tracking**: Tracks various barbell exercises during strength training sessions, eliminating the need for expensive personal trainers.
- **Repetition Counting**: Accurately counts repetitions, helping users stay on track with their fitness goals.
- **Exercise Detection**: Identifies different exercises being performed using accelerometer and gyroscope data.
- **Machine Learning Integration**: Utilizes advanced machine learning models built through a comprehensive lifecycle, including data cleaning, preprocessing, feature engineering, model selection, and evaluation.
- **Superior Performance**: Achieves better performance compared to existing solutions, such as the work done by Dave Ebbelaar in his master's degree project.
- **Deployment and MLOps**: Implements modern deployment strategies using Flask and MLOps tools like MLflow, along with continuous integration and continuous delivery (CI/CD) practices.

## Contributing
We welcome contributions to the Fitness Tracker Based on ML project! To contribute, you can:
1. **Create a Pull Request**:
    - Fork the repository.
    - Create a new branch (`git checkout -b feature-branch`).
    - Commit your changes (`git commit -m 'Add some feature`').
    - Push to the branch (`git push origin feature-branch`).
    - Open a pull request.

2. **Open an Issue**:
    - If you find a bug or have a feature request, please open an issue on the GitHub repository.

## Contact Information
For further questions or support, you can contact the team through LinkedIn:
- [Abdelrahman Mohamed](https://www.linkedin.com/in/abdelrahman-mohamed-28649120b/)
- [Esmail Essam](https://www.linkedin.com/in/esmail-essam/)
- [Mona Ahmed Nafea](https://www.linkedin.com/in/monanaf3/)



## Project Organization

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io

---

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
