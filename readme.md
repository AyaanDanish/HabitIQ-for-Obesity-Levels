# Project Title
Ethical Exploriong of the Dataset for estimation of obesity levels.

## 1. Dataset Description
### Dataset Name:
Estimation of Obesity Levels Based On Eating Habits and Physical Condition
### Source / Link:
https://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition
### Domain / Context: 
Health and Medicine (specifically related to obesity estimation based on lifestyle factors)
### Number of Instances: 
2111
### Number of Features: 17 (including the target variable)
Possible Target Variable(s): NObesity (Obesity Level), which categorizes individuals into: Insufficient Weight, Normal Weight, Overweight Level I, Overweight Level II,Obesity Type I, Obesity Type II, Obesity Type III
### Data Access & License: 
The dataset is publicly available and can be accessed via the UCI repository. It is licensed under CC BY license.

### Other Information:
#### Countries Represented: 
Mexico, Peru, and Colombia
#### Article: 
https://www.sciencedirect.com/science/article/pii/S2352340919306985?via%3Dihub
#### Who created the dataset?
The dataset was created by a team of researchers (Fabio Mendoza Palechor, Alexis de la Hoz Manotas)
#### When was the dataset created?
The dataset was donated on 8/26/2019.
#### What does each attribute mean?
The dataset comprises 17 attributes, including: Eating habits, Physical condition, Demographic information The target variable: NObesity (Obesity Level), which categorizes individuals into:Insufficient Weight, Normal Weight, Overweight Level I, Overweight Level II,Obesity Type I, Obesity Type II, Obesity Type III
#### For what purpose was the dataset created?
The dataset was created to estimate obesity levels in individuals based on their eating habits and physical condition. It is suitable for tasks such as classification, regression, and clustering.
#### Are there missing values?
No
#### Does the dataset contain metadata, e.g., a README file?
The UCI repository entry provides a brief overview of the dataset but does not include a comprehensive README file or detailed metadata.
#### Short Description:
This dataset contains information on obesity levels based on eating habits and physical conditions, collected from individuals in Mexico, Peru, and Colombia. It includes 2111 instances and 17 features, with the target variable NObesity categorizing individuals into seven obesity levels. The dataset can be used for classification, regression, and clustering tasks and is publicly available under a CC BY license.

## 2. Decision-Making Scenario

## 3. Stakeholder Analysis
### Stakeholder list:
- Patient, user 
- Nutritionist or Trainer
- ML expert

### Stakeholder Patient, User:
#### Stakeholder Knowledge:
- Limited technical knowledge.
- Basic understanding of health and nutrition.
- Personal insight into their own habits and lifestyle.

#### Goals:
- To achieve a normal weight
- To become fit and healthy

#### Objectives:
- Be informed about their own current status.
- Be self-aware of the factors that affect obesity the most.
- Achieve a specific weight based on your demographics.
- Achieve nutrition goals based on recommendations, like eating more vegetables.
- Receive personalized recommendations on current habits to change in order to reach a normal weight.

#### Tasks:
- Prepare Complete information to the Nutritionist or Trainer.
- Follow the directions of the Nutritionist or Trainer.
-Attend follow-up sessions and track progress.

#### Key Questions:
- Is my data safe and private?
- Can I trust this system's recommendations?

### Stakeholder Nutritionist or Trainer:
#### Stakeholder Knowledge:
- Deep understanding of physical health and dietary science.
Goals:
- Easily use the model and give Smart suggestions.
- Understand the model well enough to interpret results confidently.

#### Objectives:
-Understand how the results are structured and be able to find flaws/errors of the model.
-Be aware of the model’s feature importance.
- Mix his own domain knowledge with the model recommendations and give a proper suggestion.
- Customize the model to the patients personal characteristics.

#### Tasks:
- Gather patients’ or users’ inputs and input them to the system.
- Combine his knowledge with the model and give proper suggestions.
- Personalize recommendations by his knowledge and model output.

#### Key Questions:
- How reliable is this recommendation?
- What features most influenced this suggestion?
- Can I override the system’s advice if I disagree?

### Stakeholder ML expert:
#### Stakeholder Knowledge:
-Strong technical understanding of ML/AI, algorithms, and model evaluation.
-Has limited domain knowledge in nutrition or physical fitness.

#### Goals:
- Create a robust, fair, and interpretable model
- Ensure the system has real-world impact and commercial value.

#### Objectives:
-Make an accurate model that can be well interpreted.
-Build a UI/UX friendly Model for non technical people.
- Make and sell the product.
- Gather feedback about performance of model in action.

#### Tasks:
- Train and Test the Model.
- Make a proper explanation and presentation for users.
- UI and UX design.
- Fine-tune the model based on feedback and design a mechanism for it.

#### Key Questions:
- Is the model biased toward certain groups?
- Which features most influence predictions?
- How should I communicate uncertainties to users?

### Authors
- Syed Ayaan Danish
- Abisola Ajuwon
- Niloufar Neshat 
- Tanya Ignatenko 
- Yi-Hui Fan

## Realistic decision-making scenario 
What real-world decision could be made with this dataset?
What kind of recommendations we should provide to a person to help them achieve a normal weight.

What are possible constraints & requirements?
1. If our recommendation system provides recommendations that can cause harm for person's health (eg. advicess to start smoking to achieve a normal weight)
2. Store securely user's data in accordance to GDPR
3. A value is required for all of the features to make a prediction
4. Our recommendation system should provide a prediction in a reasonable amount of time

What are the stakes?
High stakes, because we can provide recommendations that can cause harm to person's health because:
1. predict wrong predictions (because of model's errors).
2. recommendations are not tailored to a user.
3. autors of software and end users don't know the domain.
4. users could overact on recommendations (eg. don't eat at all, overexercising)

What ML model might be used? 
We can use tree-based models, because they're easily explainable as the trees could be plotted.

## Scenario Brief:
We could create a software that would provide recommendations for a user that would help them to achieve a normal weight. This software could be used by a nutritionist.

1. Brainstorm a set of possible Stakeholders (Be creative)
Patients/Users, Nutrionists/Trainers, A Decision maker on deployment of app in a hospital/gym, ML experts.


