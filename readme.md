# Project Title
Ethical Exploriong of the Dataset for estimation of obesity levels.
## Description
This project works on a dataset that estimatas the obesity levels based on eating habits and physical condition in individuals from Colombia, Peru and Mexico.
## Installation
To be filled 

## Dataset details
Source: UCI Machine Learning Repository(https://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition)

Article: https://www.sciencedirect.com/science/article/pii/S2352340919306985?via%3Dihub

Number of Instances: 2,111

Number of Attributes: 17 (including the target variable)

Countries Represented: Mexico, Peru, and Colombia

Who created the dataset?
The dataset was created by a team of researchers(Fabio Mendoza Palechor, Alexis de la Hoz Manotas)
When was the dataset created? dataset has been Donated on 8/26/2019.
Who paid for the dataset creation?There is no information available regarding the funding or sponsorship for the creation of this dataset.
What does each attribute mean?
The dataset comprises 17 attributes, including:Eating habits, Physical condition, Demographic information
The target variable: NObesity (Obesity Level), which categorizes individuals into:Insufficient Weight, Normal Weight, Overweight Level I, Overweight Level II,Obesity Type I, Obesity Type II, Obesity Type III

For what purpose was the dataset created?
The dataset was created to estimate obesity levels in individuals based on their eating habits and physical condition. It is suitable for tasks such as classification, regression, and clustering.

How is the data sampled? Does the sampling make sense?
The dataset includes 2,111 instances from three countries. However, the sampling strategy and representativeness are not discussed in the provided information.

Are there missing values? no

Does the dataset contain metadata, e.g., a README file?The UCI repository entry provides a brief overview of the dataset but does not include a comprehensive README file or detailed metadata.


## Potential Questions to answer

1. ⁠Does physical activity affect obesity levels?
2. ⁠Is one gender more susceptible to obesity than the other?
3. ⁠To what extent is obesity influenced by genetics?
4. ⁠How is obesity prevalence related to age?
5. ⁠⁠Is there a relationship between alcohol consumption and obesity rates?
6. ⁠Does the quantity of water consumed daily have an effect on obesity levels?
7. ⁠How do meal characteristics such as calorie content, frequency, and snacking between meals affect obesity risk?
8. Is eating frequently high-calorie food strongly correlate with obesity?
9. Do people who are doing regularly physical activities have less obesity levels than people leading a sedentary lifestyle?
10. What factors predict obesity the most?
11. Do people with a family history with obesity have higher level of obesity than people without with similar diet and exercise?
12. Are older people more likely have obesity than younger people with similar diet and exercise?
13. What factors predict a normal weight the most?
14. ⁠Do different factors predict normal weight in men and women?
15. Do different factors predict obesity in men and women?

## Results
...

## Technologies
- Python

## Authors

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


