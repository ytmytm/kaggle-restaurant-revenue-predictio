
Kaggle Restaurant Revenue Prediction solution
============================================================
Maciej Witkowiak

#Method
Due to small amount of train data I used public leaderboard for model validation, hence all data was used for training.

After brief data exploration it was clear that taking the logarithm of revenue (predicted value) generated nice gaussian curve.

I used lasso for feature selection and random forest for actual model.
The model had clear advantages when compared to my other attempts: it used a small subset of features but performed the best.

The only improvement I could make was to add log(days-since-restaurant-open) as a predictor instead of
just using the raw number.

#Notes
[Competition on Kaggle](https://www.kaggle.com/c/restaurant-revenue-prediction)

* This code ranked 54/2257 in private leaderboard, this earned "top 10%" badge for me
* This code ranked 110/2257 in public leaderboard
