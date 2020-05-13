# Food Demand Challange
Hackathon at Analytics Vidhya, reachable at https://datahack.analyticsvidhya.com/contest/genpact-machine-learning-hackathon-1/.

I contested under nickname Rekmark, finished 193rd out of 1137 contestans. I set deadline for this hackathon for three weeks in order to test myself in "real" work conditions.

# Problem Statement
Your client is a meal delivery company which operates in multiple cities. They have various fulfillment centers in these cities for dispatching meal orders to their customers. The client wants you to help these centers with demand forecasting for upcoming weeks so that these centers will plan the stock of raw materials accordingly.

The replenishment of majority of raw materials is done on weekly basis and since the raw material is perishable, the procurement planning is of utmost importance. Secondly, staffing of the centers is also one area wherein accurate demand forecasts are really helpful. Given the following information, the task is to predict the demand for the next 10 weeks (Weeks: 146-155) for the center-meal combinations in the test set:  

* Historical data of demand for a product-center combination (Weeks: 1 to 145)
* Product(Meal) features such as category, sub-category, current price and discount
* Information for fulfillment center like center area, city information etc.

# Solution
I tried couple of solutions. I inspected time dependency and tried simple time-series models. Nevertheless it prudeced high prediction error due to valuable informations contained in other variables. This forced me to transform this task to regression problem. In order to be able to do this i had to create some synthetic variables, cummulative mean and shift from number of orders. 

I decided to go with exhaustive solution of creating model for each center separately. My idea was that this procedure will capture different variability of orders for centers. From theoretical standpoint of view it seemed that XGBoost models are the most suitable for this job because of their fundamental principals. This solution was much better than very simple TS forecasting. 

After that I have tried evaluate couple of neural networks against XGBoost models, trained on same dataset and tested on another one. DNNs proved like more accurate models but their training and predicting has been consuming significant amount of time so eventualy i have ran out of a time. In my final solution I use DNNs for centers and one "common model" wich is learned on unique meals. It was neccessary because of new meals introduced at some centers and therefore they hasnt been in train set.

My takeaway from this project is that i should schedule timetable with more time-reserve. Next steps could be better synthetic features to transform time dependant problem to regression problem, hyperparameters tunning, in depth error analysis, usage of combinated predict from different types of ML models.
