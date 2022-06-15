# Introduction 
The objective of this study is to showcase how to modernize Credit Risk Modeling (CRM) and mitigate the credit risk by implementing ML techniques to the IRB approach. This research assesses the current CRM situation by comparing the traditional approach and enhanced ML approaches. During the process, various modern ML techniques will be investigated and tuned to reach the most optimal prediction power. The model performance will be compared through evaluation metrics. In addition, the results of these models will be adopted to calculate the minimum capital requirements for banks. In doing so, the economical gains of applying different ML techniques could be estimated and compared. Thus, these models will be evaluated and validated in a statistical as well as an economical way.  

# Files in the repository
 `1. Data processing.ipynb` is the file where I cleaned, merged and processed data
 
 `2. Modeling (without sampling).ipynb` is the file where I trained models based on the training set without sampling. I also tested models' performance on the test set.
 
 `3. Modeling (with sampling).ipynb` is the file where I trained models based on the training set with sampling. I also tested models' performance on the test set and analysed feature importance through SHAP values.
 
 `4. Economical analysis.ipynb` is the file where I calculated economical gains of applying ML models compared with the baseline models
 
  To reproduce my results, please run the code in the above order.
  
  # Data needed to reproduce the results
  
  `compustat_financial_final.csv` is retrieved from the `Compustat Daily Updates - Fundamentals Annual`. The dataset downloaded from WRDS contains 355,890 observations for listed companies in North America in the period of June 1987 to February 2016.
  
  `gdp_growth_rate_north_america.csv` is the data on the growth of GDP in North America obtained from `World Bank`.
  
  `link.csv` is retrieved from `Roberts Dealscan-Compustat Linking Database`
  
  `delist_V2.csv` is retrieved from `CRSP Stock Events - Delisting Information (2022)`
  
  `rating_V2.csv` is retrieved from `Compustat Daily Updates - Ratings`. The data contains Standard & Poor’s rating from June 1989 to February 2017.
  
  `loan.csv` is retrieved from `WRDS-Reuters DealScan - Facility - Legacy`. The data contains registered loan information from June 1988 to February 2016.
  
  `interest.csv` is retrieved from `WRDS-Reuters DealScan - Current Facility Pricing - Legacy`.
  
  `dealscan_link.csv` is retrieved from `Roberts Dealscan-Compustat Linking Database`.
    
