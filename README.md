# Introduction 
The objective of this study is to showcase how to modernize Credit Risk Modeling (CRM) and mitigate the credit risk by implementing ML techniques to the IRB approach. This research assesses the current CRM situation by comparing the traditional approach and enhanced ML approaches. During the process, various modern ML techniques will be investigated and tuned to reach the most optimal prediction power. The model performance will be compared through evaluation metrics. In addition, the results of these models will be adopted to calculate the minimum capital requirements for banks. In doing so, the economical gains of applying different ML techniques could be estimated and compared. Thus, these models will be evaluated and validated in a statistical as well as an economical way.  

# Files in the repository
 `1. data processing.ipynb` is the file where I cleaned, merged and processed data
 
 `2. modelling (without sampling).ipynb` is the file where I trained models based on the training set without sampling. I also tested models' performance on the test set.
 
 `3. modelling (with sampling).ipynb` is the file where I trained models based on the training set with sampling. I also tested models' performance on the test set and analysed feature importance through SHAP values.
 
 `4. economical analysis.ipynb` is the file where I calculated economical gains of applying ML models compared with the baseline models
 
  To reproduce my results, please run the code in the above order.