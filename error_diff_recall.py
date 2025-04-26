import pandas as pd

# Extract clean and worst recall
bert_clean_recall = {"World": 0.94, "Sports": 0.98, "Business": 0.88, "Sci/Tech": 0.93}
bert_worst_recall = {"World": 0.89, "Sports": 0.90, "Business": 0.60, "Sci/Tech": 0.43}

roberta_clean_recall = {"World": 0.94, "Sports": 0.99, "Business": 0.89, "Sci/Tech": 0.92}
roberta_worst_recall = {"World": 0.92, "Sports": 0.92, "Business": 0.62, "Sci/Tech": 0.44}

logreg_clean_recall = {"World": 0.88, "Sports": 0.97, "Business": 0.86, "Sci/Tech": 0.89}
logreg_worst_recall = {"World": 0.94, "Sports": 0.90, "Business": 0.59, "Sci/Tech": 0.41}

# calculate recall drop
def calculate_drop(clean, worst):
    return {cls: round(((clean[cls] - worst[cls]) / clean[cls]) * 100, 2) for cls in clean}

bert_drop = calculate_drop(bert_clean_recall, bert_worst_recall)
roberta_drop = calculate_drop(roberta_clean_recall, roberta_worst_recall)
logreg_drop = calculate_drop(logreg_clean_recall, logreg_worst_recall)

# To DataFrame
drop_df = pd.DataFrame({
    "BERT Recall Drop (%)": bert_drop,
    "RoBERTa Recall Drop (%)": roberta_drop,
    "Logistic Regression Recall Drop (%)": logreg_drop
})

import ace_tools as tools; tools.display_dataframe_to_user(name="Correct Recall Drop per Class for Each Model", dataframe=drop_df)

print(drop_df)
