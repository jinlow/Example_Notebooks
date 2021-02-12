"""
Prep the application_train.csv dataset from the following kaggle competition:
https://www.kaggle.com/c/home-credit-default-risk

We are going to pretty naively just convert object columns with a low number of
numeric fields to numeric levels, and then just use everything for modeling.

"""
import pandas as pd
import numpy as np

df = pd.read_csv("../data/application_train.csv")

# I am going to convert all of the columns to lowercase, all uppercase
# column names annoy me, I don't want to allways have to hit the SHIFT
# key.
df.columns = [col.lower() for col in df.columns]

# Let's convert object columns with 10 or fewer levels, to numeric
# categories that we will model with.
cat_fields = (
    df.select_dtypes("object").nunique().pipe(lambda x: x[x.le(10)]).index.to_list()
)

# We will just add a group of fields with a "cat_" prefix.
df[[f"cat_{f}" for f in cat_fields]] = df[cat_fields].apply(
    lambda x: x.astype("category").cat.codes.replace({-1: np.nan})
)

# I want to use the field code_gender, as a test field for optimizing sample
# weights on, to make things cleaner, I will drop the 4 rows where this field
# is XNA. We will also use the field name_contract_type as one of our sample
# fields.
df = df[df["code_gender"].ne("XNA")].copy()

df.to_parquet("../data/application_train_proc.parquet")
