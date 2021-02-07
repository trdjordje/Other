pip install pandas
pip install mlxtend

dataset = [['Milk', 'Eggs', 'Bread'],
		   ['Milk', 'Eggs'],
		   ['Milk', 'Bread'],
		   ['Eggs', 'Apple']]

print(dataset)

#Output:
#[['Milk', 'Eggs', 'Bread'], ['Milk', 'Eggs'], ['Milk', 'Bread'], ['Eggs', 'Apple']]

import pandas as pd
from mlxtend.preprocessing import TransactionEncoder

te = TransactionEncoder()
te_array = te.fit(dataset).transform(dataset)
df = pd.DataFrame(te_array, columns=te.columns_)

print(df)

# Output:
#
#    Apple  Bread   Eggs   Milk
# 0  False   True   True   True
# 1  False  False   True   True
# 2  False   True  False   True
# 3   True  False   True  False

from mlxtend.frequent_patterns import apriori

frequent_itemsets_ap = apriori(df, min_support=0.01, use_colnames=True)

# First, we import the apriori algorithm function from the library.
# Then we apply the algorithm to our data to extract the itemsets that have a minimum support value of 0.01 
# (this parameter can be changed).

# Let’s take a look at the result:

print(frequent_itemsets_ap)

# Output:

   # support             itemsets
# 0     0.25              (Apple)
# 1     0.50              (Bread)
# 2     0.75               (Eggs)
# 3     0.75               (Milk)
# 4     0.25        (Eggs, Apple)
# 5     0.25        (Eggs, Bread)
# 6     0.50        (Bread, Milk)
# 7     0.50         (Eggs, Milk)
# 8     0.25  (Eggs, Bread, Milk)

from mlxtend.frequent_patterns import fpgrowth

frequent_itemsets_fp=fpgrowth(df, min_support=0.01, use_colnames=True)

# First, we import the F-P growth algorithm function from the library.
# Then we apply the algorithm to our data to extract the itemsets that have a minimum support value of 0.01
# (this parameter can be tuned on a case-by-case basis).

print(frequent_itemsets_fp)

# Output:

   # support             itemsets
# 0     0.75               (Milk)
# 1     0.75               (Eggs)
# 2     0.50              (Bread)
# 3     0.25              (Apple)
# 4     0.50         (Eggs, Milk)
# 5     0.50        (Bread, Milk)
# 6     0.25        (Eggs, Bread)
# 7     0.25  (Eggs, Bread, Milk)
# 8     0.25        (Eggs, Apple)


# Note: what you observe is regardless of the technique you used, you arrived at the identical itemsets and 
# support values. The only difference is the order in which they appear. You should notice that the output
# from F-P Growth appears in descending orders, hence the proof of what we mentioned in the theoretical 
# part about this algorithm.


from mlxtend.frequent_patterns import association_rules

rules_ap = association_rules(frequent_itemsets_ap, metric="confidence", min_threshold=0.8)
rules_fp = association_rules(frequent_itemsets_fp, metric="confidence", min_threshold=0.8)

# First we import the required function from the page to determine the association rules for a given dataset using some set of parameters.
# Then we apply it to the two frequent item datasets which we created in Step 3.

# Note: “metric” and “min_threshold” parameters can be tuned on a case-by-case basis, depending on the business problem requirements.

# Let’s take a look at both sets of rules:

print(rules_ap)

# Output:

     # antecedents consequents  antecedent support  consequent support  support  confidence      lift  leverage  conviction
# 0        (Bread)      (Milk)                0.50                0.75     0.50         1.0  1.333333    0.1250         inf
# 1  (Eggs, Bread)      (Milk)                0.25                0.75     0.25         1.0  1.333333    0.0625         inf
# 2        (Apple)      (Eggs)                0.25                0.75     0.25         1.0  1.333333    0.0625         inf

# From the two above we see that both algorithms found identical association rules with same coefficients, just presented in a different order.