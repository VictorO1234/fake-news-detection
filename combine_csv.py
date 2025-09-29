import pandas as pd

# Load CSV files
fake = pd.read_csv('data/Fake.csv')
true = pd.read_csv('data/True.csv')

# Add label column
fake['label'] = 'FAKE'
true['label'] = 'REAL'

# Keep only title and text
fake = fake[['title', 'text', 'label']]
true = true[['title', 'text', 'label']]

# Combine title and text
fake['text'] = fake['title'] + ' ' + fake['text']
true['text'] = true['title'] + ' ' + true['text']

# Merge datasets
df = pd.concat([fake, true], ignore_index=True)

# Save as single CSV
df[['text','label']].to_csv('data/fake_news.csv', index=False)
print("Combined dataset saved as fake_news.csv")
