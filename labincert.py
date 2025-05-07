import pandas as pd
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter
from wordcloud import WordCloud
import re


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')

df = pd.read_csv("spam_ham.csv", sep=";", encoding="windows-1252", names=["label", "message"], header=0)

df['label'] = df['label'].str.strip()
df = df[df['label'].isin(['ham', 'spam'])]  
df.dropna(subset=["message"], inplace=True)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text)  
    tokens = word_tokenize(text)  
    tokens = [word.lower() for word in tokens]  
    tokens = [word for word in tokens if word not in stop_words]  
    tokens = [lemmatizer.lemmatize(word) for word in tokens]  
    return tokens


df['tokens'] = df['message'].apply(preprocess_text)
df['clean_text'] = df['tokens'].apply(lambda x: ' '.join(x))
df['length'] = df['tokens'].apply(len)

print("Cantidad total de mensajes:", len(df))
print("\nProporción de clases:")
print(df['label'].value_counts(normalize=True))

plt.figure(figsize=(10, 5))
sns.kdeplot(df[df['label'] == 'spam']['length'], label='Spam', shade=True)
sns.kdeplot(df[df['label'] == 'ham']['length'], label='Ham', shade=True)
plt.title("Densidad de longitud de mensajes (post-preprocesamiento)")
plt.xlabel("Cantidad de palabras")
plt.legend()
plt.show()

def get_top_words(tokens_list, top_n=20):
    all_words = [word for tokens in tokens_list for word in tokens]
    return Counter(all_words).most_common(top_n)

spam_words = get_top_words(df[df['label'] == 'spam']['tokens'])
ham_words = get_top_words(df[df['label'] == 'ham']['tokens'])

spam_df = pd.DataFrame(spam_words, columns=['word', 'count'])
plt.figure(figsize=(10, 5))
sns.barplot(data=spam_df, x='count', y='word', palette='Reds_r')
plt.title("Top 20 palabras más frecuentes - SPAM")
plt.xlabel("Frecuencia")
plt.ylabel("Palabra")
plt.show()

ham_df = pd.DataFrame(ham_words, columns=['word', 'count'])
plt.figure(figsize=(10, 5))
sns.barplot(data=ham_df, x='count', y='word', palette='Blues_r')
plt.title("Top 20 palabras más frecuentes - HAM")
plt.xlabel("Frecuencia")
plt.ylabel("Palabra")
plt.show()

print("\nPalabras más frecuentes en SPAM:")
print(spam_words)

print("\nPalabras más frecuentes en HAM:")
print(ham_words)

spam_text = ' '.join(df[df['label'] == 'spam']['clean_text'])
ham_text = ' '.join(df[df['label'] == 'ham']['clean_text'])

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(WordCloud(width=600, height=400, background_color='white').generate(spam_text))
plt.title("WordCloud - SPAM")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(WordCloud(width=600, height=400, background_color='white').generate(ham_text))
plt.title("WordCloud - HAM")
plt.axis("off")

plt.tight_layout()
plt.show()
