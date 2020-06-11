import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import spacy
import random
import time


#reading file 
data = pd.read_csv("therapy_data.csv")
#Create a dataframe from the dataset and then print the first 5 rows
df = pd.DataFrame(data)

#cleaning data 
#'views', 'questionLink', 'Unnamed: 0', 'split', 'answerText', 'questionText'
df = df.drop(columns=['questionID', 'therapistInfo', 'therapistURL', 'upvotes', 'views', 'questionLink', 'Unnamed: 0'])
print(df.head())

y = df['topic']
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

vectorizer = CountVectorizer()
X_train = vectorizer.fit(X_train)
print(X_train)

#sort the dataframe by the "split" column
# this will group the rows in the order of "test", "train" , "val"
df.sort_values(by=['split'], inplace=True)
#print(df)

#Count number of rows with each "split" value
#This will allow us to split our test set into another dataframe
#It will give us numbers to see the quantity of our data
countVal = 0
countTest = 0
countTrain = 0
total = 0
for ind, row in df.iterrows():
    if row['split'] == 'test':
        countTest+=1
        total+=1
    elif row['split'] == 'val':
        countVal+=1
        total+=1
    elif row['split'] == 'train':
        countTrain+=1
        total+=1
        
print("Train: "+ str(countTrain) + " Test: " + str(countTest) + " Validation: " + str(countVal)+ " Total: " + str(total))

#Create a dataframe for the Test data only
test = df[:117]
test_data = test.drop(columns=['topic', 'split', 'questionText', 'answerText'])
test_data.head()

#Create a dataframe for the train data set
train = df[177:]
train_data = train.drop(columns=['split', 'questionText', 'answerText'])
train_data.head()

# Distribution of answers by topic
fig, ax = plt.subplots(figsize=(12, 8))
train_data.groupby("topic").agg("count")["questionTitle"].sort_values(ascending=False).plot.bar(ax=ax)
ax.set_title("Number of Responses by Topic", fontsize=25)
ax.set_xlabel("Topic", fontsize=25)
ax.set_ylabel("Number of Responses", fontsize=20)
ax.set_xticklabels(ax.get_xticklabels(), fontsize=17)
plt.tight_layout()

# select random question from train data set
nlp = spacy.load('en')

random_question = train_data['questionTitle'].sample()
question = random_question.iloc[0]

doc = nlp(question)

print(question)

for token in doc:
    print()
    print("{} : {}".format(token, token.vector[:3]))


#Create a list out of the 'questionTitle' and 'topic' series
question_list = train_data['questionTitle'].tolist()
topic_list = train_data['topic'].tolist()
#print(topic_list)

#Create a dictionary of responses

responses = {"question" : ("How do you feel about that?", "What would you like to talk about?", 
                        "Is that ok with you?", "Tell me more please", "Why?"), 
             "statement": ("You are not alone.", "I hear what you are saying.", "Tell me more about that."),
             "family-conflict": ("family-conflict"),
             "marriage": ("marriage"),
             "relationships": ("relationships"),
             "eating-disorders": ("eating-disorders"),
             "depression": ("depression"),
             "anxiety": ("anxiety"),
             "intimacy": ("intimacy"),
             "anger-management": ("anger-management"),
             "counseling-fundamentals": ("counseling-fundamentals"),
             "sleep-improvement": ("sleep-improvement"),
             "substance-abuse": ("substance-abuse"),
             "grief-and-loss": ("grief-and-loss"),
             "self-harm": ("self-harm"),
             "children-adolescents": ("children-adolescents"),
             "military-issues": ("military-issues"),
             "social-relationships": ("social-relationships"),
             "diagnosis": ("diagnosis"),
             "lgbtq": ("lgbtq"),
             "addiction": ("addiction"),
             "professional-ethics": ("professional-ethics"),
             "legal-regulatory": ("legal-regulatory"),
             "human-sexuality": ("human-sexuality"),
             "stress": ("stress"),
             "behavioral-change": ("behavioral-change"),
             "self-esteem": ("self-esteem"),
             "trauma": ("trauma"),
             "relationship-dissolution": ("relationship-dissolution"),
             "spirituality": ("spirituality"),
             "parenting": ("parenting"),
             "workplace-relationships": ("workplace-relationships"),
             "domestic-violence": ("domestic-violence"),
            }

# Create template for chat
bot_template = "Therapy Bot : {0}"
user_template = "User : {0} "

# Define respond function that takes a parameter 'message'
# bot's response to message

def respond(message):  
    print()
    if message.endswith("?"):
        # Return a random question
        return random.choice(responses["statement"])
    # Return a random statement
    return random.choice(responses["question"])
    #bot_message = random.choice(responses["statement"])
    #print()
    #return bot_message

# Test function
#print(respond(random_question))

# Define a function send_message that sends a message to the bot

def send_message(question):
    # Print user_template including the user_message
    #print(user_template.format(question))
    # Get the bot's response to the message
    response = respond(question)
    # Print the bot template including the bot's response.
    time.sleep(2)
    print(bot_template.format(response))

# Send a message to the bot
send_message(question)