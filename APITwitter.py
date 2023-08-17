import tweepy

# Configura tus credenciales de Twitter API
consumer_key = 'wNwO11gYwaKvGHu42S1gX3ytQ'
consumer_secret = '7Y8pE2aqRFdt8fMLB4NeDCsdVDD14RQF7jqhCsvlasjIfjefab'
access_token = '1691444196471939073-LaqZwU4xRFFHToliK4r7d2Vtn1DiCn'
access_token_secret = 'BhBzRpQr7bM6yNWmeyBfzaPCdsBdFhLFiJ9MZIlp7P0ce'


# Autenticación con la API de Twitter
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

user = api.get_user(screen_name='@pcl_predes')
print(user.screen_name)

# Recuperar tus propios tweets
num_tweets = 1
tweets = api.user_timeline(screen_name='pcl.proyectoredes', count=num_tweets)

