from instagram.client import InstagramAPI
import indicoio
from collections import Counter
from facepp import API as face_api
import numpy as np
import calendar
import json, requests
import pprint
'''
client_id = "bec82b4b69cc435998eb2c9f82212fb4"
client_secret = "6f7cd017a78945afaffcd992840a8fe5"
access_token = "1147536024.bec82b4.fb48b565d9ad4fe09f64f63d64d4f664"
INDICO_API_KEY = '61cdd30af4bbdfe5a21b92689a872234'

FACE_API_KEY = '57a5f94a17c3bbf07823b6e4d06dde10'
FACE_API_SECRET = '1le1_mXDShAHhwG5OPjjjbpqpT14fEKG'


api = InstagramAPI(access_token=access_token, client_secret=client_secret)
face = face_api(FACE_API_KEY, FACE_API_SECRET)
indicoio.config.api_key = INDICO_API_KEY

jsonFile = {'data': {
				'posts': []
				}
			}


isPrivate = api.user_relationship(user_id='1147536024').target_user_is_private
if isPrivate == False:
	recent_media, next = api.user_recent_media(user_id='self', count=3)
	
	
	url = 'https://api.instagram.com/v1/users/self?access_token=%s' % access_token
	resp = requests.get(url=url)
	data = resp.json()
	follows = data['data']['counts']['follows']
	followers = data['data']['counts']['followed_by']

	for media in recent_media:

		image_url = media.get_standard_resolution_url()
		#print url
		faces = len(face.detection.detect(url=image_url)['face'])
		print faces
		if  media.type != 'video' and 1 <= faces <= 4:
			#links.append(url)
			day = media.created_time.weekday()
			hour = str(media.created_time.hour) + ':' + str(media.created_time.minute)
			likes = media.like_count
			hashtags = len(media.tags)

			captionSentiment = 0.5
			if media.caption != None:
				caption = media.caption.text.replace('\n', ' ').replace('\r', ' ').encode('utf-8')
				captionSentiment = indicoio.sentiment(caption)

			fer = indicoio.fer(image_url)
			print fer
			 
			new_post = {}
			new_post['happy'] = fer['Happy']
			new_post['sad'] = fer['Sad']
			new_post['angry'] = fer['Angry']
			new_post['fear'] = fer['Fear']
			new_post['surprise'] = fer['Surprise']
			new_post['neutral'] = fer['Neutral']
			new_post['day'] = day
			new_post['hour'] = hour
			new_post['likes'] = likes 
			new_post['follows'] = follows
			new_post['followers'] = followers
			new_post['hashtags'] = hashtags
			new_post['captionSentiment'] = captionSentiment
			new_post['likeRatio'] = float(likes)/followers
			new_post['followerRatio'] = float(followers)/follows
			jsonFile['data']['posts'].append(new_post)
'''

jsonFile = {'data': {
				'posts': []
				}
			}

with open('data.json', 'w') as outfile:
    json.dump(jsonFile, outfile, sort_keys=True,indent=4)
'''
with open('data.json') as infile:
	d = json.load(infile)
	pp = pprint.PrettyPrinter(indent=4)
	pprint.pprint(d)	
'''	


	