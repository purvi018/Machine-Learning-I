from bs4 import BeautifulSoup as bs
import os
import requests
import re
import sys
import pandas as pd


class YelpRestaurantScraper:
    def __clean(self, text):
        rep = {"<br>": "\n", "<br/>": "\n", "<li>": "\n"}
        rep = dict((re.escape(k), v) for k, v in rep.items())
        pattern = re.compile("|".join(rep.keys()))
        text = pattern.sub(lambda m: rep[re.escape(m.group(0))], text)
        text = re.sub('<(.*?)>', '', text)
        return text

    def __validate(self, url):
        if not re.match(r'https?://www.yelp.ca/', url):
            print('Please enter a valid website, or make sure it is a yelp article')
            sys.exit(1)

    def __get_review_info(self, review_tag):
        reviewer_name = None
        rating = None
        text = None

        reviewer_tags = review_tag.find_all('a', 'css-166la90')
        if len(reviewer_tags) > 0:
            reviewer_name = reviewer_tags[0].string

        rating_tags = review_tag.find_all('div', 'i-stars__373c0___sZu0')
        if len(rating_tags) > 0:
            rating = rating_tags[0]['aria-label'][0]

        text_tags = review_tag.find_all('span', 'raw__373c0__tQAx6')
        if len(text_tags) > 0:
            text_string = text_tags[0].text
            if text_string:
                text = self.__clean(text_string)

        return [reviewer_name, rating, text]

    def __construct_file_name(self, name, number):
        if name and number:
            return f'{name} ({number} reviews)'
        if name:
            return name
        if number:
            return f'Restaurant {number} reviews'
        return 'Restaurant'

    def scrape(self, url):
        self.__validate(url)

        res = requests.get(url)

        res.raise_for_status()
        soup = bs(res.text, 'html.parser')

        restaurant_name = None
        number_of_reviews = None

        titles = soup.find_all('h1', 'css-m7s7xv')
        if len(titles) > 0:
            restaurant_name = titles[0].string

        number_reviews_tags = soup.find_all('span', 'css-bq71j2')
        number_reviews_tags = list(filter(lambda tag: tag.string.__contains__(' reviews'), number_reviews_tags))
        if len(number_reviews_tags) > 0:
            number_of_reviews = number_reviews_tags[0].string.replace(' reviews', '')

        review_tags = soup.find_all('div', 'review__373c0__3MsBX')
        reviews_info = list(map(self.__get_review_info, review_tags))

        df = pd.DataFrame(reviews_info, columns=['reviewer', 'rating', 'text'])
        df.to_csv(self.__construct_file_name(restaurant_name, number_of_reviews))
        return restaurant_name,number_of_reviews,df


url = 'https://www.yelp.ca/biz/pai-northern-thai-kitchen-toronto-5?osq=Restaurants'

scraper = YelpRestaurantScraper()
df = scraper.scrape(url)
print(df)

