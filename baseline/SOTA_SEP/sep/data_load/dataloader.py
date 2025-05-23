from summarize_module.summarizer import Summarizer
import os, json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm

class DataLoader:
    def __init__(self, args):
        self.price_dir = args.price_dir
        self.tweet_dir = args.tweet_dir
        self.seq_len = args.seq_len
        self.summarizer = Summarizer(args)
        # Initialize cache for summaries
        self.summary_cache = {}


    def daterange(self, start_date, end_date):
        for n in range(int((end_date - start_date).days)):
            yield start_date + timedelta(n)


    def get_sentiment(self, date_str, price_path):
        price_data = np.genfromtxt(price_path, dtype=str, skip_header=False)
        price_chg = price_data[price_data[:, 0] == date_str][0, 1].astype(float)

        if price_chg > 0.0:
            sentiment = "Positive"
        else:
            sentiment = "Negative"
        return sentiment


    def get_tweets(self, ticker, date_str):
        tweets = []
        tweet_path = os.path.join(self.tweet_dir, ticker, date_str)
        if os.path.exists(tweet_path):
            with open(tweet_path) as f:
                lines = f.readlines()
                for line in lines:
                    tweet_obj = json.loads(line)
                    tweets.append(tweet_obj['text'])
        return tweets


    def get_cached_summary(self, ticker, date_str, tweet_data):
        """Get summary from cache or generate new one if not cached"""
        cache_key = f"{ticker}_{date_str}"
        if cache_key not in self.summary_cache:
            summary = self.summarizer.get_summary(ticker, date_str, tweet_data)
            if summary and summary is not None and summary != "" and self.summarizer.is_informative(summary):
                self.summary_cache[cache_key] = summary
            else:
                self.summary_cache[cache_key] = None
        return self.summary_cache[cache_key]


    def load(self, flag):
        data = pd.DataFrame()
        stock_files = os.listdir(self.price_dir)
        
        with tqdm(total=len(stock_files), desc="Processing Stocks", position=0, leave=True) as outer_bar:
            for file in os.listdir(self.price_dir):
                price_path = os.path.join(self.price_dir, file)
                ordered_price_data = np.flip(np.genfromtxt(price_path, dtype=str, skip_header=False), 0)
                ticker = file[:-4]

                tes_idx = round(len(ordered_price_data) * 0.8)
                end_idx = len(ordered_price_data)

                if flag == "train":
                    # data_range = range(tes_idx)
                    data_range = range(14)
                else:
                    data_range = range(tes_idx, end_idx)

                with tqdm(total=len(data_range), desc=f"{ticker} Processing", position=1, leave=True) as inner_bar:
                    for idx in data_range:
                        summary_all = ""

                        end_date_str = ordered_price_data[idx, 0]
                        tqdm.write(f"End Date: {end_date_str}")
                        end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
                        start_date = end_date - timedelta(days=self.seq_len)
                        target = self.get_sentiment(end_date_str, price_path)
                        
                        for seq_date in self.daterange(start_date, end_date):
                            seq_date_str = seq_date.strftime("%Y-%m-%d")    

                            tweet_data = self.get_tweets(ticker, seq_date_str)
                            summary = self.get_cached_summary(ticker, seq_date_str, tweet_data)

                            if summary:
                                summary_all = summary_all + seq_date_str + "\n" + summary + "\n\n"

                        if summary_all != "":
                            data = pd.concat([data, pd.DataFrame([{'ticker': ticker, 'summary': summary_all.rstrip(), 'target': target}])], ignore_index=True)

                        inner_bar.update(1)
                outer_bar.update(1)

            return data
