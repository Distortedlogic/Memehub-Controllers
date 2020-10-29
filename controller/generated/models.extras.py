# RedditMeme
# @property
#     def serialize(self):
#         return {
#             "reddit_id": self.reddit_id,
#             "subreddit": self.subreddit,
#             "title": self.title,
#             "username": self.username,
#             "url": self.url,
#             "meme_text": self.meme_text,
#             "template": self.template,
#             "timestamp": self.timestamp,
#             "datetime": dump_datetime(self.datetime),
#             "upvote_ratio": self.upvote_ratio,
#             "upvotes": self.upvotes,
#             "downvotes": self.downvotes,
#             "num_comments": self.num_comments,
#         }

# RedditScore

# @property
#     def serialize(self):
#         return {
#             "username": self.username,
#             "subreddit": self.subreddit,
#             "timestamp": self.timestamp,
#             "datetime": dump_datetime(self.datetime),
#             "final_score": self.final_score,
#             "raw_score": self.raw_score,
#             "num_in_bottom": self.num_in_bottom,
#             "num_in_top": self.num_in_top,
#             "shitposter_index": self.shitposter_index,
#             "highest_upvotes": self.highest_upvotes,
#             "hu_score": self.hu_score,
#             "lowest_ratio": self.lowest_ratio,
#         }

#     @property
#     def stats(self):
#         return {
#             "username": self.username,
#             "final_score": self.final_score,
#             "num_in_bottom": self.num_in_bottom,
#             "num_in_top": self.num_in_top,
#             "shitposter_index": self.shitposter_index,
#             "highest_upvotes": self.highest_upvotes,
#             "hu_score": self.hu_score,
#             "lowest_ratio": self.lowest_ratio,
#         }


# follower = db.relationship(
#     "User", primaryjoin="Follow.followerId == User.id", backref="following"
# )
# following = db.relationship(
#     "User", primaryjoin="Follow.followingId == User.id", backref="followers"
# )
