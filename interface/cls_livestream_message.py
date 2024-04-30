import json


class LivestreamMessage:
    def __init__(self, author: str, message: str, id: str = ""):
        self.author = author
        self.message = message
        self.id = id

    def to_dict(self):
        return {"author": self.author, "message": self.message, "id": self.id}

    def __str__(self):
        return f"Author: {self.author}, Message: {self.message}, ID: {self.id}"

    def __repr__(self):
        return f"Author: {self.author}, Message: {self.message}, ID: {self.id}\n"
        # return f"LivestreamMessage(author={self.author!r}, message={self.message!r}, id={self.id!r})"