import pymongo


def get_performers_table():
    client = pymongo.MongoClient()
    return client.thrash_drums_library.performers


def get_albums_table():
    client = pymongo.MongoClient()
    return client.thrash_drums_library.albums


def get_songs_table():
    client = pymongo.MongoClient()
    return client.thrash_drums_library.songs