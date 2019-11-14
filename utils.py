
def open_by_tweets(path,sep="<user>"):
    """
    Open txt file and output list of strings based on the separator, also remove '\n'
    """
    with open(path,"r") as file:
        train_pos = file.read().replace('\n','').split(sep)
    return train_pos