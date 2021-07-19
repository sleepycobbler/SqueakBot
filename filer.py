import json

def read():
    with open("./squeak.json", "r+") as f:
        return json.load(f)

def write(data, id):
    with open("./squeak.json", "r+") as f:
        f.write(json.dumps(data))
        f.truncate(0) # Delete Everything
        f.seek(0) # Set cursor to correct place
        f.flush() # Delete Everything again?
        data[id] = list(dict.fromkeys(data[id])) # get rid of all duplicate messages
        f.write(json.dumps(data)) # Profit (stored in thing)