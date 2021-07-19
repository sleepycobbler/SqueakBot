import random
import starter
import filer

def message_check(message):
    pass

async def update(ctx):
    id = str(ctx.guild.id)
    channels = ctx.guild.text_channels
    data = filer.read()
    if id not in data.keys():
        await starter.add_guild(id, data)
    for channel in channels:
        messages = await channel.history(limit=None).flatten() # this single line of code takes 99% longer than any other line here
        for message in messages:
            if len(message.content) > 0:
                if message.content[0] not in ['-', '!', '>'] and not message.content.startswith("toddbot"):
                    data[id].append(message.content + '\n')
    filer.write(data, id)

async def grabber(ctx, links=None):
    id = str(ctx.guild.id)
    data = filer.read()

    if id not in data.keys(): # if the guild hasn't used the bot before, we gotta add it to the list
        starter.add_guild(ctx.guild.id, data)
        update(ctx)

    message = random.choice(data[id])

    if links is not None:
        if links is True:
            while message == "" or not message.startswith('http'):
                message = random.choice(data[id])
        else:
            while message == "" or message.startswith('http'):
                message = random.choice(data[id])

    await ctx.channel.send(message)
