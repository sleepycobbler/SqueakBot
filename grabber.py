import random
import starter
import filer
import csv
import discord
import profanity

def message_check(message):
    pass

async def update(ctx):
    id = str(ctx.guild.id)
    channels = ctx.guild.text_channels
    data = filer.read()
    all_message_data = []
    if id not in data.keys():
        await starter.add_guild(id, data)
    for channel in channels:
        messages = await channel.history(limit=None).flatten() # this single line of code takes 99% longer than any other line here
        for message in messages:
            if len(message.content) > 0:
                if message.content[0] not in ['-', '!', '>'] and not message.content.startswith("toddbot") and not message.author.bot and "http" not in message.content:
                    all_message_data.append({"message_id": message.id,
                                             "author": message.author.name,
                                             "created_on": message.created_at,
                                             "edited_on": message.edited_at,
                                             "num_reactions": len(message.reactions),
                                             "is_pinned": message.pinned,
                                             "content": discord.utils.remove_markdown(message.clean_content).encode("ascii", "ignore")})
                    data[id].append(message.clean_content)
    filer.write(data, id)
    print('json done')
    try:
        with open('data.csv', 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=['message_id', 'author', 'created_on', 'edited_on', 'num_reactions', 'is_pinned', 'content'])
            writer.writeheader()
            for my_data in all_message_data:
                writer.writerow(my_data)
            print('csv done')
    except IOError:
        print("I/O error")

async def grabber(ctx, links=None):
    id = str(ctx.guild.id)
    data = filer.read()

    if id not in data.keys(): # if the guild hasn't used the bot before, we gotta add it to the list
        await starter.add_guild(ctx.guild.id, data)
        await update(ctx)

    message = random.choice(data[id])

    if links is not None:
        if links is True:
            while message == "" or not message.startswith('http'):
                message = random.choice(data[id])
        else:
            while message == "" or message.startswith('http'):
                message = random.choice(data[id])

    await ctx.channel.send(message)
